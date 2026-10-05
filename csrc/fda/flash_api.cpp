/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 * Copyright (c) 2026, fast-diffusion-attention authors.
 *
 * Host API of the diffusion attention kernels, registered as the `fda` torch library.
 ******************************************************************************/

#include <Python.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <torch/all.h>

#include <cutlass/numeric_types.h>

#include "flash.h"
#include "static_switch.h"
#include "tile_size.h"
#include "cuda_check.h"


extern "C" {
/* Creates a dummy empty _C module that can be imported from Python.
    The import from Python will load the .so consisting of this file
    in this extension, so that the TORCH_LIBRARY static initializers
    below are run. */
PyObject* PyInit__C(void)
{
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        "_C",   /* name of module */
        NULL,   /* module documentation, may be NULL */
        -1,     /* size of per-interpreter state of the module,
                    or -1 if the module keeps state in global variables. */
        NULL,   /* methods */
    };
    return PyModule_Create(&module_def);
}
}

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

namespace {

inline at::cuda::CUDAGuard make_cuda_guard_from_tensor(const at::Tensor& t) {
  return at::cuda::CUDAGuard(static_cast<c10::DeviceIndex>(t.get_device()));
}

inline int round_multiple(int x, int m) { return (x + m - 1) / m * m; }

inline int round_up_headdim(int head_size) {
    #ifndef FLASHATTENTION_DISABLE_HDIM64
    if (head_size <= 64) { return 64; }
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM96
    if (head_size <= 96) { return 96; }
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM128
    if (head_size <= 128) { return 128; }
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM192
    if (head_size <= 192) { return 192; }
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM256
    if (head_size <= 256) { return 256; }
    #endif
    return 256;
}

inline int get_max_headdim() {
    #ifndef FLASHATTENTION_DISABLE_HDIM256
    return 256;
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM192
    return 192;
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM128
    return 128;
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM96
    return 96;
    #endif
    #ifndef FLASHATTENTION_DISABLE_HDIM64
    return 64;
    #endif
    return 0;
}

// Common checks of q, k, v and of the varlen arguments. Returns (batch_size, seqlen, total).
std::tuple<int, int, int> check_inputs(at::Tensor const& q, at::Tensor const& k, at::Tensor const& v,
                                        std::optional<at::Tensor> const& cu_seqlens_,
                                        std::optional<int64_t> const& max_seqlen_,
                                        int64_t block_size) {
    auto dprops = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(dprops->major == 9, "fast-diffusion-attention only supports Hopper GPUs (sm90)");
    TORCH_CHECK(k.scalar_type() == q.scalar_type(), "query and key must have the same dtype");
    TORCH_CHECK(v.scalar_type() == q.scalar_type(), "query and value must have the same dtype");
    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);
    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(block_size >= 1, "block_size must be a positive integer");

    bool const is_varlen = cu_seqlens_.has_value();
    int batch_size, seqlen, total;
    int const num_heads = q.size(-2);
    int const head_size = q.size(-1);
    int const num_heads_k = k.size(-2);
    if (!is_varlen) {
        TORCH_CHECK(q.dim() == 4, "q must have shape (batch_size, seqlen, num_heads, head_size)");
        batch_size = q.size(0);
        seqlen = q.size(1);
        total = batch_size * seqlen;
        TORCH_CHECK(seqlen % 2 == 0, "The sequence length must be even: [noisy half | clean half]");
        CHECK_SHAPE(q, batch_size, seqlen, num_heads, head_size);
        CHECK_SHAPE(k, batch_size, seqlen, num_heads_k, head_size);
        CHECK_SHAPE(v, batch_size, seqlen, num_heads_k, head_size);
    } else {
        at::Tensor cu_seqlens = cu_seqlens_.value();
        CHECK_DEVICE(cu_seqlens); CHECK_CONTIGUOUS(cu_seqlens);
        TORCH_CHECK(cu_seqlens.dtype() == torch::kInt32, "cu_seqlens must have dtype torch.int32");
        TORCH_CHECK(max_seqlen_.has_value(), "max_seqlen must be provided if cu_seqlens is provided");
        TORCH_CHECK(q.dim() == 3, "q must have shape (total, num_heads, head_size) if cu_seqlens is provided");
        batch_size = cu_seqlens.size(0) - 1;
        seqlen = max_seqlen_.value();
        total = q.size(0);
        CHECK_SHAPE(q, total, num_heads, head_size);
        CHECK_SHAPE(k, total, num_heads_k, head_size);
        CHECK_SHAPE(v, total, num_heads_k, head_size);
        CHECK_SHAPE(cu_seqlens, batch_size + 1);
    }
    TORCH_CHECK(head_size <= get_max_headdim(), "fast-diffusion-attention supports head dimension at most " + std::to_string(get_max_headdim()));
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    int const alignment = q.scalar_type() == torch::kFloat8_e4m3fn ? 16 : 8;
    TORCH_CHECK(head_size % alignment == 0, "head_size should be a multiple of " + std::to_string(alignment));
    return {batch_size, seqlen, total};
}

void set_params_fprop(Flash_fwd_params &params,
                      const int b, const int seqlen, const int total,
                      const int h, const int h_k, const int d, const int d_rounded,
                      const at::Tensor q, const at::Tensor k, const at::Tensor v, at::Tensor out,
                      void *cu_seqlens_d,
                      void *softmax_lse_d,
                      float softmax_scale,
                      int block_size,
                      const float softcap=0.f,
                      const int sm_margin=0) {

    // Reset the parameters
    params = {};

    params.is_bf16 = q.dtype() == torch::kBFloat16;
    params.is_e4m3 = q.dtype() == torch::kFloat8_e4m3fn;

    // Set the pointers and strides.
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    // All stride are in elements, not bytes.
    params.q_row_stride = q.stride(-3);
    params.k_row_stride = k.stride(-3);
    params.v_row_stride = v.stride(-3);
    params.q_head_stride = q.stride(-2);
    params.k_head_stride = k.stride(-2);
    params.v_head_stride = v.stride(-2);
    params.o_ptr = out.data_ptr();
    params.o_row_stride = out.stride(-3);
    params.o_head_stride = out.stride(-2);

    if (cu_seqlens_d == nullptr) {
        params.q_batch_stride = q.stride(0);
        params.k_batch_stride = k.stride(0);
        params.v_batch_stride = v.stride(0);
        params.o_batch_stride = out.stride(0);
    }

    params.cu_seqlens = static_cast<int *>(cu_seqlens_d);

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.seqlen = seqlen;
    params.total = total;
    params.d = d;
    params.d_rounded = d_rounded;

    // Set the different scale values.
    params.scale_softmax = softmax_scale;
    params.softcap = softcap;

    params.block_size = block_size;

    params.arch = at::cuda::getCurrentDeviceProperties()->major * 10 + at::cuda::getCurrentDeviceProperties()->minor;
    params.num_sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount - sm_margin;
    TORCH_CHECK(params.num_sm > 0, "sm_margin must be smaller than the number of SMs");
}

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    TORCH_CHECK(params.num_sm > 0);
    // HEADDIM_SWITCH(params.d, [&] {
    //     run_mha_fwd_<cutlass::half_t, kHeadSize>(params, stream);
    // });
    SOFTCAP_SWITCH(params.softcap > 0.0, Has_softcap, [&] {
        if (!params.is_e4m3) {
            if (params.is_bf16) {
                #ifndef FLASHATTENTION_DISABLE_HDIM64
                if (params.d <= 64) { return run_mha_fwd_<90, cutlass::bfloat16_t, 64, Has_softcap>(params, stream); }
                #endif
                #ifndef FLASHATTENTION_DISABLE_HDIM96
                if (params.d <= 96) { return run_mha_fwd_<90, cutlass::bfloat16_t, 96, Has_softcap>(params, stream); }
                #endif
                #ifndef FLASHATTENTION_DISABLE_HDIM128
                if (params.d <= 128) { return run_mha_fwd_<90, cutlass::bfloat16_t, 128, Has_softcap>(params, stream); }
                #endif
                #ifndef FLASHATTENTION_DISABLE_HDIM192
                if (params.d <= 192) { return run_mha_fwd_<90, cutlass::bfloat16_t, 192, Has_softcap>(params, stream); }
                #endif
                #ifndef FLASHATTENTION_DISABLE_HDIM256
                if (params.d <= 256) { return run_mha_fwd_<90, cutlass::bfloat16_t, 256, Has_softcap>(params, stream); }
                #endif
            } else {
                #ifndef FLASHATTENTION_DISABLE_FP16
                #ifndef FLASHATTENTION_DISABLE_HDIM64
                if (params.d <= 64) { return run_mha_fwd_<90, cutlass::half_t, 64, Has_softcap>(params, stream); }
                #endif
                #ifndef FLASHATTENTION_DISABLE_HDIM96
                if (params.d <= 96) { return run_mha_fwd_<90, cutlass::half_t, 96, Has_softcap>(params, stream); }
                #endif
                #ifndef FLASHATTENTION_DISABLE_HDIM128
                if (params.d <= 128) { return run_mha_fwd_<90, cutlass::half_t, 128, Has_softcap>(params, stream); }
                #endif
                #ifndef FLASHATTENTION_DISABLE_HDIM192
                if (params.d <= 192) { return run_mha_fwd_<90, cutlass::half_t, 192, Has_softcap>(params, stream); }
                #endif
                #ifndef FLASHATTENTION_DISABLE_HDIM256
                if (params.d <= 256) { return run_mha_fwd_<90, cutlass::half_t, 256, Has_softcap>(params, stream); }
                #endif
                #else
                TORCH_CHECK(false, "This build does not support FP16.");
                #endif
            }
        } else {
            #ifndef FLASHATTENTION_DISABLE_FP8
            #ifndef FLASHATTENTION_DISABLE_HDIM64
            if (params.d <= 64) { return run_mha_fwd_<90, cutlass::float_e4m3_t, 64, Has_softcap>(params, stream); }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM96
            if (params.d <= 96) { return run_mha_fwd_<90, cutlass::float_e4m3_t, 96, Has_softcap>(params, stream); }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM128
            if (params.d <= 128) { return run_mha_fwd_<90, cutlass::float_e4m3_t, 128, Has_softcap>(params, stream); }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM192
            if (params.d <= 192) { return run_mha_fwd_<90, cutlass::float_e4m3_t, 192, Has_softcap>(params, stream); }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM256
            if (params.d <= 256) { return run_mha_fwd_<90, cutlass::float_e4m3_t, 256, Has_softcap>(params, stream); }
            #endif
            #else
            TORCH_CHECK(false, "This build does not support FP8.");
            #endif
        }
    });
}

void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream) {
    #ifndef FLASHATTENTION_DISABLE_BACKWARD
    SOFTCAP_SWITCH(params.softcap > 0.0, Has_softcap, [&] {
        if (!params.is_bf16) {
            #ifndef FLASHATTENTION_DISABLE_FP16
            #ifndef FLASHATTENTION_DISABLE_HDIM64
            if (params.d <= 64) { return run_mha_bwd_<90, cutlass::half_t, 64, Has_softcap>(params, stream); }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM96
            if (params.d <= 96) { return run_mha_bwd_<90, cutlass::half_t, 96, Has_softcap>(params, stream); }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM128
            if (params.d <= 128) { return run_mha_bwd_<90, cutlass::half_t, 128, Has_softcap>(params, stream); }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM192
            if (params.d <= 192) { return run_mha_bwd_<90, cutlass::half_t, 192, Has_softcap>(params, stream); }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM256
            if (params.d <= 256) { return run_mha_bwd_<90, cutlass::half_t, 256, Has_softcap>(params, stream); }
            #endif
            #else
            TORCH_CHECK(false, "This build does not support FP16.");
            #endif
        } else {
            #ifndef FLASHATTENTION_DISABLE_HDIM64
            if (params.d <= 64) { return run_mha_bwd_<90, cutlass::bfloat16_t, 64, Has_softcap>(params, stream); }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM96
            if (params.d <= 96) { return run_mha_bwd_<90, cutlass::bfloat16_t, 96, Has_softcap>(params, stream); }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM128
            if (params.d <= 128) { return run_mha_bwd_<90, cutlass::bfloat16_t, 128, Has_softcap>(params, stream); }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM192
            if (params.d <= 192) { return run_mha_bwd_<90, cutlass::bfloat16_t, 192, Has_softcap>(params, stream); }
            #endif
            #ifndef FLASHATTENTION_DISABLE_HDIM256
            if (params.d <= 256) { return run_mha_bwd_<90, cutlass::bfloat16_t, 256, Has_softcap>(params, stream); }
            #endif
        }
    });
    #endif
}

} // namespace

// Forward pass of the diffusion attention.
// q: (b, 2n, h, d) or (total, h, d) if cu_seqlens is given; k, v: (b, 2n, h_k, d) or (total, h_k, d).
// Every sample is [noisy half (n tokens) | clean half (n tokens)].
std::tuple<at::Tensor, at::Tensor>
mha_fwd(at::Tensor q,
        at::Tensor k,
        at::Tensor v,
        std::optional<at::Tensor> out_,
        std::optional<at::Tensor> cu_seqlens_,  // b+1
        std::optional<int64_t> max_seqlen_,
        int64_t block_size,
        std::optional<at::Tensor> q_descale_,  // (b, h_k), not (b, h)
        std::optional<at::Tensor> k_descale_,  // (b, h_k)
        std::optional<at::Tensor> v_descale_,  // (b, h_k)
        std::optional<double> softmax_scale_,
        double softcap,
        int64_t sm_margin
        ) {

    auto q_type = q.scalar_type();
    TORCH_CHECK(q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16 || q_type == at::ScalarType::Float8_e4m3fn,
                "fast-diffusion-attention only supports fp16, bf16, and fp8_e4m3 data type");
    auto [batch_size, seqlen, total] = check_inputs(q, k, v, cu_seqlens_, max_seqlen_, block_size);
    bool const is_varlen = cu_seqlens_.has_value();
    int const num_heads = q.size(-2);
    int const head_size = q.size(-1);
    int const num_heads_k = k.size(-2);
    double softmax_scale = 1.0 / sqrt(double(head_size));
    if (softmax_scale_.has_value()) {
        softmax_scale = softmax_scale_.value();
    }

    auto opts = q.options();
    auto out_type = q_type == at::ScalarType::Float8_e4m3fn ? at::ScalarType::BFloat16 : q_type;
    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.scalar_type() == out_type, "For FP16/BF16 input, output must have the same dtype as inputs. For FP8 input, output must have dtype BF16");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        if (!is_varlen) {
            CHECK_SHAPE(out, batch_size, seqlen, num_heads, head_size);
        } else {
            CHECK_SHAPE(out, total, num_heads, head_size);
        }
    } else {
        out = !is_varlen
            ? torch::empty({batch_size, seqlen, num_heads, head_size}, opts.dtype(out_type))
            : torch::empty({total, num_heads, head_size}, opts.dtype(out_type));
    }

    int const head_size_rounded = round_up_headdim(head_size);

    // Otherwise the kernel will be launched from cuda:0 device
    auto device_guard = make_cuda_guard_from_tensor(q);

    at::Tensor softmax_lse;
    if (!is_varlen) {
        softmax_lse = torch::empty({batch_size, num_heads, seqlen}, opts.dtype(at::kFloat));
    } else {
        softmax_lse = torch::empty({num_heads, total}, opts.dtype(at::kFloat));
    }

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size, seqlen, total,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, out,
                     !is_varlen ? nullptr : cu_seqlens_.value().data_ptr(),
                     softmax_lse.data_ptr(),
                     softmax_scale,
                     block_size,
                     softcap,
                     sm_margin);

    // Scheduler metadata: tile counter (and for varlen: num_m_blocks, sorted sample indices, heads in L2)
    at::Tensor tile_count_semaphore;
    if (!is_varlen) {
        tile_count_semaphore = torch::zeros({1}, opts.dtype(torch::kInt32));
        params.tile_count_semaphore = tile_count_semaphore.data_ptr<int>();
    } else {
        int const b_rounded = round_multiple(batch_size, 4);  // for 16 byte alignment of pointers
        tile_count_semaphore = torch::empty({4 + 3 * b_rounded}, opts.dtype(torch::kInt32));
        params.tile_count_semaphore = tile_count_semaphore.data_ptr<int>();
        params.num_m_blocks_ptr = params.tile_count_semaphore + 4;
        params.varlen_batch_idx_ptr = params.num_m_blocks_ptr + b_rounded;
        params.num_nheads_in_l2_ptr = params.varlen_batch_idx_ptr + b_rounded;
    }

    if (q_type == at::ScalarType::Float8_e4m3fn) {
        if (q_descale_.has_value()) {
            auto q_descale = q_descale_.value();
            CHECK_DEVICE(q_descale);
            CHECK_SHAPE(q_descale, batch_size, num_heads_k);
            params.q_descale_ptr = q_descale.data_ptr<float>();
            params.q_descale_batch_stride = q_descale.stride(0);
            params.q_descale_head_stride = q_descale.stride(1);
        } else {
            params.q_descale_ptr = nullptr;
        }
        if (k_descale_.has_value()) {
            auto k_descale = k_descale_.value();
            CHECK_DEVICE(k_descale);
            CHECK_SHAPE(k_descale, batch_size, num_heads_k);
            params.k_descale_ptr = k_descale.data_ptr<float>();
            params.k_descale_batch_stride = k_descale.stride(0);
            params.k_descale_head_stride = k_descale.stride(1);
        } else {
            params.k_descale_ptr = nullptr;
        }
        if (v_descale_.has_value()) {
            auto v_descale = v_descale_.value();
            CHECK_DEVICE(v_descale);
            CHECK_SHAPE(v_descale, batch_size, num_heads_k);
            params.v_descale_ptr = v_descale.data_ptr<float>();
            params.v_descale_batch_stride = v_descale.stride(0);
            params.v_descale_head_stride = v_descale.stride(1);
        } else {
            params.v_descale_ptr = nullptr;
        }
    } else {
        TORCH_CHECK(!q_descale_.has_value() && !k_descale_.has_value() && !v_descale_.has_value(),
                    "q_descale, k_descale and v_descale are only supported for FP8 inputs");
    }

    #ifdef FLASHATTENTION_DISABLE_SOFTCAP
    TORCH_CHECK(params.softcap == 0.0, "This build does not support tanh softcapping.");
    #endif

    if (total > 0 && num_heads > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_fwd(params, stream);
    } else {
        // If seqlen is 0, there's nothing to compute.
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }
    return {out, softmax_lse};
}

// Backward pass of the diffusion attention.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>
mha_bwd(at::Tensor dout,  // (b, 2n, h, d) or (total, h, d)
        at::Tensor q,     // (b, 2n, h, d) or (total, h, d)
        at::Tensor k,     // (b, 2n, h_k, d) or (total, h_k, d)
        at::Tensor v,     // (b, 2n, h_k, d) or (total, h_k, d)
        at::Tensor out,   // (b, 2n, h, d) or (total, h, d)
        at::Tensor softmax_lse,    // (b, h, 2n) or (h, total)
        std::optional<at::Tensor> dq_,   // (b, 2n, h, d) or (total, h, d)
        std::optional<at::Tensor> dk_,   // (b, 2n, h_k, d) or (total, h_k, d)
        std::optional<at::Tensor> dv_,   // (b, 2n, h_k, d) or (total, h_k, d)
        std::optional<at::Tensor> cu_seqlens_,  // b+1
        std::optional<int64_t> max_seqlen_,
        int64_t block_size,
        std::optional<double> softmax_scale_,
        double softcap,
        bool deterministic,
        int64_t sm_margin
    ) {

    #ifdef FLASHATTENTION_DISABLE_BACKWARD
        TORCH_CHECK(false, "This fast-diffusion-attention build does not support backward.");
    #endif

    auto q_type = q.dtype();
    TORCH_CHECK(q_type == torch::kFloat16 || q_type == torch::kBFloat16,
                "fast-diffusion-attention backward only supports fp16 and bf16 data type");
    TORCH_CHECK(out.dtype() == q_type, "query and out must have the same dtype");
    TORCH_CHECK(dout.dtype() == q_type, "query and dout must have the same dtype");
    auto [batch_size, seqlen, total] = check_inputs(q, k, v, cu_seqlens_, max_seqlen_, block_size);
    bool const is_varlen = cu_seqlens_.has_value();
    CHECK_DEVICE(out); CHECK_DEVICE(dout); CHECK_DEVICE(softmax_lse);
    TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
    TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");

    int const num_heads = q.size(-2);
    int const head_size = q.size(-1);
    int const num_heads_k = k.size(-2);
    double softmax_scale = 1.0 / sqrt(double(head_size));
    if (softmax_scale_.has_value()) {
        softmax_scale = softmax_scale_.value();
    }
    int const head_size_rounded = round_up_headdim(head_size);
    // Very important that these match the kernel configs
    int const kBlockM_sm90 = head_size_rounded <= 64 ? (softcap > 0.0 ? 96 : 128)
        : (head_size_rounded <= 96 ? 64
           : (head_size_rounded <= 128 ? 64
              : (head_size_rounded <= 192 ? 64 : 64)));
    int const kBlockN_sm90 = head_size_rounded <= 128
        ? 128
        : (head_size_rounded <= 192 ? 96 : 80);
    int const kBlockM = kBlockM_sm90;
    int const kBlockN = kBlockN_sm90;
    // Accumulators: every half of every sample starts at a multiple of the tile size
    int const total_q_padded = round_multiple(total + 2 * batch_size * kBlockM, kBlockM);
    int const total_k_padded = round_multiple(total + 2 * batch_size * kBlockN, kBlockN);

    if (!is_varlen) {
        CHECK_SHAPE(out, batch_size, seqlen, num_heads, head_size);
        CHECK_SHAPE(dout, batch_size, seqlen, num_heads, head_size);
        CHECK_SHAPE(softmax_lse, batch_size, num_heads, seqlen);
    } else {
        CHECK_SHAPE(out, total, num_heads, head_size);
        CHECK_SHAPE(dout, total, num_heads, head_size);
        CHECK_SHAPE(softmax_lse, num_heads, total);
    }
    TORCH_CHECK(softmax_lse.is_contiguous(), "softmax_lse must be contiguous");

    at::Tensor dq, dk, dv;
    if (dq_.has_value()) {
        dq = dq_.value();
        TORCH_CHECK(dq.dtype() == q_type, "dq must have the same dtype as q");
        CHECK_DEVICE(dq);
        TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
        TORCH_CHECK(dq.sizes() == q.sizes(), "dq must have the same shape as q");
    } else {
        dq = torch::empty_like(q);
    }
    if (dk_.has_value()) {
        dk = dk_.value();
        TORCH_CHECK(dk.dtype() == q_type, "dk must have the same dtype as q");
        CHECK_DEVICE(dk);
        TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
        TORCH_CHECK(dk.sizes() == k.sizes(), "dk must have the same shape as k");
    } else {
        dk = torch::empty_like(k);
    }
    if (dv_.has_value()) {
        dv = dv_.value();
        TORCH_CHECK(dv.dtype() == q_type, "dv must have the same dtype as q");
        CHECK_DEVICE(dv);
        TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
        TORCH_CHECK(dv.sizes() == v.sizes(), "dv must have the same shape as v");
    } else {
        dv = torch::empty_like(v);
    }

    // Otherwise the kernel will be launched from cuda:0 device
    auto device_guard = make_cuda_guard_from_tensor(q);

    auto opts = q.options();
    // Need softmax_d to have total_q_padded since we want its address to be aligned by 16/8 bytes for TMA / LDG.64
    at::Tensor softmax_d = torch::empty({num_heads, total_q_padded}, opts.dtype(at::kFloat));
    at::Tensor softmax_lse_log2 = torch::empty({num_heads, total_q_padded}, opts.dtype(at::kFloat));
    at::Tensor dq_accum = torch::empty({num_heads, total_q_padded * head_size_rounded}, opts.dtype(at::kFloat));
    at::Tensor dk_accum, dv_accum;
    if (num_heads_k != num_heads) {  // MQA / GQA
        dk_accum = torch::zeros({num_heads_k, total_k_padded * head_size_rounded}, opts.dtype(at::kFloat));
        dv_accum = torch::zeros({num_heads_k, total_k_padded * head_size_rounded}, opts.dtype(at::kFloat));
    }

    Flash_bwd_params params;
    set_params_fprop(params,
                     batch_size, seqlen, total,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, out,
                     !is_varlen ? nullptr : cu_seqlens_.value().data_ptr(),
                     softmax_lse.data_ptr(),
                     softmax_scale,
                     block_size,
                     softcap,
                     sm_margin);
    // Set the pointers and strides.
    params.do_ptr = dout.data_ptr();
    params.do_row_stride = dout.stride(-3);
    params.do_head_stride = dout.stride(-2);
    params.dq_ptr = dq.data_ptr();
    params.dk_ptr = dk.data_ptr();
    params.dv_ptr = dv.data_ptr();
    params.dq_row_stride = dq.stride(-3);
    params.dk_row_stride = dk.stride(-3);
    params.dv_row_stride = dv.stride(-3);
    params.dq_head_stride = dq.stride(-2);
    params.dk_head_stride = dk.stride(-2);
    params.dv_head_stride = dv.stride(-2);
    if (!is_varlen) {
        params.do_batch_stride = dout.stride(0);
        params.dq_batch_stride = dq.stride(0);
        params.dk_batch_stride = dk.stride(0);
        params.dv_batch_stride = dv.stride(0);
    }
    params.dq_accum_ptr = dq_accum.data_ptr();
    params.dk_accum_ptr = num_heads_k != num_heads ? dk_accum.data_ptr() : nullptr;
    params.dv_accum_ptr = num_heads_k != num_heads ? dv_accum.data_ptr() : nullptr;
    params.dsoftmax_sum = softmax_d.data_ptr();
    params.softmax_lse_log2_ptr = softmax_lse_log2.data_ptr();
    params.total_q_padded = total_q_padded;
    params.total_k_padded = total_k_padded;
    params.deterministic = deterministic;

    at::Tensor dq_semaphore, dk_semaphore, dv_semaphore;
    if (deterministic) {
        // One semaphore per query tile and head (dQ), per key tile and kv head (dK, dV for GQA)
        dq_semaphore = torch::zeros({total_q_padded / kBlockM, num_heads}, opts.dtype(torch::kInt32));
        params.dq_semaphore = dq_semaphore.data_ptr<int>();
        if (num_heads_k != num_heads) {
            dk_semaphore = torch::zeros({total_k_padded / kBlockN, num_heads_k}, opts.dtype(torch::kInt32));
            dv_semaphore = torch::zeros({total_k_padded / kBlockN, num_heads_k}, opts.dtype(torch::kInt32));
            params.dk_semaphore = dk_semaphore.data_ptr<int>();
            params.dv_semaphore = dv_semaphore.data_ptr<int>();
        }
    }

    #ifdef FLASHATTENTION_DISABLE_SOFTCAP
    TORCH_CHECK(params.softcap == 0.0, "This build does not support tanh softcapping.");
    #endif

    if (total > 0 && num_heads > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_bwd(params, stream);
    } else {
        // If seqlen is 0, there's nothing to compute.
        dq.zero_();
        dk.zero_();
        dv.zero_();
        softmax_d.zero_();
    }

    return {dq, dk, dv, softmax_d};
}

TORCH_LIBRARY(fda, m) {
    m.def("fwd("
        "Tensor q,"
        "Tensor k,"
        "Tensor v,"
        "Tensor(out!)? out = None,"
        "Tensor? cu_seqlens = None,"
        "int? max_seqlen = None,"
        "int block_size = 1,"
        "Tensor? q_descale = None,"
        "Tensor? k_descale = None,"
        "Tensor? v_descale = None,"
        "float? softmax_scale = None,"
        "float softcap = 0.0,"
        "int sm_margin = 0) -> (Tensor(out!), Tensor)");
    m.def("bwd("
        "Tensor dout,"
        "Tensor q,"
        "Tensor k,"
        "Tensor v,"
        "Tensor out,"
        "Tensor softmax_lse,"
        "Tensor(dq!)? dq = None,"
        "Tensor(dk!)? dk = None,"
        "Tensor(dv!)? dv = None,"
        "Tensor? cu_seqlens = None,"
        "int? max_seqlen = None,"
        "int block_size = 1,"
        "float? softmax_scale = None,"
        "float softcap = 0.0,"
        "bool deterministic = False,"
        "int sm_margin = 0) -> (Tensor(dq!), Tensor(dk!), Tensor(dv!), Tensor)");
}

TORCH_LIBRARY_IMPL(fda, CUDA, m) {
    m.impl("fwd", &mha_fwd);
    m.impl("bwd", &mha_bwd);
}
