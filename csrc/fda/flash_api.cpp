/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Copyright (c) 2026, fast-diffusion-attention authors.
 *
 * C++ API of the diffusion attention kernels, adapted from FlashAttention-2 (csrc/flash_attn/flash_api.cpp).
 * Self-attention on samples laid out as [noisy (n tokens) | clean (n tokens)], see src/diffusion.h.
 ******************************************************************************/

// Include these 2 headers instead of torch/extension.h since we don't need all of the torch headers.
#include <torch/python.h>
#include <torch/nn/functional.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#ifndef FLASHATTENTION_DISABLE_DROPOUT
#include <ATen/cuda/CUDAGeneratorImpl.h>  // For at::PhiloxCudaState / at::CUDAGeneratorImpl (default-generator dropout path)
#include "philox_unpack.cuh"  // For at::cuda::philox::unpack
#include <type_traits>  // For std::is_trivially_copyable (philox_args buffer assert below)
#endif

#include <cutlass/numeric_types.h>

#include "namespace_config.h"
#include "hardware_info.h"
#include "flash.h"
#include "static_switch.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

namespace FLASH_NAMESPACE {

#ifndef FLASHATTENTION_DISABLE_DROPOUT
// Flash_fwd_params keeps philox state as an opaque uint64_t buffer (see flash.h) so the
// shared header avoids ATen Generator types. Validate the buffer against the real
// at::PhiloxCudaState layout here, where the type is actually visible.
static_assert(sizeof(at::PhiloxCudaState) <= sizeof(Flash_fwd_params::philox_args),
              "Flash_fwd_params::philox_args buffer is too small for at::PhiloxCudaState");
static_assert(alignof(at::PhiloxCudaState) <= alignof(decltype(Flash_fwd_params::philox_args)),
              "Flash_fwd_params::philox_args buffer is under-aligned for at::PhiloxCudaState");
static_assert(std::is_trivially_copyable<at::PhiloxCudaState>::value,
              "at::PhiloxCudaState must be trivially copyable: it is placement-new'd into "
              "philox_args and the whole Flash_fwd_params is copied by value to the device kernel");
#endif

namespace {

int round_multiple(int x, int m) { return (x + m - 1) / m * m; }

// Head dimension of the kernel used for head dimension d: 32, 64, 96, 128, 192 or 256 (as FlashAttention-2).
int round_head_size(int d) { return round_multiple(d, d <= 128 ? 32 : 64); }

void check_arch() {
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    TORCH_CHECK(cc_major >= 8, "fast-diffusion-attention only supports Ampere GPUs or newer (sm80, sm86, sm89).");
}

}  // namespace

void set_params_fprop(Flash_fwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      at::Tensor out,
                      void *cu_seqlens_d,
                      void *p_d,
                      void *softmax_lse_d,
                      float p_dropout,
                      float softmax_scale,
                      int block_size,
                      const float softcap,
                      const bool unpadded_lse=false) {

    // Reset the parameters
    params = {};

    params.is_bf16 = q.dtype() == torch::kBFloat16;

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

    // P = softmax(QK^T)
    params.p_ptr = p_d;

    // Softmax sum
    params.softmax_lse_ptr = softmax_lse_d;

    // Set the dimensions.
    params.b = b;
    params.h = h;
    params.h_k = h_k;
    params.h_h_k_ratio = h / h_k;
    params.seqlen = seqlen;
    params.seqlen_half_rounded = round_multiple(seqlen / 2, 128);
    params.d = d;
    params.d_rounded = d_rounded;

    // The diffusion blocks
    params.block_size = block_size;
    params.block_size_divmod = cutlass::FastDivmod(block_size);

    // Set the different scale values.
    #ifdef FLASHATTENTION_DISABLE_SOFTCAP
        TORCH_CHECK(softcap <= 0.0, "This build does not support softcap.");
    #endif
    if (softcap > 0.0) {
        params.softcap = softmax_scale / softcap;
        params.scale_softmax = softcap;
        params.scale_softmax_log2 = softcap * M_LOG2E;
    } else{
        // Remove potential NaN
        params.softcap = 0.0;
        params.scale_softmax = softmax_scale;
        params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    }

    // Set this to probability of keeping an element to simplify things.
    params.p_dropout = 1.f - p_dropout;
    // Convert p from float to int so we don't have to convert the random uint to float to compare.
    // [Minor] We want to round down since when we do the comparison we use <= instead of <
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;
    TORCH_CHECK(p_dropout < 1.f);
    #ifdef FLASHATTENTION_DISABLE_DROPOUT
        TORCH_CHECK(p_dropout == 0.0f, "This build does not support dropout.");
    #endif

    #ifdef FLASHATTENTION_DISABLE_UNEVEN_K
        TORCH_CHECK(d == d_rounded, "This build does not support headdim not being a multiple of 32.");
    #endif

    params.unpadded_lse = unpadded_lse;
}

void set_params_dgrad(Flash_bwd_params &params,
                      // sizes
                      const size_t b,
                      const size_t seqlen,
                      const size_t h,
                      const size_t h_k,
                      const size_t d,
                      const size_t d_rounded,
                      const size_t total_padded,
                      // device pointers
                      const at::Tensor q,
                      const at::Tensor k,
                      const at::Tensor v,
                      const at::Tensor out,
                      const at::Tensor dout,
                      at::Tensor dq,
                      at::Tensor dk,
                      at::Tensor dv,
                      void *cu_seqlens_d,
                      void *dq_accum_d,
                      void *softmax_lse_d,
                      void *softmax_lse_padded_d,
                      void *dsoftmax_sum_d,
                      float p_dropout,
                      float softmax_scale,
                      int block_size,
                      const float softcap,
                      bool deterministic,
                      const bool unpadded_lse) {

    set_params_fprop(params,
                     b, seqlen, h, h_k, d, d_rounded,
                     q, k, v, out,
                     cu_seqlens_d,
                     nullptr,
                     softmax_lse_d,
                     p_dropout,
                     softmax_scale,
                     block_size,
                     softcap,
                     unpadded_lse);

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

    if (cu_seqlens_d == nullptr) {
        params.do_batch_stride = dout.stride(0);
        params.dq_batch_stride = dq.stride(0);
        params.dk_batch_stride = dk.stride(0);
        params.dv_batch_stride = dv.stride(0);
    }

    params.dq_accum_ptr = dq_accum_d;

    // Softmax sum
    params.dsoftmax_sum = dsoftmax_sum_d;
    params.softmax_lse_padded_ptr = softmax_lse_padded_d;
    params.total_padded = total_padded;

    params.deterministic = deterministic;
}

void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d_rounded, [&] {
            if (params.num_splits <= 1) {  // If we don't set it num_splits == 0
                run_mha_fwd_<elem_type, kHeadDim>(params, stream);
            } else {
                run_mha_fwd_splitkv_dispatch<elem_type, kHeadDim>(params, stream);
            }
        });
    });
}

// Find the number of splits that maximizes the occupancy. For example, if we have
// batch * n_heads = 48 and we have 108 SMs, having 2 splits (efficiency = 0.89) is
// better than having 3 splits (efficiency = 0.67). However, we also don't want too many
// splits as that would incur more HBM reads/writes.
// So we find the best efficiency, then find the smallest number of splits that gets 85%
// of the best efficiency.
// (FlashAttention-2's num_splits_heuristic.)
inline int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks, int max_splits) {
    // If we have enough to almost fill the SMs, then just use 1 split
    if (batch_nheads_mblocks >= 0.8f * num_SMs) { return 1; }
    max_splits = std::min({max_splits, num_SMs, num_n_blocks});
    float max_efficiency = 0.f;
    std::vector<float> efficiency;
    efficiency.reserve(max_splits);
    auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
    // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
    // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
    // (i.e. it's 11 splits anyway).
    // So we check if the number of blocks per split is the same as the previous num_splits.
    auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
        return num_splits == 1 || ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
    };
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) {
            efficiency.push_back(0.f);
        } else {
            float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
            float eff = n_waves / ceil(n_waves);
            if (eff > max_efficiency) { max_efficiency = eff; }
            efficiency.push_back(eff);
        }
    }
    for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
        if (!is_split_eligible(num_splits)) { continue; }
        if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
            return num_splits;
        }
    }
    return 1;
}

// FlashAttention-2's set_params_splitkv for the batched forward (FlashAttention-2's varlen forward only splits for a
// query length of 1, which samples [noisy | clean] never have). Its inputs are those of FlashAttention-2 on the same
// work, causal attention on the halves (2b sequences of length n): 2b * h * ceil(n / 64) query tiles of the split
// kernel, ceil(n / block_n) key tiles. FDA_FWD_NUM_SPLITS overrides the number of splits (tests).
std::tuple<at::Tensor, at::Tensor> set_params_splitkv(Flash_fwd_params &params, const int batch_size,
    const int num_heads, const int head_size, const int seqlen, const int head_size_rounded, const float p_dropout,
    const int num_sm, struct c10::TensorOptions opts) {

    const int seqlen_half = seqlen / 2;
    // This needs to match with run_mha_fwd_splitkv_dispatch
    const int block_n = head_size <= 64 ? 256 : (head_size <= 128 ? 128 : 64);
    const int num_n_blocks = (seqlen_half + block_n - 1) / block_n;
    const int num_m_blocks = 2 * ((seqlen_half + 64 - 1) / 64);
    params.num_splits = 1;
    at::Tensor softmax_lse_accum;
    at::Tensor out_accum;

    if (p_dropout == 0.0f) {  // SplitKV is not implemented for dropout
        // We multiply number of SMs by 2 to hard-code the fact that we're using 128 threads per block.
        params.num_splits = num_splits_heuristic(batch_size * num_heads * num_m_blocks, num_sm * 2, num_n_blocks, 128);
        const char *env = std::getenv("FDA_FWD_NUM_SPLITS");
        if (env != nullptr) { params.num_splits = std::max(1, std::atoi(env)); }
        if (params.num_splits > 1) {
            softmax_lse_accum = torch::empty({params.num_splits, batch_size, num_heads, seqlen}, opts.dtype(at::kFloat));
            out_accum = torch::empty({params.num_splits, batch_size, num_heads, seqlen, head_size_rounded}, opts.dtype(at::kFloat));
            params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
            params.oaccum_ptr = out_accum.data_ptr();
        }
        TORCH_CHECK(params.num_splits <= 128, "num_splits > 128 not supported");
    }
    return std::make_tuple(softmax_lse_accum, out_accum);
}

void set_params_dropout(Flash_fwd_params &params, at::Tensor &rng_state, const float p_dropout) {
    // Forward kernel will populate memory with the seed and offset.
    params.rng_state = reinterpret_cast<uint64_t*>(rng_state.data_ptr());
#ifndef FLASHATTENTION_DISABLE_DROPOUT
    if (p_dropout > 0.0)  {
        // number of times random will be generated per thread, to offset philox counter in thc random
        // state
        // We use a custom RNG that increases the offset by batch_size * nheads * 32.
        int64_t counter_offset = params.b * params.h * 32;
        auto gen = at::cuda::detail::getDefaultCUDAGenerator();
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen.mutex());
        new (params.philox_args) at::PhiloxCudaState(gen.get<at::CUDAGeneratorImpl>()->philox_cuda_state(counter_offset));
    }
#endif
}

void check_qkv(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, const int block_size) {
    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "fast-diffusion-attention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v);

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(block_size >= 1, "block_size must be a positive integer");
}

std::vector<at::Tensor>
mha_fwd(at::Tensor &q,         // batch_size x seqlen x num_heads x round_multiple(head_size, 8)
        const at::Tensor &k,         // batch_size x seqlen x num_heads_k x round_multiple(head_size, 8)
        const at::Tensor &v,         // batch_size x seqlen x num_heads_k x round_multiple(head_size, 8)
        std::optional<at::Tensor> &out_,             // batch_size x seqlen x num_heads x round_multiple(head_size, 8)
        const int block_size,
        const float p_dropout,
        const float softmax_scale,
        const float softcap,
        const bool return_softmax) {

    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{q.device()};
    check_arch();
    check_qkv(q, k, v, block_size);
    TORCH_CHECK(q.dim() == 4, "q must have shape (batch_size, seqlen, num_heads, head_size)");

    const auto sizes = q.sizes();

    const int batch_size = sizes[0];
    const int seqlen = sizes[1];
    const int num_heads = sizes[2];
    const int head_size = sizes[3];
    const int num_heads_k = k.size(2);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(seqlen % 2 == 0, "The sequence length must be even: [noisy half | clean half]");
    TORCH_CHECK(head_size <= 256, "fast-diffusion-attention forward only supports head dimension at most 256");
    TORCH_CHECK(head_size % 8 == 0, "query, key, value, and out_ must have a head_size that is a multiple of 8");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }

    CHECK_SHAPE(q, batch_size, seqlen, num_heads, head_size);
    CHECK_SHAPE(k, batch_size, seqlen, num_heads_k, head_size);
    CHECK_SHAPE(v, batch_size, seqlen, num_heads_k, head_size);

    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q.dtype(), "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, batch_size, seqlen, num_heads, head_size);
    } else {
        out = torch::empty_like(q);
    }

    const int head_size_rounded = round_head_size(head_size);

    auto opts = q.options();

    auto softmax_lse = torch::empty({batch_size, num_heads, seqlen}, opts.dtype(at::kFloat));
    at::Tensor p;
    // Only return softmax if there's dropout to reduce compilation time
    if (return_softmax) {
        TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
        const int seqlen_half_rounded = round_multiple(seqlen / 2, 128);
        p = torch::empty({ batch_size, num_heads, 2 * seqlen_half_rounded, 2 * seqlen_half_rounded }, opts);
    }
    else {
        p = torch::empty({ 0 }, opts);
    }

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     seqlen,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, out,
                     /*cu_seqlens_d=*/nullptr,
                     return_softmax ? p.data_ptr() : nullptr,
                     softmax_lse.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     block_size,
                     softcap
                     );

    // Keep references to these tensors to extend their lifetime
    at::Tensor softmax_lse_accum, out_accum;
    std::tie(softmax_lse_accum, out_accum) = set_params_splitkv(
        params, batch_size, num_heads, head_size, seqlen, head_size_rounded, p_dropout, get_num_sm(get_current_device()), opts);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
    set_params_dropout(params, rng_state, p_dropout);

    if (seqlen > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_fwd(params, stream);
    } else {
        // If seqlen == 0, then we have an empty tensor. We need to set the output to 0.
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    return {out, softmax_lse, p, rng_state};
}

std::vector<at::Tensor>
mha_varlen_fwd(at::Tensor &q,  // total x num_heads x head_size, total := \sum_{i=0}^{b} s_i
               const at::Tensor &k,  // total x num_heads_k x head_size
               const at::Tensor &v,  // total x num_heads_k x head_size
               std::optional<at::Tensor> &out_, // total x num_heads x head_size
               const at::Tensor &cu_seqlens,  // b+1
               int max_seqlen,
               const int block_size,
               const float p_dropout,
               const float softmax_scale,
               const bool zero_tensors,
               const float softcap,
               const bool return_softmax) {

    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{q.device()};
    check_arch();
    check_qkv(q, k, v, block_size);
    TORCH_CHECK(q.dim() == 3, "q must have shape (total, num_heads, head_size)");
    TORCH_CHECK(cu_seqlens.dtype() == torch::kInt32, "cu_seqlens must have dtype int32");
    CHECK_DEVICE(cu_seqlens);
    CHECK_CONTIGUOUS(cu_seqlens);

    const auto sizes = q.sizes();

    const int batch_size = cu_seqlens.numel() - 1;
    const int total = sizes[0];
    const int num_heads = sizes[1];
    const int head_size = sizes[2];
    const int num_heads_k = k.size(1);

    if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(max_seqlen >= 0, "max_seqlen must be non-negative");
    TORCH_CHECK(head_size <= 256, "fast-diffusion-attention forward only supports head dimension at most 256");
    TORCH_CHECK(head_size % 8 == 0, "query, key, value, and out_ must have a head_size that is a multiple of 8");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    // Every sample has an even length; the kernels only need an upper bound on the lengths.
    max_seqlen = round_multiple(max_seqlen, 2);

    CHECK_SHAPE(q, total, num_heads, head_size);
    CHECK_SHAPE(k, total, num_heads_k, head_size);
    CHECK_SHAPE(v, total, num_heads_k, head_size);
    CHECK_SHAPE(cu_seqlens, batch_size + 1);

    at::Tensor out;
    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.dtype() == q.dtype(), "Output must have the same dtype as inputs");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        CHECK_SHAPE(out, total, num_heads, head_size);
    } else {
        out = torch::empty_like(q);
    }

    const int head_size_rounded = round_head_size(head_size);

    auto opts = q.options();
    auto softmax_lse = torch::empty({num_heads, total}, opts.dtype(at::kFloat));
    at::Tensor p;
    // Only return softmax if there's dropout to reduce compilation time
    if (return_softmax) {
        TORCH_CHECK(p_dropout > 0.0f, "return_softmax is only supported when p_dropout > 0.0");
        const int seqlen_half_rounded = round_multiple(max_seqlen / 2, 128);
        p = torch::empty({ batch_size, num_heads, 2 * seqlen_half_rounded, 2 * seqlen_half_rounded }, opts);
    }
    else {
        p = torch::empty({ 0 }, opts);
    }

    if (zero_tensors) {
        out.zero_();
        softmax_lse.fill_(-std::numeric_limits<float>::infinity());
        if (return_softmax) {p.zero_();}
    }

    Flash_fwd_params params;
    set_params_fprop(params,
                     batch_size,
                     max_seqlen,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, out,
                     cu_seqlens.data_ptr(),
                     return_softmax ? p.data_ptr() : nullptr,
                     softmax_lse.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     block_size,
                     softcap,
                     /*unpadded_lse*/true);
    params.total_q = total;

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
    set_params_dropout(params, rng_state, p_dropout);

    if (max_seqlen > 0 && total > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_fwd(params, stream);
    } else {
        // If max_seqlen == 0, then we have an empty tensor. We need to set the output to 0.
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    return {out, softmax_lse, p, rng_state};
}

void run_mha_bwd(Flash_bwd_params &params, cudaStream_t stream) {
#ifndef FLASHATTENTION_DISABLE_BACKWARD  // the backward kernel files are not built (FDA_DISABLE_BACKWARD=1)
    FP16_SWITCH(!params.is_bf16, [&] {
        HEADDIM_SWITCH(params.d_rounded, [&] {
            run_mha_bwd_<elem_type, kHeadDim>(params, stream);
        });
    });
#endif
}

void set_params_bwd_dropout(Flash_bwd_params &params, std::optional<at::Tensor> &rng_state, const float p_dropout) {
    if ( rng_state.has_value() ) {
        params.rng_state = reinterpret_cast<uint64_t*>(rng_state.value().data_ptr());
#ifndef FLASHATTENTION_DISABLE_DROPOUT
    } else if( p_dropout > 0.0 ) {
        // We use a custom RNG that increases the offset by batch_size * nheads * 32.
        int64_t counter_offset = params.b * params.h * 32;
        auto gen = at::cuda::detail::getDefaultCUDAGenerator();
        // See Note [Acquire lock when using random generators]
        std::lock_guard<std::mutex> lock(gen.mutex());
        new (params.philox_args) at::PhiloxCudaState(gen.get<at::CUDAGeneratorImpl>()->philox_cuda_state(counter_offset));
        auto seeds = at::cuda::philox::unpack(*reinterpret_cast<at::PhiloxCudaState const*>(params.philox_args));
        params.rng_state[0] = std::get<0>(seeds);
        params.rng_state[1] = std::get<1>(seeds);
#endif
    }
}

std::vector<at::Tensor>
mha_bwd_impl(const at::Tensor &dout,  // batch_size x seqlen x num_heads x head_size, or total x num_heads x head_size
             const at::Tensor &q,   // batch_size x seqlen x num_heads x head_size, or total x num_heads x head_size
             const at::Tensor &k,   // batch_size x seqlen x num_heads_k x head_size, or total x num_heads_k x head_size
             const at::Tensor &v,   // batch_size x seqlen x num_heads_k x head_size, or total x num_heads_k x head_size
             const at::Tensor &out,   // batch_size x seqlen x num_heads x head_size, or total x num_heads x head_size
             const at::Tensor &softmax_lse,     // b x h x seqlen, or h x total
             std::optional<at::Tensor> &dq_,
             std::optional<at::Tensor> &dk_,
             std::optional<at::Tensor> &dv_,
             std::optional<const at::Tensor> cu_seqlens_,  // b+1 (varlen)
             int max_seqlen,                               // varlen only
             const int block_size,
             const float p_dropout,         // probability to drop
             const float softmax_scale,
             const bool zero_tensors,
             const float softcap,
             const bool deterministic,
             std::optional<at::Tensor> &rng_state) {

    #ifdef FLASHATTENTION_DISABLE_BACKWARD
        TORCH_CHECK(false, "This build does not support backward.");
    #endif

    // Otherwise the kernel will be launched from cuda:0 device
    at::cuda::CUDAGuard device_guard{q.device()};
    check_arch();
    check_qkv(q, k, v, block_size);

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto q_dtype = q.dtype();
    TORCH_CHECK(out.dtype() == q_dtype, "query and out must have the same dtype");
    TORCH_CHECK(dout.dtype() == q_dtype, "query and dout must have the same dtype");

    CHECK_DEVICE(out); CHECK_DEVICE(dout); CHECK_DEVICE(softmax_lse);

    TORCH_CHECK(out.stride(-1) == 1, "out tensor must have contiguous last dimension");
    TORCH_CHECK(dout.stride(-1) == 1, "dout tensor must have contiguous last dimension");
    TORCH_CHECK(softmax_lse.dtype() == torch::kFloat32, "softmax_lse must have dtype float32");
    CHECK_CONTIGUOUS(softmax_lse);

    const bool is_varlen = cu_seqlens_.has_value();
    int batch_size, seqlen, total;
    if (!is_varlen) {
        TORCH_CHECK(q.dim() == 4, "q must have shape (batch_size, seqlen, num_heads, head_size)");
        batch_size = q.size(0);
        seqlen = q.size(1);
        total = batch_size * seqlen;
        TORCH_CHECK(seqlen % 2 == 0, "The sequence length must be even: [noisy half | clean half]");
    } else {
        const at::Tensor &cu_seqlens = cu_seqlens_.value();
        TORCH_CHECK(q.dim() == 3, "q must have shape (total, num_heads, head_size)");
        TORCH_CHECK(cu_seqlens.dtype() == torch::kInt32, "cu_seqlens must have dtype int32");
        CHECK_DEVICE(cu_seqlens);
        CHECK_CONTIGUOUS(cu_seqlens);
        batch_size = cu_seqlens.numel() - 1;
        TORCH_CHECK(max_seqlen >= 0, "max_seqlen must be non-negative");
        seqlen = round_multiple(max_seqlen, 2);
        total = q.size(0);
        CHECK_SHAPE(cu_seqlens, batch_size + 1);
    }
    const int num_heads = q.size(-2);
    const int head_size = q.size(-1);
    const int num_heads_k = k.size(-2);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
    TORCH_CHECK(head_size <= 256, "fast-diffusion-attention backward only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");
    if (softcap > 0.f) { TORCH_CHECK(p_dropout == 0.f, "Softcapping does not support dropout for now"); }

    const int head_size_rounded = round_head_size(head_size);
    // Rows of the padded accumulators (every half of every sample is followed by kAccumPad rows)
    const int total_padded = total + 2 * kAccumPad * batch_size;

    if (!is_varlen) {
        CHECK_SHAPE(q, batch_size, seqlen, num_heads, head_size);
        CHECK_SHAPE(k, batch_size, seqlen, num_heads_k, head_size);
        CHECK_SHAPE(v, batch_size, seqlen, num_heads_k, head_size);
        CHECK_SHAPE(out, batch_size, seqlen, num_heads, head_size);
        CHECK_SHAPE(dout, batch_size, seqlen, num_heads, head_size);
        CHECK_SHAPE(softmax_lse, batch_size, num_heads, seqlen);
    } else {
        CHECK_SHAPE(q, total, num_heads, head_size);
        CHECK_SHAPE(k, total, num_heads_k, head_size);
        CHECK_SHAPE(v, total, num_heads_k, head_size);
        CHECK_SHAPE(out, total, num_heads, head_size);
        CHECK_SHAPE(dout, total, num_heads, head_size);
        CHECK_SHAPE(softmax_lse, num_heads, total);
    }

    at::Tensor dq, dk, dv;
    if (dq_.has_value()) {
        dq = dq_.value();
        TORCH_CHECK(dq.dtype() == q_dtype, "dq must have the same dtype as q");
        CHECK_DEVICE(dq);
        TORCH_CHECK(dq.stride(-1) == 1, "dq must have contiguous last dimension");
        TORCH_CHECK(dq.sizes() == q.sizes(), "dq must have the same shape as q");
    } else {
        dq = torch::empty_like(q);
    }
    if (dk_.has_value()) {
        dk = dk_.value();
        TORCH_CHECK(dk.dtype() == q_dtype, "dk must have the same dtype as q");
        CHECK_DEVICE(dk);
        TORCH_CHECK(dk.stride(-1) == 1, "dk must have contiguous last dimension");
        TORCH_CHECK(dk.sizes() == k.sizes(), "dk must have the same shape as k");
    } else {
        dk = torch::empty_like(k);
    }
    if (dv_.has_value()) {
        dv = dv_.value();
        TORCH_CHECK(dv.dtype() == q_dtype, "dv must have the same dtype as q");
        CHECK_DEVICE(dv);
        TORCH_CHECK(dv.stride(-1) == 1, "dv must have contiguous last dimension");
        TORCH_CHECK(dv.sizes() == v.sizes(), "dv must have the same shape as v");
    } else {
        dv = torch::empty_like(v);
    }

    auto opts = q.options();
    // dPsum and LSE in the padded layout (written by the pre-processing kernel)
    auto softmax_d = torch::empty({num_heads, total_padded}, opts.dtype(at::kFloat));
    auto softmax_lse_padded = torch::empty({num_heads, total_padded}, opts.dtype(at::kFloat));
    at::Tensor dq_accum;
    if (!deterministic) {
        dq_accum = torch::empty({total_padded, num_heads, head_size_rounded}, opts.dtype(at::kFloat));
    } else {
        // FlashAttention-2: one wave of thread blocks, each accumulating into its own buffer (must match gridDimx in
        // run_flash_bwd_seqk_parallel)
        const int nsplits = (get_num_sm(get_current_device()) + batch_size * num_heads - 1) / (batch_size * num_heads);
        dq_accum = torch::zeros({nsplits, total_padded, num_heads, head_size_rounded}, opts.dtype(at::kFloat));
    }

    // MQA / GQA: dK / dV per query head, summed over the query heads of a KV head afterwards (FlashAttention-2's
    // dk_expanded / dv_expanded)
    at::Tensor dk_expanded, dv_expanded;
    if (num_heads_k != num_heads) {
        auto expanded_sizes = k.sizes().vec();
        expanded_sizes[expanded_sizes.size() - 2] = num_heads;
        dk_expanded = torch::empty(expanded_sizes, opts);
        dv_expanded = torch::empty(expanded_sizes, opts);
    } else {
        dk_expanded = dk;
        dv_expanded = dv;
    }

    if (zero_tensors) {
        dq.zero_();
        dk_expanded.zero_();
        dv_expanded.zero_();
        softmax_d.zero_();
    }

    Flash_bwd_params params;

    set_params_dgrad(params,
                     batch_size,
                     seqlen,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     total_padded,
                     q, k, v, out,
                     dout, dq, dk_expanded, dv_expanded,
                     is_varlen ? cu_seqlens_.value().data_ptr() : nullptr,
                     dq_accum.data_ptr(),
                     softmax_lse.data_ptr(),
                     softmax_lse_padded.data_ptr(),
                     softmax_d.data_ptr(),
                     p_dropout,
                     softmax_scale,
                     block_size,
                     softcap,
                     deterministic,
                     /*unpadded_lse*/is_varlen);
    params.dq_accum_split_stride = !deterministic ? 0 : dq_accum.stride(0);
    params.total_q = total;

    set_params_bwd_dropout(params, rng_state, p_dropout);

    if (seqlen > 0 && total > 0) {
        run_mha_bwd(params, stream);
    } else {
        // If seqlen == 0, then we have an empty tensor. We need to set the output to 0.
        dq.zero_();
        dk_expanded.zero_();
        dv_expanded.zero_();
        softmax_d.zero_();
    }

    // For MQA/GQA we need to sum dK and dV across the groups
    if (num_heads_k != num_heads) {
        auto grouped_sizes = k.sizes().vec();
        grouped_sizes.insert(grouped_sizes.end() - 1, num_heads / num_heads_k);
        const int64_t group_dim = grouped_sizes.size() - 2;
        at::sum_out(dk, at::reshape(dk_expanded, grouped_sizes), {group_dim});
        at::sum_out(dv, at::reshape(dv_expanded, grouped_sizes), {group_dim});
    }

    return { dq, dk, dv, softmax_d };
}

std::vector<at::Tensor>
mha_bwd(const at::Tensor &dout,  // batch_size x seqlen x num_heads, x multiple_of(head_size_og, 8)
        const at::Tensor &q,   // batch_size x seqlen x num_heads x head_size
        const at::Tensor &k,   // batch_size x seqlen x num_heads_k x head_size
        const at::Tensor &v,   // batch_size x seqlen x num_heads_k x head_size
        const at::Tensor &out,   // batch_size x seqlen x num_heads x head_size
        const at::Tensor &softmax_lse,     // b x h x seqlen
        std::optional<at::Tensor> &dq_,   // batch_size x seqlen x num_heads x head_size
        std::optional<at::Tensor> &dk_,   // batch_size x seqlen x num_heads_k x head_size
        std::optional<at::Tensor> &dv_,   // batch_size x seqlen x num_heads_k x head_size
        const int block_size,
        const float p_dropout,         // probability to drop
        const float softmax_scale,
        const float softcap,
        const bool deterministic,
        std::optional<at::Tensor> &rng_state) {
    return mha_bwd_impl(dout, q, k, v, out, softmax_lse, dq_, dk_, dv_, std::nullopt, 0, block_size, p_dropout,
                        softmax_scale, /*zero_tensors=*/false, softcap, deterministic, rng_state);
}

std::vector<at::Tensor>
mha_varlen_bwd(const at::Tensor &dout,  // total x num_heads, x head_size
               const at::Tensor &q,   // total x num_heads x head_size, total := \sum_{i=0}^{b} s_i
               const at::Tensor &k,   // total x num_heads_k x head_size
               const at::Tensor &v,   // total x num_heads_k x head_size
               const at::Tensor &out,   // total x num_heads x head_size
               const at::Tensor &softmax_lse,    // h x total, softmax logsumexp
               std::optional<at::Tensor> &dq_,   // total x num_heads x head_size
               std::optional<at::Tensor> &dk_,   // total x num_heads_k x head_size
               std::optional<at::Tensor> &dv_,   // total x num_heads_k x head_size
               const at::Tensor &cu_seqlens,  // b+1
               const int max_seqlen,          // max sequence length to choose the kernel
               const int block_size,
               const float p_dropout,         // probability to drop
               const float softmax_scale,
               const bool zero_tensors,
               const float softcap,
               const bool deterministic,
               std::optional<at::Tensor> &rng_state) {
    return mha_bwd_impl(dout, q, k, v, out, softmax_lse, dq_, dk_, dv_, cu_seqlens, max_seqlen, block_size, p_dropout,
                        softmax_scale, zero_tensors, softcap, deterministic, rng_state);
}

} // namespace FLASH_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "fast-diffusion-attention (FlashAttention-2 generation): diffusion attention kernels for sm80 / sm86 / sm89";
    m.def("fwd", &FLASH_NAMESPACE::mha_fwd, "Forward pass");
    m.def("varlen_fwd", &FLASH_NAMESPACE::mha_varlen_fwd, "Forward pass (variable length)");
    m.def("bwd", &FLASH_NAMESPACE::mha_bwd, "Backward pass");
    m.def("varlen_bwd", &FLASH_NAMESPACE::mha_varlen_bwd, "Backward pass (variable length)");
}
