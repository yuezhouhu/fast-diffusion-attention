/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Copyright (c) 2026, fast-diffusion-attention authors.
 *
 * Launch and dispatch of the backward kernels (FlashAttention-2 flash_bwd_launch_template.h).
 ******************************************************************************/

#pragma once

#include "namespace_config.h"
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK
#include <cstdlib>
#include <type_traits>

#include "static_switch.h"
#include "hardware_info.h"
#include "flash.h"
#include "flash_bwd_preprocess_kernel.h"
#include "flash_bwd_kernel.h"
#ifdef FDA_TUNING
#include "tuning.h"
#endif

namespace FLASH_NAMESPACE {

// Determine if the architecture supports FLASH and define a macro to handle parameter modifiers
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define ARCH_SUPPORTS_FLASH
#define KERNEL_PARAM_MODIFIER __grid_constant__
#else
#define KERNEL_PARAM_MODIFIER
#endif

// Define a macro for unsupported architecture handling to centralize the error message
#define FLASH_UNSUPPORTED_ARCH printf("FATAL: fast-diffusion-attention requires building with sm version sm80-sm90, but was built for < 8.0!");

// Use a macro to clean up kernel definitions
#define DEFINE_FLASH_BACKWARD_KERNEL(kernelName, ...) \
template<typename Kernel_traits, __VA_ARGS__> \
__global__ void kernelName(KERNEL_PARAM_MODIFIER const Flash_bwd_params params)

DEFINE_FLASH_BACKWARD_KERNEL(flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel, bool Is_dropout, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Is_grouped) {
    #if defined(ARCH_SUPPORTS_FLASH)
        FLASH_NAMESPACE::compute_dq_dk_dv_seqk_parallel<Kernel_traits, Is_dropout, Is_even_MN, Is_even_K, Is_softcap, Is_grouped>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

template<bool Clear_dQaccum=true, typename Kernel_traits>
__global__ void flash_bwd_dot_do_o_kernel(const Flash_bwd_params params) {
    FLASH_NAMESPACE::compute_dot_do_o<Clear_dQaccum, Kernel_traits>(params);
}

template<typename Kernel_traits>
__global__ void flash_bwd_convert_dq_kernel(const Flash_bwd_params params, const int nsplits) {
    FLASH_NAMESPACE::convert_dQ<Kernel_traits>(params, nsplits);
}

template<typename Kernel_traits, bool Is_dropout>
void run_flash_bwd_dq_dk_dv(Flash_bwd_params &params, cudaStream_t stream, const int gridDimx) {
    dim3 grid_n(gridDimx, params.b, params.h / params.bwd_heads_per_cta);
    const int seqlen_half = params.seqlen / 2;
    // We want to specialize to is_even_MN and not just is_even_M, since in the case where N is not
    // a multiple of kBlockN, we'll need to apply mask in the loop.
    const bool is_even_MN = params.cu_seqlens == nullptr && seqlen_half % Kernel_traits::kBlockM == 0 && seqlen_half % Kernel_traits::kBlockN == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    constexpr int smem_size_dq_dk_dv = Kernel_traits::kSmemSize1colblock;
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                // Several query heads per thread block (GQA, see bwd_heads_per_cta in flash_api.cpp): only without
                // dropout and softcapping (flash_api.cpp then uses one head per thread block), to reduce the number of
                // templates.
                BOOL_SWITCH(params.bwd_heads_per_cta > 1, IsGroupedConst, [&] {
                    // If not IsEvenKConst, we also set IsEvenMNConst to false to reduce number of templates.
                    // If head dim > 128, set IsEvenMNConst to false to reduce number of templates
                    auto kernel = &flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Kernel_traits, Is_dropout && !Is_softcap, IsEvenMNConst && IsEvenKConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst, Is_softcap, IsGroupedConst && !Is_dropout && !Is_softcap>;
                    if (smem_size_dq_dk_dv >= 48 * 1024)  {
                        C10_CUDA_CHECK(cudaFuncSetAttribute(
                            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_dq_dk_dv));
                    }
                    kernel<<<grid_n, Kernel_traits::kNThreads, smem_size_dq_dk_dv, stream>>>(params);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                });
            });
        });
    });
}

template<typename Kernel_traits, bool Is_dropout>
void run_flash_bwd_seqk_parallel(Flash_bwd_params &params, cudaStream_t stream) {
    // The query tiles and key tiles of the two halves of every sample.
    const int seqlen_half = params.seqlen / 2;
    const int num_m_block = (seqlen_half + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid_m(2 * num_m_block, params.b, params.h);
    const int num_n_block = (seqlen_half + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
    int gridDimx = 2 * num_n_block;
    if (params.deterministic) {
        // Thread blocks for about 2 waves, each with its own dQaccum buffer (must match the number of buffers
        // allocated in flash_api.cpp; no GQA grouping in deterministic mode)
        gridDimx = get_deterministic_splits(get_num_sm(get_current_device()), params.b * params.h, 2 * ((seqlen_half + 63) / 64));
    }

    if (!params.deterministic) {
        flash_bwd_dot_do_o_kernel<true, Kernel_traits><<<grid_m, Kernel_traits::kNThreads, 0, stream>>>(params);
    } else {
        flash_bwd_dot_do_o_kernel<false, Kernel_traits><<<grid_m, Kernel_traits::kNThreads, 0, stream>>>(params);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Order of the thread blocks (tile_of_block in diffusion.h): groups of about 4 waves, 2 for head dimensions > 128
    // (the memory-bound noisy diagonal key tiles need the Q / dO tiles of their group in L2). The data shared by the
    // thread blocks of a (batch, head), Q and dO (16 bits) and the FP32 dQ accumulators of the sample, for the
    // bwd_heads_per_cta query heads of a thread block, must fit in twice the L2 for a group: binds on GPUs with little
    // L2 (RTX A6000: GQA grouping 5-10% faster with it at short sequences; A800 neutral, L40S never binding).
    {
        const int64_t bytes_per_bh = int64_t(8) * params.bwd_heads_per_cta * params.seqlen * params.d;
        params.sched_heads = get_sched_heads("FDA_SCHED_HEADS_BWD", 2 * num_n_block, params.b * (params.h / params.bwd_heads_per_cta),
                                             get_num_sm(get_current_device()), /*waves=*/Kernel_traits::kHeadDim > 128 ? 2 : 4, bytes_per_bh);
    }
    run_flash_bwd_dq_dk_dv<Kernel_traits, Is_dropout>(params, stream, gridDimx);

    auto kernel_dq = &flash_bwd_convert_dq_kernel<Kernel_traits>;
    if (Kernel_traits::kSmemdQSize >= 48 * 1024)  {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_dq, cudaFuncAttributeMaxDynamicSharedMemorySize, Kernel_traits::kSmemdQSize));
    }
    kernel_dq<<<grid_m, Kernel_traits::kNThreads, Kernel_traits::kSmemdQSize, stream>>>(params, !params.deterministic ? 1 : gridDimx);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename Kernel_traits, bool Is_dropout>
void run_flash_bwd(Flash_bwd_params &params, cudaStream_t stream) {
#ifndef FLASHATTENTION_DISABLE_BACKWARD
    run_flash_bwd_seqk_parallel<Kernel_traits, Is_dropout>(params, stream);
#endif
}

// The dispatch of FlashAttention-2 (flash-attention 2.8, csrc/flash_attn/src/flash_bwd_launch_template.h) in its format:
// the tile sizes per head dimension, selected with the shared memory available per thread block (163 KB on sm80 /
// sm87, 99 KB on sm86 / sm89). Tuned as the forward (flash_fwd_launch_template.h).

template<typename T>
void run_mha_bwd_hdim32(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 32;
    int max_smem_per_block = get_max_smem_per_block(get_current_device());
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if (max_smem_per_block >= 2 * ((3 * 128 + 2 * 128) * Headdim + 2 * 128 * 128)) { // 104 KB
            if constexpr(!Is_dropout) {  // We can afford more registers to keep V in registers
                run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, true, false, T>, Is_dropout>(params, stream);
            } else {
                run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, false, false, T>, Is_dropout>(params, stream);
            }
        } else {  // 96 KB
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, true, false, T>, Is_dropout>(params, stream);
        }
    });
}

template<typename T>
void run_mha_bwd_hdim64(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    int max_smem_per_block = get_max_smem_per_block(get_current_device());
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // This is slightly faster. We want to split M more so we need fewer registers to store LSE.
        if (max_smem_per_block >= 144 * 1024) {
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, false, false, T>, Is_dropout>(params, stream);
        } else {
            if constexpr(!Is_dropout) {
                run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, true, false, T>, Is_dropout>(params, stream);
            } else {
                // sm86 with dropout: 64 x 128 with AtomLayoutMdQ = 2, V not in registers, is 4.0% faster
                run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 2, false, false, T>, Is_dropout>(params, stream);
            }
        }
    });
}

template<typename T>
void run_mha_bwd_hdim96(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 96;
    int max_smem_per_block = get_max_smem_per_block(get_current_device());
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if (max_smem_per_block >= 116 * 1024) {
            if constexpr(!Is_dropout) {  // 92KB
                run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, true, false, T>, Is_dropout>(params, stream);
            } else {  // 116 KB
                // This is faster for dropout since we don't have many registers to spare
                run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, false, false, T>, Is_dropout>(params, stream);
            }
        } else {
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, true, false, T>, Is_dropout>(params, stream);
        }
    });
}

template<typename T>
void run_mha_bwd_hdim128(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    int max_smem_per_block = get_max_smem_per_block(get_current_device());
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if (max_smem_per_block >= 144 * 1024) {
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 2, false, false, T>, Is_dropout>(params, stream);
        } else {
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, true, false, T>, Is_dropout>(params, stream);
        }
    });
}

template<typename T>
void run_mha_bwd_hdim192(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 192;
    int max_smem_per_block = get_max_smem_per_block(get_current_device());
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if (max_smem_per_block >= 136 * 1024) {
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, false, false, T>, Is_dropout>(params, stream);
        } else {
            if constexpr(!Is_dropout) {
                // FlashAttention-2's 64 x 64 (V in registers, no double buffering: the only 64 x 64 that fits in 99 KB)
                // spills more in our kernel than in FlashAttention-2's, and the local memory traffic (~11 GB per call
                // at 16K tokens) does not stay in an L2 of 6 MB. 32 x 64 tiles fit without V in registers (RTX A6000:
                // 1.1-1.4x FlashAttention-2 instead of 0.6-0.8x; also the best configuration on L40S). They cannot do
                // dropout (16 x 32 blocks of random numbers).
                run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 32, 64, 8, 2, 2, 2, false, false, T>, Is_dropout>(params, stream);
            } else {
                run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, true, true, T>, Is_dropout>(params, stream);
            }
        }
    });
}

template<typename T>
void run_mha_bwd_hdim256(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;
    int max_smem_per_block = get_max_smem_per_block(get_current_device());
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if (max_smem_per_block >= 176 * 1024) {  // H100
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, false, false, T>, Is_dropout>(params, stream);
        } else if (max_smem_per_block >= 144 * 1024) {  // A100, we don't do double buffering to save smem
            run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, false, true, T>, Is_dropout>(params, stream);
        } else { // sm86 and sm89, max smem is 99 KB. Only works without dropout. V in regs and no double buffering.
            if constexpr (!Is_dropout) {
                run_flash_bwd<Flash_bwd_kernel_traits<Headdim, 64, 32, 8, 4, 1, 2, true, true, T>, false>(params, stream);
            } else {
                TORCH_CHECK(false, "Dropout with head dimension 256 in the backward pass needs more than 99 KB of shared memory per thread block (sm80, sm87, sm90)");
            }
        }
    });
}

} // namespace FLASH_NAMESPACE
