/******************************************************************************
 * Copyright (c) 2026, fast-diffusion-attention authors.
 *
 * Definitions of the launch functions of the tuning candidates (tuning.h), included by the generated files
 * flash_{fwd,bwd}_hdim<d>_<dtype>_tune<i>_sm80.cu of tuning builds.
 ******************************************************************************/

#pragma once

#include "tuning.h"

namespace FLASH_NAMESPACE {

// Shared memory of the kernel traits within kMaxSmemSm8x, only checked for candidates whose tiles are valid (the
// kernel traits of the others are not instantiated: they may not compile).
template<bool Tiles_ok, bool Bwd, typename Kernel_traits>
constexpr bool smem_ok() {
    if constexpr (!Tiles_ok) {
        return false;
    } else if constexpr (Bwd) {
        return int(Kernel_traits::kSmemSize1colblock) <= kMaxSmemSm8x;
    } else {
        return int(Kernel_traits::kSmemSize) <= kMaxSmemSm8x;
    }
}

template<int I, int Headdim, typename T> struct FwdCandidate;
#define FDA_DEFINE_FWD_CANDIDATE(I, M, N, W, QR, SH)                                                \
    template<int Headdim, typename T> struct FwdCandidate<I, Headdim, T> {                          \
        using type = Flash_fwd_kernel_traits<Headdim, M, N, W, QR, SH, T>;                          \
        static constexpr bool valid = smem_ok<fwd_valid<Headdim, M, N, W>(), false, type>();       \
    };
FDA_FWD_CANDIDATES(FDA_DEFINE_FWD_CANDIDATE)
#undef FDA_DEFINE_FWD_CANDIDATE

template<int I, int Headdim, typename T> struct BwdCandidate;
#define FDA_DEFINE_BWD_CANDIDATE(I, M, N, W, A1, A2, A3, VR, NDB)                                   \
    template<int Headdim, typename T> struct BwdCandidate<I, Headdim, T> {                          \
        using type = Flash_bwd_kernel_traits<Headdim, M, N, W, A1, A2, A3, VR, NDB, T>;             \
        static constexpr bool valid = smem_ok<bwd_valid<Headdim, M, N, W, A1, A2, A3>(), true, type>(); \
        static constexpr bool dropout_valid = bwd_dropout_valid<N, W, A1>();                        \
    };
FDA_BWD_CANDIDATES(FDA_DEFINE_BWD_CANDIDATE)
#undef FDA_DEFINE_BWD_CANDIDATE

#ifdef FDA_TUNING_FWD_CANDIDATES
template<typename T, int Headdim, int I>
void run_mha_fwd_candidate(Flash_fwd_params &params, cudaStream_t stream) {
    using Candidate = FwdCandidate<I, Headdim, T>;
    if constexpr (Candidate::valid) {
        using Kernel_traits = typename Candidate::type;
        TORCH_CHECK(int(Kernel_traits::kSmemSize) <= get_max_smem_per_block(get_current_device()),
                    "FDA_TUNE_FWD: the candidate needs too much shared memory");
        DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
            run_flash_fwd<Kernel_traits, Is_dropout>(params, stream);
        });
    } else {
        TORCH_CHECK(false, "FDA_TUNE_FWD: the candidate is not valid for this head dimension (tiles or shared memory)");
    }
}
#endif

#ifdef FDA_TUNING_BWD_CANDIDATES
template<typename T, int Headdim, int I>
void run_mha_bwd_candidate(Flash_bwd_params &params, cudaStream_t stream) {
    using Candidate = BwdCandidate<I, Headdim, T>;
    if constexpr (Candidate::valid) {
        using Kernel_traits = typename Candidate::type;
        TORCH_CHECK(int(Kernel_traits::kSmemSize1colblock) <= get_max_smem_per_block(get_current_device()),
                    "FDA_TUNE_BWD: the candidate needs too much shared memory");
        DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
            if constexpr (!Is_dropout || Candidate::dropout_valid) {
                run_flash_bwd<Kernel_traits, Is_dropout>(params, stream);
            } else {
                TORCH_CHECK(false, "FDA_TUNE_BWD: the candidate does not support dropout");
            }
        });
    } else {
        TORCH_CHECK(false, "FDA_TUNE_BWD: the candidate is not valid for this head dimension (tiles, warps or shared memory)");
    }
}
#endif

}  // namespace FLASH_NAMESPACE
