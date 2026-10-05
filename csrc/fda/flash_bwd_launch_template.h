/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 * Copyright (c) 2026, fast-diffusion-attention authors.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include "cutlass/device_kernel.h"  // For device_kernel
#include "cutlass/kernel_launch.h"  // For kernel_launch

#include "cuda_check.h"
#include "static_switch.h"
#include "flash.h"
#include "flash_bwd_preprocess_kernel.h"
#include "flash_bwd_postprocess_kernel.h"
#include "tile_scheduler.hpp"
#include "mainloop_bwd_sm90_tma_gmma_ws.hpp"
#include "epilogue_bwd.hpp"
#include "flash_bwd_kernel_sm90.h"

using namespace cute;

template <int kHeadDim, int kBlockM, int kBlockN, typename Element,
          bool Has_softcap, bool Varlen, bool Deterministic, bool GQA,
          int Stages_dO=2, int Stages_dS=2,
          bool SdP_swapAB=true, bool dKV_swapAB=false, bool dQ_swapAB=false,
          int NumMmaWarpGroups=2, int AtomLayoutMSdP=1, int AtomLayoutNdKV=2, int AtomLayoutMdQ=1,
          bool V_in_regs=false>
void run_flash_bwd(Flash_bwd_params &params, cudaStream_t stream) {
    using ElementAccum = float;
    using ArchTag = cutlass::arch::Sm90;

    bool const is_varlen = params.cu_seqlens;
    int const seqlen = !is_varlen ? params.seqlen : params.total;
    int const batch = !is_varlen ? params.b : 1;
    // Maximum length of a half of a sample
    int const n_max = (params.seqlen + 1) / 2;

    using TileShape_MK = cute::Shape<Int<kBlockM>, Int<kHeadDim>>;
    using PreprocessKernel = flash::FlashAttnBwdPreprocess<TileShape_MK, Element, ElementAccum, ArchTag, /*Clear_dQaccum=*/true, Varlen>;
    typename PreprocessKernel::Arguments preprocess_args {
        static_cast<Element const*>(params.o_ptr),
        {seqlen, params.d, params.h, batch},  // shape_O
        {params.o_row_stride, _1{}, params.o_head_stride, !is_varlen ? params.o_batch_stride : 0},  // stride_O
        static_cast<Element const*>(params.do_ptr),
        {params.do_row_stride, _1{}, params.do_head_stride, !is_varlen ? params.do_batch_stride : 0},  // stride_dO
        static_cast<float*>(params.dsoftmax_sum),
        {params.total_q_padded, params.h},  // shape_dPsum
        {_1{}, params.total_q_padded},  // stride_dPsum
        static_cast<float*>(params.softmax_lse_ptr),
        {_1{}, seqlen, !is_varlen ? params.h * seqlen : 0},  // stride_LSE
        static_cast<float*>(params.softmax_lse_log2_ptr),
        {_1{}, params.total_q_padded},  // stride_LSE_log2
        static_cast<ElementAccum*>(params.dq_accum_ptr),
        {params.total_q_padded * params.d_rounded, params.h},  // shape_dQaccum
        {_1{}, params.total_q_padded * params.d_rounded},  // stride_dQaccum
        params.seqlen,
        params.cu_seqlens
    };
    typename PreprocessKernel::Params preprocess_params = PreprocessKernel::to_underlying_arguments(preprocess_args);
    int num_m_block = 2 * cute::ceil_div(n_max, kBlockM);
    dim3 grid_m(num_m_block, params.h, params.b);
    CHECK_CUTLASS(cutlass::kernel_launch<PreprocessKernel>(grid_m, PreprocessKernel::MaxThreadsPerBlock, PreprocessKernel::SharedStorageSize, stream, preprocess_params, false /*launch_with_pdl*/));

    using TileShape_MNK = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    static constexpr int Stages = 2;
    using CollectiveMainloop = flash::CollectiveMainloopBwdSm90<Stages, Stages_dO, Stages_dS, TileShape_MNK, Element, ElementAccum, ArchTag,
            Has_softcap, Varlen, Deterministic,
            SdP_swapAB, dKV_swapAB, dQ_swapAB, NumMmaWarpGroups, AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ, V_in_regs>;
    using CollectiveEpilogue = std::conditional_t<
        !GQA,
        flash::CollectiveEpilogueBwd<TileShape_MNK, Element, ArchTag, CollectiveMainloop::NumMmaThreads, Varlen, dKV_swapAB, NumMmaWarpGroups / AtomLayoutNdKV>,
        flash::CollectiveEpilogueBwdGQA<TileShape_MNK, ElementAccum, ArchTag, CollectiveMainloop::NumMmaThreads, Varlen, Deterministic>
    >;
    using Scheduler = flash::DiffusionBwdScheduler<Varlen, kBlockN>;
    using AttnKernel = flash::enable_sm90<flash::FlashAttnBwdSm90<CollectiveMainloop, CollectiveEpilogue, Scheduler>>;

    typename CollectiveMainloop::Arguments mainloop_args {
        static_cast<Element const*>(params.q_ptr),
        {seqlen, params.d, params.h, batch},  // shape_Q
        {params.q_row_stride, _1{}, params.q_head_stride, !is_varlen ? params.q_batch_stride : 0},  // stride_Q
        static_cast<Element const*>(params.k_ptr),
        {seqlen, params.d, params.h_k, batch},  // shape_K
        {params.k_row_stride, _1{}, params.k_head_stride, !is_varlen ? params.k_batch_stride : 0},  // stride_K
        static_cast<Element const*>(params.v_ptr),
        {params.v_row_stride, _1{}, params.v_head_stride, !is_varlen ? params.v_batch_stride : 0},  // stride_V
        static_cast<Element const*>(params.do_ptr),
        {params.do_row_stride, _1{}, params.do_head_stride, !is_varlen ? params.do_batch_stride : 0},  // stride_dO
        static_cast<ElementAccum*>(params.dq_accum_ptr),
        {params.total_q_padded * params.d_rounded, params.h},  // shape_dQaccum
        {_1{}, params.total_q_padded * params.d_rounded}, // stride_dQaccum
        static_cast<float*>(params.softmax_lse_log2_ptr),
        {params.total_q_padded, params.h},  // shape_LSE
        {_1{}, params.total_q_padded},  // stride_LSE_log2
        static_cast<float*>(params.dsoftmax_sum),
        {_1{}, params.total_q_padded},  // stride_dPsum
        params.scale_softmax,
        params.softcap,
        params.block_size,
        params.dq_semaphore,
        params.cu_seqlens
    };
    // The case work with GQA is ugly but idk how to fix it.
    typename CollectiveEpilogue::Arguments epilogue_args = [&] {
        if constexpr (!GQA) {
            return typename CollectiveEpilogue::Arguments {
                static_cast<Element*>(params.dk_ptr),
                {seqlen, params.d, params.h, batch},  // shape_dK
                {params.dk_row_stride, _1{}, params.dk_head_stride, !is_varlen ? params.dk_batch_stride : 0},  // stride_dK
                static_cast<Element*>(params.dv_ptr),
                {params.dv_row_stride, _1{}, params.dv_head_stride, !is_varlen ? params.dv_batch_stride : 0},  // stride_dV
                params.h,
                params.dk_semaphore,
                params.dv_semaphore,
                params.cu_seqlens
            };
        } else {
            return typename CollectiveEpilogue::Arguments {
                static_cast<ElementAccum*>(params.dk_accum_ptr),
                {params.total_k_padded * params.d_rounded, params.h_k},  // shape_dKaccum
                {_1{}, params.total_k_padded * params.d_rounded},  // stride_dKaccum
                static_cast<ElementAccum*>(params.dv_accum_ptr),
                {_1{}, params.total_k_padded * params.d_rounded},  // stride_dVaccum
                params.h,
                params.seqlen,
                params.dk_semaphore,
                params.dv_semaphore,
                params.cu_seqlens
            };
        }
    }();

    // Key tiles of the clean half, then of the noisy half
    int const num_blocks_half = cutlass::ceil_div(n_max, kBlockN);
    typename flash::TileSchedulerArguments scheduler_args {
        2 * num_blocks_half, params.h, params.b,
        params.h / params.h_k,
        params.seqlen,
        params.d, sizeof(Element),
        params.tile_count_semaphore, params.cu_seqlens,
        nullptr, nullptr, nullptr,
        num_blocks_half /*num_blocks_noisy*/,
        Deterministic /*spt*/
    };

    int device;
    cudaGetDevice(&device);
    typename AttnKernel::Params kernel_params = AttnKernel::to_underlying_arguments({
        mainloop_args, epilogue_args, {device, params.num_sm}, scheduler_args
    });

    dim3 grid_dims = AttnKernel::get_grid_shape(kernel_params);
    dim3 block_dims = AttnKernel::get_block_shape();
    int smem_size = AttnKernel::SharedStorageSize;
    if (smem_size >= 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(cutlass::device_kernel<AttnKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    CHECK_CUTLASS(cutlass::kernel_launch<AttnKernel>(grid_dims, block_dims, smem_size, stream, kernel_params, false /*launch_with_pdl*/));

    using PostprocessKernel = flash::FlashAttnBwdPostprocessConvertdQ<TileShape_MK, Element, ElementAccum, ArchTag,
        AttnKernel::CollectiveMainloop::NumMmaThreads,
        typename AttnKernel::CollectiveMainloop::TiledMmadQ,
        AttnKernel::CollectiveMainloop::dQ_swapAB,
        Varlen
        >;
    typename PostprocessKernel::Arguments postprocess_args {
        static_cast<ElementAccum const*>(params.dq_accum_ptr),
        {params.total_q_padded * params.d_rounded, params.h},  // shape_dQaccum
        {_1{}, params.total_q_padded * params.d_rounded}, // stride_dQaccum
        static_cast<Element*>(params.dq_ptr),
        {seqlen, params.d, params.h, batch},  // shape_dQ
        {params.dq_row_stride, _1{}, params.dq_head_stride, params.dq_batch_stride},  // stride_dQ
        params.scale_softmax,
        params.seqlen,
        params.cu_seqlens
    };
    typename PostprocessKernel::Params postprocess_params = PostprocessKernel::to_underlying_arguments(postprocess_args);
    dim3 grid_m_postprocess(num_m_block, params.h, params.b);
    int smem_size_postprocess = PostprocessKernel::SharedStorageSize;
    if (smem_size_postprocess >= 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(cutlass::device_kernel<PostprocessKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_postprocess));
    }
    CHECK_CUTLASS(cutlass::kernel_launch<PostprocessKernel>(grid_m_postprocess, PostprocessKernel::MaxThreadsPerBlock, smem_size_postprocess, stream, postprocess_params, false /*launch_with_pdl*/));

    if constexpr (GQA) {
        using TileShape_NK = cute::Shape<Int<kBlockN>, Int<kHeadDim>>;
        using PostprocessKerneldKV = flash::FlashAttnBwdPostprocessConvertdQ<TileShape_NK, Element, ElementAccum, ArchTag,
            AttnKernel::CollectiveEpilogue::NumEpilogueThreads,
            typename AttnKernel::CollectiveMainloop::TiledMmadKV,
            AttnKernel::CollectiveMainloop::dKV_swapAB,
            Varlen
            >;
        typename PostprocessKerneldKV::Arguments postprocess_dK_args {
            static_cast<ElementAccum const*>(params.dk_accum_ptr),
            {params.total_k_padded * params.d_rounded, params.h_k},  // shape_dKaccum
            {_1{}, params.total_k_padded * params.d_rounded},  // stride_dKaccum
            static_cast<Element*>(params.dk_ptr),
            {seqlen, params.d, params.h_k, batch},  // shape_dK
            {params.dk_row_stride, _1{}, params.dk_head_stride, params.dk_batch_stride},  // stride_dK
            1.f,
            params.seqlen,
            params.cu_seqlens
        };
        typename PostprocessKerneldKV::Params postprocess_dK_params = PostprocessKerneldKV::to_underlying_arguments(postprocess_dK_args);
        typename PostprocessKerneldKV::Arguments postprocess_dV_args {
            static_cast<ElementAccum const*>(params.dv_accum_ptr),
            {params.total_k_padded * params.d_rounded, params.h_k},  // shape_dVaccum
            {_1{}, params.total_k_padded * params.d_rounded},  // stride_dVaccum
            static_cast<Element*>(params.dv_ptr),
            {seqlen, params.d, params.h_k, batch},  // shape_dV
            {params.dv_row_stride, _1{}, params.dv_head_stride, params.dv_batch_stride},  // stride_dV
            1.f,
            params.seqlen,
            params.cu_seqlens
        };
        typename PostprocessKerneldKV::Params postprocess_dV_params = PostprocessKerneldKV::to_underlying_arguments(postprocess_dV_args);
        dim3 grid_n_postprocess(2 * num_blocks_half, params.h_k, params.b);
        int smem_size_postprocess = PostprocessKerneldKV::SharedStorageSize;
        if (smem_size_postprocess >= 48 * 1024) {
            CHECK_CUDA(cudaFuncSetAttribute(cutlass::device_kernel<PostprocessKerneldKV>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_postprocess));
        }
        CHECK_CUTLASS(cutlass::kernel_launch<PostprocessKerneldKV>(grid_n_postprocess, PostprocessKerneldKV::MaxThreadsPerBlock, smem_size_postprocess, stream, postprocess_dK_params, false /*launch_with_pdl*/));
        CHECK_CUTLASS(cutlass::kernel_launch<PostprocessKerneldKV>(grid_n_postprocess, PostprocessKerneldKV::MaxThreadsPerBlock, smem_size_postprocess, stream, postprocess_dV_params, false /*launch_with_pdl*/));
    }

}

template<typename T, int kBlockM, int kBlockN, int kHeadDim, bool Has_softcap,
         int Stages_dO=2, int Stages_dS=2,
         bool SdP_swapAB=true, bool dKV_swapAB=false, bool dQ_swapAB=false,
         int NumMmaWarpGroups=2, int AtomLayoutMSdP=1, int AtomLayoutNdKV=2, int AtomLayoutMdQ=1,
         bool V_in_regs=false>
void run_mha_bwd_dispatch(Flash_bwd_params &params, cudaStream_t stream) {
    VARLEN_SWITCH(params.cu_seqlens != nullptr, Varlen, [&] {
        BOOL_SWITCH(params.h != params.h_k, GQA, [&] {
            BOOL_SWITCH(params.deterministic, Deterministic_, [&] {
                static constexpr bool Deterministic = Deterministic_ && kHeadDim < 256;
                run_flash_bwd<kHeadDim, kBlockM, kBlockN, T, Has_softcap, Varlen, Deterministic, GQA, Stages_dO, Stages_dS, SdP_swapAB, dKV_swapAB, dQ_swapAB, NumMmaWarpGroups, AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ, V_in_regs>(params, stream);
            });
        });
    });
}

// Tile sizes: those of causal attention of FlashAttention-3. kBlockM and kBlockN must match flash_api.cpp.

template<int Arch, typename T, bool Has_softcap>
void run_mha_bwd_hdim64(Flash_bwd_params &params, cudaStream_t stream) {
    if constexpr (Has_softcap) {
        // register spill with 128 x 128
        run_mha_bwd_dispatch<T, 96, 128, 64, Has_softcap, 2, 2, true, false, true, 2, 1, 2, 2, false>(params, stream);
    } else {
        run_mha_bwd_dispatch<T, 128, 128, 64, Has_softcap, 2, 2, true, false, false, 2, 1, 2, 2, false>(params, stream);
    }
}

template<int Arch, typename T, bool Has_softcap>
void run_mha_bwd_hdim96(Flash_bwd_params &params, cudaStream_t stream) {
    run_mha_bwd_dispatch<T, 64, 128, 96, Has_softcap, 2, 2, true, false, false, 2, 1, 2, 1, true>(params, stream);
}

template<int Arch, typename T, bool Has_softcap>
void run_mha_bwd_hdim128(Flash_bwd_params &params, cudaStream_t stream) {
    run_mha_bwd_dispatch<T, 64, 128, 128, Has_softcap, 2, 2, true, false, false, 2, 1, 2, 1, false>(params, stream);
}

template<int Arch, typename T, bool Has_softcap>
void run_mha_bwd_hdim192(Flash_bwd_params &params, cudaStream_t stream) {
    run_mha_bwd_dispatch<T, 64, 96, 192, Has_softcap, 1, 1, false, true, false, 3, 1, 1, 1, false>(params, stream);
}

template<int Arch, typename T, bool Has_softcap>
void run_mha_bwd_hdim256(Flash_bwd_params &params, cudaStream_t stream) {
    run_mha_bwd_dispatch<T, 64, 80, 256, Has_softcap, 1, 1, false, true, true, 2, 1, 1, 1, false>(params, stream);
}
