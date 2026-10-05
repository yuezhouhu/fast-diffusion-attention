/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 * Copyright (c) 2026, fast-diffusion-attention authors.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"  // For device_kernel
#include <cutlass/kernel_hardware_info.h>
#include "cutlass/kernel_launch.h"

#include "cuda_check.h"
#include "static_switch.h"
#include "flash.h"
#include "tile_size.h"
#include "tile_scheduler.hpp"
#include "flash_fwd_kernel_sm90.h"
#include "mainloop_fwd_sm90_tma_gmma_ws.hpp"
#include "epilogue_fwd.hpp"

using namespace cute;

template <int kHeadDim, typename Element, typename ElementOut, bool Has_softcap, bool Varlen>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    static constexpr bool Is_FP8 = cute::is_same_v<Element, cutlass::float_e4m3_t> || cute::is_same_v<Element, cutlass::float_e5m2_t>;
    using ArchTag = cutlass::arch::Sm90;

    // Can't use structured binding since it's not compatible with constexpr
    static constexpr std::tuple<int, int, bool, bool> kBlockMN_RS_IntraWGOverlap = tile_size_fwd_sm90(kHeadDim, sizeof(Element) /*element_size*/, Has_softcap);
    static constexpr int kBlockM = std::get<0>(kBlockMN_RS_IntraWGOverlap);
    static constexpr int kBlockN = std::get<1>(kBlockMN_RS_IntraWGOverlap);
    static constexpr bool MmaPV_is_RS = std::get<2>(kBlockMN_RS_IntraWGOverlap);
    static constexpr bool IntraWGOverlap = std::get<3>(kBlockMN_RS_IntraWGOverlap);
    static constexpr int kStages = 2;

    using TileShape_MNK = cute::Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
    using TileShape_MNK_PV = cute::Shape<Int<kBlockM>, Int<kHeadDim>, Int<kBlockN>>;
    using CollectiveMainloop = flash::CollectiveMainloopFwdSm90<kStages, TileShape_MNK, Element, float, ArchTag, Has_softcap, Varlen, MmaPV_is_RS, IntraWGOverlap>;
    using CollectiveEpilogue = flash::CollectiveEpilogueFwd<TileShape_MNK_PV, ElementOut, ArchTag, CollectiveMainloop::NumMmaThreads, Varlen, Is_FP8 /*FP8PermuteCol*/>;

    static constexpr int NumProducerThreads = CollectiveMainloop::NumProducerThreads;
    using Scheduler = std::conditional_t<Varlen,
        flash::VarlenDynamicPersistentTileScheduler<kBlockM, kBlockN, CollectiveMainloop::NumMmaThreads, NumProducerThreads>,
        flash::DiffusionDynamicPersistentTileScheduler<CollectiveMainloop::NumMmaThreads, NumProducerThreads>
    >;
    using AttnKernel = flash::enable_sm90<flash::FlashAttnFwdSm90<CollectiveMainloop, CollectiveEpilogue, Scheduler>>;

    bool const is_varlen = params.cu_seqlens;
    int seqlen = !is_varlen ? params.seqlen : params.total;
    int batch = !is_varlen ? params.b : 1;
    typename CollectiveMainloop::Arguments mainloop_args {
        static_cast<Element const*>(params.q_ptr),
        {seqlen, params.d, params.h, batch},  // shape_Q
        {params.q_row_stride, _1{}, params.q_head_stride, !is_varlen ? params.q_batch_stride : 0},  // stride_Q
        static_cast<Element const*>(params.k_ptr),
        {seqlen, params.d, params.h_k, batch},  // shape_K
        {params.k_row_stride, _1{}, params.k_head_stride, !is_varlen ? params.k_batch_stride : 0},  // stride_K
        static_cast<Element const*>(params.v_ptr),
        {params.v_row_stride, _1{}, params.v_head_stride, !is_varlen ? params.v_batch_stride : 0},  // stride_V
        params.scale_softmax,
        params.q_descale_ptr, params.k_descale_ptr, params.v_descale_ptr,
        {params.q_descale_batch_stride, params.q_descale_head_stride},
        {params.k_descale_batch_stride, params.k_descale_head_stride},
        {params.v_descale_batch_stride, params.v_descale_head_stride},
        params.softcap,
        params.block_size,
        params.cu_seqlens
    };
    typename CollectiveEpilogue::Arguments epilogue_args {
        static_cast<ElementOut*>(params.o_ptr),
        {seqlen, params.d, params.h, batch},  // shape_O
        {params.o_row_stride, _1{}, params.o_head_stride, !is_varlen ? params.o_batch_stride : 0}, // stride_O
        static_cast<float*>(params.softmax_lse_ptr),
        {_1{}, seqlen, !is_varlen ? params.h * seqlen : 0},  // stride_LSE
        params.cu_seqlens
    };

    // Query tiles of both halves of a sample
    int num_blocks_m = 2 * cutlass::ceil_div(params.seqlen / 2, kBlockM);
    typename flash::TileSchedulerArguments scheduler_args {
        num_blocks_m, params.h, params.b,
        params.h / params.h_k,
        params.seqlen,
        params.d, sizeof(Element),
        params.tile_count_semaphore, params.cu_seqlens,
        params.num_m_blocks_ptr,
        params.varlen_batch_idx_ptr,
        params.num_nheads_in_l2_ptr
    };

    if (Varlen) {
        prepare_varlen_num_blocks(params, stream, kBlockM, kBlockN);
        CHECK_CUDA_KERNEL_LAUNCH();
    }

    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    typename AttnKernel::Params kernel_params = AttnKernel::to_underlying_arguments({
        mainloop_args, epilogue_args, {device, params.num_sm}, scheduler_args
    });

    dim3 grid_dims = AttnKernel::get_grid_shape(kernel_params);
    dim3 block_dims = AttnKernel::get_block_shape();
    int smem_size = AttnKernel::SharedStorageSize;
    auto kernel = cutlass::device_kernel<AttnKernel>;
    if (smem_size >= 48 * 1024) {
        CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    CHECK_CUTLASS(cutlass::kernel_launch<AttnKernel>(grid_dims, block_dims, smem_size, stream, kernel_params, false /*launch_with_pdl*/));
}

template<int Arch, typename T, int kHeadDim, bool Has_softcap>
void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream) {
    static_assert(sizeof(T) == 2 || sizeof(T) == 1, "Only 16bit and 8bit are supported");
    static constexpr bool Is_FP8 = cute::is_same_v<T, cutlass::float_e4m3_t> || cute::is_same_v<T, cutlass::float_e5m2_t>;
    using T_out = std::conditional_t<!Is_FP8, T, cutlass::bfloat16_t>;
    VARLEN_SWITCH(params.cu_seqlens, Varlen, [&] {
        run_flash_fwd<kHeadDim, T, T_out, Has_softcap, Varlen>(params, stream);
    });
}
