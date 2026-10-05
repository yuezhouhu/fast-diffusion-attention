/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 * Copyright (c) 2026, fast-diffusion-attention authors.
 ******************************************************************************/

#pragma once

#include "cute/tensor.hpp"

#include <cutlass/cutlass.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/kernel_hardware_info.h>
#include "cutlass/pipeline/pipeline.hpp"

#include "cutlass/arch/grid_dependency_control.h"

#include "diffusion.h"
#include "utils.h"
#include "softmax.h"

namespace flash {

using namespace cute;

template <class CollectiveMainloop_, class CollectiveEpilogue_, class TileScheduler_>
class FlashAttnFwdSm90 {

public:

    // Type Aliases
    using CollectiveMainloop = CollectiveMainloop_;
    using CollectiveEpilogue = CollectiveEpilogue_;
    static_assert(CollectiveMainloop::Varlen == CollectiveEpilogue::Varlen);
    static constexpr bool Has_softcap = CollectiveMainloop::Has_softcap;
    static constexpr bool Varlen = CollectiveMainloop::Varlen;
    static constexpr bool Is_FP8 = CollectiveMainloop::Is_FP8;
    static constexpr bool Transpose_V = CollectiveMainloop::Transpose_V;
    static constexpr int NumProducerThreads = CollectiveMainloop::NumProducerThreads;
    using SeqlenInfo_t = typename CollectiveMainloop::SeqlenInfo_t;

    // Mainloop derived types
    using TileShape_MNK_PV = typename CollectiveMainloop::TileShape_MNK_PV;
    using TiledMmaPV = typename CollectiveMainloop::TiledMmaPV;
    using ArchTag = typename CollectiveMainloop::ArchTag;
    using ClusterShape = typename CollectiveMainloop::ClusterShape;
    using MainloopArguments = typename CollectiveMainloop::Arguments;
    using MainloopParams = typename CollectiveMainloop::Params;
    using BarrierQ = cutlass::arch::ClusterTransactionBarrier;

    // Epilogue derived types
    using EpilogueArguments = typename CollectiveEpilogue::Arguments;
    using EpilogueParams = typename CollectiveEpilogue::Params;

    static_assert(ArchTag::kMinComputeCapability >= 90);

    using TileScheduler = TileScheduler_;
    using TileSchedulerArguments = typename flash::TileSchedulerArguments;
    using TileSchedulerParams = typename TileScheduler::Params;

    static constexpr uint32_t NumLoadWarpGroups = 1;
    static constexpr uint32_t NumMmaWarpGroups = CUTE_STATIC_V(size(TiledMmaPV{})) / cutlass::NumThreadsPerWarpGroup;
    static constexpr uint32_t MaxThreadsPerBlock = CUTE_STATIC_V(size(TiledMmaPV{})) + (NumLoadWarpGroups * cutlass::NumThreadsPerWarpGroup);
    static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
    static_assert(NumMmaWarpGroups == 1 || NumMmaWarpGroups == 2 || NumMmaWarpGroups == 3);

    /// Register requirement for Load and Math WGs
    static constexpr uint32_t LoadRegisterRequirement = NumMmaWarpGroups == 1 ? 56 : (NumMmaWarpGroups == 2 ? 24 : 32);
    static constexpr uint32_t MmaRegisterRequirement = NumMmaWarpGroups == 1 ? 256 : (NumMmaWarpGroups == 2 ? 240 : 160);
    // If you want to print from the producer warp, you'd need to increase the number of registers
    // Otherwise you'll get CUDA error.
    // static constexpr uint32_t LoadRegisterRequirement = 40;
    // static constexpr uint32_t MmaRegisterRequirement = NumMmaWarpGroups == 2 ? 232 : 152;

    // Kernel level shared memory storage
    // We overlap the shared memory for the mainloop and epilogue. However, we only want smem_o to overlap with smem_v
    // and nothing else, so we'll pad in case sizeof(smem_o) > sizeof(smem_v).
    static constexpr int mainloop_smem_padding_ = int(sizeof(typename CollectiveEpilogue::TensorStorage)) - int(sizeof(decltype((typename CollectiveMainloop::TensorStorage{}).smem_v)));
    static constexpr int mainloop_smem_padding = mainloop_smem_padding_ < 0 ? 0 : mainloop_smem_padding_;
    struct SharedStorage {
        struct TensorStorage : cute::aligned_struct<128, _1> {
            union {
                struct {
                    cute::array<uint32_t, mainloop_smem_padding / sizeof(uint32_t)> padding_;
                    typename CollectiveMainloop::TensorStorage mainloop;
                };
                // We want smem_o to line up with the start of smem_v
                typename CollectiveEpilogue::TensorStorage epilogue;
            };
        } tensors;
        struct PipelineStorage : cute::aligned_struct<16, _1> {
            alignas(16) BarrierQ barrier_Q;
            alignas(16) cutlass::arch::ClusterBarrier barrier_O;
            alignas(16) typename CollectiveMainloop::MainloopPipelineK::SharedStorage pipeline_k;
            alignas(16) typename CollectiveMainloop::MainloopPipelineV::SharedStorage pipeline_v;
            alignas(16) typename CollectiveMainloop::MainloopPipelineVt::SharedStorage pipeline_vt;
            alignas(16) typename TileScheduler::SharedStorage smem_scheduler;
        } pipelines;

    };

    static constexpr int SharedStorageSize = sizeof(SharedStorage);

    // Device side arguments
    struct Arguments {
        MainloopArguments mainloop{};
        EpilogueArguments epilogue{};
        cutlass::KernelHardwareInfo hw_info{};
        TileSchedulerArguments scheduler{};
    };

    // Kernel entry point API
    struct Params {
        MainloopParams mainloop{};
        EpilogueParams epilogue{};
        cutlass::KernelHardwareInfo hw_info{};
        TileSchedulerParams scheduler{};
    };

    //
    // Methods
    //

    // Convert to underlying arguments. In this case, a simple copy for the aliased type.
    static
    Params
    to_underlying_arguments(Arguments const& args) {
        CUTLASS_TRACE_HOST("to_underlying_arguments():");

        // Get SM count if needed, otherwise use user supplied SM count
        int sm_count = args.hw_info.sm_count;
        if (sm_count <= 0) {
            CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
                "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
            sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
        }

        CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);

        cutlass::KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count};
        return {
            CollectiveMainloop::to_underlying_arguments(args.mainloop),
            CollectiveEpilogue::to_underlying_arguments(args.epilogue),
            hw_info,
            TileScheduler::to_underlying_arguments(args.scheduler)
        };
    }

    // Computes the kernel launch grid shape based on runtime parameters
    static dim3
    get_grid_shape(Params const& params) {
        return TileScheduler::get_grid_shape(params.scheduler, params.hw_info.sm_count);
    }

    static dim3
    get_block_shape() {
        return dim3(MaxThreadsPerBlock, 1, 1);
    }

    CUTLASS_DEVICE
    void
    operator()(Params const& params, char* smem_buf) {

        static constexpr int NumMmaThreads = NumMmaWarpGroups * cutlass::NumThreadsPerWarpGroup;
        static constexpr int MmaThreadOffset = NumLoadWarpGroups * cutlass::NumThreadsPerWarpGroup;
        static constexpr int kBlockM = get<0>(TileShape_MNK_PV{});

        using MainloopPipelineK = typename CollectiveMainloop::MainloopPipelineK;
        using MainloopPipelineV = typename CollectiveMainloop::MainloopPipelineV;
        using MainloopPipelineVt = typename CollectiveMainloop::MainloopPipelineVt;
        using PipelineState = typename CollectiveMainloop::PipelineState;
        using PipelineParamsK = typename MainloopPipelineK::Params;
        using PipelineParamsV = typename MainloopPipelineV::Params;
        using PipelineParamsVt = typename MainloopPipelineVt::Params;

        SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

        int const lane_predicate = cute::elect_one_sync();
        int const warp_idx = cutlass::canonical_warp_idx_sync();

        // Issue Tma Descriptor Prefetch from a single thread
        if (warp_idx == 0 && lane_predicate) {
            CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
            CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
        }

        // Obtain warp index
        int const warp_group_thread_idx = threadIdx.x % cutlass::NumThreadsPerWarpGroup;
        int warp_group_idx = cutlass::canonical_warp_group_idx();

        if (warp_idx == 0 && lane_predicate) {
            shared_storage.pipelines.barrier_Q.init(1 /*numThreads*/);
            shared_storage.pipelines.barrier_O.init(size(ClusterShape{}) * NumMmaThreads /*numThreads*/);
        }

        // We're counting on pipeline_k to call cutlass::arch::fence_barrier_init();
        PipelineParamsK pipeline_params_k;
        pipeline_params_k.role = warp_group_idx == 0
            ? MainloopPipelineK::ThreadCategory::Producer
            : MainloopPipelineK::ThreadCategory::Consumer;
        pipeline_params_k.transaction_bytes = CollectiveMainloop::TmaTransactionBytesK;
        pipeline_params_k.is_leader = warp_group_thread_idx == 0;
        pipeline_params_k.num_consumers = NumMmaThreads;

        static_assert(is_same_v<PipelineParamsK, PipelineParamsVt>);
        PipelineParamsVt pipeline_params_vt = pipeline_params_k;

        MainloopPipelineK pipeline_k(shared_storage.pipelines.pipeline_k, pipeline_params_k, ClusterShape{});
        MainloopPipelineV pipeline_v = [&] {
            if constexpr (!Transpose_V) {
                static_assert(is_same_v<PipelineParamsK, PipelineParamsV>);
                return MainloopPipelineV(shared_storage.pipelines.pipeline_v, pipeline_params_vt, ClusterShape{});
            } else {
                PipelineParamsV pipeline_params_v;
                pipeline_params_v.role = warp_group_idx == 0
                    ? MainloopPipelineV::ThreadCategory::Producer
                    : MainloopPipelineV::ThreadCategory::Consumer;
                pipeline_params_v.producer_arv_count = NumProducerThreads;
                pipeline_params_v.consumer_arv_count = NumMmaThreads;
                return MainloopPipelineV(shared_storage.pipelines.pipeline_v, pipeline_params_v);
            }
        }();
        // If we need to transpose V (e.g. FP8 and V is row-major), we use pipeline_vt for the TMA, then
        // the producer WG will read from pipeline_vt and write to pipeline_v.
        // If we don't need to transpose V, we use pipeline_v for the TMA, and pipeline_vt won't be used.
        // Technically for pipeline_params_vt, warp0 of WG0 is the producer and all of WG0 are consumers.
        // However, the thread role isn't used in the pipeline implementation.
        MainloopPipelineVt pipeline_vt = [&] {
            pipeline_params_vt.num_consumers = NumProducerThreads; // TMA_V is only consumed by the producer WG
            return MainloopPipelineVt(shared_storage.pipelines.pipeline_vt, pipeline_params_vt, ClusterShape{});
        }();

        CollectiveMainloop mainloop;
        CollectiveEpilogue epilogue;

        // We need this to guarantee that the Pipeline init is visible to all producers and consumer blocks in the Cluster
        __syncthreads();

        TileScheduler scheduler(reinterpret_cast<typename TileScheduler::SharedStorage*>(&shared_storage.pipelines.smem_scheduler));

        if (warp_group_idx == 0) {  // Producer
            cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();

            PipelineState smem_pipe_write = cutlass::make_producer_start_state<MainloopPipelineK>();
            int work_idx = 0;
            int warp_idx_in_warpgroup = __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
            static constexpr bool SingleProducerWarp = NumProducerThreads == cutlass::NumThreadsPerWarp;
            if constexpr (SingleProducerWarp) {
                if (warp_idx_in_warpgroup != 0) { return; }
            }
            if (!SingleProducerWarp && warp_idx_in_warpgroup != 0) { scheduler.init_consumer(); }

            cutlass::arch::wait_on_dependent_grids();

            // Load Q, K, V
            for (auto work_tile_info = SingleProducerWarp || warp_idx_in_warpgroup == 0 ? scheduler.template get_initial_work</*IsProducerWarp=*/true>(params.scheduler) : scheduler.template get_initial_work</*IsProducerWarp=*/false>(params.scheduler);
                 work_tile_info.is_valid(params.scheduler);
                 work_tile_info = SingleProducerWarp || warp_idx_in_warpgroup == 0 ? scheduler.template get_next_work</*IsProducerWarp=*/true>(params.scheduler, work_tile_info) : scheduler.template get_next_work</*IsProducerWarp=*/false>(params.scheduler, work_tile_info)) {

                auto block_coord = work_tile_info.get_block_coord(params.scheduler);
                SeqlenInfo_t seqlen_info{get<2>(block_coord) /*bidb*/, get<0>(params.mainloop.shape_Q), params.mainloop.cu_seqlens};
                auto scheduler_prefetch = [&scheduler, &params, &work_tile_info]() {
                    scheduler.prefetch_next_work(params.scheduler, work_tile_info);
                };
                // pipeline_vt won't be used if we don't need to transpose V.
                mainloop.load(params.mainloop, pipeline_k, pipeline_v, pipeline_vt, smem_pipe_write,
                              shared_storage, scheduler_prefetch, seqlen_info, block_coord, work_idx);
            }
            mainloop.load_tail(pipeline_k, pipeline_v, pipeline_vt, smem_pipe_write, shared_storage, work_idx);
        } else {  // Consumer
            cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();

            // Initialize matmul objects.
            TiledMmaPV tiled_mma_pv;

            PipelineState smem_pipe_read;
            // We don't need separate variables smem_pipe_release_k and smem_pipe_release_v
            // (like in Cutlass's gemm) because the read and release pipeline states are always the same.

            scheduler.init_consumer();
            mainloop.mma_init();

            int work_idx = 0;
            CUTLASS_PRAGMA_NO_UNROLL
            for (auto work_tile_info = scheduler.template get_initial_work</*IsProducerWarp=*/false>(params.scheduler);
                 work_tile_info.is_valid(params.scheduler);
                 // get_next_work will be called before the epilogue
                 ) {
                auto block_coord = work_tile_info.get_block_coord(params.scheduler);
                int const bidb = get<2>(block_coord);
                SeqlenInfo_t seqlen_info{bidb, get<0>(params.mainloop.shape_Q), params.mainloop.cu_seqlens};
                // If there's tanh softcap, the scaling will be done before tanh.
                float softmax_scale_log2 = params.mainloop.softmax_scale_log2;
                if constexpr (Is_FP8 && !Has_softcap) {
                    int const bidh = get<1>(block_coord);
                    int const bidh_kv = params.mainloop.qhead_per_khead_divmod.divide(bidh);
                    float const q_descale = params.mainloop.ptr_q_descale == nullptr ? 1.0f : params.mainloop.ptr_q_descale[bidb * get<0>(params.mainloop.stride_q_descale) + bidh_kv * get<1>(params.mainloop.stride_q_descale)];
                    float const k_descale = params.mainloop.ptr_k_descale == nullptr ? 1.0f : params.mainloop.ptr_k_descale[bidb * get<0>(params.mainloop.stride_k_descale) + bidh_kv * get<1>(params.mainloop.stride_k_descale)];
                    softmax_scale_log2 *= q_descale * k_descale;
                }
                flash::Softmax<2 * (2 * kBlockM / NumMmaThreads), /*Max_offset=*/!Is_FP8 ? 0 : 8> softmax(softmax_scale_log2);
                // Attention output (GEMM-II) accumulator.
                Tensor tOrO = partition_fragment_C(tiled_mma_pv, select<0, 1>(TileShape_MNK_PV{}));
                bool tile_valid = mainloop.mma(
                    params.mainloop, pipeline_k, pipeline_v, smem_pipe_read,
                    tOrO, softmax, threadIdx.x - MmaThreadOffset, work_idx, seqlen_info, block_coord, shared_storage);
                // Do this here before the epilogue so that the next tile is ready to go.
                work_tile_info = scheduler.template get_next_work</*IsProducerWarp=*/false>(params.scheduler, work_tile_info);
                if (tile_valid) {
                    epilogue.store(params.epilogue, tOrO, softmax.row_sum, shared_storage, tiled_mma_pv,
                                   threadIdx.x - MmaThreadOffset, block_coord, seqlen_info);
                } else {
                    // Write 0 to gO and -inf to gLSE.
                    epilogue.store_zero(params.epilogue, threadIdx.x - MmaThreadOffset, block_coord, seqlen_info);
                }
            }
            epilogue.store_tail();
        }

    }

};

} // namespace flash
