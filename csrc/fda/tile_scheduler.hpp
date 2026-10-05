/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 * Copyright (c) 2026, fast-diffusion-attention authors.
 *
 * Tile schedulers of the diffusion attention kernels (adapted from FlashAttention-3).
 *
 * Forward: a work tile is (m_block, head, sample, half). Within a sample the 2 * ceil(n / kBlockM) query tiles are
 * enumerated by a block index; longest-processing-time-first order visits block indices downwards, and a block
 * index b encodes m_block = b / 2 and half = noisy if b is odd, clean if b is even (the noisy query tile m visits
 * one more key tile than the clean query tile m). Noisy and clean tiles with the same m_block are processed close
 * in time and share the clean key tiles in L2.
 *
 * Backward: a work tile is (n_block, head, sample, half) for a key tile; see DiffusionBwdScheduler.
 ******************************************************************************/

#pragma once

#include "cutlass/fast_math.h"
#include "cutlass/arch/barrier.h"

#include "named_barrier.hpp"
#include "utils.h"

namespace flash {

///////////////////////////////////////////////////////////////////////////////

// Host side kernel arguments
struct TileSchedulerArguments {
    int const num_blocks, num_head, num_batch;
    int const qhead_per_khead;
    int const seqlen;  // Length of a sample (2n) if not varlen, max length of a sample if varlen
    int const headdim, element_size;  // Used to calculate L2 swizzling
    int* const tile_count_semaphore = nullptr;
    int const* const cu_seqlens = nullptr;
    int const* const num_m_blocks_ptr = nullptr;
    int const* const varlen_batch_idx_ptr = nullptr;
    int const* const num_nheads_in_l2_ptr = nullptr;
    int const num_blocks_noisy = 0;  // Backward only: number of noisy key tile slots (0 if handled elsewhere)
    bool const spt = false;          // Backward only: shortest-processing-time-first (for deterministic)
};

// Decode the (LPT-ordered) block index of a forward query tile.
CUTLASS_DEVICE
cute::tuple<int32_t, int32_t> diffusion_fwd_decode_block(int const block_lpt) {
    return {block_lpt >> 1, (block_lpt & 1) ? 0 /*noisy*/ : 1 /*clean*/};
}

///////////////////////////////////////////////////////////////////////////////

template<int NumMmaThreads=2 * cutlass::NumThreadsPerWarpGroup, int NumProducerThreads=cutlass::NumThreadsPerWarp>
class DiffusionDynamicPersistentTileScheduler {

    // Each tile takes a different amount of time. We use longest-processing-time-first scheduling:
    // the longest remaining tile is assigned to the first SM that's free.
    // SM indicates they are free by incrementing a semaphore.
    // However, we have to make sure K & V still fit into L2 cache, so we perform scheduling
    // on "sections" of the head & batch dimension, each section consisting of e.g. 8 heads.
    // This is the L2 swizzling part. The size of each section is precomputed based on the
    // size of K & V and the L2 cache size.

    static constexpr int NumThreads = NumMmaThreads + NumProducerThreads;

public:
    using SharedStorage = int;

protected:
    SharedStorage* const tile_count_smem;

public:

    // Device side kernel params
    struct Params {
        int const total_blocks;
        cutlass::FastDivmod const m_block_divmod, head_divmod;
        cutlass::FastDivmod const l2_minor_divmod, l2_major_divmod;
        cutlass::FastDivmod const l2_minor_residual_divmod;
        int const num_hb_quotient;
        int* const tile_count_semaphore;
    };

    static Params
    to_underlying_arguments(TileSchedulerArguments const& args) {
        long long const size_one_kv_head = long(args.seqlen) * long(2 * args.headdim) * long(args.element_size);
        int const size_l2 = 32 * 1024 * 1024;  // 32 MB for K & V
        // Swizzle is the size of each "section". Round swizzle to a power of 2
        // Need to be careful about the case where only one head will fit
        auto find_log2_floor = [&](int n) { return 31 - cutlass::clz(n); };
        // Seems faster if swizzle if a power of 2
        int const swizzle = (size_l2 < size_one_kv_head ? 1 : (1 << find_log2_floor(size_l2 / size_one_kv_head))) * args.qhead_per_khead;
        // If we're in the last section (called residual), we don't want to divide by
        // swizzle. Instead we want to divide by the remainder.
        int const num_hb_remainder = (args.num_head * args.num_batch) % swizzle;
        assert(args.tile_count_semaphore != nullptr);
        return {args.num_blocks * args.num_head * args.num_batch,
                cutlass::FastDivmod(args.num_blocks), cutlass::FastDivmod(args.num_head),
                cutlass::FastDivmod(swizzle), cutlass::FastDivmod(swizzle * args.num_blocks),
                // don't divide by 0
                cutlass::FastDivmod(num_hb_remainder > 0 ? num_hb_remainder : 1),
                (args.num_head * args.num_batch) / swizzle,
                args.tile_count_semaphore};
    }

    static dim3
    get_grid_shape(Params const& params, int num_sm) {
        return {uint32_t(num_sm)};
    }

    struct WorkTileInfo {
        int tile_idx;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return tile_idx < params.total_blocks;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            int block, bidh, bidb;
            int l2_mod, bidhb, bidhb_residual;
            bidhb = params.l2_major_divmod.divmod(l2_mod, tile_idx);
            // If we're in the last section (called residual), we don't want to divide by
            // swizzle. Instead we want to divide by the remainder.
            if (bidhb < params.num_hb_quotient) {
                block = params.l2_minor_divmod.divmod(bidhb_residual, l2_mod);
            } else {
                block = params.l2_minor_residual_divmod.divmod(bidhb_residual, l2_mod);
            }
            bidb = params.head_divmod.divmod(bidh, bidhb * params.l2_minor_divmod.divisor + bidhb_residual);
            // Longest-processing-time-first
            block = params.m_block_divmod.divisor - 1 - block;
            auto [m_block, half] = diffusion_fwd_decode_block(block);
            return {m_block, bidh, bidb, half};
        }

    };

    CUTLASS_DEVICE
    DiffusionDynamicPersistentTileScheduler(SharedStorage* const smem_scheduler) : tile_count_smem(smem_scheduler) {};

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        return {int(blockIdx.x)};
    }

    CUTLASS_DEVICE
    void
    init_consumer() const {
        flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 /*id*/);  // TileCountSmemEmpty
    }

    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {
        if (threadIdx.x % NumProducerThreads == 0) {
            current_work.tile_idx = atomicAdd(params.tile_count_semaphore, 1) + int(gridDim.x);
        }
    }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        if constexpr (IsProducerWarp) {
            // thread 0 already has the right tile_idx, just need to broadcast to the rest of warp 0
            int new_tile_idx = __shfl_sync(0xffffffff, current_work.tile_idx, 0 /*lane*/);
            flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 /*id*/);  // TileCountSmemEmpty
            if (threadIdx.x % NumProducerThreads == 0) {
                *tile_count_smem = current_work.tile_idx;
            }
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 /*id*/);  // TileCountSmemFull
            return {new_tile_idx};
        } else {
            flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 /*id*/);  // TileCountSmemFull
            int tile_idx = *tile_count_smem;
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 /*id*/);  // TileCountSmemEmpty
            return {tile_idx};
        }
    }

};

///////////////////////////////////////////////////////////////////////////////

// Varlen forward scheduler. num_m_blocks_ptr[b] = 2 * ceil(n_b / kBlockM) (both halves) for the b-th longest sample
// (varlen_batch_idx_ptr maps it to the actual sample), computed by prepare_varlen_num_blocks.
template<int kBlockM, int kBlockN, int NumMmaThreads=2 * cutlass::NumThreadsPerWarpGroup, int NumProducerThreads=cutlass::NumThreadsPerWarp>
class VarlenDynamicPersistentTileScheduler {

    static constexpr int NumThreads = NumMmaThreads + NumProducerThreads;

public:
    using SharedStorage = int4;

protected:
    SharedStorage* const work_info_smem;

public:

    // Device side kernel params
    struct Params {
        int num_head, num_batch;
        int* const tile_count_semaphore;
        int const* const num_m_blocks_ptr;
        int const* const varlen_batch_idx_ptr;
        int const* const num_nheads_in_l2_ptr;
    };

    static Params
    to_underlying_arguments(TileSchedulerArguments const& args) {
        assert(args.tile_count_semaphore != nullptr);
        return {args.num_head, args.num_batch,
                args.tile_count_semaphore,
                args.num_m_blocks_ptr,
                args.varlen_batch_idx_ptr,
                args.num_nheads_in_l2_ptr};
    }

    static dim3
    get_grid_shape(Params const& params, int num_sm) {
        return {uint32_t(num_sm)};
    }

    struct WorkTileInfo {
        int tile_idx, block, bidh, bidb;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return bidb < params.num_batch;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            auto [m_block, half] = diffusion_fwd_decode_block(block);
            return {m_block, bidh, params.varlen_batch_idx_ptr[bidb], half};
        }
    };

    CUTLASS_DEVICE
    VarlenDynamicPersistentTileScheduler(SharedStorage* const smem_scheduler) : work_info_smem(smem_scheduler) {};

    CUTLASS_DEVICE
    WorkTileInfo
    tile_idx_to_work_tile(Params const& params, int next_tile_idx, WorkTileInfo const& current_work) const {
        int lane = threadIdx.x % cutlass::NumThreadsPerWarp;
        auto get_num_m_blocks = [&] (int bidb_start) {
            int batch_idx = lane + bidb_start;
            return batch_idx < params.num_batch && lane < cutlass::NumThreadsPerWarp - 1
                ? params.num_m_blocks_ptr[batch_idx] : 0;
        };

        int num_m_blocks = get_num_m_blocks(current_work.bidb);  // Different for each lane
        // Cumulative number of blocks for the next 31 batches
        int num_m_blocks_cumulative = warp_prefix_sum(num_m_blocks);
        // Total number of blocks for the next 31 batches
        int m_blocks_in_group = __shfl_sync(0xffffffff, num_m_blocks_cumulative, cutlass::NumThreadsPerWarp - 1);
        // current_work.tile_idx holds group_start_tile for starting batch
        int group_end_tile = current_work.tile_idx + m_blocks_in_group * params.num_head;  // Same for all lanes
        int bidb = current_work.bidb;
        while (group_end_tile <= next_tile_idx) {
            bidb += cutlass::NumThreadsPerWarp - 1;
            if (bidb >= params.num_batch) {
                return {next_tile_idx, 0, 0, params.num_batch};
            }
            num_m_blocks = get_num_m_blocks(bidb);
            num_m_blocks_cumulative = warp_prefix_sum(num_m_blocks);
            m_blocks_in_group = __shfl_sync(0xffffffff, num_m_blocks_cumulative, cutlass::NumThreadsPerWarp - 1);
            group_end_tile += m_blocks_in_group * params.num_head;
        }
        int group_start_tile = group_end_tile - m_blocks_in_group * params.num_head;
        // The next problem to process is the first one that does not have ending tile position
        // that is greater than or equal to tile index.
        int batch_idx_in_group = __popc(__ballot_sync(0xffffffff, group_start_tile + num_m_blocks_cumulative * params.num_head <= next_tile_idx));
        bidb += batch_idx_in_group;
        num_m_blocks = __shfl_sync(0xffffffff, num_m_blocks, batch_idx_in_group);
        group_start_tile += (batch_idx_in_group == 0 ? 0 : __shfl_sync(0xffffffff, num_m_blocks_cumulative, batch_idx_in_group - 1)) * params.num_head;
        int mh_block = next_tile_idx - group_start_tile;
        int block, bidh;
        // Longest-processing-time-first within a sample, in sections of heads whose K & V fit in L2
        int nheads_in_l2 = params.num_nheads_in_l2_ptr[bidb];
        int mh_in_l2 = nheads_in_l2 * num_m_blocks;
        int section_idx = mh_block / mh_in_l2;
        int l2_mod = mh_block - section_idx * mh_in_l2;
        // tail section
        int nheads_remainder = params.num_head - section_idx * nheads_in_l2;
        int nheads_in_this_section = nheads_in_l2 <= nheads_remainder ? nheads_in_l2 : nheads_remainder;
        block = l2_mod / nheads_in_this_section;
        int bidh_residual = l2_mod - block * nheads_in_this_section;
        bidh = section_idx * nheads_in_l2 + bidh_residual;
        block = num_m_blocks - 1 - block;
        return {group_start_tile, block, bidh, bidb};
    }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        if constexpr (IsProducerWarp) {
            WorkTileInfo work_info = tile_idx_to_work_tile(params, int(blockIdx.x), {0, 0, 0, 0});
            if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
                *work_info_smem = make_int4(work_info.tile_idx, work_info.block, work_info.bidh, work_info.bidb);
            }
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 /*id*/);  // TileCountSmemFull
            return work_info;
        } else {
            return get_next_work<false>(params, {0, 0, 0, 0});
        }
    }

    CUTLASS_DEVICE
    void
    init_consumer() const {
        // Don't arrive at the TileCountSmemEmpty barrier here, because get_initial_work will do that
    }

    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {
        if (threadIdx.x % NumProducerThreads == 0) {
            current_work.tile_idx = atomicAdd(params.tile_count_semaphore, 1) + int(gridDim.x);
        }
    }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        if constexpr (IsProducerWarp) {
            // thread 0 has the next tile_idx, just need to broadcast to the rest of warp 0
            int new_tile_idx = __shfl_sync(0xffffffff, current_work.tile_idx, 0 /*lane*/);
            WorkTileInfo work_info = {__shfl_sync(0xffffffff, current_work.tile_idx, 1 /*lane*/), current_work.block, current_work.bidh, current_work.bidb};
            work_info = tile_idx_to_work_tile(params, new_tile_idx, work_info);
            flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 /*id*/);  // TileCountSmemEmpty
            if (threadIdx.x % cutlass::NumThreadsPerWarp == 0) {
                *work_info_smem = make_int4(work_info.tile_idx, work_info.block, work_info.bidh, work_info.bidb);
            }
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 /*id*/);  // TileCountSmemFull
            return work_info;
        } else {
            flash::named_barrier_sync(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier1 /*id*/);  // TileCountSmemFull
            int4 work_info = *work_info_smem;
            flash::named_barrier_arrive(NumThreads, cutlass::arch::ReservedNamedBarriers::StreamkBarrier0 /*id*/);  // TileCountSmemEmpty
            return WorkTileInfo{work_info.x, work_info.y, work_info.z, work_info.w};
        }
    }

};

///////////////////////////////////////////////////////////////////////////////

// Backward scheduler (one CTA per key tile, no persistence), with L2 swizzling over sections of heads.
// Block indices [0, num_blocks_clean) are the clean key tiles and [num_blocks_clean, num_blocks) the noisy ones.
// LPT (default): clean key tile j = block (the clean key tile 0 is seen by the most query tiles), then the noisy
// key tiles. SPT (deterministic backward): noisy key tiles first, then clean key tiles j = n_clean - 1 down to 0, so
// that the CTAs accumulating into a dQ tile are launched in the order of their accumulation.
template <bool Varlen, int kBlock>
class DiffusionBwdScheduler {

public:

    using SharedStorage = int;

    // Device side kernel params
    struct Params {
        int const total_blocks;
        cutlass::FastDivmod const block_divmod, head_divmod;
        cutlass::FastDivmod const l2_minor_divmod, l2_major_divmod;
        cutlass::FastDivmod const l2_minor_residual_divmod;
        int const num_hb_quotient;
        int const num_blocks_noisy;
        int const seqlen;
        bool const spt;
        int const* const cu_seqlens;
    };

    static Params
    to_underlying_arguments(TileSchedulerArguments const& args) {
        // Q, dO and dQaccum of a head
        long long const size_one_qdo_head = long(args.seqlen) * long(2 * args.headdim) * long(args.element_size);
        long long const size_one_dqaccum_head = long(args.seqlen) * long(args.headdim) * sizeof(float);
        long long const size_one_head = size_one_qdo_head + size_one_dqaccum_head;
        int const size_l2 = 40 * 1024 * 1024;  // 40 MB for Q, dO, and dQaccum
        // Swizzle is the size of each "section". Round swizzle to a power of 2
        // Need to be careful about the case where only one head will fit
        auto find_log2_floor = [&](int n) { return 31 - cutlass::clz(n); };
        // Seems faster if swizzle if a power of 2
        int const swizzle = size_l2 < size_one_head ? 1 : (1 << find_log2_floor(size_l2 / size_one_head));
        // If we're in the last section (called residual), we don't want to divide by
        // swizzle. Instead we want to divide by the remainder.
        int const num_hb_remainder = (args.num_head * args.num_batch) % swizzle;
        return {args.num_blocks * args.num_head * args.num_batch,
                cutlass::FastDivmod(args.num_blocks), cutlass::FastDivmod(args.num_head),
                cutlass::FastDivmod(swizzle), cutlass::FastDivmod(swizzle * args.num_blocks),
                // don't divide by 0
                cutlass::FastDivmod(num_hb_remainder > 0 ? num_hb_remainder : 1),
                (args.num_head * args.num_batch) / swizzle,
                args.num_blocks_noisy,
                args.seqlen, args.spt, !Varlen ? nullptr : args.cu_seqlens};
    }

    static dim3
    get_grid_shape(Params const& params, int num_sm) {
        return {uint32_t(params.total_blocks)};
    }

    struct WorkTileInfo {
        int block;
        int bidh;
        int bidb;
        int half;

        CUTLASS_DEVICE
        bool
        is_valid(Params const& params) const {
            return bidb >= 0;
        }

        CUTLASS_DEVICE
        cute::tuple<int32_t, int32_t, int32_t, int32_t>
        get_block_coord(Params const& params) const {
            return {block, bidh, bidb, half};
        }

    };

    CUTLASS_DEVICE
    DiffusionBwdScheduler(SharedStorage* const smem_scheduler) { }

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_initial_work(Params const& params) const {
        int tile_idx = blockIdx.x;
        int block, bidh, bidb;
        int l2_mod, bidhb, bidhb_residual;
        bidhb = params.l2_major_divmod.divmod(l2_mod, tile_idx);
        // If we're in the last section (called residual), we don't want to divide by
        // swizzle. Instead we want to divide by the remainder.
        if (bidhb < params.num_hb_quotient) {
            block = params.l2_minor_divmod.divmod(bidhb_residual, l2_mod);
        } else {
            block = params.l2_minor_residual_divmod.divmod(bidhb_residual, l2_mod);
        }
        bidb = params.head_divmod.divmod(bidh, bidhb * params.l2_minor_divmod.divisor + bidhb_residual);
        int const num_blocks_clean_max = params.block_divmod.divisor - params.num_blocks_noisy;
        // Number of key tiles of each half of this sample
        int const n = !Varlen ? params.seqlen / 2 : (params.cu_seqlens[bidb + 1] - params.cu_seqlens[bidb]) / 2;
        int const num_blocks_sample = cute::ceil_div(n, kBlock);
        int half, n_block;
        if (!params.spt) {
            half = block < num_blocks_clean_max ? 1 : 0;
            n_block = block < num_blocks_clean_max ? block : block - num_blocks_clean_max;
        } else {
            half = block < params.num_blocks_noisy ? 0 : 1;
            n_block = block < params.num_blocks_noisy ? block : params.block_divmod.divisor - 1 - block;
        }
        bool const is_valid_tile = n_block < num_blocks_sample;
        return {n_block, bidh, is_valid_tile ? bidb : -1, half};
    }

    CUTLASS_DEVICE
    void
    init_consumer() const {}

    CUTLASS_DEVICE
    void
    prefetch_next_work(Params const& params, WorkTileInfo& current_work) const {}

    template<bool IsProducerWarp=false>
    CUTLASS_DEVICE
    WorkTileInfo
    get_next_work(Params const& params, WorkTileInfo const& current_work) const {
        return {0, 0, -1, 0};
    }

};

///////////////////////////////////////////////////////////////////////////////

} // flash
