/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 * Copyright (c) 2026, fast-diffusion-attention authors.
 *
 * Scheduling metadata of the varlen forward kernel: number of query tiles of each sample (both halves), samples
 * sorted by length (longest first), number of heads whose K & V fit in L2.
 ******************************************************************************/

#include <cub/cub.cuh>
#include "cutlass/fast_math.h"

#include "flash.h"

#include "static_switch.h"

namespace flash {

// Sort in descending order
struct PrepareSortOp {
    __device__ __forceinline__ bool operator()(int4 const & lhs, int4 const & rhs) const {
        return lhs.x > rhs.x;
    }
};

template <int NumWarps>
__global__ void prepare_varlen_num_blocks_kernel(
        int const* const cu_seqlens, int num_batch, int num_head, int qhead_per_khead,
        cutlass::FastDivmod blockm_divmod, cutlass::FastDivmod blockn_divmod,
        int* const tile_count_semaphore,
        int* const num_m_blocks_ptr,
        int* const varlen_batch_idx_ptr,
        int* const num_nheads_in_l2_ptr,
        int max_kvblocks_in_l2) {

    static constexpr int kNumBatchPerWarp = cutlass::NumThreadsPerWarp - 1;
    static constexpr int BLOCK_DIM_X = NumWarps * 32;
    static constexpr int ITEMS_PER_THREAD = 1;
    using BlockMergeSort = cub::BlockMergeSort<int4, BLOCK_DIM_X, ITEMS_PER_THREAD>;

    // Allocate shared memory for BlockMergeSort operations
    __shared__ typename BlockMergeSort::TempStorage temp_storage;

    if (threadIdx.x == 0 && blockIdx.x == 0 && tile_count_semaphore) { *tile_count_semaphore = 0; }

    int lane = threadIdx.x % cutlass::NumThreadsPerWarp;
    int warp_idx = threadIdx.x / cutlass::NumThreadsPerWarp;
    int batch_cta_idx_offset = int(blockIdx.x) * 992;
    int bidb_start = batch_cta_idx_offset + kNumBatchPerWarp * warp_idx;
    int batch_idx = lane + bidb_start;

    // Length of each half of the sample
    int cur_cu_seqlen = batch_idx <= num_batch ? cu_seqlens[batch_idx] : 0;
    int next_cu_seqlen = __shfl_down_sync(0xffffffff, cur_cu_seqlen, 1);
    int const n = (next_cu_seqlen - cur_cu_seqlen) / 2;
    bool const valid = batch_idx < num_batch && lane < kNumBatchPerWarp;
    // Query tiles of both halves
    int const num_m_blocks = valid ? 2 * blockm_divmod.div(n + blockm_divmod.divisor - 1) : 0;
    // Key tiles of the sample
    int const num_n_blocks = valid ? 2 * blockn_divmod.div(n + blockn_divmod.divisor - 1) : 0;

    auto get_nheads_in_l2 = [&](int n_blocks) {
        int nheads_in_l2 = n_blocks * 16 <= max_kvblocks_in_l2 ? 16
            : n_blocks * 8 <= max_kvblocks_in_l2 ? 8
            : n_blocks * 4 <= max_kvblocks_in_l2 ? 4
            : n_blocks * 2 <= max_kvblocks_in_l2 ? 2
            : 1;
        nheads_in_l2 *= qhead_per_khead;
        return min(nheads_in_l2, num_head);
    };

    // Sort samples by length in descending order
    int4 batch_coords[ITEMS_PER_THREAD]; // 1 item per thread
    batch_coords[0] = make_int4(valid ? n : INT_MIN, num_m_blocks, num_n_blocks, batch_idx);
    BlockMergeSort(temp_storage).Sort(batch_coords, PrepareSortOp());

    // When sorting, we re-index some metadata by 'virtual batch index'
    // and also store the vbidx -> bidx mapping.
    batch_idx = batch_cta_idx_offset + threadIdx.x;
    if (batch_idx < num_batch && threadIdx.x < 992) {
        num_nheads_in_l2_ptr[batch_idx] = get_nheads_in_l2(max(batch_coords[0].z, 1));
        num_m_blocks_ptr[batch_idx] = batch_coords[0].y;
        varlen_batch_idx_ptr[batch_idx] = batch_coords[0].w;
    }
}

} // flash

void prepare_varlen_num_blocks(Flash_fwd_params &params, cudaStream_t stream, int blockM, int blockN) {
    int qhead_per_khead = cutlass::ceil_div(params.h, params.h_k);
    int num_warps = cutlass::ceil_div(params.b, 31); // warp switch will cap this at 32
    int num_ctas = cutlass::ceil_div(params.b, 31 * 32);
    int const size_l2 = 8 * 1024 * 1024; // underestimate seems better in practice
    int const element_size = params.is_e4m3 ? 1 : 2;
    int const size_one_kvblock = blockN * (2 * params.d) * element_size;
    int const max_kvblocks_in_l2 = size_l2 / size_one_kvblock;
    NUM_WARP_SWITCH(num_warps, NumWarps, [&] {
        flash::prepare_varlen_num_blocks_kernel<NumWarps><<<num_ctas /*grid*/, 32 * NumWarps /*block*/, 0, stream>>>(
            params.cu_seqlens, params.b, params.h, qhead_per_khead,
            cutlass::FastDivmod(blockM), cutlass::FastDivmod(blockN),
            params.tile_count_semaphore,
            params.num_m_blocks_ptr,
            params.varlen_batch_idx_ptr,
            params.num_nheads_in_l2_ptr,
            max_kvblocks_in_l2);
    });
}
