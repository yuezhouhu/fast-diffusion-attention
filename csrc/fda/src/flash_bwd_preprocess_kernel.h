/***************************************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Copyright (c) 2026, fast-diffusion-attention authors.
 *
 * Pre- and post-processing kernels of the backward pass (FlashAttention-2 flash_bwd_preprocess_kernel.h), on the
 * query tiles of each half of the samples and the padded row layout of the accumulators (diffusion.h).
 ******************************************************************************/

#pragma once

#include "namespace_config.h"
#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include "diffusion.h"
#include "kernel_traits.h"
#include "utils.h"

namespace FLASH_NAMESPACE {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int THREADS_PER_ROW, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
inline __device__ void dot_do_o(Tensor<Engine0, Layout0> const &do_, Tensor<Engine0, Layout0> const &o,
                                Tensor<Engine1, Layout1> &dP_sum, const int gdP_col_stride, const float scale) {
    static_assert(Layout0::rank == 3, "Only support 3D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(do_.layout() == o.layout());
    // Reshape do_ and o from (8, kBlockM / 32, kHeadDim / 64) to (kBlockM / 32, 8 * kHeadDim / 64)
    // The last coordinate is the "page".
    Tensor do_reshaped = make_tensor(do_.data(), make_layout(get<1>(do_.layout()),
                                                             make_layout(get<0>(do_.layout()),
                                                                         get<2>(do_.layout()))));
    Tensor o_reshaped = make_tensor(o.data(), do_reshaped.layout());
    Tensor do_fp32 = FLASH_NAMESPACE::convert_type<float>(do_reshaped);
    Tensor o_fp32 = FLASH_NAMESPACE::convert_type<float>(o_reshaped);
    #pragma unroll
    for (int mi = 0; mi < size<0>(do_reshaped); ++mi) {
        float dP_sum_cur = do_fp32(mi, 0) * o_fp32(mi, 0);
        #pragma unroll
        for (int ni = 1; ni < size<1>(do_reshaped); ni++) {
            dP_sum_cur += do_fp32(mi, ni) * o_fp32(mi, ni);
        }
        FLASH_NAMESPACE::SumOp<float> sum_op;
        dP_sum_cur = FLASH_NAMESPACE::Allreduce<THREADS_PER_ROW>::run(dP_sum_cur, sum_op) * scale;
        if (threadIdx.x % THREADS_PER_ROW == 0) {
            dP_sum(mi * gdP_col_stride + threadIdx.x / THREADS_PER_ROW) = dP_sum_cur;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Query tile of blockIdx.x: the tiles of the noisy half, then those of the clean half.
__forceinline__ __device__ void query_tile_of_block(bool &q_noisy, int &m_block) {
    const int num_m_blocks = gridDim.x / 2;
    q_noisy = int(blockIdx.x) < num_m_blocks;
    m_block = q_noisy ? int(blockIdx.x) : int(blockIdx.x) - num_m_blocks;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Compute dot(do, o) and write the result (softmax_d) to global memory, copy the LSE to the padded layout and
// clear dQaccum, as a separate kernel (the backward is parallelized across the key tiles).
template<bool Clear_dQaccum=true, typename Kernel_traits, typename Params>
inline __device__ void compute_dot_do_o(const Params &params) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    bool q_noisy;
    int m_block;
    query_tile_of_block(q_noisy, m_block);
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;
    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;

    const DiffusionBlockInfo<> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.n) return;

    const int q_row0 = binfo.half_row(!q_noisy) + m_block * kBlockM;
    const int q_row0_padded = binfo.padded_row(bidb, !q_noisy) + m_block * kBlockM;
    const int m_residue = binfo.n - m_block * kBlockM;

    const index_t row_offset_do = binfo.offset(params.do_batch_stride, params.do_row_stride, bidb)
        + index_t(q_row0) * params.do_row_stride + bidh * params.do_head_stride;
    const index_t row_offset_o = binfo.offset(params.o_batch_stride, params.o_row_stride, bidb)
        + index_t(q_row0) * params.o_row_stride + bidh * params.o_head_stride;
    const index_t dq_accum_row_stride = static_cast<index_t>(params.h) * params.d_rounded;
    const index_t row_offset_dq_accum = index_t(q_row0_padded) * dq_accum_row_stride + bidh * params.d_rounded;
    const index_t row_offset_dpsum = index_t(bidh) * params.total_padded + q_row0_padded;

    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                             Shape<Int<kBlockM>, Int<kHeadDim>>{},
                             make_stride(params.do_row_stride, _1{}));
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.o_row_stride, _1{}));
    Tensor gdQaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dq_accum_ptr) + row_offset_dq_accum),
                                  Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                  make_stride(dq_accum_row_stride, _1{}));
    Tensor dP_sum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});

    // LSE in the padded layout, +inf beyond the end of the half: the probabilities of those rows are 0.
    {
        const index_t lse_offset = params.unpadded_lse
            ? index_t(bidh) * params.total_q + binfo.start + q_row0
            : index_t(bidb * params.h + bidh) * params.seqlen + q_row0;
        const ElementAccum *lse_ptr = reinterpret_cast<const ElementAccum *>(params.softmax_lse_ptr) + lse_offset;
        ElementAccum *lse_padded_ptr = reinterpret_cast<ElementAccum *>(params.softmax_lse_padded_ptr) + row_offset_dpsum;
        for (int row = tidx; row < kBlockM; row += Kernel_traits::kNThreads) {
            lse_padded_ptr[row] = row < m_residue ? lse_ptr[row] : INFINITY;
        }
    }

    typename Kernel_traits::GmemTiledCopydO gmem_tiled_copy_dO;
    auto gmem_thr_copy_dO = gmem_tiled_copy_dO.get_thread_slice(tidx);
    // TODO: careful, we're zeroing out dQaccum with type float4, but when
    // we do atomicAdds, we use type float. The layouts are different. Check this.
    typename Kernel_traits::GmemTiledCopydQaccum gmem_tiled_copy_dQaccum;
    auto gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_thread_slice(tidx);

    Tensor tdOgdO = gmem_thr_copy_dO.partition_S(gdO);
    Tensor tdOgO = gmem_thr_copy_dO.partition_S(gO);
    Tensor tdQgdQaccum = gmem_thr_copy_dQaccum.partition_D(gdQaccum);

    Tensor cdO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor tdOcdO = gmem_thr_copy_dO.partition_S(cdO);

    // Allocate predicate tensors for k
    Tensor tdOpdO = make_tensor<bool>(make_shape(size<2>(tdOgdO)));
    // Set predicates for k bounds
    #pragma unroll
    for (int k = 0; k < size(tdOpdO); ++k) {tdOpdO(k) = get<1>(tdOcdO(0, 0, k)) < params.d;}

    Tensor tdOrdO = make_fragment_like(tdOgdO);
    Tensor tdOrO = make_fragment_like(tdOgO);
    FLASH_NAMESPACE::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_dO, tdOgdO, tdOrdO, tdOcdO, tdOpdO, m_residue
    );
    FLASH_NAMESPACE::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_dO, tdOgO, tdOrO, tdOcdO, tdOpdO, m_residue
    );
    // By right we need to scale dP up by 1/p_dropout, but instead we don't and only scale the final
    // results (dQ and dK) by 1/p_dropout. So we need to keep dP_sum scaled down by p_dropout here,
    // so that (dP - dP_sum) is on the same scale.
    // The rows beyond the end of the half get dP_sum = 0 (in the padding).
    dot_do_o<Kernel_traits::kGmemThreadsPerRow>(tdOrdO, tdOrO, dP_sum,
                                                Kernel_traits::kNThreads / (Kernel_traits::kGmemThreadsPerRow), params.p_dropout);
    if (Clear_dQaccum) {
        // We're actually not zero'ing out all of dQaccum, but only the part that we're going to
        // do atomicAdds on.
        Tensor zero = make_fragment_like(tdQgdQaccum);
        clear(zero);
        cute::copy(gmem_tiled_copy_dQaccum, zero, tdQgdQaccum);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert dQ from dQaccum (in float) to fp16/bf16.
// This is used in the case where we want to parallelize the backward across seqlen_k.
template<typename Kernel_traits, typename Params>
inline __device__ void convert_dQ(const Params &params, const int nsplits) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    bool q_noisy;
    int m_block;
    query_tile_of_block(q_noisy, m_block);
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;
    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;

    const DiffusionBlockInfo<> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.n) return;

    const int q_row0 = binfo.half_row(!q_noisy) + m_block * kBlockM;
    const int q_row0_padded = binfo.padded_row(bidb, !q_noisy) + m_block * kBlockM;
    const int m_residue = binfo.n - m_block * kBlockM;

    const index_t row_offset_dq = binfo.offset(params.dq_batch_stride, params.dq_row_stride, bidb)
        + index_t(q_row0) * params.dq_row_stride + bidh * params.dq_head_stride;
    const index_t dq_accum_row_stride = static_cast<index_t>(params.h) * params.d_rounded;
    const index_t row_offset_dq_accum = index_t(q_row0_padded) * dq_accum_row_stride + bidh * params.d_rounded;

    Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                             Shape<Int<kBlockM>, Int<kHeadDim>>{},
                             make_stride(params.dq_row_stride, _1{}));
    Tensor gdQaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dq_accum_ptr) + row_offset_dq_accum),
                                  Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                  make_stride(dq_accum_row_stride, _1{}));

    Tensor sdQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                             typename Kernel_traits::SmemLayoutdQ{});

    typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dQ;
    auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopydQaccumAtomicAdd gmem_tiled_copy_dQaccum;
    auto gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_thread_slice(tidx);

    typename Kernel_traits::TiledMmadQ tiled_mma_dq;
    auto smem_tiled_copy_dQ = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdQ{}, tiled_mma_dq);
    auto smem_thr_copy_dQ = smem_tiled_copy_dQ.get_thread_slice(tidx);
    Tensor taccdQsdQ = smem_thr_copy_dQ.partition_D(sdQ);  // ((Atom,AtomNum),PIPE_M,PIPE_N)

    Tensor tdQsdQ = gmem_thr_copy_dQ.partition_S(sdQ);    // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);
    Tensor tdQgdQaccum = gmem_thr_copy_dQaccum.partition_S(gdQaccum);

    Tensor acc_dq = partition_fragment_C(tiled_mma_dq, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_N, MMA_K
    CUTE_STATIC_ASSERT_V(size(acc_dq) == size(tdQgdQaccum));

    Tensor tdQrdQaccum = make_fragment_like(tdQgdQaccum);
    clear(acc_dq);
    for (int s = 0; s < nsplits; ++s) {
        cute::copy(gmem_tiled_copy_dQaccum, tdQgdQaccum, tdQrdQaccum);
        #pragma unroll
        for (int i = 0; i < size(acc_dq); ++i) { acc_dq(i) += tdQrdQaccum(i); }
        tdQgdQaccum.data() = tdQgdQaccum.data() + params.dq_accum_split_stride;
    }
    #pragma unroll
    for (int i = 0; i < size(acc_dq); ++i) { acc_dq(i) *= params.scale_softmax_rp_dropout; }
    // Convert acc_dq from fp32 to fp16
    Tensor rdQ = FLASH_NAMESPACE::convert_type<Element>(acc_dq);
    Tensor taccdQrdQ = smem_thr_copy_dQ.retile_S(rdQ);  // ((Atom,AtomNum), MMA_N, MMA_N)
    cute::copy(smem_tiled_copy_dQ, taccdQrdQ, taccdQsdQ);
    __syncthreads();
    Tensor tdQrdQ = make_tensor<Element>(shape(tdQgdQ));
    cute::copy(gmem_tiled_copy_dQ, tdQsdQ, tdQrdQ);

    Tensor cdQ = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor tdQcdQ = gmem_thr_copy_dQ.partition_D(cdQ);
    Tensor tdQpdQ = make_tensor<bool>(make_shape(size<2>(tdQgdQ)));
    #pragma unroll
    for (int k = 0; k < size(tdQpdQ); ++k) { tdQpdQ(k) = get<1>(tdQcdQ(0, 0, k)) < params.d; }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    FLASH_NAMESPACE::copy</*Is_even_MN=*/false, /*Is_even_K=*/false, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dQ, tdQrdQ, tdQgdQ, tdQcdQ, tdQpdQ, m_residue
    );
}

} // namespace FLASH_NAMESPACE
