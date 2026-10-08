/***************************************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Copyright (c) 2026, fast-diffusion-attention authors.
 *
 * Backward kernel of the diffusion attention: compute_dq_dk_dv_1colblock of FlashAttention-2 (flash_bwd_kernel.h,
 * sequence-parallel version: one thread block per key tile, dQ accumulated with atomicAdd in FP32) with the query
 * tiles of the diffusion mask (DiffusionBwdTiles, diffusion.h) instead of the causal / local ones.
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
#include "softmax.h"
#include "dropout.h"

namespace FLASH_NAMESPACE {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int MMA_N,
          class... Args,
          class TiledMMA>
CUTE_HOST_DEVICE
auto
make_tiled_copy_B_warpcontiguousN(Copy_Atom<Args...> const& copy_atom,
                                  TiledMMA           const& tiled_mma) {
    constexpr int TileShape_N = decltype(tiled_mma.template tile_size_mnk<1>())::value;
    constexpr int TileShape_K = decltype(tiled_mma.template tile_size_mnk<2>())::value;
    using AtomShape_MNK = typename TiledMMA::AtomShape_MNK;
    constexpr int AtomShape_N = decltype(size<1>(AtomShape_MNK{}))::value;
    // Divide by 2 because right now we always use 2 for the ValLayout
    constexpr int kNWarpsN = TileShape_N / AtomShape_N / 2;
    constexpr int MMAStride_N = MMA_N * AtomShape_N * 2;
    // This gives the correct layout, idk why.
    // auto t = make_tile(Layout<Shape<Shape<_8, _2>, _2>,
    //                           Stride<Stride<_1, _64>, _8> >{},
    // auto t = make_tile(Layout<Shape<_8, _2, _2>,
    //                           Stride<_1, _64, _8> >{},
    auto t = make_tile(Layout<Shape<Int<AtomShape_N>, Int<kNWarpsN>, _2>,   // (8, 2, 2) or (8, 4, 2)
                              Stride<_1, Int<MMAStride_N>, _8> >{},       // (1, 64, 8) or (1, 32, 8)
                       make_layout(Int<TileShape_K>{}));
    return make_tiled_copy_impl(copy_atom, tiled_mma.get_layoutB_TV(), t);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int MMA_N,
          class... Args,
          class TiledMMA>
CUTE_HOST_DEVICE
auto
make_tiled_copy_C_warpcontiguousN(Copy_Atom<Args...> const& copy_atom,
                                  TiledMMA           const& tiled_mma) {
    constexpr int TileShape_M = decltype(tiled_mma.template tile_size_mnk<0>())::value;
    constexpr int TileShape_N = decltype(tiled_mma.template tile_size_mnk<1>())::value;
    using AtomShape_MNK = typename TiledMMA::AtomShape_MNK;
    constexpr int AtomShape_N = decltype(size<1>(AtomShape_MNK{}))::value;
    // Divide by 2 because right now we always use 2 for the ValLayout
    constexpr int kNWarpsN = TileShape_N / AtomShape_N / 2;
    constexpr int MMAStride_N = MMA_N * AtomShape_N * 2;
    auto t = make_tile(make_layout(Int<TileShape_M>{}),
                       Layout<Shape<Int<AtomShape_N>, Int<kNWarpsN>, _2>,   // (8, 2, 2) or (8, 4, 2)
                              Stride<_1, Int<MMAStride_N>, _8> >{});       // (1, 64, 8) or (1, 32, 8)
    return make_tiled_copy_impl(copy_atom, tiled_mma.get_layoutC_TV(), t);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Is_grouped, typename Params>
inline __device__ void compute_dq_dk_dv_1colblock(const Params &params, const int bidb, const int bidh_group, const bool k_noisy, const int n_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int MMA_N_SdP = kBlockN / decltype(typename Kernel_traits::TiledMmaSdP{}.template tile_size_mnk<1>())::value;
    constexpr int AtomLayoutMS = Kernel_traits::AtomLayoutMSdP;
    constexpr bool Double_buffer = !Kernel_traits::No_double_buffer;
    static_assert(kBlockM <= kAccumPad, "The padding of the accumulators must hold a partial query tile");

    const DiffusionBlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    const int n = binfo.n;
    if (n_block * kBlockN >= n) return;

    const DiffusionGeometry geo(params.block_size_divmod, n);
    const DiffusionBwdTiles<kBlockM, kBlockN> tiles(geo, k_noisy, n_block);
    // Every key is seen by at least the queries of its own block: m_block_max >= 1.
    const int m_block_max = tiles.m_block_max();
    // The query tiles u < n_mask need masking.
    const int n_mask = tiles.n_mask();

    // Row of the first key of the tile in the sample, number of valid keys of the tile.
    const int k_row0 = binfo.half_row(!k_noisy) + n_block * kBlockN;
    const int n_residue = n - n_block * kBlockN;
    // Rows of the clean half in the dropout coordinates.
    const int hr = params.seqlen_half_rounded;

    // The thread block processes the key tile for the params.bwd_heads_per_cta query heads [bidh_group *
    // bwd_heads_per_cta, ...) that share a KV head (GQA), accumulating dK / dV over these heads: the iterations
    // run over (query tile u, query head g), the heads inner. bwd_heads_per_cta = h / h_k gives the final dK / dV of the
    // KV head, without the reduction over the query heads of FlashAttention-2 (dk_expanded).
    // Is_grouped = false: one query head per thread block (bwd_heads_per_cta = 1), the head loop is compiled out.
    const int heads_per_cta = Is_grouped ? params.bwd_heads_per_cta : 1;
    const int num_iters = heads_per_cta * m_block_max;
    int bidh = bidh_group * heads_per_cta;
    int g = 0;
    // The current query tile u: its half, its index inside the half, its first row in the sample and in the padded
    // accumulators.
    int u = 0;
    // The query tiles are visited in decreasing order of the enumeration of DiffusionBwdTiles (the masked tiles last),
    // like FlashAttention-2 visits m_block downwards: the thread blocks of the key tiles of a head start with the same
    // query tiles. Measured on A100: 1-2% faster than the increasing order for short sequences.
    auto eidx = [&](const int x) { return m_block_max - 1 - x; };
    bool q_noisy = tiles.q_is_noisy(eidx(u), k_noisy);
    int m_block = tiles.tile_idx(eidx(u));
    int q_row = binfo.half_row(!q_noisy) + m_block * kBlockM;
    int q_row_padded = binfo.padded_row(bidb, !q_noisy) + m_block * kBlockM;

    const index_t row_offset_q = binfo.offset(params.q_batch_stride, params.q_row_stride, bidb)
        + index_t(q_row) * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_k = binfo.offset(params.k_batch_stride, params.k_row_stride, bidb)
        + index_t(k_row0) * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
    const index_t row_offset_v = binfo.offset(params.v_batch_stride, params.v_row_stride, bidb)
        + index_t(k_row0) * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;
    const index_t row_offset_do = binfo.offset(params.do_batch_stride, params.do_row_stride, bidb)
        + index_t(q_row) * params.do_row_stride + bidh * params.do_head_stride;
    const index_t dq_accum_row_stride = static_cast<index_t>(params.h) * params.d_rounded;
    const index_t row_offset_dq_accum = index_t(q_row_padded) * dq_accum_row_stride + bidh * params.d_rounded
        // If deterministic, each thread block will do atomicAdd to a different dQ_accum buffer.
        + (!params.deterministic ? 0 : blockIdx.x * params.dq_accum_split_stride);
    // LSE and dPsum in the padded layout (h, total_padded)
    const index_t row_offset_lse = index_t(bidh) * params.total_padded + q_row_padded;

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));
    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                             Shape<Int<kBlockM>, Int<kHeadDim>>{},
                             make_stride(params.do_row_stride, _1{}));
    Tensor gdQaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dq_accum_ptr) + row_offset_dq_accum),
                                  Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                  make_stride(dq_accum_row_stride, _1{}));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_padded_ptr) + row_offset_lse),
                              Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_lse),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQdO{});
    Tensor sQt = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOtransposed{});
    Tensor sQtNoSwizzle = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOtransposedNoSwizzle{});
    // Double buffer for sQ
    Tensor sdO = make_tensor(sQ.data() + (Double_buffer ? 2 : 1) * size(sQ), typename Kernel_traits::SmemLayoutQdO{});
    Tensor sdOt = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutQdOtransposed{});
    Tensor sdOtransposedNoSwizzle = make_tensor(sdO.data(),
                                                typename Kernel_traits::SmemLayoutQdOtransposedNoSwizzle{});
    Tensor sK = make_tensor(sdO.data() + size(sdO), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sKt = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKtransposed{});
    Tensor sKtNoSwizzle = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKtransposedNoSwizzle{});
    Tensor sdS = make_tensor(!Kernel_traits::Is_V_in_regs ? sV.data() + size(sV) : sK.data() + size(sK),
                             typename Kernel_traits::SmemLayoutPdS{});
    Tensor sdSt = make_tensor(sdS.data(), typename Kernel_traits::SmemLayoutPdStransposed{});
    Tensor sdStNoSwizzle = make_tensor(sdS.data(), typename Kernel_traits::SmemLayoutPdStransposedNoSwizzle{});
    Tensor sP = make_tensor(sdS.data() + size(sdS), typename Kernel_traits::SmemLayoutPdS{});
    Tensor sPt = make_tensor(sP.data(), typename Kernel_traits::SmemLayoutPdStransposed{});
    Tensor sPtNoSwizzle = make_tensor(sP.data(), typename Kernel_traits::SmemLayoutPdStransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    // dO is loaded with cp.async, like Q (the Is_first path of FlashAttention-2 is not used)
    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_dO;
    auto gmem_thr_copy_dO = gmem_tiled_copy_dO.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopydQaccumAtomicAdd gmem_tiled_copy_dQaccum;
    auto gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tdOgdO = gmem_thr_copy_dO.partition_S(gdO);
    Tensor tdOsdO = gmem_thr_copy_dO.partition_D(sdO);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
    Tensor tdQgdQaccum = gmem_thr_copy_dQaccum.partition_D(gdQaccum);

    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_A(sQ);         // (MMA,MMA_N,MMA_K)
    Tensor tSrK = thr_mma_sdp.partition_fragment_B(sK);         // (MMA,MMA_N,MMA_K)
    Tensor tdPrdO = thr_mma_sdp.partition_fragment_A(sdO);      // (MMA,MMA_N,MMA_K)
    Tensor tdPrV = thr_mma_sdp.partition_fragment_B(sV);        // (MMA,MMA_N,MMA_K)

    typename Kernel_traits::TiledMmadKV tiled_mma_dkv;
    auto thr_mma_dkv = tiled_mma_dkv.get_thread_slice(tidx);
    Tensor tdKrdSt = thr_mma_dkv.partition_fragment_A(sdStNoSwizzle); // (MMA, MMA_N, MMA_N)
    Tensor tdKrQt = thr_mma_dkv.partition_fragment_B(sQtNoSwizzle);   // (MMA, MMA_K, MMA_N)
    Tensor tdVrPt = thr_mma_dkv.partition_fragment_A(sPtNoSwizzle);   // (MMA, MMA_N, MMA_N)
    Tensor tdVrdO = thr_mma_dkv.partition_fragment_B(sdOtransposedNoSwizzle); // (MMA, MMA_K, MMA_N)

    typename Kernel_traits::TiledMmadQ tiled_mma_dq;
    auto thr_mma_dq = tiled_mma_dq.get_thread_slice(tidx);
    Tensor tdQrdS = thr_mma_dq.partition_fragment_A(sdS);                      // (MMA, MMA_N, MMA_N)
    Tensor tdQrKt = thr_mma_dq.partition_fragment_B(sKtNoSwizzle);    // (MMA, MMA_K, MMA_N)

    Tensor acc_dk = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});  // MMA, MMA_N, MMA_K
    Tensor acc_dv = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});  // MMA, MMA_N, MMA_K

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_QdO = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_QdO = smem_tiled_copy_QdO.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_QdO.partition_S(sQ);
    Tensor tdPsdO = smem_thr_copy_QdO.partition_S(sdO);

    auto smem_tiled_copy_KV = make_tiled_copy_B_warpcontiguousN<MMA_N_SdP>(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_KV = smem_tiled_copy_KV.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_KV.partition_S(sK);
    Tensor tdPsV = smem_thr_copy_KV.partition_S(sV);

    // Partition sP and sdS to match the accumulator partitioning
    // This has to be tiled_mma_sdp, not tiled_mma_dkv
    auto smem_tiled_copy_PdS = make_tiled_copy_C_warpcontiguousN<MMA_N_SdP>(typename Kernel_traits::SmemCopyAtomPdS{}, tiled_mma_sdp);
    auto smem_thr_copy_PdS = smem_tiled_copy_PdS.get_thread_slice(tidx);
    Tensor tPsP = smem_thr_copy_PdS.partition_D(sP);      // ((Atom,AtomNum),PIPE_M,PIPE_N)
    Tensor tdSsdS = smem_thr_copy_PdS.partition_D(sdS);   // ((Atom,AtomNum),PIPE_M,PIPE_N)

    auto smem_tiled_copy_PdSt = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dkv);
    auto smem_thr_copy_PdSt = smem_tiled_copy_PdSt.get_thread_slice(tidx);
    Tensor tdVsPt = smem_thr_copy_PdSt.partition_S(sPt);
    Tensor tdKsdSt = smem_thr_copy_PdSt.partition_S(sdSt);

    auto smem_tiled_copy_QdOt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dkv);
    auto smem_thr_copy_QdOt = smem_tiled_copy_QdOt.get_thread_slice(tidx);
    Tensor tdVsdOt = smem_thr_copy_QdOt.partition_S(sdOt);
    Tensor tdKsQt = smem_thr_copy_QdOt.partition_S(sQt);

    auto smem_tiled_copy_dS = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_dq);
    auto smem_thr_copy_dS = smem_tiled_copy_dS.get_thread_slice(tidx);
    Tensor tdQsdS = smem_thr_copy_dS.partition_S(sdS);

    auto smem_tiled_copy_Kt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dq);
    auto smem_thr_copy_Kt = smem_tiled_copy_Kt.get_thread_slice(tidx);
    Tensor tdQsKt = smem_thr_copy_Kt.partition_S(sKt);

    //
    // PREDICATES
    //

    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tQcQ = gmem_thr_copy_QKV.partition_D(cQ);
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_D(cKV);

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Prologue

    // The query tiles alternate between the two buffers of sQ: tile u uses buffer u % 2.

    if (params.deterministic) { __syncthreads(); }

    if (Kernel_traits::Is_V_in_regs) {
        // Clear the smem tiles to account for predicated off loads
        FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, n_residue
        );
        FLASH_NAMESPACE::cp_async_fence();
    }

    // Only the last query tile of each half can extend beyond the end of the half: the loads are predicated (and the
    // rows beyond the end cleared) unless Is_even_MN. Their LSE in the padded layout is +inf, so that their
    // probabilities (and dS) are 0, and their dQ lands in the padding of dQaccum.
    // Clear the smem tiles to account for predicated off loads
    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_dO, tdOgdO, tdOsdO, tQcQ, tQpQ, n - m_block * kBlockM
    );
    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ, n - m_block * kBlockM
    );

    Tensor caccS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});    // (BLK_M,BLK_N) -> (blk_m,blk_n)
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);                           // (MMA,MMA_N,MMA_N)
    static_assert(decltype(size<0>(taccScS))::value == 4);
    // Convert to ((2, 2), MMA_N, MMA_N) then take only the row indices.
    Tensor taccScS_row = logical_divide(taccScS, Shape<_2>{})(make_coord(0, _), _, 0);
    Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        const int row = get<0>(taccScS_row(mi));
        lse(mi) = gLSE(row);
    }
    // We want LSE = inf if the row is OOB. In that case Q would be zero, K would be zero,
    // and scores would be zero. With LSE = inf, probs are always 0.

    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV, n_residue
    );
    if (!Kernel_traits::Is_V_in_regs) {
        FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, n_residue
        );
    }
    FLASH_NAMESPACE::cp_async_fence();

    if (Kernel_traits::Is_V_in_regs) {
        cute::cp_async_wait<1>();
        __syncthreads();
        Tensor tdPrV_copy_view = smem_thr_copy_KV.retile_D(tdPrV);
        CUTE_STATIC_ASSERT_V(size<1>(tdPsV) == size<1>(tdPrV_copy_view));            // M
        cute::copy(smem_tiled_copy_KV, tdPsV, tdPrV_copy_view);
    }

    clear(acc_dv);
    clear(acc_dk);

    // Offsets between consecutive query tiles of a half
    const index_t q_tile_stride = index_t(kBlockM) * params.q_row_stride;
    const index_t do_tile_stride = index_t(kBlockM) * params.do_row_stride;
    const index_t dq_accum_tile_stride = index_t(kBlockM) * dq_accum_row_stride;
    // The (at most 4) iterations u where a new run of consecutive tiles starts: the enumeration of DiffusionBwdTiles
    // is made of runs of consecutive tiles of a half, starting at the indices x below, i.e. at u = m_block_max - x in
    // decreasing order.
    struct RunStarts {
        int a, b, c, d;
        __forceinline__ __device__ bool test(int const x) const { return x == a || x == b || x == c || x == d; }
    };
    RunStarts run_starts;
    {
        int const n_vis0 = tiles.vis0_hi - tiles.vis0_lo;
        run_starts.a = m_block_max - ((n_vis0 > 0 ? tiles.vis0_lo : tiles.m0_hi) - tiles.m0_lo);
        run_starts.b = m_block_max - tiles.n_m0_mask();
        run_starts.c = m_block_max - (tiles.n_m0_mask() + tiles.n_m1_mask);
        run_starts.d = m_block_max - (tiles.n_m0_mask() + tiles.n_m1_mask + n_vis0);
    }

    for (int it = 0; it < num_iters; ++it) {
        Tensor acc_s = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_N)
        clear(acc_s);
        cute::cp_async_wait<0>();
        __syncthreads();

        Tensor dP_sum = make_fragment_like(lse);
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) { dP_sum(mi) = gdPsum(get<0>(taccScS_row(mi))); }

        FLASH_NAMESPACE::gemm(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma_sdp,
                              smem_tiled_copy_QdO, smem_tiled_copy_KV, smem_thr_copy_QdO, smem_thr_copy_KV);

        if constexpr (Is_softcap) {
            FLASH_NAMESPACE::apply_softcap(acc_s, params.softcap);
        }

        // Reshape acc_s from (MMA=4, MMA_N, MMA_N) to (row=(2, MMA_N), col=(2, MMA_N))
        Tensor scores = make_tensor(acc_s.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_s.layout()));

        // Softcapping - calculating dTanh and scaling dS later with it
        [[maybe_unused]] Tensor dtanh = make_tensor_like(scores);
        if constexpr (Is_softcap) {
            FLASH_NAMESPACE::calculate_dtanh(scores, dtanh, params.softcap);
        }

        // TD [2023-07-29]: I was thinking that we don't need to mask out the elements beyond
        // actual_seqlen_k, because acc_s would be some finite value for those indices.
        // In the end when we multiply with K to get dQ, the corresponding values of K would be 0,
        // so the result would still be correct.
        // However, it's possible that the values in acc_s are so large that they overflow
        // when we multiply with dP and convert to fp16, resulting in Inf in dS and NaNs in dQ.
        // So we need to mask out the elements beyond actual_seqlen_k.
        // DiffusionBwdTiles puts the key tiles extending beyond the end of the half in the masked iterations.
        if (eidx(u) < n_mask) {
            // The columns of a warp are contiguous (make_tiled_copy_B_warpcontiguousN).
            FLASH_NAMESPACE::apply_diffusion_mask(
                acc_s, geo, q_noisy, k_noisy, m_block * kBlockM + get<0>(taccScS_row(0)), AtomLayoutMS * 16,
                n_block * kBlockN + (tidx / 32 / AtomLayoutMS) * MMA_N_SdP * 16
            );
        }

        // Compute the exponential value.
        FLASH_NAMESPACE::scale_apply_exp2</*scale_max=*/false>(scores, lse, params.scale_softmax_log2);
        if constexpr (Is_dropout) {
            FLASH_NAMESPACE::Dropout dropout(params.rng_state[0], params.rng_state[1], params.p_dropout_in_uint8_t,
                                             bidb, bidh, tidx, params.h);
            int warp_id = tidx / 32;
            // Coordinates of the 16 x 32 blocks of the dropout random numbers: rows and columns of the clean half
            // start at hr.
            int block_row_idx = ((q_noisy ? 0 : hr) + m_block * kBlockM) / 16 + warp_id % AtomLayoutMS;
            // Need col to be multiples of 32, since we're doing dropout with block of 16 x 32
            static_assert(MMA_N_SdP % 2 == 0);
            int block_col_idx = ((k_noisy ? 0 : hr) + n_block * kBlockN) / 32 + (warp_id / AtomLayoutMS) * (MMA_N_SdP / 2);
            dropout.template apply_dropout</*encode_dropout_in_sign_bit=*/true>(
                acc_s, block_row_idx, block_col_idx, AtomLayoutMS
            );
        }
        // Convert scores from fp32 to fp16/bf16
        Tensor rP = !Is_dropout
            ? FLASH_NAMESPACE::convert_type<Element>(acc_s)
            : FLASH_NAMESPACE::convert_type_relu<Element>(acc_s);
        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_N, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_N, MMA_N) if using m16n8k8.
        Tensor tPrP = make_tensor(rP.data(), FLASH_NAMESPACE::convert_layout_acc_Aregs<typename Kernel_traits::TiledMmaSdP>(rP.layout()));
        Tensor tPaP = smem_thr_copy_PdS.retile_S(tPrP);     // ((Atom,AtomNum), MMA_N, MMA_N)
        cute::copy(smem_tiled_copy_PdS, tPaP, tPsP);

        Tensor acc_dp = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_N, MMA_N)
        CUTE_STATIC_ASSERT_V(size<0>(acc_dp) == size<0>(acc_s));                     // MMA
        CUTE_STATIC_ASSERT_V(size<1>(acc_dp) == size<1>(acc_s));                     // MMA
        CUTE_STATIC_ASSERT_V(size<2>(acc_dp) == size<2>(acc_s));                     // MMA

        clear(acc_dp);

        FLASH_NAMESPACE::gemm</*A_in_regs=*/false, /*B_in_regs=*/Kernel_traits::Is_V_in_regs>(
            acc_dp, tdPrdO, tdPrV, tdPsdO, tdPsV, tiled_mma_sdp,
            smem_tiled_copy_QdO, smem_tiled_copy_KV, smem_thr_copy_QdO, smem_thr_copy_KV
        );

        // Reshape acc_dp from (MMA=4, MMA_N, MMA_N) to (row=(2, MMA_N), col=(2, MMA_N))
        Tensor dS = make_tensor(acc_dp.data(), scores.layout());
        auto pointwise_mult = [](float p, float dp, float d) {
            return p * (!Is_dropout || p >= 0 ? dp - d : d);
        };
        #pragma unroll
        for (int mi = 0; mi < size<0>(dS); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(dS); ++ni) {
                float scaled_ds = pointwise_mult(scores(mi, ni), dS(mi, ni), dP_sum(mi));
                if constexpr (Is_softcap) { scaled_ds *= dtanh(mi, ni); }
                dS(mi, ni) = scaled_ds;
            }
        }

        Tensor acc_dq = partition_fragment_C(tiled_mma_dq, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_N, MMA_K
        clear(acc_dq);

        // The next iteration. Grouped: the same query tile for the next head, then the next query tile for the first
        // head. The thread blocks of the key tiles of a KV head then visit the query tiles in step, as those of a head
        // without grouping, and share the Q / dO / dQaccum tiles in the L2. Head by head instead, they drift apart (the
        // key tiles with fewer query tiles move on to the next heads first) and the working set grows to all the heads
        // of the group: measured 1-15% slower (A800, L40S, RTX A6000), up to 40% with many heads per thread block.
        const bool has_next = it + 1 < num_iters;
        bool head_step = false;
        int u_next = u + 1, g_next = g;
        if constexpr (Is_grouped) {
            head_step = g + 1 < heads_per_cta;
            u_next = head_step ? u : u + 1;
            g_next = head_step ? g + 1 : 0;
        }
        bool q_noisy_next = q_noisy;
        int m_block_next = m_block, q_row_next = q_row, q_row_padded_next = q_row_padded;
        // Offsets of the next tile w.r.t. the current one (in elements). Inside a run of the enumeration, the next
        // tile is the previous tile of the same half and head (FlashAttention-2's m_block - 1): no index computation
        // in the common case, which matters (measured: 1.5% of the backward for long sequences).
        index_t q_delta = -q_tile_stride, do_delta = -do_tile_stride, dq_accum_delta = -dq_accum_tile_stride, lse_delta = -kBlockM;
        if (has_next) {
            if (head_step) {
                q_delta = 0;
                do_delta = 0;
                dq_accum_delta = 0;
                lse_delta = 0;
            } else if (u_next == u + 1 && !run_starts.test(u_next)) {
                m_block_next = m_block - 1;
                q_row_next = q_row - kBlockM;
                q_row_padded_next = q_row_padded - kBlockM;
            } else {
                q_noisy_next = tiles.q_is_noisy(eidx(u_next), k_noisy);
                m_block_next = tiles.tile_idx(eidx(u_next));
                q_row_next = binfo.half_row(!q_noisy_next) + m_block_next * kBlockM;
                q_row_padded_next = binfo.padded_row(bidb, !q_noisy_next) + m_block_next * kBlockM;
                q_delta = index_t(q_row_next - q_row) * params.q_row_stride;
                do_delta = index_t(q_row_next - q_row) * params.do_row_stride;
                dq_accum_delta = index_t(q_row_padded_next - q_row_padded) * dq_accum_row_stride;
                lse_delta = index_t(q_row_padded_next - q_row_padded);
            }
            if constexpr (Is_grouped) {
                const int head_delta = g_next - g;
                q_delta += index_t(head_delta) * params.q_head_stride;
                do_delta += index_t(head_delta) * params.do_head_stride;
                dq_accum_delta += index_t(head_delta) * params.d_rounded;
                lse_delta += index_t(head_delta) * params.total_padded;
            }
        }

        if (Double_buffer && has_next) {
            // Double buffer for sQ
            const int sQ_offset = it % 2 == 0 ? size(sQ) : -size(sQ);
            tQsQ.data() = tQsQ.data() + sQ_offset;
            tSsQ.data() = tSsQ.data() + sQ_offset;
            // Advance gQ
            tQgQ.data() = tQgQ.data() + q_delta;
            FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ, n - m_block_next * kBlockM
            );
            FLASH_NAMESPACE::cp_async_fence();
        }

        Tensor dS_reshaped = make_tensor(dS.data(), acc_dp.layout());
        // Convert dS from fp32 to fp16
        Tensor tdSrdS = FLASH_NAMESPACE::convert_type<Element>(dS_reshaped);
        Tensor tdSadS = smem_thr_copy_PdS.retile_S(tdSrdS);                                          // ((Atom,AtomNum), MMA_N, MMA_N)
        cute::copy(smem_tiled_copy_PdS, tdSadS, tdSsdS);
        __syncthreads();

        FLASH_NAMESPACE::gemm(acc_dv, tdVrPt, tdVrdO, tdVsPt, tdVsdOt, tiled_mma_dkv,
                              smem_tiled_copy_PdSt, smem_tiled_copy_QdOt, smem_thr_copy_PdSt, smem_thr_copy_QdOt);

        __syncthreads(); // Need syncthreads since we're writing to the same sdO location

        if (has_next) {
            // Advance gdO
            tdOgdO.data() = tdOgdO.data() + do_delta;
            FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_dO, tdOgdO, tdOsdO, tQcQ, tQpQ, n - m_block_next * kBlockM
            );
            FLASH_NAMESPACE::cp_async_fence();
        }

        FLASH_NAMESPACE::gemm(acc_dq, tdQrdS, tdQrKt, tdQsdS, tdQsKt, tiled_mma_dq,
                              smem_tiled_copy_dS, smem_tiled_copy_Kt, smem_thr_copy_dS, smem_thr_copy_Kt);

        if (has_next) {
            gLSE.data() = gLSE.data() + lse_delta;
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) { lse(mi) = gLSE(get<0>(taccScS_row(mi))); }
            gdPsum.data() = gdPsum.data() + lse_delta;
        }

        CUTE_STATIC_ASSERT_V(size(acc_dq) == size(tdQgdQaccum));
        #pragma unroll
        for (int i = 0; i < size(acc_dq); ++i) { atomicAdd(&tdQgdQaccum(i), acc_dq(i)); }
        if (has_next) { tdQgdQaccum.data() = tdQgdQaccum.data() + dq_accum_delta; }

        FLASH_NAMESPACE::gemm(acc_dk, tdKrdSt, tdKrQt, tdKsdSt, tdKsQt, tiled_mma_dkv,
                              smem_tiled_copy_PdSt, smem_tiled_copy_QdOt, smem_thr_copy_PdSt, smem_thr_copy_QdOt);
        if (Double_buffer) {  // Double buffer for sQ
            tdKsQt.data() = tdKsQt.data() + (it % 2 == 0 ? size(sQ) : -size(sQ));
        }
        if (!Double_buffer && has_next) {
            __syncthreads();
            // Advance gQ
            tQgQ.data() = tQgQ.data() + q_delta;
            FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ, n - m_block_next * kBlockM
            );
            FLASH_NAMESPACE::cp_async_fence();
        }

        u = u_next;
        bidh += g_next - g;
        g = g_next;
        q_noisy = q_noisy_next;
        m_block = m_block_next;
        q_row = q_row_next;
        q_row_padded = q_row_padded_next;
    }

    // Epilogue

    if (Is_dropout) {
        #pragma unroll
        for (int i = 0; i < size(acc_dv); ++i) { acc_dv(i) *= params.rp_dropout; }
    }
    #pragma unroll
    for (int i = 0; i < size(acc_dk); ++i) { acc_dk(i) *= params.scale_softmax_rp_dropout; }

    // Convert acc_dv from fp32 to fp16
    Tensor rdK = FLASH_NAMESPACE::convert_type<Element>(acc_dk);
    Tensor rdV = FLASH_NAMESPACE::convert_type<Element>(acc_dv);

    Tensor sdK = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutdKV{});  // (SMEM_N, SMEM_K)
    Tensor sdV = make_tensor(sdK.data() + size(sdK), typename Kernel_traits::SmemLayoutdKV{}); // (SMEM_N, SMEM_K)

    // Partition sdV and sdK to match the accumulator partitioning
    auto smem_tiled_copy_dKV = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdKV{}, tiled_mma_dkv);
    auto smem_thr_copy_dKV = smem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor taccdKrdK = smem_thr_copy_dKV.retile_S(rdK);       // ((Atom,AtomNum), MMA_N, MMA_N)
    Tensor taccdKsdK = smem_thr_copy_dKV.partition_D(sdK);   // ((Atom,AtomNum),PIPE_M,PIPE_N)
    Tensor taccdVrdV = smem_thr_copy_dKV.retile_S(rdV);       // ((Atom,AtomNum), MMA_N, MMA_N)
    Tensor taccdVsdV = smem_thr_copy_dKV.partition_D(sdV);    // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // We need syncthreads here since we're writing to the same location as sK and sV.
    // Without syncthreads, some thread might modify the location of sK while another thread
    // is reading it for dQ gemm, leading to a race condition.
    __syncthreads();

    cute::copy(smem_tiled_copy_dKV, taccdKrdK, taccdKsdK);
    cute::copy(smem_tiled_copy_dKV, taccdVrdV, taccdVsdV);

    // dK / dV summed over the query heads of the group: head bidh_group of the (partial) dK / dV buffer
    const index_t row_offset_dk = binfo.offset(params.dk_batch_stride, params.dk_row_stride, bidb)
       + index_t(k_row0) * params.dk_row_stride + bidh_group * params.dk_head_stride;
    const index_t row_offset_dv = binfo.offset(params.dv_batch_stride, params.dv_row_stride, bidb)
       + index_t(k_row0) * params.dv_row_stride + bidh_group * params.dv_head_stride;
    Tensor gdK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_ptr) + row_offset_dk),
                             Shape<Int<kBlockN>, Int<kHeadDim>>{},
                             make_stride(params.dk_row_stride, _1{}));
    Tensor gdV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_ptr) + row_offset_dv),
                             Shape<Int<kBlockN>, Int<kHeadDim>>{},
                             make_stride(params.dv_row_stride, _1{}));

    typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
    auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
    Tensor tdKsdK = gmem_thr_copy_dKV.partition_S(sdK);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
    Tensor tdVsdV = gmem_thr_copy_dKV.partition_S(sdV);   // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);

    __syncthreads();
    Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
    cute::copy(gmem_tiled_copy_dKV, tdKsdK, tdKrdK);
    Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
    cute::copy(gmem_tiled_copy_dKV, tdVsdV, tdVrdV);
    Tensor cdKV = make_identity_tensor(make_shape(size<0>(sdK), size<1>(sdK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
    Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKgdK)));
    #pragma unroll
    for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(0, 0, k)) < params.d; }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKVcdKV, tdKVpdKV, n_residue
    );
    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdKVcdKV, tdKVpdKV, n_residue
    );

}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Is_grouped, typename Params>
inline __device__ void compute_dq_dk_dv_seqk_parallel(const Params &params) {

    // Key tiles of both halves, the most expensive ones first: the clean key tiles (seen by the queries of both
    // halves, tile 0 by all of them), then the noisy key tiles (seen by the noisy queries of their blocks).
    const int num_n_blocks = (params.seqlen / 2 + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
    // A single call site of compute_dq_dk_dv_1colblock (inlined; two call sites double the code of the kernel, ~1%
    // slower): one key tile per thread block (in the order of tile_of_block), or, if deterministic, the key tiles
    // blockIdx.x, blockIdx.x + gridDim.x, ... (each thread block does atomicAdd to a different dQ_accum buffer).
    int t, bidb, bidh_group, stride;
    if (!params.deterministic) {
        tile_of_block(params.sched_heads, params.h / params.bwd_heads_per_cta, t, bidb, bidh_group);
        stride = 2 * num_n_blocks;
    } else {
        t = blockIdx.x;
        bidb = blockIdx.y;
        bidh_group = blockIdx.z;
        stride = gridDim.x;
    }
    for (; t < 2 * num_n_blocks; t += stride) {
        const bool k_noisy = t >= num_n_blocks;
        const int n_block = k_noisy ? t - num_n_blocks : t;
        FLASH_NAMESPACE::compute_dq_dk_dv_1colblock<Kernel_traits, Is_dropout, Is_even_MN, Is_even_K, Is_softcap, Is_grouped>(params, bidb, bidh_group, k_noisy, n_block);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
} // namespace FLASH_NAMESPACE
