/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Copyright (c) 2026, fast-diffusion-attention authors.
 *
 * Forward kernel of the diffusion attention: compute_attn_1rowblock of FlashAttention-2 (flash_fwd_kernel.h) with the
 * key tiles of the diffusion mask (DiffusionFwdTiles, diffusion.h) instead of the causal / local ones.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"
#ifndef FLASHATTENTION_DISABLE_DROPOUT
#include "philox_unpack.cuh" // For at::cuda::philox::unpack
#endif

#include <tuple>
#include <type_traits>
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

// Split (FlashAttention-2's compute_attn_1rowblock_splitkv): split n_split_idx of num_n_splits of the key tiles, the
// normalized partial output and its log-sum-exp in FP32 (params.oaccum_ptr, params.softmax_lseaccum_ptr).
template<typename Kernel_traits, bool Is_dropout, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, bool Split=false, typename Params>
inline __device__ void compute_attn_1rowblock(const Params &params, const int bidb, const int bidh, const bool q_noisy, const int m_block,
                                              const int n_split_idx=0, const int num_n_splits=1) {

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
    constexpr int kNWarps = Kernel_traits::kNWarps;

#ifndef FLASHATTENTION_DISABLE_DROPOUT
    auto seed_offset = at::cuda::philox::unpack(*reinterpret_cast<at::PhiloxCudaState const*>(params.philox_args));
#else
    auto seed_offset = std::make_tuple(uint64_t(0), uint64_t(0));
#endif
    FLASH_NAMESPACE::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                                     bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    const DiffusionBlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    const int n = binfo.n;
    if (m_block * kBlockM >= n) return;

    const DiffusionGeometry geo(params.block_size_divmod, n);
    const DiffusionFwdTiles<kBlockM, kBlockN> tiles(geo, q_noisy, m_block);
    // Every query sees at least its own block: n_block_max >= 1.
    const int n_block_max = tiles.n_block_max();
    // Split-KV: the key tiles [v_lo, v_hi) of the enumeration, n_blocks_per_split per split as FlashAttention-2 splits
    // the key tiles of causal attention on a half (ceil(n / kBlockN) of them); the last split also takes the extra
    // tiles of the noisy query tiles (their noisy diagonal).
    static_assert(!Split || !Is_dropout, "Split-KV is not implemented for dropout (as in FlashAttention-2)");
    const int n_blocks_per_split = Split ? (cute::ceil_div(n, kBlockN) + num_n_splits - 1) / num_n_splits : 0;
    const int v_lo = Split ? n_split_idx * n_blocks_per_split : 0;
    const int v_hi = !Split || n_split_idx == num_n_splits - 1 ? n_block_max : std::min(v_lo + n_blocks_per_split, n_block_max);
    // Noisy key tiles of a noisy query tile when the block size divides 16 (the diagonal of the noisy half): the 16
    // query rows of an MMA row group of a warp see 16 keys of a key tile at most (their blocks), at the same offset in
    // the key tile for all the row groups of the warp when the row groups are kNWarps * 16 rows apart (a multiple of
    // kBlockN). Each warp then computes only these 16 keys (diag_sub: their 16-key sub-tile of the key tile): a quarter
    // of the MMAs of a 64-key tile. Used with 128-row query tiles (two diagonal key tiles per noisy query tile, at the
    // register limit): measured on A100, +2-6% of the forward (hdim 64 - 256, 4K-16K tokens); on L40S also for the
    // 128 x 64 hdim 32 kernel (+3-6%). With 64-row query tiles, between -2% and +2% depending on the head dimension
    // and length (and with hdim 32 the 64 x 64 kernel needs more than the 128 registers of 4 thread blocks per SM:
    // -10%): FlashAttention-2's loop structure (with its special first iteration). With dropout, a 16-key sub-tile uses one half of its 32-column block of random
    // numbers (Dropout::apply_dropout_half): the same dropout pattern as the whole tile, and as the backward.
    constexpr bool kDiagSubtiles = kBlockM == 128 && kBlockN % 16 == 0
        && (16 * kNWarps) % kBlockN == 0;
    static_assert(!Split || !kDiagSubtiles, "The split-KV kernels use FlashAttention-2's 64-row tiles");

    // The key tiles v >= v_mask_end need masking. Without kDiagSubtiles, the first iteration is always a masking one
    // (it initializes the softmax statistics), applying the mask to a fully visible tile is a no-op.
    const int v_mask_end = kDiagSubtiles ? tiles.n_block_mask_min : std::min(tiles.n_block_mask_min, n_block_max - 1);
    const int v_mask_lo = Split ? std::max(v_mask_end, v_lo) : v_mask_end;

    // Row of the first query of the tile in the sample, number of valid rows of the tile.
    const int q_row0 = binfo.half_row(!q_noisy) + m_block * kBlockM;
    const int m_residue = n - m_block * kBlockM;
    // Rows of the clean half in S_dmask and in the dropout coordinates.
    const int hr = params.seqlen_half_rounded;
    const int p_row0 = (q_noisy ? 0 : hr) + m_block * kBlockM;
    // Split-KV: rows of the partial outputs (num_splits, b, h, seqlen, d_rounded) and LSEs (num_splits, b, h, seqlen)
    const index_t row_offset_lseaccum = Split ? ((index_t(n_split_idx) * params.b + bidb) * params.h + bidh) * params.seqlen + q_row0 : 0;
    if constexpr (Split) {
        if (v_lo >= v_hi) {
            // No key tile in this split: write 0 to the partial output and -inf to its LSE (FlashAttention-2)
            Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.oaccum_ptr) + row_offset_lseaccum * params.d_rounded),
                                         Shape<Int<kBlockM>, Int<kHeadDim>>{}, Stride<Int<kHeadDim>, _1>{});
            Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lseaccum_ptr) + row_offset_lseaccum),
                                           Shape<Int<kBlockM>>{}, Stride<_1>{});
            typename Kernel_traits::GmemTiledCopyOaccum gmem_tiled_copy_Oaccum;
            auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
            Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);
            Tensor tOrOaccum = make_tensor<ElementAccum>(shape(tOgOaccum));
            clear(tOrOaccum);
            Tensor cO = make_identity_tensor(make_shape(size<0>(gOaccum), size<1>(gOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
            Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);
            Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
            if (!Is_even_K) {
                #pragma unroll
                for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
            }
            // Clear_OOB_K must be false since we don't want to write zeros to gmem
            FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO, m_residue
            );
            #pragma unroll
            for (int m = 0; m < size<1>(tOgOaccum); ++m) {
                const int row = get<0>(tOcO(0, m, 0));
                if (row < m_residue && get<1>(tOcO(0, m, 0)) == 0) { gLSEaccum(row) = -INFINITY; }
            }
            return;
        }
    }

    // We iterate over the key tiles in the order of DiffusionFwdTiles (masked tiles first), like FlashAttention-2
    // iterates over n_block in reverse order: only the masked tiles can extend beyond the end of a half, the K and V
    // loads of the other iterations are not predicated.
    int v = v_hi - 1;
    int row_v = tiles.key_row(v, n);  // Row of the first key of tile v in the sample

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo.offset(params.q_batch_stride, params.q_row_stride, bidb)
                                          + index_t(q_row0) * params.q_row_stride + bidh * params.q_head_stride),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gQ = mQ;  // (kBlockM, kHeadDim)
    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)
                                          + binfo.offset(params.k_batch_stride, params.k_row_stride, bidb)
                                          + index_t(row_v) * params.k_row_stride
                                          + (bidh / params.h_h_k_ratio) * params.k_head_stride),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.k_row_stride, _1{}));  // (kBlockN, kHeadDim)
    Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)
                                          + binfo.offset(params.v_batch_stride, params.v_row_stride, bidb)
                                          + index_t(row_v) * params.v_row_stride
                                          + (bidh / params.h_h_k_ratio) * params.v_head_stride),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(params.v_row_stride, _1{}));  // (kBlockN, kHeadDim)
    // S_dmask: (b, h, 2 * hr, 2 * hr), the clean half starts at row / column hr.
    const index_t p_row_offset = (index_t(bidb * params.h + bidh) * (2 * hr) + p_row0) * (2 * hr);

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    // Careful we're using the same smem for sQ and sK | sV if Share_Q_K_smem;
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    //
    // PREDICATES
    //

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

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

    // Number of valid keys of the tile starting at row `row` of the sample.
    auto key_residue = [&](const int row) { return (row >= n ? 2 * n : n) - row; };

    // Prologue

    // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ, m_residue);
    if (Kernel_traits::Is_Q_in_regs) { cute::cp_async_fence(); }

    if (Kernel_traits::Share_Q_K_smem) {
        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV, key_residue(row_v));
    cute::cp_async_fence();

    if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        FLASH_NAMESPACE::cp_async_wait<1>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    clear(acc_o);

    FLASH_NAMESPACE::Softmax<2 * size<1>(acc_o)> softmax;

    // Offsets of the previous key tile of the same half
    const index_t k_tile_back = -index_t(kBlockN) * params.k_row_stride;
    const index_t v_tile_back = -index_t(kBlockN) * params.v_row_stride;

    // The masked iterations process their key tile in kSplitMasked sub-tiles of kBlockN / kSplitMasked keys (one
    // online-softmax update each, same shared memory tile): when the scores of a tile take 64 registers per thread or
    // more, the mask (two bounds per row) on top of the scores and the output accumulator otherwise exceeds the 255
    // registers of a thread, and the spills of the softmax statistics then also slow down the unmasked iterations.
    // Measured on A100 (sm86 tiles emulated): +5% of the forward with hdim 96, +1-3% with hdim 64 / 128 / 192; slower
    // when the scores take 32 registers per thread (-0.5-1%) and with hdim 32 (small output accumulator: -3%, -10%
    // with dropout).
    constexpr int kSplitMasked = kHeadDim >= 64 && kBlockN >= 64 && kBlockM * kBlockN / Kernel_traits::kNThreads >= 64 ? 2 : 1;

    // One iteration of the mainloop, on key tile v. We separate out three kinds of iterations: those that don't need
    // masking on S (Kind 0), those that do (Kind 1, with predicated loads of K and V), and the noisy diagonal tiles
    // of kDiagSubtiles (Kind 2, masked, 16 keys per warp).
    // Must be inlined: the accumulators it captures by reference would otherwise live in local memory.
    auto fwd_step = [&](auto is_first_c, auto kind_c) __attribute__((always_inline)) {
        constexpr bool Is_first = decltype(is_first_c)::value;
        constexpr int Kind = decltype(kind_c)::value;
        constexpr bool Is_masked = Kind != 0;
        constexpr bool Is_diag = Kind == 2;
        constexpr int kSplit = Kind == 1 ? kSplitMasked : 1;
        constexpr int kSubN = Is_diag ? 16 : kBlockN / kSplit;
        const bool k_noisy = row_v < n;
        const int k0 = row_v - binfo.half_row(!k_noisy);  // first key of the tile inside its half
        // Kind 2: the 16-key sub-tile of the key tile seen by the rows of this warp
        const int diag_sub = Is_diag ? ((m_block * kBlockM - k0 + 16 * (tidx / 32)) & (kBlockN - 1)) / 16 : 0;
        // Offset of rows [16 * diag_sub, 16 * diag_sub + 16) of the K and V tiles in shared memory (atoms of 8 rows x
        // kBlockKSmem columns, arranged rows first): a multiple of the swizzle period.
        const int diag_sub_offset = diag_sub * 16 * Kernel_traits::kBlockKSmem;

        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();

        // Load V of tile v
        if constexpr (Is_masked) {
            // Clear the smem tiles to account for predicated off loads
            FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, n - k0
            );
        } else {
            FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV);
        }
        cute::cp_async_fence();

        int row_next = row_v;
        // Offset of the next key tile (in rows). In the unmasked iterations, the next tile is the previous tile of
        // the same half (FlashAttention-2's n_block - 1), except when going from the visible noisy tiles to the
        // visible clean tiles: no index computation in the common case (measured: up to 1.4% of the forward).
        bool next_is_prev_tile = false;
        if constexpr (!Is_masked) { next_is_prev_tile = v != tiles.clean_mask_lo; }

        // Sub-tile S of the key tile: keys [k0 + S * kSubN, k0 + (S + 1) * kSubN)
        auto sub_step = [&](auto s_c) __attribute__((always_inline)) {
            constexpr int S = decltype(s_c)::value;
            Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kSubN>>{});  // (MMA=4, MMA_M, MMA_N)
            clear(acc_s);
            if constexpr (Is_diag) {
                // Sub-tile diag_sub of 16 rows: a whole number of swizzle periods away from sub-tile 0, a plain
                // pointer offset (diag_sub_offset) on the layout of sub-tile 0.
                Tensor sK_0 = local_tile(sK, Shape<Int<kSubN>, Int<kHeadDim>>{}, make_coord(_0{}, _0{}));
                Tensor sK_s = make_tensor(sK_0.data() + diag_sub_offset, sK_0.layout());
                Tensor tSsK_s = smem_thr_copy_K.partition_S(sK_s);
                Tensor tSrK_s = thr_mma.partition_fragment_B(sK_s);
                FLASH_NAMESPACE::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
                    acc_s, tSrQ, tSrK_s, tSsQ, tSsK_s, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
                    smem_thr_copy_Q, smem_thr_copy_K
                );
            } else if constexpr (kSplit == 1) {
                FLASH_NAMESPACE::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
                    acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
                    smem_thr_copy_Q, smem_thr_copy_K
                );
            } else {
                Tensor sK_s = local_tile(sK, Shape<Int<kSubN>, Int<kHeadDim>>{}, make_coord(Int<S>{}, _0{}));
                Tensor tSsK_s = smem_thr_copy_K.partition_S(sK_s);
                Tensor tSrK_s = thr_mma.partition_fragment_B(sK_s);
                FLASH_NAMESPACE::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
                    acc_s, tSrQ, tSrK_s, tSsQ, tSsK_s, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
                    smem_thr_copy_Q, smem_thr_copy_K
                );
            }
            if constexpr (Is_softcap){
                FLASH_NAMESPACE::apply_softcap(acc_s, params.softcap);
            }

            if constexpr (Is_masked) {
                FLASH_NAMESPACE::apply_diffusion_mask(
                    acc_s, geo, q_noisy, k_noisy, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16,
                    k0 + (Is_diag ? diag_sub : S) * kSubN
                );
            }

            // First sub-tile: V of tile v is ready. Last sub-tile: all warps are done with K of tile v.
            FLASH_NAMESPACE::cp_async_wait<0>();
            __syncthreads();
            if constexpr (S == kSplit - 1) {
                // Load K of the next tile
                if (v > v_lo) {
                    if (next_is_prev_tile) {
                        row_next = row_v - kBlockN;
                        tKgK.data() = tKgK.data() + k_tile_back;
                    } else {
                        row_next = tiles.key_row(v - 1, n);
                        tKgK.data() = tKgK.data() + index_t(row_next - row_v) * params.k_row_stride;
                    }
                    if constexpr (Is_masked) {
                        FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV, key_residue(row_next));
                    } else {
                        FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);
                    }
                    // This cp_async_fence needs to be in the if block, otherwise the synchronization
                    // isn't right and we get race conditions.
                    cute::cp_async_fence();
                }
            }

            // Rows can be fully masked in the masked tiles: Check_inf
            softmax.template softmax_rescale_o<Is_first && S == 0, /*Check_inf=*/Is_masked>(acc_s, acc_o, params.scale_softmax_log2);

            // Convert acc_s from fp32 to fp16/bf16
            Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
            // Coordinates of the 16 x 32 blocks of the dropout random numbers: rows and columns of the clean half start at hr.
            const int col0 = (k_noisy ? 0 : hr) + k0 + (Is_diag ? diag_sub : S) * kSubN;
            int block_row_idx = p_row0 / 16 + tidx / 32;
            int block_col_idx = col0 / 32;
            // Kind 2: the 16-column sub-tile is one half of a 32-column block of random numbers
            auto apply_dropout = [&](auto &t, auto encode_c) __attribute__((always_inline)) {
                constexpr bool Encode = decltype(encode_c)::value;
                if constexpr (Is_diag) {
                    dropout.template apply_dropout_half<Encode>(t, block_row_idx, block_col_idx, (col0 / 16) % 2, kNWarps);
                } else {
                    dropout.template apply_dropout<Encode>(t, block_row_idx, block_col_idx, kNWarps);
                }
            };
            if constexpr (Return_softmax) {
                Tensor rP_drop = make_fragment_like(rP);
                cute::copy(rP, rP_drop);
                apply_dropout(rP_drop, /*encode_dropout_in_sign_bit=*/std::true_type{});
                Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + p_row_offset + col0),
                                        Shape<Int<kBlockM>, Int<kSubN>>{},
                                        make_stride(2 * hr, _1{}));
                Tensor tSgS = thr_mma.partition_C(gP);
                // Only the rows and keys inside the halves: tiles larger than 128 (the alignment of the halves in
                // S_dmask) would otherwise write into the other half
                Tensor tScS = thr_mma.partition_C(make_identity_tensor(Shape<Int<kBlockM>, Int<kSubN>>{}));
                const int key_residue = n - (col0 - (k_noisy ? 0 : hr));
                #pragma unroll
                for (int i = 0; i < size(tSgS); ++i) {
                    if (get<0>(tScS(i)) < m_residue && get<1>(tScS(i)) < key_residue) { tSgS(i) = rP_drop(i); }
                }
            }
            if constexpr (Is_dropout) {
                apply_dropout(rP, std::false_type{});
            }

            // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
            // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
            Tensor tOrP = make_tensor(rP.data(), FLASH_NAMESPACE::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));
            if constexpr (Is_diag) {
                Tensor sVt_0 = local_tile(sVt, Shape<Int<kHeadDim>, Int<kSubN>>{}, make_coord(_0{}, _0{}));
                Tensor sVt_s = make_tensor(sVt_0.data() + diag_sub_offset, sVt_0.layout());
                Tensor sVtNoSwizzle_s = local_tile(sVtNoSwizzle, Shape<Int<kHeadDim>, Int<kSubN>>{}, make_coord(_0{}, _0{}));
                Tensor tOsVt_s = smem_thr_copy_V.partition_S(sVt_s);
                Tensor tOrVt_s = thr_mma.partition_fragment_B(sVtNoSwizzle_s);
                FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt_s, tOsVt_s, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
            } else if constexpr (kSplit == 1) {
                FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
            } else {
                Tensor sVt_s = local_tile(sVt, Shape<Int<kHeadDim>, Int<kSubN>>{}, make_coord(_0{}, Int<S>{}));
                Tensor sVtNoSwizzle_s = local_tile(sVtNoSwizzle, Shape<Int<kHeadDim>, Int<kSubN>>{}, make_coord(_0{}, Int<S>{}));
                Tensor tOsVt_s = smem_thr_copy_V.partition_S(sVt_s);
                Tensor tOrVt_s = thr_mma.partition_fragment_B(sVtNoSwizzle_s);
                FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt_s, tOsVt_s, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
            }
        };
        sub_step(Int<0>{});
        if constexpr (kSplit == 2) { sub_step(Int<1>{}); }

        // Advance gV to the next tile
        if (next_is_prev_tile) {
            if (v > v_lo) { tVgV.data() = tVgV.data() + v_tile_back; }
        } else {
            tVgV.data() = tVgV.data() + index_t(row_next - row_v) * params.v_row_stride;
        }
        row_v = row_next;
    };

    if constexpr (kDiagSubtiles) {
        // The softmax statistics start at (-inf, 0) instead of FlashAttention-2's special first iteration (two copies
        // less of the iteration code: fewer registers; the same results for a positive softmax scale: the first
        // rescaling multiplies zeros).
        #pragma unroll
        for (int mi = 0; mi < size(softmax.row_max); ++mi) {
            softmax.row_max(mi) = -INFINITY;
            softmax.row_sum(mi) = 0.f;
        }
        if (q_noisy && 16 % geo.bs() == 0) {
            // The noisy key tiles [n_clean, n_block_max), the first iterations: all masked (none is fully visible
            // with these block sizes).
            for (; v >= tiles.n_clean; --v) {
                fwd_step(std::false_type{}, std::integral_constant<int, 2>{});
            }
        }
    } else {
        // The first iteration (masked, initializes the softmax statistics)
        fwd_step(std::true_type{}, std::integral_constant<int, 1>{});
        --v;
    }
    // The other iterations that need masking
    for (; v >= v_mask_lo; --v) {
        fwd_step(std::false_type{}, std::integral_constant<int, 1>{});
    }
    // These are the iterations where we don't need masking on S
    for (; v >= v_lo; --v) {
        fwd_step(std::false_type{}, std::integral_constant<int, 0>{});
    }

    // Epilogue

    if constexpr (Split) {
        // The normalized partial output (FP32) and its LSE (-inf if no key is visible), as FlashAttention-2
        Tensor lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false, /*Split=*/true>(acc_o, params.scale_softmax);
        Tensor sOaccum = make_tensor(make_smem_ptr(reinterpret_cast<ElementAccum *>(smem_)), typename Kernel_traits::SmemLayoutO{}); // (SMEM_M,SMEM_N)
        // Partition sO to match the accumulator partitioning
        auto smem_tiled_copy_Oaccum = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomOaccum{}, tiled_mma);
        auto smem_thr_copy_Oaccum = smem_tiled_copy_Oaccum.get_thread_slice(tidx);
        Tensor rO = FLASH_NAMESPACE::convert_type<ElementAccum>(acc_o);
        Tensor taccOrOaccum = smem_thr_copy_Oaccum.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
        Tensor taccOsOaccum = smem_thr_copy_Oaccum.partition_D(sOaccum);     // ((Atom,AtomNum),PIPE_M,PIPE_N)
        // sOaccum is larger than sQ, so we need to syncthreads here
        __syncthreads();
        cute::copy(smem_tiled_copy_Oaccum, taccOrOaccum, taccOsOaccum);

        Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.oaccum_ptr) + row_offset_lseaccum * params.d_rounded),
                                     Shape<Int<kBlockM>, Int<kHeadDim>>{}, Stride<Int<kHeadDim>, _1>{});
        Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lseaccum_ptr) + row_offset_lseaccum),
                                       Shape<Int<kBlockM>>{}, Stride<_1>{});
        typename Kernel_traits::GmemTiledCopyOaccum gmem_tiled_copy_Oaccum;
        auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
        Tensor tOsOaccum = gmem_thr_copy_Oaccum.partition_S(sOaccum);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
        Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);

        __syncthreads();

        Tensor tOrOaccum = make_tensor<ElementAccum>(shape(tOgOaccum));
        cute::copy(gmem_tiled_copy_Oaccum, tOsOaccum, tOrOaccum);

        Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
        static_assert(decltype(size<0>(taccOcO))::value == 4);
        // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
        Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
        CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
        if (get<1>(taccOcO_row(0)) == 0) {
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) {
                const int row = get<0>(taccOcO_row(mi));
                if (row < m_residue) { gLSEaccum(row) = lse(mi); }
            }
        }

        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(sOaccum), size<1>(sOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO, m_residue
        );
        return;
    }

    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);

    // Convert acc_o from fp32 to fp16/bf16
    Tensor rO = FLASH_NAMESPACE::convert_type<Element>(acc_o);
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // sO has the same size as sQ, so we don't need to sync here.
    if (Kernel_traits::Share_Q_K_smem) { __syncthreads(); }

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                          + binfo.offset(params.o_batch_stride, params.o_row_stride, bidb)
                                          + index_t(q_row0) * params.o_row_stride + bidh * params.o_head_stride),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.o_row_stride, _1{}));  // (kBlockM, kHeadDim)
    // LSE: (b, h, seqlen) or (h, total) if unpadded_lse (varlen)
    const index_t lse_offset = params.unpadded_lse
        ? index_t(bidh) * params.total_q + binfo.start + q_row0
        : index_t(bidb * params.h + bidh) * params.seqlen + q_row0;
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr) + lse_offset),
                              Shape<Int<kBlockM>>{}, Stride<_1>{});

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    __syncthreads();

    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    static_assert(decltype(size<0>(taccOcO))::value == 4);
    // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
    Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
    if (get<1>(taccOcO_row(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            if (row < m_residue) { gLSE(row) = lse(mi); }
        }
    }

    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
    }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, m_residue
    );
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_attn(const Params &params) {
    // blockIdx.x enumerates the query tiles of both halves, the most expensive ones first: query tile m of the
    // noisy half visits the clean key tiles before its blocks and its noisy diagonal, query tile m of the clean half
    // the clean key tiles up to its own (like causal attention).
    int tile, bidb, bidh;
    tile_of_block(params.sched_heads, params.h, tile, bidb, bidh);
    const int num_m_blocks = gridDim.x / 2;
    const int m_block = num_m_blocks - 1 - tile / 2;
    const bool q_noisy = tile % 2 == 0;

    // We want the fwd and bwd to generate the same dropout pattern (RNG), without restricting
    // them to have the same number of threads or have to traverse the attention matrix
    // in the same order.
    // In the Philox RNG, we use the offset to store the batch, head, and the lane id
    // (within a warp). We use the subsequence to store the location of the 16 x 32 blocks within
    // the attention matrix. This way, as long as we have the batch, head, and the location of
    // the 16 x 32 block within the attention matrix, we can generate the exact same dropout pattern.

    FLASH_NAMESPACE::compute_attn_1rowblock<Kernel_traits, Is_dropout, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, q_noisy, m_block);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_even_MN, bool Is_even_K, bool Is_softcap, typename Params>
inline __device__ void compute_attn_splitkv(const Params &params) {
    // FlashAttention-2's grid of the split-KV kernel: (2 * num_m_blocks, num_splits, b * h)
    const int num_m_blocks = gridDim.x / 2;
    const bool q_noisy = int(blockIdx.x) < num_m_blocks;
    const int m_block = q_noisy ? int(blockIdx.x) : int(blockIdx.x) - num_m_blocks;
    const int n_split_idx = blockIdx.y;
    const int bidb = blockIdx.z / params.h;
    const int bidh = blockIdx.z - bidb * params.h;
    FLASH_NAMESPACE::compute_attn_1rowblock<Kernel_traits, /*Is_dropout=*/false, Is_even_MN, Is_even_K, Is_softcap, /*Return_softmax=*/false, /*Split=*/true>(
        params, bidb, bidh, q_noisy, m_block, n_split_idx, gridDim.y);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// FlashAttention-2's combine_attn_seqk_parallel: the output and LSE of every row from the num_splits partial results
// (batched inputs only: LSE (b, h, seqlen), the rows of a sample in order).
template<typename Kernel_traits, int kBlockM, int Log_max_splits, bool Is_even_K, typename Params>
inline __device__ void combine_attn_seqk_parallel(const Params &params) {
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    constexpr int kMaxSplits = 1 << Log_max_splits;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNThreads = Kernel_traits::kNThreads;

    static_assert(kMaxSplits <= 128, "kMaxSplits must be <= 128");
    static_assert(kBlockM == 4 || kBlockM == 8 || kBlockM == 16 || kBlockM == 32, "kBlockM must be 4, 8, 16 or 32");
    static_assert(kNThreads == 128, "We assume that each block has 128 threads");

    // Shared memory.
    // kBlockM + 1 instead of kBlockM to reduce bank conflicts.
    __shared__ ElementAccum sLSE[kMaxSplits][kBlockM + 1];

    // The thread and block index.
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;

    const index_t lse_size = index_t(params.b) * params.h * params.seqlen;

    const index_t row_offset_lse = index_t(bidx) * kBlockM;
    Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lseaccum_ptr) + row_offset_lse),
                                   Shape<Int<kMaxSplits>, Int<kBlockM>>{},
                                   make_stride(lse_size, _1{}));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                              Shape<Int<kBlockM>>{}, Stride<_1>{});

    constexpr int kNLsePerThread = (kMaxSplits * kBlockM + kNThreads - 1) / kNThreads;

    // Read the LSE values from gmem and store them in shared memory, then transpose them.
    constexpr int kRowsPerLoadLSE = kNThreads / kBlockM;
    #pragma unroll
    for (int l = 0; l < kNLsePerThread; ++l) {
        const int row = l * kRowsPerLoadLSE + tidx / kBlockM;
        const int col = tidx % kBlockM;
        ElementAccum lse = (row < params.num_splits && col < lse_size - bidx * kBlockM) ? gLSEaccum(row, col) : -INFINITY;
        if (row < kMaxSplits) { sLSE[row][col] = lse; }
    }
    __syncthreads();
    Tensor lse_accum = make_tensor<ElementAccum>(Shape<Int<kNLsePerThread>>{});
    constexpr int kRowsPerLoadTranspose = std::min(kRowsPerLoadLSE, kMaxSplits);
    // To make sure that kMaxSplits is within 1 warp: we decide how many elements within kMaxSplits
    // each thread should hold. If kMaxSplits = 16, then each thread holds 2 elements (128 threads,
    // kBlockM rows, so each time we load we can load 128 / kBlockM rows).
    static_assert(kRowsPerLoadTranspose <= 32);
    static_assert(kNLsePerThread * kRowsPerLoadTranspose <= kMaxSplits);
    #pragma unroll
    for (int l = 0; l < kNLsePerThread; ++l) {
        const int row = l * kRowsPerLoadTranspose + tidx % kRowsPerLoadTranspose;
        const int col = tidx / kRowsPerLoadTranspose;
        lse_accum(l) = (row < kMaxSplits && col < kBlockM) ? sLSE[row][col] : -INFINITY;
    }

    // Compute the logsumexp of the LSE along the split dimension.
    ElementAccum lse_max = lse_accum(0);
    #pragma unroll
    for (int l = 1; l < kNLsePerThread; ++l) { lse_max = max(lse_max, lse_accum(l)); }
    MaxOp<float> max_op;
    lse_max = Allreduce<kRowsPerLoadTranspose>::run(lse_max, max_op);
    lse_max = lse_max == -INFINITY ? 0.0f : lse_max;  // In case all local LSEs are -inf
    float lse_sum = expf(lse_accum(0) - lse_max);
    #pragma unroll
    for (int l = 1; l < kNLsePerThread; ++l) { lse_sum += expf(lse_accum(l) - lse_max); }
    SumOp<float> sum_op;
    lse_sum = Allreduce<kRowsPerLoadTranspose>::run(lse_sum, sum_op);
    // For the case where all local lse == -INFINITY, we want to set lse_logsum to INFINITY. Otherwise
    // lse_logsum is log(0.0) = -INFINITY and we get NaN when we do lse_accum(l) - lse_logsum.
    ElementAccum lse_logsum = (lse_sum == 0.f || lse_sum != lse_sum) ? INFINITY : logf(lse_sum) + lse_max;
    if (tidx % kRowsPerLoadTranspose == 0 && tidx / kRowsPerLoadTranspose < kBlockM) {
        if (row_offset_lse + tidx / kRowsPerLoadTranspose < lse_size) { gLSE(tidx / kRowsPerLoadTranspose) = lse_logsum; }
    }
    // Store the scales exp(lse - lse_logsum) in shared memory.
    #pragma unroll
    for (int l = 0; l < kNLsePerThread; ++l) {
        const int row = l * kRowsPerLoadTranspose + tidx % kRowsPerLoadTranspose;
        const int col = tidx / kRowsPerLoadTranspose;
        if (row < params.num_splits && col < kBlockM) { sLSE[row][col] = expf(lse_accum(l) - lse_logsum); }
    }
    __syncthreads();

    const index_t row_offset_oaccum = index_t(bidx) * kBlockM * params.d_rounded;
    Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.oaccum_ptr) + row_offset_oaccum),
                                 Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                 Stride<Int<kHeadDim>, _1>{});
    constexpr int kBlockN = kNThreads / kBlockM;
    using GmemLayoutAtomOaccum = Layout<Shape<Int<kBlockM>, Int<kBlockN>>, Stride<Int<kBlockN>, _1>>;
    using GmemTiledCopyOaccum = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementAccum>{},
                        GmemLayoutAtomOaccum{},
                        Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
    GmemTiledCopyOaccum gmem_tiled_copy_Oaccum;
    auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
    Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_S(gOaccum);
    Tensor tOrO = make_tensor<ElementAccum>(shape(tOgOaccum));
    Tensor tOrOaccum = make_tensor<ElementAccum>(shape(tOgOaccum));
    clear(tOrO);

    // Predicates
    Tensor cOaccum = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    // Repeat the partitioning with identity layouts
    Tensor tOcOaccum = gmem_thr_copy_Oaccum.partition_S(cOaccum);
    Tensor tOpOaccum = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tOpOaccum); ++k) { tOpOaccum(k) = get<1>(tOcOaccum(0, 0, k)) < params.d; }
    }
    // Load Oaccum in then scale and accumulate to O
    for (int split = 0; split < params.num_splits; ++split) {
        FLASH_NAMESPACE::copy</*Is_even_MN=*/false, Is_even_K>(
            gmem_tiled_copy_Oaccum, tOgOaccum, tOrOaccum, tOcOaccum, tOpOaccum, lse_size - bidx * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOrOaccum); ++m) {
            int row = get<0>(tOcOaccum(0, m, 0));
            ElementAccum lse_scale = sLSE[split][row];
            #pragma unroll
            for (int k = 0; k < size<2>(tOrOaccum); ++k) {
                #pragma unroll
                for (int i = 0; i < size<0>(tOrOaccum); ++i) {
                    tOrO(i, m, k) += lse_scale * tOrOaccum(i, m, k);
                }
            }
        }
        tOgOaccum.data() = tOgOaccum.data() + lse_size * params.d_rounded;
    }

    Tensor rO = FLASH_NAMESPACE::convert_type<Element>(tOrO);
    // Write to gO
    #pragma unroll
    for (int m = 0; m < size<1>(rO); ++m) {
        const index_t idx = index_t(bidx) * kBlockM + get<0>(tOcOaccum(0, m, 0));
        if (idx < lse_size) {
            const int batch_idx = idx / (params.h * params.seqlen);
            const int head_idx = (idx - index_t(batch_idx) * (params.h * params.seqlen)) / params.seqlen;
            // The index to the rows of Q
            const int row = idx - index_t(batch_idx) * (params.h * params.seqlen) - index_t(head_idx) * params.seqlen;
            auto o_ptr = reinterpret_cast<Element *>(params.o_ptr) + batch_idx * params.o_batch_stride
                + head_idx * params.o_head_stride + row * params.o_row_stride;
            #pragma unroll
            for (int k = 0; k < size<2>(rO); ++k) {
                if (Is_even_K || tOpOaccum(k)) {
                    const int col = get<1>(tOcOaccum(0, m, k));
                    Tensor gO = make_tensor(make_gmem_ptr(o_ptr + col),
                                            Shape<Int<decltype(size<0>(rO))::value>>{}, Stride<_1>{});
                    copy(rO(_, m, k), gO);
                }
            }
        }
    }
}

} // namespace FLASH_NAMESPACE
