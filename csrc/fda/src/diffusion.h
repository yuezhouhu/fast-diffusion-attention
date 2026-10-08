/******************************************************************************
 * Copyright (c) 2026, fast-diffusion-attention authors.
 *
 * Geometry of the diffusion attention mask (FlashAttention-2 kernels).
 *
 * Every sample has the layout  [ noisy (n tokens) | clean (n tokens) ]  (length 2n). Positions of each half are
 * split into blocks of `block_size` (bs) tokens, counted from the start of the half (the last block may be partial).
 * With i, j two positions of the noisy half and c, c' two positions of the clean half (counted from the start of
 * their half), block(x) = x / bs:
 *
 *     noisy i -> noisy j  :  block(j) == block(i)            (bidirectional inside the block)
 *     noisy i -> clean c' :  block(c') <  block(i)           (all the previous clean blocks)
 *     clean c -> clean c' :  block(c') <= block(c)           (block-causal)
 *     clean c -> noisy j  :  never
 *
 * This is the SDAR / BD3-LM block diffusion mask (`block_diff_mask(..., use_regular_causal=False)` of the I-DLM
 * training code). For every query, the visible keys of each half form an interval, and so do the queries that
 * see a given key. The kernels only visit the tiles that intersect these intervals and apply the mask only on the
 * tiles that cross an interval boundary.
 *
 * The two halves of a sample are tiled separately: a query (or key) tile never mixes noisy and clean rows. The FP32
 * accumulators of the backward pass (dQaccum, LSE, dPsum) use a padded row layout where every half of every sample is
 * followed by kAccumPad rows of padding (see DiffusionBlockInfo::padded_row), like the varlen layout of
 * FlashAttention-2 with each half being a sequence: the rows of a partial tile beyond the end of a half land in the
 * padding.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>

#include "cutlass/fast_math.h"  // For cutlass::FastDivmod

#include "namespace_config.h"
#include "flash.h"
#include "kernel_traits.h"  // Before utils.h (using namespace cute), as in FlashAttention-2
#include "utils.h"

namespace FLASH_NAMESPACE {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Sequence information of one sample (BlockInfo of FlashAttention-2).
////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool Varlen=true>
struct DiffusionBlockInfo {

    template<typename Params>
    __device__ DiffusionBlockInfo(const Params &params, const int bidb)
        : sum_s(!Varlen || params.cu_seqlens == nullptr ? -1 : params.cu_seqlens[bidb])
        , n(!Varlen || params.cu_seqlens == nullptr ? params.seqlen / 2 : (params.cu_seqlens[bidb + 1] - sum_s) / 2)
        , start(sum_s == -1 ? bidb * params.seqlen : sum_s)
        {
        }

    // Offset of the first row of the sample in a (batch, seqlen, ...) or (total, ...) tensor.
    template <typename index_t>
    __forceinline__ __device__ index_t offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        return sum_s == -1 ? bidb * batch_stride : uint32_t(sum_s) * row_stride;
    }

    // Row of the first token of half r (0: noisy, 1: clean) in the sample.
    __forceinline__ __device__ int half_row(const int r) const { return r ? n : 0; }

    // Row of the first token of half r in the padded accumulators: every half is followed by kAccumPad rows.
    // The accumulators hold total + 2 * kAccumPad * b rows.
    __forceinline__ __device__ int padded_row(const int bidb, const int r) const {
        return start + (r ? n : 0) + (2 * bidb + r) * kAccumPad;
    }

    const int sum_s;  // -1 if not varlen
    const int n;      // length of each half
    const int start;  // index of the first token of the sample among all the tokens of the batch
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Visibility intervals.
////////////////////////////////////////////////////////////////////////////////////////////////////

struct DiffusionGeometry {

    cutlass::FastDivmod bs_divmod;  // divisor = block size
    int n;                          // length of each half

    __forceinline__ __device__
    DiffusionGeometry(cutlass::FastDivmod const& bs_divmod, int const n) : bs_divmod(bs_divmod), n(n) {}

    __forceinline__ __device__ int bs() const { return bs_divmod.divisor; }
    // Start of the block containing position x of a half.
    __forceinline__ __device__ int block_start(int const x) const { return bs_divmod.div(x) * bs_divmod.divisor; }

    // Keys of region k_noisy visible from query qi (< n) of half q_noisy: [lo, hi).
    // Never called with (q clean, k noisy).
    __forceinline__ __device__ void key_interval(bool const q_noisy, bool const k_noisy, int const qi, int &lo, int &hi) const {
        int const blk = block_start(qi);
        if (!q_noisy) {         // clean -> clean: block-causal
            lo = 0;
            hi = min(blk + bs(), n);
        } else if (!k_noisy) {  // noisy -> clean: strictly previous blocks
            lo = 0;
            hi = blk;
        } else {                // noisy -> noisy: same block
            lo = blk;
            hi = min(blk + bs(), n);
        }
    }

};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Forward: the key tiles visited by a query tile.
//
// A query tile visits the clean key tiles [0, n_clean) and, for a noisy query tile, the noisy key tiles
// [noisy_lo, noisy_hi). They are enumerated by a virtual index v in [0, n_block_max()), fully visible key tiles
// (visible from every query of the tile) first, then the key tiles that need masking:
//   [clean visible [0, clean_mask_lo) | noisy visible [vis_lo, vis_hi) |
//    clean masked [clean_mask_lo, n_clean) | noisy masked [noisy_lo, vis_lo) and [vis_hi, noisy_hi)]
// The mainloop iterates v downwards (like FlashAttention-2 iterates n_block): the key tiles v >= n_block_mask_min
// need masking and come first. A key tile extending beyond the end of its half always needs masking.
// A noisy key tile is fully visible only if the query tile lies in a single block that contains the key tile
// (block sizes >= kBlockN); for smaller block sizes the noisy key tiles always need masking.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kBlockM, int kBlockN>
struct DiffusionFwdTiles {

    int n_clean = 0;
    int clean_mask_lo = 0;
    int noisy_lo = 0, noisy_hi = 0;
    int vis_lo = 0, vis_hi = 0;  // fully visible noisy key tiles, vis_lo == vis_hi if none
    int n_block_mask_min = 0;

    __forceinline__ __device__ int n_block_max() const { return n_clean + (noisy_hi - noisy_lo); }
    __forceinline__ __device__ bool is_noisy(int const v) const {
        int const n_vis = vis_hi - vis_lo;
        return v < n_block_mask_min ? v >= clean_mask_lo : v >= n_clean + n_vis;
    }
    // Index of key tile v inside its half.
    __forceinline__ __device__ int tile_idx(int const v) const {
        int const n_vis = vis_hi - vis_lo;
        if (v < clean_mask_lo) { return v; }                                // clean, visible
        if (v < n_block_mask_min) { return vis_lo + (v - clean_mask_lo); }  // noisy, visible
        if (v < n_clean + n_vis) { return v - n_vis; }                      // clean, masked
        int const w = v - (n_clean + n_vis);                                // noisy, masked
        int const n_low = (n_vis > 0 ? vis_lo : noisy_hi) - noisy_lo;
        return w < n_low ? noisy_lo + w : vis_hi + (w - n_low);
    }
    // Row of the first key of tile v in the sample.
    __forceinline__ __device__ int key_row(int const v, int const n) const {
        return (is_noisy(v) ? 0 : n) + tile_idx(v) * kBlockN;
    }

    __forceinline__ __device__
    DiffusionFwdTiles(DiffusionGeometry const& g, bool const q_noisy, int const m_block) {
        int const i0 = m_block * kBlockM;
        int const i_last = min(i0 + kBlockM, g.n) - 1;
        if (i_last < i0) { return; }  // empty tile
        int const blk_first = g.block_start(i0), blk_last = g.block_start(i_last);
        if (!q_noisy) {
            // Clean keys [0, block_end(i)).
            n_clean = cute::ceil_div(min(blk_last + g.bs(), g.n), kBlockN);
            clean_mask_lo = min(min(blk_first + g.bs(), g.n) / kBlockN, n_clean);
        } else {
            // Clean keys [0, block_start(i)).
            n_clean = cute::ceil_div(blk_last, kBlockN);
            clean_mask_lo = min(blk_first / kBlockN, n_clean);
            // Noisy keys [block_start(i), block_end(i)). Visible from every query: [blk_last, block_end(i0)).
            noisy_lo = blk_first / kBlockN;
            noisy_hi = cute::ceil_div(min(blk_last + g.bs(), g.n), kBlockN);
            vis_lo = max(cute::ceil_div(blk_last, kBlockN), noisy_lo);
            vis_hi = min(min(blk_first + g.bs(), g.n) / kBlockN, noisy_hi);
            if (vis_hi <= vis_lo) { vis_lo = vis_hi = 0; }
        }
        n_block_mask_min = clean_mask_lo + (vis_hi - vis_lo);
    }

};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Backward: the query tiles visited by a key tile.
//
// The query tiles of a key tile form two lists: list 0 = the query tiles [m0_lo, m0_hi) of the same half as the key
// tile, list 1 = the noisy query tiles [m1_lo, m1_lo + n_m1) (clean key tiles only). In list 0, the query tiles
// [vis0_lo, vis0_hi) see the whole key tile, the others need masking; in list 1 the first n_m1_mask tiles need
// masking. Everything needs masking if the key tile extends beyond the end of the half.
// The query tiles are enumerated by a virtual index u in [0, m_block_max()), masked tiles first:
//   [list 0 masked: [m0_lo, vis0_lo) and [vis0_hi, m0_hi) | list 1 masked | list 0 visible | list 1 visible]
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kBlockM, int kBlockN>
struct DiffusionBwdTiles {

    int m0_lo = 0, m0_hi = 0;
    int vis0_lo = 0, vis0_hi = 0;  // vis0_lo == vis0_hi if none
    int n_m1 = 0;
    int m1_lo = 0;
    int n_m1_mask = 0;

    __forceinline__ __device__ int n_m0_mask() const { return (m0_hi - m0_lo) - (vis0_hi - vis0_lo); }
    __forceinline__ __device__ int m_block_max() const { return (m0_hi - m0_lo) + n_m1; }
    // Query tiles u < n_mask() need masking.
    __forceinline__ __device__ int n_mask() const { return n_m0_mask() + n_m1_mask; }
    // Is query tile u in list 1?
    __forceinline__ __device__ bool in_list1(int const u) const {
        return u < n_mask() ? u >= n_m0_mask() : u >= n_mask() + (vis0_hi - vis0_lo);
    }
    // Half of query tile u (k_noisy: half of the key tile).
    __forceinline__ __device__ bool q_is_noisy(int const u, bool const k_noisy) const { return k_noisy || in_list1(u); }
    // Index of query tile u inside its half.
    __forceinline__ __device__ int tile_idx(int const u) const {
        int const n_vis0 = vis0_hi - vis0_lo;
        int const n_low0 = (n_vis0 > 0 ? vis0_lo : m0_hi) - m0_lo;
        int const n_mask0 = n_m0_mask();
        if (u < n_low0) { return m0_lo + u; }
        if (u < n_mask0) { return vis0_hi + (u - n_low0); }
        if (u < n_mask0 + n_m1_mask) { return m1_lo + (u - n_mask0); }
        int const v = u - (n_mask0 + n_m1_mask);
        return v < n_vis0 ? vis0_lo + v : m1_lo + n_m1_mask + (v - n_vis0);
    }

    __forceinline__ __device__
    DiffusionBwdTiles(DiffusionGeometry const& g, bool const k_noisy, int const n_block) {
        int const j0 = n_block * kBlockN;
        if (j0 >= g.n) { return; }  // empty tile
        bool const partial = j0 + kBlockN > g.n;  // keys beyond the end of the half must be masked everywhere
        int const j_last = min(j0 + kBlockN, g.n) - 1;
        int const num_m_blocks = cute::ceil_div(g.n, kBlockM);
        int const blk_first = g.block_start(j0), blk_last = g.block_start(j_last);
        if (!k_noisy) {
            // Clean queries [block_start(j), n), visible from every key: [blk_last, n)
            m0_lo = blk_first / kBlockM;
            m0_hi = num_m_blocks;
            vis0_lo = min(cute::ceil_div(blk_last, kBlockM), m0_hi);
            vis0_hi = m0_hi;
            // Noisy queries [block_start(j) + bs, n).
            if (blk_first + g.bs() < g.n) {
                m1_lo = (blk_first + g.bs()) / kBlockM;
                n_m1 = num_m_blocks - m1_lo;
                n_m1_mask = !partial ? cute::ceil_div(min(blk_last + g.bs(), g.n), kBlockM) - m1_lo : n_m1;
            }
        } else {
            // Noisy queries [block_start(j), block_end(j)). Visible from every key if the key tile is in a single
            // block: the query tiles inside that block.
            m0_lo = blk_first / kBlockM;
            m0_hi = cute::ceil_div(min(blk_last + g.bs(), g.n), kBlockM);
            if (blk_first == blk_last) {
                vis0_lo = max(cute::ceil_div(blk_first, kBlockM), m0_lo);
                vis0_hi = min(min(blk_first + g.bs(), g.n) / kBlockM, m0_hi);
            }
        }
        if (partial || vis0_hi <= vis0_lo) { vis0_lo = vis0_hi = 0; }
        n_m1_mask = max(min(n_m1_mask, n_m1), 0);
    }

};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Register-level mask, in the accumulator layout of the FlashAttention-2 kernels (apply_mask_local of mask.h).
//
// tensor_ is the (MMA=4, MMA_M, MMA_N) accumulator of S = Q K^T of a query tile of half q_noisy and a key tile of
// half k_noisy. row_idx_offset: row (inside the query half) of the first row of this thread; warp_row_stride: row
// distance between the MMA_M row groups of a thread; col_idx_offset_: column (inside the key half) of the first
// column of this warp. Masked scores are set to -inf, including the keys beyond the end of the half.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine, typename Layout>
__forceinline__ __device__ void apply_diffusion_mask(Tensor<Engine, Layout> &tensor_, DiffusionGeometry const& g,
                                                     const bool q_noisy, const bool k_noisy,
                                                     const int row_idx_offset, const int warp_row_stride,
                                                     const int col_idx_offset_) {
    static_assert(Layout::rank == 3, "Only support 3D Tensor");
    static_assert(decltype(size<0>(tensor_))::value == 4, "First dimension must be 4");
    // Reshape tensor_ from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
    Tensor tensor = make_tensor(tensor_.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(tensor_.layout()));
    const int lane_id = threadIdx.x % 32;
    const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
    #pragma unroll
    for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
        const int row_idx_base = row_idx_offset + mi * warp_row_stride;
        #pragma unroll
        for (int i = 0; i < size<0, 0>(tensor); ++i) {
            const int row_idx = row_idx_base + i * 8;
            int lo, hi;
            g.key_interval(q_noisy, k_noisy, row_idx, lo, hi);
            // Compare with the column indices relative to this thread (known at compile time)
            lo -= col_idx_offset;
            hi -= col_idx_offset;
            #pragma unroll
            for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                #pragma unroll
                for (int j = 0; j < size<1, 0>(tensor); ++j) {
                    const int col = nj * 8 + j;
                    if (col < lo || col >= hi) {
                        tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                    }
                }
            }
        }
    }
}

}  // namespace FLASH_NAMESPACE
