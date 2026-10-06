/******************************************************************************
 * Copyright (c) 2026, fast-diffusion-attention authors.
 *
 * Geometry of the diffusion attention mask.
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
 * Inside the kernels, the two halves of a sample are tiled separately: a query (or key) tile never mixes noisy
 * and clean rows. Accumulators (dQaccum, LSE_log2, dPsum, dKaccum, dVaccum) use a padded layout where every half
 * of every sample starts at a multiple of the tile size, like the varlen layout of FlashAttention-3 with each half
 * being a sequence.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>

#include "cutlass/fast_math.h"  // For cutlass::FastDivmod

#include "utils.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Sequence information of one sample.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Varlen>
struct DiffusionSeqlenInfo {

    int const offset;  // Row of the first token of the sample in the tensors (0 if not varlen: use the batch index)
    int const start;   // Index of the first token of the sample among all the tokens of the batch
    int const n;       // Length of each half

    CUTLASS_DEVICE
    DiffusionSeqlenInfo(int const bidb, int const seqlen_static, int const* const cu_seqlens)
        : offset(!Varlen ? 0 : cu_seqlens[bidb])
        , start(!Varlen ? bidb * seqlen_static : cu_seqlens[bidb])
        , n(!Varlen ? seqlen_static / 2 : (cu_seqlens[bidb + 1] - cu_seqlens[bidb]) / 2)
    {
    }

    // Row of the first token of half r (0: noisy, 1: clean) in the tensors.
    CUTLASS_DEVICE int offset_half(int const r) const { return offset + (r ? n : 0); }

    // Row of the first token of half r in the padded accumulators: every half starts at a multiple of kBlock and
    // has room for a full last tile. The accumulators hold round_up(total + 2 * b * kBlock, kBlock) rows.
    template <int kBlock>
    CUTLASS_DEVICE int offset_padded(int const bidb, int const r) const {
        int const v = 2 * bidb + r;
        return (start + (r ? n : 0) + v * kBlock) / kBlock * kBlock;
    }

};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Visibility intervals.
////////////////////////////////////////////////////////////////////////////////////////////////////

struct DiffusionGeometry {

    cutlass::FastDivmod bs_divmod;  // divisor = block size
    int n;                          // length of each half

    CUTLASS_DEVICE
    DiffusionGeometry(cutlass::FastDivmod const& bs_divmod, int const n) : bs_divmod(bs_divmod), n(n) {}

    CUTLASS_DEVICE int bs() const { return bs_divmod.divisor; }
    // Start of the block containing position x of a half.
    CUTLASS_DEVICE int block_start(int const x) const { return bs_divmod.div(x) * bs_divmod.divisor; }
    // End (exclusive) of the block containing position x of a half.
    CUTLASS_DEVICE int block_end(int const x) const { return cute::min(block_start(x) + bs_divmod.divisor, n); }

    // Keys of region k_noisy visible from query qi (< n) of half q_noisy: [lo, hi).
    // Never called with (q clean, k noisy).
    CUTLASS_DEVICE void key_interval(bool const q_noisy, bool const k_noisy, int const qi, int &lo, int &hi) const {
        int const blk = block_start(qi);
        if (!q_noisy) {         // clean -> clean: block-causal
            lo = 0;
            hi = cute::min(blk + bs(), n);
        } else if (!k_noisy) {  // noisy -> clean: strictly previous blocks
            lo = 0;
            hi = blk;
        } else {                // noisy -> noisy: same block
            lo = blk;
            hi = cute::min(blk + bs(), n);
        }
    }

    // Queries of half q_noisy that see key kj of region k_noisy: [lo, hi). Empty if kj >= n.
    // Never called with (q clean, k noisy).
    CUTLASS_DEVICE void query_interval(bool const q_noisy, bool const k_noisy, int const kj, int &lo, int &hi) const {
        int const blk = block_start(kj);
        if (!k_noisy) {
            lo = !q_noisy ? blk : blk + bs();  // clean -> clean: same and later blocks; noisy -> clean: later blocks
            hi = n;
        } else {                               // noisy -> noisy: same block
            lo = blk;
            hi = cute::min(blk + bs(), n);
        }
        if (kj >= n) { hi = 0; }
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
// The mainloop iterates v downwards: the key tiles v >= n_block_mask_min need masking.
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

    CUTLASS_DEVICE int n_block_max() const { return n_clean + (noisy_hi - noisy_lo); }
    CUTLASS_DEVICE bool is_noisy(int const v) const {
        int const n_vis = vis_hi - vis_lo;
        return v < n_block_mask_min ? v >= clean_mask_lo : v >= n_clean + n_vis;
    }
    // Index of key tile v inside its half.
    CUTLASS_DEVICE int tile_idx(int const v) const {
        int const n_vis = vis_hi - vis_lo;
        if (v < clean_mask_lo) { return v; }                         // clean, visible
        if (v < n_block_mask_min) { return vis_lo + (v - clean_mask_lo); }  // noisy, visible
        if (v < n_clean + n_vis) { return v - n_vis; }               // clean, masked
        int const w = v - (n_clean + n_vis);                         // noisy, masked
        int const n_low = (n_vis > 0 ? vis_lo : noisy_hi) - noisy_lo;
        return w < n_low ? noisy_lo + w : vis_hi + (w - n_low);
    }

    CUTLASS_DEVICE
    DiffusionFwdTiles(DiffusionGeometry const& g, bool const q_noisy, int const m_block) {
        int const i0 = m_block * kBlockM;
        int const i_last = cute::min(i0 + kBlockM, g.n) - 1;
        if (i_last < i0) { return; }  // empty tile
        int const blk_first = g.block_start(i0), blk_last = g.block_start(i_last);
        if (!q_noisy) {
            // Clean keys [0, block_end(i)).
            n_clean = cute::ceil_div(cute::min(blk_last + g.bs(), g.n), kBlockN);
            clean_mask_lo = cute::min(cute::min(blk_first + g.bs(), g.n) / kBlockN, n_clean);
        } else {
            // Clean keys [0, block_start(i)).
            n_clean = cute::ceil_div(blk_last, kBlockN);
            clean_mask_lo = cute::min(blk_first / kBlockN, n_clean);
            // Noisy keys [block_start(i), block_end(i)). Visible from every query: [blk_last, block_end(i0)).
            noisy_lo = blk_first / kBlockN;
            noisy_hi = cute::ceil_div(cute::min(blk_last + g.bs(), g.n), kBlockN);
            vis_lo = cute::max(cute::ceil_div(blk_last, kBlockN), noisy_lo);
            vis_hi = cute::min(cute::min(blk_first + g.bs(), g.n) / kBlockN, noisy_hi);
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
// The query tiles are enumerated by a virtual index u in [0, m_block_max()), iterated upwards, masked tiles first:
//   [list 0 masked: [m0_lo, vis0_lo) and [vis0_hi, m0_hi) | list 1 masked | list 0 visible | list 1 visible]
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kBlockM, int kBlockN>
struct DiffusionBwdTiles {

    int m0_lo = 0, m0_hi = 0;
    int vis0_lo = 0, vis0_hi = 0;  // vis0_lo == vis0_hi if none
    int n_m1 = 0;
    int m1_lo = 0;
    int n_m1_mask = 0;

    CUTLASS_DEVICE int n_m0_mask() const { return (m0_hi - m0_lo) - (vis0_hi - vis0_lo); }
    CUTLASS_DEVICE int m_block_max() const { return (m0_hi - m0_lo) + n_m1; }
    // Query tiles u < n_mask() need masking.
    CUTLASS_DEVICE int n_mask() const { return n_m0_mask() + n_m1_mask; }
    // Is query tile u in list 1?
    CUTLASS_DEVICE bool in_list1(int const u) const {
        return u < n_mask() ? u >= n_m0_mask() : u >= n_mask() + (vis0_hi - vis0_lo);
    }
    // Half of query tile u (k_noisy: half of the key tile).
    CUTLASS_DEVICE bool q_is_noisy(int const u, bool const k_noisy) const { return k_noisy || in_list1(u); }
    // Index of query tile u inside its half.
    CUTLASS_DEVICE int tile_idx(int const u) const {
        int const n_vis0 = vis0_hi - vis0_lo;
        int const n_low0 = (n_vis0 > 0 ? vis0_lo : m0_hi) - m0_lo;
        int const n_mask0 = n_m0_mask();
        if (u < n_low0) { return m0_lo + u; }
        if (u < n_mask0) { return vis0_hi + (u - n_low0); }
        if (u < n_mask0 + n_m1_mask) { return m1_lo + (u - n_mask0); }
        int const v = u - (n_mask0 + n_m1_mask);
        return v < n_vis0 ? vis0_lo + v : m1_lo + n_m1_mask + (v - n_vis0);
    }

    CUTLASS_DEVICE
    DiffusionBwdTiles(DiffusionGeometry const& g, bool const k_noisy, int const n_block) {
        int const j0 = n_block * kBlockN;
        if (j0 >= g.n) { return; }  // empty tile
        bool const partial = j0 + kBlockN > g.n;  // keys beyond the end of the half must be masked everywhere
        int const j_last = cute::min(j0 + kBlockN, g.n) - 1;
        int const num_m_blocks = cute::ceil_div(g.n, kBlockM);
        int const blk_first = g.block_start(j0), blk_last = g.block_start(j_last);
        if (!k_noisy) {
            // Clean queries [block_start(j), n), visible from every key: [blk_last, n)
            m0_lo = blk_first / kBlockM;
            m0_hi = num_m_blocks;
            vis0_lo = cute::min(cute::ceil_div(blk_last, kBlockM), m0_hi);
            vis0_hi = m0_hi;
            // Noisy queries [block_start(j) + bs, n).
            if (blk_first + g.bs() < g.n) {
                m1_lo = (blk_first + g.bs()) / kBlockM;
                n_m1 = num_m_blocks - m1_lo;
                n_m1_mask = !partial ? cute::ceil_div(cute::min(blk_last + g.bs(), g.n), kBlockM) - m1_lo : n_m1;
            }
        } else {
            // Noisy queries [block_start(j), block_end(j)). Visible from every key if the key tile is in a single
            // block: the query tiles inside that block.
            m0_lo = blk_first / kBlockM;
            m0_hi = cute::ceil_div(cute::min(blk_last + g.bs(), g.n), kBlockM);
            if (blk_first == blk_last) {
                vis0_lo = cute::max(cute::ceil_div(blk_first, kBlockM), m0_lo);
                vis0_hi = cute::min(cute::min(blk_first + g.bs(), g.n) / kBlockM, m0_hi);
            }
        }
        if (partial || vis0_hi <= vis0_lo) { vis0_lo = vis0_hi = 0; }
        n_m1_mask = cute::max(cute::min(n_m1_mask, n_m1), 0);
    }

};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Register-level mask application.
//
// tSrS is the accumulator of S = Q K^T (S^T = K Q^T if SwapAB) for query tile m_block of half q_noisy and key
// tile n_block of half k_noisy (tile indices inside their half).
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kBlockM, int kBlockN, typename TiledMma, bool SwapAB=false>
struct DiffusionMask {

    int const thread_idx;
    DiffusionGeometry const g;

    CUTLASS_DEVICE
    DiffusionMask(int const thread_idx, DiffusionGeometry const& g) : thread_idx(thread_idx), g(g) {}

    template <typename Engine, typename Layout>
    CUTLASS_DEVICE
    void apply(Tensor<Engine, Layout> &tSrS, bool const q_noisy, int const m_block,
               bool const k_noisy, int const n_block) const {
        static_assert(Layout::rank == 3, "Only support 3D Tensor");
        auto thread_mma = TiledMma{}.get_thread_slice(thread_idx);
        auto thread0_mma = TiledMma{}.get_thread_slice(_0{});

        static constexpr int Row = !SwapAB ? 0 : 1, Col = !SwapAB ? 1 : 0;

        Tensor cS = cute::make_identity_tensor(Shape<Int<!SwapAB ? kBlockM : kBlockN>, Int<!SwapAB ? kBlockN : kBlockM>>{});
        Tensor tScS = thread_mma.partition_C(cS);
        // In the (row, col) view, rows index queries and cols index keys.
        Tensor tSrS_rowcol = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tSrS.layout()));
        Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tScS.layout()));
        Tensor t0ScS = thread0_mma.partition_C(cS);
        Tensor t0ScS_rowcol = make_tensor(t0ScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(t0ScS.layout()));

        if constexpr (!SwapAB) {
            // Few rows per thread: compute the key interval of each row and compare it with the column indices
            // of thread 0 (known at compile time), shifted by the column offset of this thread.
            int const col_base = n_block * kBlockN + int(get<Col>(tScS_rowcol(_0{}, _0{})));
            #pragma unroll
            for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                int const qi = int(get<Row>(tScS_rowcol(m, _0{}))) + m_block * kBlockM;
                int lo, hi;
                g.key_interval(q_noisy, k_noisy, qi, lo, hi);
                lo -= col_base;
                hi -= col_base;
                #pragma unroll
                for (int ni = 0; ni < size<1>(tSrS_rowcol); ++ni) {
                    int const col = int(get<Col>(t0ScS_rowcol(_0{}, ni)));
                    if (col < lo || col >= hi) { tSrS_rowcol(m, ni) = -INFINITY; }
                }
            }
        } else {
            // Few columns per thread: compute the query interval of each column.
            int const row_base = m_block * kBlockM + int(get<Row>(tScS_rowcol(_0{}, _0{})));
            #pragma unroll
            for (int ni = 0; ni < size<1>(tSrS_rowcol); ++ni) {
                int const kj = int(get<Col>(tScS_rowcol(_0{}, ni))) + n_block * kBlockN;
                int lo, hi;
                g.query_interval(q_noisy, k_noisy, kj, lo, hi);
                lo -= row_base;
                hi -= row_base;
                #pragma unroll
                for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                    int const row = int(get<Row>(t0ScS_rowcol(m, _0{})));
                    if (row < lo || row >= hi) { tSrS_rowcol(m, ni) = -INFINITY; }
                }
            }
        }
    }

};

} // namespace flash
