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
// The key tiles of a query tile are enumerated by a virtual index v in [0, n_block_max()):
//   v in [0, n_clean)                    -> clean key tile v
//   v in [n_clean, n_clean + n_noisy)    -> noisy key tile noisy_lo + (v - n_clean)   (noisy query tiles only)
// The mainloop iterates v downwards. The key tiles v >= n_block_mask_min need masking, the ones below are fully
// visible from every query of the tile.
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kBlockM, int kBlockN>
struct DiffusionFwdTiles {

    int n_clean = 0;
    int n_noisy = 0;
    int noisy_lo = 0;
    int n_block_mask_min = 0;

    CUTLASS_DEVICE int n_block_max() const { return n_clean + n_noisy; }
    CUTLASS_DEVICE bool is_noisy(int const v) const { return v >= n_clean; }
    // Index of key tile v inside its half.
    CUTLASS_DEVICE int tile_idx(int const v) const { return v < n_clean ? v : noisy_lo + (v - n_clean); }

    CUTLASS_DEVICE
    DiffusionFwdTiles(DiffusionGeometry const& g, bool const q_noisy, int const m_block) {
        int const i0 = m_block * kBlockM;
        int const i_last = cute::min(i0 + kBlockM, g.n) - 1;
        if (i_last < i0) { return; }  // empty tile
        int const blk_first = g.block_start(i0), blk_last = g.block_start(i_last);
        if (!q_noisy) {
            // Clean keys [0, block_end(i)).
            n_clean = cute::ceil_div(cute::min(blk_last + g.bs(), g.n), kBlockN);
            n_block_mask_min = cute::min(blk_first + g.bs(), g.n) / kBlockN;
        } else {
            // Clean keys [0, block_start(i)).
            n_clean = cute::ceil_div(blk_last, kBlockN);
            n_block_mask_min = blk_first / kBlockN;
            // Noisy keys [block_start(i), block_end(i)).
            noisy_lo = blk_first / kBlockN;
            n_noisy = cute::ceil_div(cute::min(blk_last + g.bs(), g.n), kBlockN) - noisy_lo;
        }
    }

};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Backward: the query tiles visited by a key tile.
//
// The query tiles of a key tile form two lists: list 0 = the query tiles [m0_lo, m0_lo + n_m0) of the same half as
// the key tile, list 1 = the noisy query tiles [m1_lo, m1_lo + n_m1) (clean key tiles only). The first n_m0_mask
// (resp. n_m1_mask) tiles of each list need masking (all of them if the key tile extends beyond the end of the half).
// They are enumerated by a virtual index u in [0, m_block_max()), iterated upwards, masked tiles first:
//   [list 0 masked | list 1 masked | list 0 fully visible | list 1 fully visible]
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kBlockM, int kBlockN>
struct DiffusionBwdTiles {

    int n_m0 = 0;
    int m0_lo = 0;
    int n_m0_mask = 0;
    int n_m1 = 0;
    int m1_lo = 0;
    int n_m1_mask = 0;

    CUTLASS_DEVICE int m_block_max() const { return n_m0 + n_m1; }
    // Query tiles u < n_mask() need masking.
    CUTLASS_DEVICE int n_mask() const { return n_m0_mask + n_m1_mask; }
    // Is query tile u in list 1?
    CUTLASS_DEVICE bool in_list1(int const u) const {
        return u < n_mask() ? u >= n_m0_mask : u >= n_mask() + (n_m0 - n_m0_mask);
    }
    // Half of query tile u (k_noisy: half of the key tile).
    CUTLASS_DEVICE bool q_is_noisy(int const u, bool const k_noisy) const { return k_noisy || in_list1(u); }
    // Index of query tile u inside its half.
    CUTLASS_DEVICE int tile_idx(int const u) const {
        if (u < n_m0_mask) { return m0_lo + u; }
        if (u < n_mask()) { return m1_lo + (u - n_m0_mask); }
        int const v = u - n_mask();
        return v < n_m0 - n_m0_mask ? m0_lo + n_m0_mask + v : m1_lo + n_m1_mask + (v - (n_m0 - n_m0_mask));
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
            // Clean queries [block_start(j), n).
            m0_lo = blk_first / kBlockM;
            n_m0 = num_m_blocks - m0_lo;
            n_m0_mask = !partial ? cute::ceil_div(blk_last, kBlockM) - m0_lo : n_m0;
            // Noisy queries [block_start(j) + bs, n).
            if (blk_first + g.bs() < g.n) {
                m1_lo = (blk_first + g.bs()) / kBlockM;
                n_m1 = num_m_blocks - m1_lo;
                n_m1_mask = !partial ? cute::ceil_div(cute::min(blk_last + g.bs(), g.n), kBlockM) - m1_lo : n_m1;
            }
        } else {
            // Noisy queries [block_start(j), block_end(j)).
            m0_lo = blk_first / kBlockM;
            n_m0 = cute::ceil_div(cute::min(blk_last + g.bs(), g.n), kBlockM) - m0_lo;
            n_m0_mask = n_m0;
        }
        n_m0_mask = cute::max(cute::min(n_m0_mask, n_m0), 0);
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
