/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 * Copyright (c) 2026, fast-diffusion-attention authors.
 ******************************************************************************/

#pragma once

#include <tuple>

// Forward tile sizes. Return {kBlockM, kBlockN, MmaPV_is_RS, IntraWGOverlap}.
// The diffusion mask has the structure of causal attention (a staircase of blocks), so we start from the tile sizes
// of causal attention of FlashAttention-3. The query tiles of the noisy half also visit the noisy key tiles of their
// blocks, a fixed cost per query tile: for some configurations smaller tiles are faster when the halves are short
// (long_seq = false, see fwd_long_seq_threshold), and the larger tiles of FA3 when they are long.
constexpr std::tuple<int, int, bool, bool> tile_size_fwd_sm90(int headdim, int element_size=2, bool softcap=false,
                                                              bool long_seq=true) {
    using Tile = std::tuple<int, int, bool, bool>;
    if (element_size == 2) {
        if (headdim <= 64) {
            // 128 x 128 also enables the half-width noisy diagonal tiles
            return long_seq ? Tile{192, 128, true, true} : Tile{128, 128, true, true};
        } else if (headdim <= 96) {
            return {192, 128, false, true};
        } else if (headdim <= 128) {
            return {128, 128, true, true};
        } else if (headdim <= 192) {
            return {128, 112, true, true};  // 128 x 112 hits the limit of smem
        } else {
            return {128, 80, true, true};  // 128 x 80 hits the limit of smem
        }
    } else {
        if (headdim <= 64) {
            return {192, 128, true, true};
        } else if (headdim <= 96) {
            return {192, 128, true, true};
        } else if (headdim <= 128) {
            return long_seq ? Tile{128, 224, true, true} : Tile{128, 128, true, true};
        } else if (headdim <= 192) {
            return {128, 128, true, true};
        } else {
            return {128, 128, true, true};
        }
    }
}

// The long-sequence forward tiles are used when the (longest) half has more than this many tokens (measured on H800
// with batches of 16K tokens: crossover between n = 2048 and 4096 for hdim 64 FP16 / BF16, around 4096 for FP8).
constexpr int fwd_long_seq_threshold(int headdim, int element_size) {
    return element_size == 2 ? 2048 : 4096;
}
