/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 * Copyright (c) 2026, fast-diffusion-attention authors.
 ******************************************************************************/

#pragma once

#include <tuple>

// Forward tile sizes. Return {kBlockM, kBlockN, MmaPV_is_RS, IntraWGOverlap}.
// The diffusion mask has the structure of causal attention (a staircase of blocks), so we use the tile sizes of
// causal attention of FlashAttention-3.
constexpr std::tuple<int, int, bool, bool> tile_size_fwd_sm90(int headdim, int element_size=2, bool softcap=false) {
    if (element_size == 2) {
        if (headdim <= 64) {
            return {192, 128, true, true};
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
            return {128, 128, true, true};
        } else if (headdim <= 192) {
            return {128, 128, true, true};
        } else {
            return {128, 128, true, true};
        }
    }
}
