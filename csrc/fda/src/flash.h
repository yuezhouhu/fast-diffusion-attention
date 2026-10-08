/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 * Copyright (c) 2026, fast-diffusion-attention authors.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"

#include <cuda.h>
#include <cstdint>
#include <vector>

#include "cutlass/fast_math.h"  // For cutlass::FastDivmod

namespace FLASH_NAMESPACE {
constexpr int TOTAL_DIM = 0;
constexpr int H_DIM = 1;
constexpr int D_DIM = 2;

// Rows of padding after every half of every sample in the padded accumulators of the backward pass (dQaccum, LSE,
// dPsum): at least the largest kBlockM of the backward kernels (see diffusion.h).
constexpr int kAccumPad = 128;

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Qkv_params {
    using index_t = int64_t;
    // The QKV matrices.
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;

    // The number of heads.
    int h, h_k;
    // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
    // different from nheads (query).
    int h_h_k_ratio; // precompute h / h_k,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_fwd_params : public Qkv_params {

    // The O matrix (output).
    void * __restrict__ o_ptr;

    // The stride between rows of O.
    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;

    // The pointer to the P matrix (S_dmask, only for testing dropout).
    void * __restrict__ p_ptr;

    // The pointer to the softmax log-sum-exp.
    void * __restrict__ softmax_lse_ptr;

    // The dimensions. Self-attention on samples [noisy (n) | clean (n)]: seqlen is the length 2n of the samples
    // (batched) or the maximum length of a sample (varlen).
    int b, seqlen, d, d_rounded, total_q;
    // Maximum length of a half (n), rounded up to a multiple of 128: row / column offset of the clean half in
    // S_dmask and in the coordinates of the dropout random numbers.
    int seqlen_half_rounded;

    // The diffusion block size.
    int block_size;
    cutlass::FastDivmod block_size_divmod;

    // The scaling factors for the kernel.
    float scale_softmax;
    float scale_softmax_log2;

    // array of length b+1 holding starting offset of each sample (varlen), nullptr otherwise.
    int * __restrict__ cu_seqlens;

    // The dropout probability (probability of keeping an activation).
    float p_dropout;
    uint8_t p_dropout_in_uint8_t;

    // Scale factor of 1 / (1 - p_dropout).
    float rp_dropout;
    float scale_softmax_rp_dropout;

    float softcap;

    // Random state, stored as an opaque buffer that will potentially hold at::PhiloxCudaState.
    // flash_api.cpp writes into this buffer via placement-new of an at::PhiloxCudaState and the forward kernel
    // reads it back via reinterpret_cast (as in FlashAttention-2).
    uint64_t philox_args[4];

    // Pointer to the RNG seed (idx 0) and offset (idx 1).
    uint64_t * rng_state;

    bool is_bf16;

    bool unpadded_lse;  // For varlen paths: LSE is in [nheads, total] format instead of [b, nheads, seqlen].

    // Split-KV forward (FlashAttention-2's set_params_splitkv): the number of splits of the key tiles of a query tile
    // (1: the standard kernel), the FP32 partial outputs (num_splits, b, h, seqlen, d_rounded) and log-sum-exps
    // (num_splits, b, h, seqlen), combined by flash_fwd_splitkv_combine_kernel.
    int num_splits;
    // (batch, head) pairs per group of the thread block order (tile_of_block in diffusion.h)
    int sched_heads;
    void * __restrict__ oaccum_ptr;
    void * __restrict__ softmax_lseaccum_ptr;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_bwd_params : public Flash_fwd_params {

    // The dO and dQKV matrices.
    void *__restrict__ do_ptr;
    void *__restrict__ dq_ptr;
    void *__restrict__ dk_ptr;
    void *__restrict__ dv_ptr;

    // To accumulate dQ, (total_padded, h, d_rounded) in FP32 (padded row layout, see diffusion.h).
    void *__restrict__ dq_accum_ptr;

    // The stride between rows of the dO, dQ, dK and dV matrices.
    index_t do_batch_stride;
    index_t do_row_stride;
    index_t do_head_stride;
    index_t dq_batch_stride;
    index_t dk_batch_stride;
    index_t dv_batch_stride;
    index_t dq_row_stride;
    index_t dk_row_stride;
    index_t dv_row_stride;
    index_t dq_head_stride;
    index_t dk_head_stride;
    index_t dv_head_stride;

    // The pointer to the softmax d sum, (h, total_padded) in FP32 (padded row layout).
    void *__restrict__ dsoftmax_sum;
    // The softmax log-sum-exp in the padded row layout, (h, total_padded) in FP32 (+inf in the padding).
    void *__restrict__ softmax_lse_padded_ptr;
    // Number of rows of the padded accumulators: total + 2 * kAccumPad * b.
    int total_padded;

    bool deterministic;
    index_t dq_accum_split_stride;

    // Number of query heads (dividing h / h_k) processed by a thread block of the backward kernel, which sums their
    // dK / dV: dK / dV point to a (..., h / bwd_heads_per_cta, d) buffer, the final dK / dV if bwd_heads_per_cta =
    // h / h_k.
    int bwd_heads_per_cta;
};


////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int Headdim> void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream);
template<typename T, int Headdim> void run_mha_fwd_splitkv_dispatch(Flash_fwd_params &params, cudaStream_t stream);

template<typename T, int Headdim> void run_mha_bwd_(Flash_bwd_params &params, cudaStream_t stream);

}  // namespace FLASH_NAMESPACE
