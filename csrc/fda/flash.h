/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 * Copyright (c) 2026, fast-diffusion-attention authors.
 *
 * Kernel parameters of the diffusion attention kernels (adapted from FlashAttention-3).
 ******************************************************************************/

#pragma once

#include <cuda.h>
#include <vector>

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
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_fwd_params : public Qkv_params {
    using index_t = int64_t;

    // The O matrix (output).
    void * __restrict__ o_ptr;

    // The stride between rows of O.
    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;

    // The pointer to the softmax log-sum-exp.
    void * __restrict__ softmax_lse_ptr;

    // For FP8 scaling
    float * __restrict__ q_descale_ptr;
    float * __restrict__ k_descale_ptr;
    float * __restrict__ v_descale_ptr;
    index_t q_descale_batch_stride;
    index_t q_descale_head_stride;
    index_t k_descale_batch_stride;
    index_t k_descale_head_stride;
    index_t v_descale_batch_stride;
    index_t v_descale_head_stride;

    // The dimensions.
    // b: number of samples. seqlen: length of a sample, 2 * n (n noisy tokens followed by n clean tokens);
    // for varlen, the maximum length of a sample.
    int b, seqlen, d, d_rounded;
    int total;  // total number of tokens (b * seqlen if not varlen)

    // The scaling factors for the kernel.
    float scale_softmax;
    float softcap;

    // Array of length b+1 holding the starting offset of each sample (varlen), or nullptr.
    int * __restrict__ cu_seqlens;

    // Diffusion block size.
    int block_size;

    bool is_bf16;
    bool is_e4m3;

    // Scheduler metadata
    int * __restrict__ tile_count_semaphore;
    int * __restrict__ num_m_blocks_ptr;
    int * __restrict__ varlen_batch_idx_ptr;  // virtual -> actual
    int * __restrict__ num_nheads_in_l2_ptr;

    int arch;
    int num_sm;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Flash_bwd_params : public Flash_fwd_params {
    using index_t = int64_t;

    // The dO and dQKV matrices.
    void *__restrict__ do_ptr;
    void *__restrict__ dq_ptr;
    void *__restrict__ dk_ptr;
    void *__restrict__ dv_ptr;

    // To accumulate dQ (and dK, dV for GQA)
    void *__restrict__ dq_accum_ptr;
    void *__restrict__ dk_accum_ptr;
    void *__restrict__ dv_accum_ptr;

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

    // Padded lengths of the accumulators (the noisy and clean halves of every sample start at a multiple of
    // the tile size, see DiffusionSeqlenInfo::offset_padded).
    int total_q_padded;   // padding to multiples of kBlockM
    int total_k_padded;   // padding to multiples of kBlockN

    // The pointer to the softmax d sum.
    void *__restrict__ dsoftmax_sum;
    void *__restrict__ softmax_lse_log2_ptr;

    int *__restrict__ dq_semaphore;
    int *__restrict__ dk_semaphore;
    int *__restrict__ dv_semaphore;

    bool deterministic;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int Arch, typename T, int kHeadDim, bool Has_softcap>
void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream);
void prepare_varlen_num_blocks(Flash_fwd_params &params, cudaStream_t stream, int blockM, int blockN);
template <int Arch, typename T, int kHeadDim, bool Has_softcap>
void run_mha_bwd_(Flash_bwd_params &params, cudaStream_t stream);
