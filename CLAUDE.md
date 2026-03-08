# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pip-installable Python package (`fast_diffusion_attention`) providing GPU-accelerated attention kernels for diffusion LM training, implemented with Triton.

Two public functions:
1. `diff_attn_with_noisy_input` — forward+backward, full prompt/clean/noisy staircase mask, GQA support
2. `diff_attn_varlen_v2` — forward-only staircase attention with GQA support

## Project Structure

```
fast_diffusion_attention/     # pip-installable package
    __init__.py               # exports both public functions
    diff_attn_noisy.py        # main kernel (fwd+bwd)
    block_prefill_attention.py # staircase fwd-only kernel
    ref_mask.py               # reference mask generator
tests/
    test_correctness.py       # correctness tests (fwd+bwd, fp16+bf16)
benchmarks/
    bench_gqa.py              # vs flash_attn_varlen_func
    tune_kernels.py           # manual tuning script
docs/
    技术规格书.md              # original spec
pyproject.toml
README.md
```

## Commands

```bash
# Install
pip install -e .

# Run tests
python tests/test_correctness.py

# Run benchmarks
python benchmarks/bench_gqa.py

# Tune kernels for current GPU (results go into _TUNED_CONFIGS)
python benchmarks/tune_kernels.py
```

## Architecture

### `diff_attn_noisy.py` — main kernel
Per-document layout: `[Prompt(L0), CleanResponse(L1), NoisyResponse(L1)]`

**Mask rules** (S = staircase_size):
- Prompt block b → sees prompt blocks [0..b]
- Clean block b → sees ALL prompt + clean blocks [0..b]
- Noisy block b → sees ALL prompt + clean blocks [0..b-1] + own noisy block b

**Key optimization**: each query row's visible keys form at most 2 contiguous ranges.
The kernel iterates over each range separately, skipping the gap.

**Kernels**: `_diff_noisy_attn_fwd_kernel`, `_diff_noisy_attn_bwd_dq_kernel`, `_diff_noisy_attn_bwd_dkdv_kernel`

### Usage
```python
from fast_diffusion_attention import diff_attn_with_noisy_input
output = diff_attn_with_noisy_input(
    q, k, v,           # q: [T, n_q_heads, D], k/v: [T, n_kv_heads, D] fp16/bf16
    cu_seqlens,         # [batch+1] int32
    L0_arr, L1_arr,     # [batch] int32
    staircase_size=4,   # 1,2,4,8,16,32,64
    max_seqlen=total,   # always pass to avoid GPU sync
)
# GQA: n_q_heads must be divisible by n_kv_heads
```

## Key Implementation Notes

- Supported dtypes: `float16`, `bfloat16`. Input validation raises `ValueError` for others.
- `sm_scale` includes `* 1.44269504` (log2(e)) for exp2-based online softmax
- LSE stored in float32; backward D_i precomputed in float32
- `max_seqlen` avoids `.item()` GPU sync — always pass it
- dkdv backward uses gap-skipping for clean key tiles
- GQA: fwd/dQ use `pid_kv = pid_h // N_KV_GROUPS` for K/V access; dKdV parallelizes across Q heads then reduces
- Block sizes and launch params are tuned per (cc_major, head_dim) via `_TUNED_CONFIGS` / `_get_kernel_config()`
- Ampere (sm_8x) tuned: fwd BM=64/128 BN=32, dq BM=32 BN=64, dkdv BM=32 BN=32
- Fallback for unknown GPUs: fwd BM=128 BN=64 W=8, bwd BM=64 BN=64 W=4

## Environment

```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH
export LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LIBRARY_PATH
```

PyTorch 2.6.0+cu124, Triton 3.2.0, flash-attn 2.8.3, NVIDIA RTX A6000.
