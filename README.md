# Fast Diffusion Attention (FlashAttention-2 generation)

FlashAttention-2-class attention kernels for training block diffusion language models on NVIDIA Ampere and Ada GPUs
(sm80 / sm86 / sm89: A100, A800, A30, A10, A40, RTX A6000, RTX 30xx, L4, L40(S), RTX 40xx).

Block diffusion training (BD3-LM, SDAR, RCD, ...) runs the model on the concatenation of a noisy copy and a clean copy
of every sequence, with a block-structured attention mask. This library computes that attention with kernels derived
from [FlashAttention-2](https://github.com/Dao-AILab/flash-attention/tree/main/csrc/flash_attn) (CUTLASS / CuTe,
`mma.sync`, `ldmatrix`, `cp.async`, online softmax): the mask is never materialized, only the visible part of the
attention matrix is computed, and a training step runs at about the throughput of FlashAttention-2 on the same amount
of work or faster (geometric mean per head dimension: A100 0.99-1.03x, RTX A6000 0.95-1.22x, L40S 1.07-1.16x; see
[Performance](#performance)).

This is the `fa2` branch, for sm80 / sm86 / sm89. The `main` branch has the FlashAttention-3 generation for Hopper
(sm90).

- Forward and backward in FP16 / BF16 (training), with the numerics of FlashAttention-2
- Batched interface (pretraining) and varlen interface (packed samples of different lengths, SFT)
- Any block size, head dimensions up to 256, MQA / GQA, dropout, softcapping, deterministic backward, `torch.compile`
- Interface of FlashAttention-2 (`flash_attn_func`, `flash_attn_varlen_func` and the packed variants)

## The attention pattern

A sample of length `L = 2n` is laid out as `[noisy half (n tokens) | clean half (n tokens)]`. Each half is split into
blocks of `block_size` tokens (prompt tokens are blocks like any other):

- a noisy block attends to itself and to all the previous clean blocks;
- a clean block attends to itself and to all the previous clean blocks;
- within a block, attention is fully bidirectional.

```
                    keys:  noisy blocks       clean blocks
                           x0  x1  x2  x3     c0  c1  c2  c3
   queries  noisy  x0      ##  .   .   .      .   .   .   .
                   x1      .   ##  .   .      ##  .   .   .
                   x2      .   .   ##  .      ##  ##  .   .
                   x3      .   .   .   ##     ##  ##  ##  .
            clean  c0      .   .   .   .      ##  .   .   .
                   c1      .   .   .   .      ##  ##  .   .
                   c2      .   .   .   .      ##  ##  ##  .
                   c3      .   .   .   .      ##  ##  ##  ##
```

(each cell is a `block_size x block_size` sub-matrix, `##` visible, `.` masked; the last block of a half may be
shorter). This is the BD3-LM block diffusion training mask, i.e. `block_diff_mask(..., use_regular_causal=False)` of
SDAR-style models (`tests/test_diffusion_attn.py::test_mask_matches_idlm` checks the equivalence). A sample has about
`n^2` visible (query, key) pairs: as many as causal attention on its two halves, half of causal attention on its `2n`
tokens.

## Installation

Requirements: an Ampere or Ada GPU (compute capability 8.x), CUDA >= 12 (tested with 13.0), PyTorch >= 2.4 (tested
with 2.14), Linux. The CUTLASS submodule is the one of FlashAttention-2.

```bash
git clone --recursive -b fa2 https://github.com/yuezhouhu/fast-diffusion-attention.git
cd fast-diffusion-attention
pip install --no-build-isolation -v .
```

As FlashAttention-2, the build generates sm80 SASS (plus compute_80 PTX), which runs on every sm8x GPU. The full build
compiles 24 kernel files (about 2 minutes with `MAX_JOBS=32`). Environment variables:

| Variable | Effect |
| --- | --- |
| `FDA_HDIMS=64,128` | head dimensions to build (out of 32, 64, 96, 128, 192, 256; a head dimension `d` uses the smallest kernel `>= d`) |
| `FDA_DISABLE_FP16=1` | no FP16 kernels (BF16 only) |
| `FDA_DISABLE_DROPOUT=1` | no dropout kernels |
| `FDA_DISABLE_SOFTCAP=1` | no softcapping kernels |
| `FDA_DISABLE_BACKWARD=1` | forward only |
| `FDA_CUDA_ARCHS="80;86;89"` | additional SASS targets (default `80`) |
| `FDA_TUNING=1` | tuning build (see [Tuning](#tuning)) |
| `MAX_JOBS`, `NVCC_THREADS` | parallelism of the build |

## Usage

### Batched (pretraining)

```python
import torch
from fast_diffusion_attention import diffusion_attn_func

B, n, H, H_kv, D = 8, 1024, 32, 8, 128
# every sample: [noisy x_t (n tokens) | clean x_0 (n tokens)]
q = torch.randn(B, 2 * n, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(B, 2 * n, H_kv, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(B, 2 * n, H_kv, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)

out = diffusion_attn_func(q, k, v, block_size=4)       # (B, 2n, H, D)
out.backward(torch.randn_like(out))
```

### Varlen (packed samples)

```python
from fast_diffusion_attention import diffusion_attn_varlen_func

# three samples with n = 300, 1000, 50: sample i occupies rows [cu_seqlens[i], cu_seqlens[i + 1])
# and is laid out as [noisy (n_i) | clean (n_i)]
ns = [300, 1000, 50]
cu_seqlens = torch.tensor([0, 600, 2600, 2700], device="cuda", dtype=torch.int32)
total = 2 * sum(ns)
q = torch.randn(total, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(total, H_kv, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(total, H_kv, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)

out = diffusion_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen_q=2000, max_seqlen_k=2000,
                                 block_size=4)    # (total, H, D)
```

### API

The functions of FlashAttention-2 with `causal` / `window_size` / `alibi_slopes` replaced by `block_size`:

```python
diffusion_attn_func(q, k, v, block_size, dropout_p=0.0, softmax_scale=None, softcap=0.0, deterministic=False,
                    return_attn_probs=False)
diffusion_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, block_size,
                           dropout_p=0.0, softmax_scale=None, softcap=0.0, deterministic=False,
                           return_attn_probs=False)
diffusion_attn_qkvpacked_func(qkv, block_size, ...)            # qkv: (batch, 2n, 3, nheads, headdim)
diffusion_attn_kvpacked_func(q, kv, block_size, ...)           # kv: (batch, 2n, 2, nheads_kv, headdim)
diffusion_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, block_size, ...)
diffusion_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, block_size, ...)
```

| Argument | |
| --- | --- |
| `q`, `k`, `v` | `(batch, 2n, nheads, headdim)` / `(batch, 2n, nheads_kv, headdim)` (batched), `(total, nheads, headdim)` / `(total, nheads_kv, headdim)` (varlen), FP16 or BF16. `nheads` must be a multiple of `nheads_kv`, `headdim <= 256` (padded to a multiple of 8 if needed). The last dimension must be contiguous. |
| `block_size` | size of the diffusion blocks, any positive integer |
| `cu_seqlens_q` | `(batch + 1,)` int32 cumulative sample lengths; every length is even (`2 n_i`). Self-attention only: `cu_seqlens_k` must equal `cu_seqlens_q` and `max_seqlen_k` must equal `max_seqlen_q` |
| `dropout_p` | dropout probability on the attention weights (as FlashAttention-2, set it to 0 for evaluation) |
| `softmax_scale` | scale of `QK^T`, default `1 / sqrt(headdim)` |
| `softcap` | `> 0` enables `softcap * tanh(scores / softcap)` |
| `deterministic` | bitwise reproducible backward (slightly slower, more memory) |
| `return_attn_probs` | also return `softmax_lse` (`(batch, nheads, 2n)` or `(nheads, total)`) and, with dropout, `S_dmask` (testing only, as in FlashAttention-2: `(batch, nheads, 2 hr, 2 hr)` with the clean half at row / column `hr = round_up(max n, 128)`, the dropped elements negative) |

The output has the shape and dtype of `q`. All the functions are differentiable and work with `torch.compile` (they
are registered as custom ops with fake implementations).

## Accuracy

`tests/test_diffusion_attn.py` follows the methodology of the FlashAttention-2 test suite: for the output, the
log-sum-exp and dQ / dK / dV, the error of the kernel with respect to a float32 reference (computed from the same
low-precision inputs) must be at most twice the error of a straightforward PyTorch implementation in the same
low precision, plus a few ulps. It covers both dtypes, all head dimensions (and padded ones: 40, 59, 80, 111, 160,
224), block sizes from 1 to larger than `n`, odd `n`, MQA / GQA, dropout (the reference applies the dropout pattern
returned in `S_dmask`, as in FlashAttention-2), softcapping, varlen batches with very different and zero-length
samples, packed inputs, non-contiguous inputs, custom softmax scales, deterministic mode and `torch.compile`, and
the sequence lengths and block sizes on both sides of every length-dependent kernel choice.

Beyond the max-error bound, `test_no_systematic_error` checks over many random problems that the errors have no
systematic component: the relative bias and the relative scale error (least-squares slope of the kernel result against
the reference, minus 1) of the output and of every gradient average to zero within 4 standard errors, and are on
average no larger than twice those of PyTorch in the same precision.

`test_accuracy_matches_fa2` compares directly with FlashAttention-2: with `block_size = 1` the clean half of a sample is
causal attention on the clean tokens, and with a zero output gradient on the noisy half its gradients are those of that
causal attention. On this subproblem the errors of the output, the LSE and dQ / dK / dV must match those of
FlashAttention-2 (for all dtypes and head dimensions). Deterministic mode is checked to be bitwise reproducible, and the
varlen kernels to give the results of the batched ones on equal-length samples.

```bash
pytest tests/ -n 8                           # 1501 tests pass on the full build (+36 skipped) on an A100
FDA_EMULATE_ARCH=86 pytest tests/ -n 8       # the kernels of sm86 / sm89 GPUs (99 KB of shared memory), on an A100
FDA_TEST_HDIMS=128 pytest tests/             # only the head dimensions that were built
```

`FDA_EMULATE_ARCH=86` / `89` makes the dispatch select the kernels that an sm86 / sm89 GPU would run (all of them fit
the A100), so that their numerics are tested too (1389 tests pass, +148 skipped: the configurations an sm86 GPU cannot
run, e.g. head dimension 256 with dropout, as in FlashAttention-2). On an RTX A6000 (sm86) and an L40S (sm89) the
suite gives the same result natively (1389 passed, 148 skipped).

## Performance

Measured with `benchmarks/benchmark_diffusion.py` with CUDA 13.0, PyTorch 2.14 and FlashAttention-2 2.8.4 on an A100
80GB PCIe (sm80: 108 SMs, 40 MB L2, 300 W), an RTX A6000 (sm86: 84 SMs, 6 MB L2, 300 W) and an L40S (sm89: 142 SMs,
96 MB L2, 350 W), on the shapes of FlashAttention-2's benchmark (`benchmarks/benchmark_flash_attention.py`): MHA with
hidden size 2048 (`nheads = 2048 // headdim`: 16 heads of dimension 128), BF16, head dimension 128 unless noted. For
every shape `B x L` of FlashAttention-2 (16K tokens: `B = 16K / L`, L = 512 ... 16K), the diffusion attention runs on
`B` samples of length `2L` (`[noisy L | clean L]`) and the baseline is FlashAttention-2 *causal* attention on `2B x L`,
the halves of the samples as separate sequences: the same number of visible pairs and the same tile structure as the
diffusion mask, so the ratio of the throughputs is the ratio of the times (causal attention on the `2L` tokens of a
sample does twice the work). Rows are labelled with FlashAttention-2's `L` and `B`. Times cover the whole forward /
backward call (including the pre- and post-processing kernels of the backward and the reduction of dK / dV with
GQA). TFLOPS count only the visible (query, key) pairs: `4 * pairs * headdim * nheads` for the forward, 2.5x that for
the backward.

Variants are measured in interleaved rounds (median of 5) with steady-state timing (`--min-ms 300`: every measurement
runs for about 300 ms after a warm-up of the same length): these GPUs are power-capped and change their clock with the
load, and FlashAttention-2's own times vary by a few percent between runs.

**Summary.** Throughput of a training step (forward + backward) relative to FlashAttention-2 (> 1: faster) over
L = 512 ... 16K (samples of 1K ... 32K tokens) and block sizes 4, 16, 64: range and (geometric mean).

| head dimension (heads) | A100 | RTX A6000 | L40S |
|---|---|---|---|
| 32 (64) | 0.89 - 1.02 (**0.94**) | 0.92 - 1.06 (**0.96**) | 0.94 - 1.10 (**0.99**) |
| 64 (32) | 0.90 - 1.01 (**0.96**) | 0.92 - 1.05 (**0.96**) | 0.94 - 1.12 (**0.99**) |
| 96 (21) | 0.91 - 1.06 (**0.99**) | 0.95 - 1.08 (**0.99**) | 0.97 - 1.12 (**1.00**) |
| 128 (16) | 0.92 - 1.06 (**0.98**) | 0.95 - 1.12 (**1.05**) | 0.98 - 1.14 (**1.02**) |
| 192 (10) | 0.93 - 1.08 (**0.98**) | 0.63 - 0.82 (**0.69**) | 0.90 - 1.06 (**0.96**) |
| 256 (8) | 0.92 - 1.08 (**0.97**) | 0.88 - 1.13 (**0.95**) | 0.96 - 1.13 (**1.00**) |

The lowest values are at short sequences with small blocks: the noisy query tiles visit their noisy diagonal key
tiles in addition to the tiles of causal attention (one more key tile per query tile of the noisy half: 11-20% more
tiles at L = 512, under 1% at 16K); in the backward these block-diagonal tiles are memory-bound. With larger blocks
(64), the diffusion mask has more visible pairs per tile and runs faster than causal attention. The configurations are
FlashAttention-2's except in four cells where ours with them was more than 3% slower than FlashAttention-2 (see
[the dispatch](#how-kernels-are-chosen-dispatch)): with FlashAttention-2's configurations the sm8x forward is up to
1.9x faster than FlashAttention-2 (RTX A6000), and the RTX A6000 backward of head dimensions 32 / 64 runs at
0.87-1.00x, where no configuration of FlashAttention-2's search space is 3% faster. The backward of head dimension
192 on sm8x runs FlashAttention-2's configuration, the only one of its search space that fits in 99 KB: it spills more
in our kernel than in FlashAttention-2's, and the local memory traffic does not stay in the 6 MB of L2 of the RTX
A6000, where it runs at 0.54-0.74x FlashAttention-2 (a training step at 0.63-0.82x; L40S: 0.90-1.05x).

<details>
<summary><b>A100 80GB PCIe (sm80)</b>: forward / backward / training step per head dimension, length and block size</summary>

| head dimension | forward | backward | training step (fwd + bwd) | training step, geometric mean |
|---|---|---|---|---|
| 32 | 0.93 - 1.18 | 0.86 - 0.97 | 0.89 - 1.02 | 0.94 |
| 64 | 0.95 - 1.06 | 0.89 - 1.00 | 0.90 - 1.01 | 0.96 |
| 96 | 0.87 - 0.99 | 0.92 - 1.08 | 0.91 - 1.06 | 0.99 |
| 128 | 0.98 - 1.08 | 0.89 - 1.05 | 0.92 - 1.06 | 0.98 |
| 192 | 0.92 - 1.04 | 0.92 - 1.09 | 0.93 - 1.08 | 0.98 |
| 256 | 0.94 - 1.04 | 0.91 - 1.09 | 0.92 - 1.08 | 0.97 |

#### BF16, head dimension 128, 16 heads (A100)

| | block size | forward ms (TFLOPS) | FA2 forward | rel. throughput | backward ms (TFLOPS) | FA2 backward | rel. throughput | fwd + bwd |
|---|---|---|---|---|---|---|---|---|
| L = 512, B = 32 | 4 | 0.669 (104) | 0.649 (106) | **0.98** | 2.331 (74) | 2.066 (83) | **0.89** | 0.92 |
|  | 16 | 0.669 (106) | 0.649 (106) | **1.00** | 2.342 (76) | 2.066 (83) | **0.92** | 0.93 |
|  | 64 | 0.674 (115) | 0.649 (106) | **1.08** | 2.221 (87) | 2.066 (83) | **1.05** | 1.06 |
| L = 1K, B = 16 | 4 | 1.033 (134) | 1.031 (133) | **1.01** | 3.366 (102) | 3.055 (113) | **0.90** | 0.94 |
|  | 16 | 1.034 (135) | 1.031 (133) | **1.02** | 3.366 (104) | 3.055 (113) | **0.92** | 0.95 |
|  | 64 | 1.039 (141) | 1.031 (133) | **1.06** | 3.254 (112) | 3.055 (113) | **0.99** | 1.02 |
| L = 2K, B = 8 | 4 | 1.757 (157) | 1.773 (155) | **1.01** | 5.401 (127) | 5.018 (137) | **0.93** | 0.95 |
|  | 16 | 1.756 (158) | 1.773 (155) | **1.02** | 5.410 (128) | 5.018 (137) | **0.93** | 0.96 |
|  | 64 | 1.755 (162) | 1.773 (155) | **1.05** | 5.283 (134) | 5.018 (137) | **0.98** | 1.00 |
| L = 4K, B = 4 | 4 | 3.204 (172) | 3.237 (170) | **1.01** | 9.406 (146) | 8.949 (154) | **0.95** | 0.97 |
|  | 16 | 3.203 (172) | 3.237 (170) | **1.01** | 9.411 (147) | 8.949 (154) | **0.95** | 0.97 |
|  | 64 | 3.211 (174) | 3.237 (170) | **1.02** | 9.285 (150) | 8.949 (154) | **0.97** | 0.99 |
| L = 8K, B = 2 | 4 | 6.089 (181) | 6.192 (178) | **1.02** | 17.271 (159) | 16.737 (164) | **0.97** | 0.98 |
|  | 16 | 6.092 (181) | 6.192 (178) | **1.02** | 17.280 (159) | 16.737 (164) | **0.97** | 0.98 |
|  | 64 | 6.107 (181) | 6.192 (178) | **1.02** | 17.134 (162) | 16.737 (164) | **0.99** | 0.99 |
| L = 16K, B = 1 | 4 | 11.858 (185) | 12.107 (182) | **1.02** | 32.772 (168) | 32.238 (171) | **0.98** | 0.99 |
|  | 16 | 11.864 (186) | 12.107 (182) | **1.02** | 32.889 (167) | 32.238 (171) | **0.98** | 0.99 |
|  | 64 | 11.848 (186) | 12.107 (182) | **1.02** | 32.596 (169) | 32.238 (171) | **0.99** | 1.00 |

#### Other head dimensions (BF16, A100): relative throughput forward / backward / training step

| | L | bs=4: fwd / bwd / step | bs=16: fwd / bwd / step | bs=64: fwd / bwd / step |
|---|---|---|---|---|
| d = 32 | 512 | 0.96 / 0.86 / 0.89 | 0.99 / 0.88 / 0.91 | 1.18 / 0.97 / 1.02 |
|  | 1K | 0.95 / 0.89 / 0.90 | 0.96 / 0.89 / 0.91 | 1.06 / 0.94 / 0.97 |
|  | 2K | 0.95 / 0.91 / 0.92 | 0.95 / 0.92 / 0.93 | 1.00 / 0.94 / 0.95 |
|  | 4K | 0.93 / 0.94 / 0.93 | 0.93 / 0.94 / 0.94 | 0.97 / 0.95 / 0.95 |
|  | 8K | 0.94 / 0.95 / 0.95 | 0.93 / 0.95 / 0.94 | 0.95 / 0.95 / 0.95 |
|  | 16K | 0.93 / 0.96 / 0.94 | 0.93 / 0.96 / 0.95 | 0.93 / 0.96 / 0.95 |
| d = 64 | 512 | 0.95 / 0.89 / 0.90 | 0.96 / 0.92 / 0.92 | 1.06 / 1.00 / 1.01 |
|  | 1K | 0.98 / 0.91 / 0.93 | 0.98 / 0.92 / 0.93 | 1.03 / 0.96 / 0.98 |
|  | 2K | 0.99 / 0.94 / 0.94 | 1.01 / 0.94 / 0.96 | 1.03 / 0.95 / 0.97 |
|  | 4K | 1.02 / 0.95 / 0.97 | 1.02 / 0.95 / 0.97 | 1.03 / 0.97 / 0.98 |
|  | 8K | 1.01 / 0.96 / 0.97 | 1.01 / 0.97 / 0.97 | 1.02 / 0.97 / 0.98 |
|  | 16K | 1.02 / 0.97 / 0.99 | 1.02 / 0.97 / 0.99 | 1.02 / 0.98 / 0.99 |
| d = 96 | 512 | 0.87 / 0.92 / 0.91 | 0.89 / 0.93 / 0.93 | 0.98 / 1.08 / 1.06 |
|  | 1K | 0.92 / 0.95 / 0.94 | 0.93 / 0.96 / 0.95 | 0.98 / 1.04 / 1.03 |
|  | 2K | 0.95 / 0.98 / 0.97 | 0.96 / 0.98 / 0.98 | 0.98 / 1.02 / 1.02 |
|  | 4K | 0.97 / 0.99 / 0.99 | 0.97 / 1.00 / 0.99 | 0.98 / 1.03 / 1.02 |
|  | 8K | 0.98 / 1.01 / 1.00 | 0.98 / 1.02 / 1.01 | 0.99 / 1.03 / 1.02 |
|  | 16K | 0.99 / 1.03 / 1.02 | 0.99 / 1.03 / 1.02 | 0.99 / 1.03 / 1.02 |
| d = 192 | 512 | 0.92 / 0.92 / 0.93 | 0.94 / 0.94 / 0.94 | 1.04 / 1.09 / 1.08 |
|  | 1K | 0.95 / 0.92 / 0.93 | 0.96 / 0.94 / 0.94 | 1.02 / 1.03 / 1.03 |
|  | 2K | 0.97 / 0.94 / 0.95 | 0.98 / 0.95 / 0.96 | 1.01 / 1.00 / 1.00 |
|  | 4K | 0.99 / 0.95 / 0.97 | 0.99 / 0.95 / 0.97 | 1.01 / 0.98 / 0.99 |
|  | 8K | 0.99 / 0.97 / 0.98 | 0.99 / 0.97 / 0.98 | 1.01 / 0.98 / 0.99 |
|  | 16K | 1.00 / 0.97 / 0.98 | 0.99 / 0.98 / 0.98 | 1.00 / 0.99 / 0.98 |
| d = 256 | 512 | 0.94 / 0.91 / 0.92 | 0.95 / 0.93 / 0.94 | 1.04 / 1.09 / 1.08 |
|  | 1K | 0.96 / 0.92 / 0.93 | 0.97 / 0.94 / 0.94 | 1.02 / 1.02 / 1.02 |
|  | 2K | 0.98 / 0.95 / 0.95 | 0.99 / 0.96 / 0.96 | 1.01 / 1.00 / 1.00 |
|  | 4K | 0.99 / 0.94 / 0.96 | 0.99 / 0.95 / 0.96 | 1.01 / 0.98 / 0.99 |
|  | 8K | 1.01 / 0.96 / 0.98 | 1.01 / 0.97 / 0.98 | 1.01 / 0.98 / 0.99 |
|  | 16K | 1.01 / 0.97 / 0.99 | 1.01 / 0.98 / 0.99 | 1.01 / 0.99 / 0.99 |

#### FP16 and GQA (A100)

GQA: 32 query heads, 8 KV heads, head dimension 128 (BF16).

| | L | bs=4: fwd / bwd / step | bs=16: fwd / bwd / step | bs=64: fwd / bwd / step |
|---|---|---|---|---|
| FP16, d = 128 | 512 | 0.99 / 0.89 / 0.92 | 1.00 / 0.90 / 0.93 | 1.10 / 1.04 / 1.06 |
|  | 1K | 1.00 / 0.91 / 0.93 | 1.01 / 0.92 / 0.94 | 1.05 / 1.00 / 1.02 |
|  | 2K | 1.01 / 0.93 / 0.95 | 1.02 / 0.94 / 0.96 | 1.04 / 0.99 / 1.00 |
|  | 4K | 1.01 / 0.95 / 0.96 | 1.02 / 0.95 / 0.97 | 1.02 / 0.97 / 0.99 |
|  | 8K | 1.02 / 0.96 / 0.98 | 1.02 / 0.96 / 0.98 | 1.03 / 0.98 / 0.99 |
|  | 16K | 1.03 / 0.98 / 1.00 | 1.02 / 0.98 / 0.99 | 1.03 / 0.99 / 1.00 |
| FP16, d = 64 | 512 | 0.95 / 0.89 / 0.91 | 0.97 / 0.90 / 0.92 | 1.07 / 0.99 / 1.01 |
|  | 1K | 0.98 / 0.91 / 0.93 | 0.99 / 0.92 / 0.94 | 1.04 / 0.96 / 0.98 |
|  | 2K | 0.99 / 0.93 / 0.94 | 1.00 / 0.94 / 0.95 | 1.03 / 0.96 / 0.98 |
|  | 4K | 1.01 / 0.96 / 0.97 | 1.01 / 0.96 / 0.97 | 1.03 / 0.97 / 0.98 |
|  | 8K | 1.02 / 0.97 / 0.98 | 1.02 / 0.97 / 0.98 | 1.03 / 0.97 / 0.99 |
|  | 16K | 1.02 / 0.97 / 0.99 | 1.02 / 0.97 / 0.99 | 1.03 / 0.97 / 1.00 |
| GQA 32 / 8 | 512 | 0.96 / 0.89 / 0.91 | 0.98 / 0.92 / 0.93 | 1.07 / 1.05 / 1.06 |
|  | 1K | 0.99 / 0.92 / 0.94 | 1.00 / 0.93 / 0.95 | 1.04 / 1.01 / 1.02 |
|  | 2K | 0.99 / 0.95 / 0.95 | 1.01 / 0.95 / 0.96 | 1.02 / 1.00 / 1.00 |
|  | 4K | 1.01 / 0.96 / 0.97 | 1.01 / 0.97 / 0.98 | 1.02 / 0.99 / 0.99 |
|  | 8K | 1.01 / 0.98 / 0.98 | 1.01 / 0.98 / 0.99 | 1.02 / 0.99 / 1.00 |
|  | 16K | 1.01 / 0.99 / 0.99 | 1.02 / 0.99 / 1.00 | 1.02 / 1.00 / 1.01 |

#### Varlen (BF16, head dimension 128, A100)

Equal lengths: the varlen interface on samples of length `2L`. Random lengths: samples whose halves have random
lengths in `[L / 8, L]`, packed into 32K tokens (SFT-like packing); the baseline is FlashAttention-2's varlen causal
attention on the halves.

| | L | bs=4: fwd / bwd / step | bs=16: fwd / bwd / step | bs=64: fwd / bwd / step |
|---|---|---|---|---|
| equal lengths | 512 | 1.06 / 0.88 / 0.92 | 1.09 / 0.90 / 0.94 | 1.19 / 1.04 / 1.07 |
|  | 1K | 1.07 / 0.89 / 0.93 | 1.07 / 0.90 / 0.94 | 1.13 / 0.98 / 1.02 |
|  | 2K | 1.05 / 0.91 / 0.94 | 1.05 / 0.91 / 0.94 | 1.08 / 0.96 / 0.99 |
|  | 4K | 1.04 / 0.92 / 0.95 | 1.05 / 0.93 / 0.96 | 1.06 / 0.95 / 0.98 |
|  | 8K | 1.04 / 0.94 / 0.97 | 1.04 / 0.94 / 0.97 | 1.05 / 0.96 / 0.98 |
|  | 16K | 1.03 / 0.95 / 0.97 | 1.03 / 0.95 / 0.97 | 1.04 / 0.95 / 0.98 |
| random lengths |  [64, 512] | 1.07 / 0.87 / 0.93 | 1.12 / 0.90 / 0.95 | 1.25 / 1.06 / 1.12 |
|  |  [128, 1024] | 1.07 / 0.88 / 0.93 | 1.09 / 0.90 / 0.94 | 1.16 / 1.00 / 1.04 |
|  |  [256, 2048] | 1.06 / 0.90 / 0.94 | 1.07 / 0.91 / 0.95 | 1.10 / 0.97 / 1.00 |
|  |  [512, 4096] | 1.05 / 0.92 / 0.95 | 1.05 / 0.92 / 0.95 | 1.07 / 0.95 / 0.98 |
|  |  [1024, 8192] | 1.04 / 0.93 / 0.96 | 1.04 / 0.93 / 0.96 | 1.04 / 0.95 / 0.97 |
|  |  [2048, 16384] | 1.04 / 0.94 / 0.96 | 1.04 / 0.94 / 0.96 | 1.04 / 0.95 / 0.98 |

#### Dropout and deterministic backward (BF16, A100)

Dropout 0.1 for both (FlashAttention-2 with `dropout_p=0.1`); deterministic: both with `deterministic=True`.


| | L | bs=4: fwd / bwd / step | bs=16: fwd / bwd / step | bs=64: fwd / bwd / step |
|---|---|---|---|---|
| dropout, d = 128 | 512 | 0.95 / 0.88 / 0.90 | 0.97 / 0.91 / 0.92 | 1.15 / 1.05 / 1.07 |
|  | 1K | 0.96 / 0.91 / 0.92 | 0.97 / 0.92 / 0.94 | 1.07 / 1.01 / 1.03 |
|  | 2K | 0.96 / 0.94 / 0.94 | 0.97 / 0.94 / 0.95 | 1.03 / 0.98 / 1.00 |
|  | 4K | 0.97 / 0.96 / 0.96 | 0.97 / 0.96 / 0.96 | 0.99 / 0.98 / 0.98 |
|  | 8K | 0.97 / 0.97 / 0.97 | 0.97 / 0.97 / 0.97 | 0.99 / 0.99 / 0.98 |
|  | 16K | 0.98 / 0.98 / 0.98 | 0.98 / 0.98 / 0.98 | 0.99 / 0.99 / 0.99 |
| dropout, d = 64 | 512 | 0.85 / 0.90 / 0.88 | 0.87 / 0.91 / 0.90 | 0.98 / 1.00 / 0.99 |
|  | 1K | 0.92 / 0.92 / 0.93 | 0.93 / 0.94 / 0.94 | 0.98 / 0.98 / 0.98 |
|  | 2K | 0.96 / 0.95 / 0.95 | 0.97 / 0.95 / 0.96 | 0.99 / 0.97 / 0.98 |
|  | 4K | 0.98 / 0.97 / 0.97 | 0.98 / 0.97 / 0.98 | 0.99 / 0.98 / 0.98 |
|  | 8K | 0.99 / 0.99 / 0.98 | 0.99 / 0.99 / 0.99 | 0.99 / 0.99 / 0.99 |
|  | 16K | 1.00 / 0.99 / 1.00 | 1.00 / 0.99 / 1.00 | 1.01 / 0.99 / 1.00 |
| dropout, d = 256 | 512 | 0.97 / 0.91 / 0.92 | 0.99 / 0.93 / 0.95 | 1.08 / 1.09 / 1.10 |
|  | 1K | 0.98 / 0.93 / 0.94 | 0.99 / 0.94 / 0.95 | 1.03 / 1.03 / 1.02 |
|  | 2K | 0.99 / 0.95 / 0.96 | 0.99 / 0.95 / 0.96 | 1.02 / 1.00 / 1.01 |
|  | 4K | 0.99 / 0.95 / 0.96 | 0.99 / 0.95 / 0.96 | 1.00 / 0.98 / 0.98 |
|  | 8K | 0.99 / 0.95 / 0.96 | 0.99 / 0.95 / 0.96 | 1.00 / 0.97 / 0.98 |
|  | 16K | 0.99 / 0.97 / 0.98 | 1.00 / 0.97 / 0.98 | 1.00 / 0.98 / 0.98 |
| deterministic, d = 128 | 512 | 0.98 / 0.90 / 0.92 | 1.00 / 0.93 / 0.95 | 1.10 / 1.07 / 1.08 |
|  | 1K | 1.00 / 0.86 / 0.89 | 1.02 / 0.88 / 0.90 | 1.06 / 0.95 / 0.97 |
|  | 2K | 1.01 / 0.81 / 0.84 | 1.01 / 0.82 / 0.85 | 1.04 / 0.85 / 0.89 |
|  | 4K | 1.01 / 0.98 / 0.99 | 1.02 / 0.98 / 0.99 | 1.02 / 1.01 / 1.01 |
|  | 8K | 1.02 / 0.97 / 0.98 | 1.02 / 0.97 / 0.98 | 1.03 / 0.99 / 0.99 |
|  | 16K | 1.02 / 0.88 / 0.90 | 1.03 / 0.88 / 0.90 | 1.03 / 0.89 / 0.90 |

</details>

<details>
<summary><b>RTX A6000 (sm86)</b>: forward / backward / training step per head dimension, length and block size</summary>

| head dimension | forward | backward | training step (fwd + bwd) | training step, geometric mean |
|---|---|---|---|---|
| 32 | 0.99 - 1.44 | 0.87 - 1.00 | 0.92 - 1.06 | 0.96 |
| 64 | 0.98 - 1.20 | 0.87 - 1.00 | 0.92 - 1.05 | 0.96 |
| 96 | 1.04 - 1.91 | 0.81 - 0.97 | 0.95 - 1.08 | 0.99 |
| 128 | 1.02 - 1.79 | 0.81 - 1.09 | 0.95 - 1.12 | 1.05 |
| 192 | 1.02 - 1.39 | 0.54 - 0.74 | 0.63 - 0.82 | 0.69 |
| 256 | 0.99 - 1.43 | 0.85 - 1.08 | 0.88 - 1.13 | 0.95 |

#### BF16, head dimension 128, 16 heads (RTX A6000)

| | block size | forward ms (TFLOPS) | FA2 forward | rel. throughput | backward ms (TFLOPS) | FA2 backward | rel. throughput | fwd + bwd |
|---|---|---|---|---|---|---|---|---|
| L = 512, B = 32 | 4 | 0.980 (71) | 1.474 (47) | **1.51** | 4.408 (39) | 3.601 (48) | **0.81** | 0.95 |
|  | 16 | 1.010 (70) | 1.474 (47) | **1.49** | 4.420 (40) | 3.601 (48) | **0.83** | 0.95 |
|  | 64 | 0.923 (84) | 1.474 (47) | **1.79** | 4.135 (47) | 3.601 (48) | **0.98** | 1.12 |
| L = 1K, B = 16 | 4 | 1.663 (83) | 1.694 (81) | **1.02** | 5.372 (64) | 5.319 (65) | **0.98** | 1.00 |
|  | 16 | 1.659 (84) | 1.694 (81) | **1.04** | 5.363 (65) | 5.319 (65) | **1.00** | 1.01 |
|  | 64 | 1.589 (92) | 1.694 (81) | **1.14** | 5.123 (71) | 5.319 (65) | **1.09** | 1.11 |
| L = 2K, B = 8 | 4 | 2.915 (94) | 2.999 (92) | **1.02** | 8.516 (81) | 8.697 (79) | **1.03** | 1.02 |
|  | 16 | 2.916 (95) | 2.999 (92) | **1.03** | 8.508 (81) | 8.697 (79) | **1.03** | 1.03 |
|  | 64 | 2.841 (100) | 2.999 (92) | **1.09** | 8.278 (86) | 8.697 (79) | **1.09** | 1.08 |
| L = 4K, B = 4 | 4 | 5.516 (100) | 5.793 (95) | **1.05** | 14.791 (93) | 15.505 (89) | **1.04** | 1.05 |
|  | 16 | 5.518 (100) | 5.793 (95) | **1.05** | 14.831 (93) | 15.505 (89) | **1.04** | 1.05 |
|  | 64 | 5.437 (103) | 5.793 (95) | **1.08** | 14.539 (96) | 15.505 (89) | **1.08** | 1.08 |
| L = 8K, B = 2 | 4 | 10.903 (101) | 11.495 (96) | **1.05** | 27.216 (101) | 28.947 (95) | **1.06** | 1.06 |
|  | 16 | 10.849 (102) | 11.495 (96) | **1.06** | 27.184 (101) | 28.947 (95) | **1.06** | 1.07 |
|  | 64 | 10.739 (103) | 11.495 (96) | **1.07** | 26.890 (103) | 28.947 (95) | **1.08** | 1.08 |
| L = 16K, B = 1 | 4 | 22.598 (97) | 23.186 (95) | **1.02** | 52.968 (104) | 56.645 (97) | **1.07** | 1.05 |
|  | 16 | 22.447 (98) | 23.186 (95) | **1.03** | 52.924 (104) | 56.645 (97) | **1.07** | 1.06 |
|  | 64 | 22.465 (98) | 23.186 (95) | **1.03** | 52.687 (105) | 56.645 (97) | **1.08** | 1.06 |

#### Other head dimensions (BF16, RTX A6000): relative throughput forward / backward / training step

| | L | bs=4: fwd / bwd / step | bs=16: fwd / bwd / step | bs=64: fwd / bwd / step |
|---|---|---|---|---|
| d = 32 | 512 | 1.18 / 0.89 / 0.97 | 1.18 / 0.92 / 0.97 | 1.29 / 1.00 / 1.06 |
|  | 1K | 1.36 / 0.87 / 0.99 | 1.38 / 0.87 / 1.00 | 1.44 / 0.93 / 1.05 |
|  | 2K | 0.99 / 0.89 / 0.92 | 1.00 / 0.90 / 0.93 | 1.01 / 0.91 / 0.94 |
|  | 4K | 0.99 / 0.90 / 0.93 | 0.99 / 0.90 / 0.93 | 1.00 / 0.91 / 0.94 |
|  | 8K | 1.01 / 0.92 / 0.94 | 1.01 / 0.92 / 0.94 | 1.02 / 0.93 / 0.95 |
|  | 16K | 1.02 / 0.92 / 0.95 | 1.02 / 0.92 / 0.95 | 1.02 / 0.93 / 0.95 |
| d = 64 | 512 | 1.03 / 0.87 / 0.92 | 1.06 / 0.89 / 0.94 | 1.14 / 1.00 / 1.05 |
|  | 1K | 1.14 / 0.88 / 0.96 | 1.14 / 0.89 / 0.96 | 1.20 / 0.97 / 1.04 |
|  | 2K | 1.13 / 0.90 / 0.96 | 1.14 / 0.90 / 0.98 | 1.17 / 0.95 / 1.02 |
|  | 4K | 1.00 / 0.91 / 0.94 | 1.00 / 0.92 / 0.94 | 1.01 / 0.95 / 0.96 |
|  | 8K | 1.00 / 0.93 / 0.95 | 1.00 / 0.93 / 0.95 | 1.00 / 0.94 / 0.96 |
|  | 16K | 0.98 / 0.94 / 0.95 | 0.98 / 0.94 / 0.96 | 0.98 / 0.94 / 0.96 |
| d = 96 | 512 | 1.65 / 0.81 / 0.97 | 1.71 / 0.83 / 1.00 | 1.91 / 0.88 / 1.08 |
|  | 1K | 1.23 / 0.88 / 0.95 | 1.24 / 0.89 / 0.97 | 1.35 / 0.92 / 1.01 |
|  | 2K | 1.05 / 0.93 / 0.95 | 1.06 / 0.94 / 0.97 | 1.11 / 0.97 / 1.01 |
|  | 4K | 1.07 / 0.95 / 0.98 | 1.07 / 0.95 / 0.98 | 1.09 / 0.97 / 1.00 |
|  | 8K | 1.07 / 0.96 / 0.99 | 1.07 / 0.96 / 1.00 | 1.08 / 0.97 / 1.01 |
|  | 16K | 1.04 / 0.97 / 0.98 | 1.05 / 0.97 / 0.99 | 1.05 / 0.97 / 0.99 |
| d = 192 | 512 | 1.16 / 0.61 / 0.69 | 1.18 / 0.65 / 0.70 | 1.39 / 0.74 / 0.82 |
|  | 1K | 1.26 / 0.54 / 0.63 | 1.27 / 0.56 / 0.64 | 1.34 / 0.60 / 0.70 |
|  | 2K | 1.20 / 0.56 / 0.65 | 1.22 / 0.58 / 0.65 | 1.24 / 0.61 / 0.69 |
|  | 4K | 1.17 / 0.61 / 0.68 | 1.17 / 0.61 / 0.68 | 1.18 / 0.64 / 0.71 |
|  | 8K | 1.09 / 0.60 / 0.67 | 1.09 / 0.60 / 0.67 | 1.10 / 0.62 / 0.68 |
|  | 16K | 1.02 / 0.71 / 0.75 | 1.02 / 0.71 / 0.75 | 1.02 / 0.71 / 0.76 |
| d = 256 | 512 | 1.19 / 0.89 / 0.95 | 1.21 / 0.92 / 0.96 | 1.43 / 1.08 / 1.13 |
|  | 1K | 1.00 / 0.92 / 0.93 | 1.01 / 0.92 / 0.94 | 1.10 / 1.02 / 1.02 |
|  | 2K | 1.04 / 0.92 / 0.94 | 1.05 / 0.94 / 0.95 | 1.10 / 0.98 / 1.00 |
|  | 4K | 1.03 / 0.93 / 0.94 | 1.05 / 0.93 / 0.95 | 1.07 / 0.97 / 0.98 |
|  | 8K | 1.01 / 0.92 / 0.94 | 1.02 / 0.92 / 0.95 | 1.03 / 0.93 / 0.96 |
|  | 16K | 0.99 / 0.85 / 0.88 | 1.00 / 0.85 / 0.88 | 1.00 / 0.86 / 0.88 |

#### FP16 and GQA (RTX A6000)

GQA: 32 query heads, 8 KV heads, head dimension 128 (BF16).

| | L | bs=4: fwd / bwd / step | bs=16: fwd / bwd / step | bs=64: fwd / bwd / step |
|---|---|---|---|---|
| FP16, d = 128 | 512 | 1.38 / 0.85 / 0.95 | 1.40 / 0.87 / 0.96 | 1.68 / 1.02 / 1.14 |
|  | 1K | 1.01 / 0.98 / 1.00 | 1.03 / 1.00 / 1.01 | 1.13 / 1.10 / 1.11 |
|  | 2K | 1.03 / 1.01 / 1.02 | 1.05 / 1.03 / 1.03 | 1.09 / 1.08 / 1.08 |
|  | 4K | 1.05 / 1.05 / 1.04 | 1.07 / 1.05 / 1.05 | 1.09 / 1.08 / 1.08 |
|  | 8K | 1.08 / 1.07 / 1.07 | 1.08 / 1.07 / 1.07 | 1.09 / 1.09 / 1.09 |
|  | 16K | 1.04 / 1.06 / 1.06 | 1.04 / 1.06 / 1.05 | 1.06 / 1.07 / 1.07 |
| FP16, d = 64 | 512 | 1.03 / 0.85 / 0.92 | 1.06 / 0.87 / 0.94 | 1.14 / 1.00 / 1.05 |
|  | 1K | 1.14 / 0.88 / 0.96 | 1.14 / 0.89 / 0.96 | 1.20 / 0.97 / 1.03 |
|  | 2K | 1.10 / 0.90 / 0.95 | 1.10 / 0.90 / 0.95 | 1.13 / 0.95 / 1.00 |
|  | 4K | 1.01 / 0.91 / 0.95 | 1.01 / 0.91 / 0.95 | 1.02 / 0.93 / 0.97 |
|  | 8K | 1.00 / 0.93 / 0.95 | 1.00 / 0.93 / 0.95 | 1.00 / 0.95 / 0.96 |
|  | 16K | 0.98 / 0.94 / 0.95 | 0.98 / 0.94 / 0.95 | 0.98 / 0.95 / 0.95 |
| GQA 32 / 8 | 512 | 1.43 / 0.83 / 0.95 | 1.45 / 0.86 / 0.96 | 1.72 / 1.00 / 1.12 |
|  | 1K | 1.00 / 1.00 / 1.00 | 1.01 / 1.02 / 1.01 | 1.11 / 1.10 / 1.10 |
|  | 2K | 1.03 / 1.03 / 1.03 | 1.04 / 1.04 / 1.04 | 1.09 / 1.09 / 1.09 |
|  | 4K | 1.05 / 1.06 / 1.05 | 1.05 / 1.06 / 1.05 | 1.08 / 1.08 / 1.08 |
|  | 8K | 1.05 / 1.06 / 1.06 | 1.06 / 1.06 / 1.07 | 1.07 / 1.08 / 1.08 |
|  | 16K | 1.06 / 1.06 / 1.07 | 1.06 / 1.06 / 1.07 | 1.06 / 1.07 / 1.07 |

#### Varlen (BF16, head dimension 128, RTX A6000)

Equal lengths: the varlen interface on samples of length `2L`. Random lengths: samples whose halves have random
lengths in `[L / 8, L]`, packed into 32K tokens (SFT-like packing); the baseline is FlashAttention-2's varlen causal
attention on the halves.

| | L | bs=4: fwd / bwd / step | bs=16: fwd / bwd / step | bs=64: fwd / bwd / step |
|---|---|---|---|---|
| equal lengths | 512 | 1.35 / 0.83 / 0.93 | 1.39 / 0.85 / 0.95 | 1.65 / 1.00 / 1.12 |
|  | 1K | 1.01 / 0.97 / 0.99 | 1.01 / 0.98 / 0.99 | 1.12 / 1.06 / 1.09 |
|  | 2K | 1.02 / 0.99 / 1.00 | 1.03 / 1.00 / 1.01 | 1.09 / 1.05 / 1.06 |
|  | 4K | 1.04 / 1.01 / 1.02 | 1.04 / 1.01 / 1.02 | 1.07 / 1.05 / 1.05 |
|  | 8K | 1.05 / 1.02 / 1.03 | 1.06 / 1.02 / 1.04 | 1.07 / 1.04 / 1.05 |
|  | 16K | 1.02 / 1.02 / 1.02 | 1.02 / 1.02 / 1.02 | 1.02 / 1.03 / 1.03 |
| random lengths |  [64, 512] | 1.32 / 0.83 / 0.93 | 1.38 / 0.86 / 0.96 | 1.68 / 1.03 / 1.14 |
|  |  [128, 1024] | 1.03 / 0.96 / 0.98 | 1.04 / 0.98 / 1.00 | 1.18 / 1.09 / 1.12 |
|  |  [256, 2048] | 1.00 / 0.99 / 0.98 | 1.01 / 0.99 / 0.99 | 1.08 / 1.06 / 1.06 |
|  |  [512, 4096] | 1.03 / 1.00 / 1.01 | 1.04 / 1.01 / 1.02 | 1.08 / 1.05 / 1.06 |
|  |  [1024, 8192] | 1.04 / 1.01 / 1.02 | 1.05 / 1.01 / 1.02 | 1.06 / 1.03 / 1.04 |
|  |  [2048, 16384] | 1.04 / 0.99 / 1.00 | 1.04 / 0.99 / 1.00 | 1.05 / 1.01 / 1.02 |

#### Dropout and deterministic backward (BF16, RTX A6000)

Dropout 0.1 for both (FlashAttention-2 with `dropout_p=0.1`); deterministic: both with `deterministic=True`.
Head dimension 256 with dropout: forward only (the backward needs more than 99 KB of shared memory, as in FlashAttention-2).

| | L | bs=4: fwd / bwd / step | bs=16: fwd / bwd / step | bs=64: fwd / bwd / step |
|---|---|---|---|---|
| dropout, d = 128 | 512 | 1.02 / 0.89 / 0.93 | 1.04 / 0.91 / 0.95 | 1.18 / 1.07 / 1.09 |
|  | 1K | 1.16 / 1.02 / 1.05 | 1.17 / 1.03 / 1.07 | 1.25 / 1.13 / 1.17 |
|  | 2K | 1.15 / 1.05 / 1.07 | 1.15 / 1.05 / 1.07 | 1.19 / 1.11 / 1.12 |
|  | 4K | 0.99 / 1.06 / 1.04 | 1.00 / 1.06 / 1.04 | 1.02 / 1.10 / 1.07 |
|  | 8K | 1.01 / 1.07 / 1.05 | 1.02 / 1.07 / 1.06 | 1.03 / 1.09 / 1.08 |
|  | 16K | 1.01 / 1.05 / 1.04 | 1.01 / 1.07 / 1.05 | 1.02 / 1.07 / 1.06 |
| dropout, d = 64 | 512 | 0.97 / 0.87 / 0.89 | 0.98 / 0.89 / 0.91 | 1.07 / 1.02 / 1.03 |
|  | 1K | 0.96 / 0.90 / 0.92 | 0.96 / 0.92 / 0.93 | 1.00 / 1.00 / 1.00 |
|  | 2K | 0.99 / 0.92 / 0.94 | 1.00 / 0.92 / 0.95 | 1.01 / 0.97 / 0.98 |
|  | 4K | 1.00 / 0.93 / 0.95 | 1.01 / 0.93 / 0.96 | 1.01 / 0.97 / 0.98 |
|  | 8K | 1.02 / 0.96 / 0.97 | 1.02 / 0.96 / 0.97 | 1.03 / 0.98 / 0.99 |
|  | 16K | 1.02 / 0.96 / 0.98 | 1.03 / 0.96 / 0.99 | 1.03 / 0.97 / 1.00 |
| dropout, d = 256 | 512 | 1.23 / - / - | 1.25 / - / - | 1.48 / - / - |
|  | 1K | 1.03 / - / - | 1.04 / - / - | 1.15 / - / - |
|  | 2K | 1.05 / - / - | 1.05 / - / - | 1.10 / - / - |
|  | 4K | 1.06 / - / - | 1.07 / - / - | 1.10 / - / - |
|  | 8K | 1.02 / - / - | 1.01 / - / - | 1.02 / - / - |
|  | 16K | 1.00 / - / - | 1.00 / - / - | 1.00 / - / - |
| deterministic, d = 128 | 512 | 1.43 / 0.91 / 0.97 | 1.45 / 0.95 / 0.99 | 1.70 / 1.14 / 1.16 |
|  | 1K | 1.01 / 0.93 / 0.94 | 1.02 / 0.93 / 0.95 | 1.12 / 1.04 / 1.04 |
|  | 2K | 1.03 / 1.03 / 1.04 | 1.03 / 1.03 / 1.04 | 1.09 / 1.10 / 1.09 |
|  | 4K | 1.04 / 1.00 / 1.00 | 1.05 / 1.00 / 1.01 | 1.08 / 1.03 / 1.03 |
|  | 8K | 1.06 / 0.86 / 0.90 | 1.06 / 0.86 / 0.90 | 1.07 / 0.89 / 0.91 |
|  | 16K | 1.03 / 1.00 / 1.01 | 1.03 / 1.00 / 1.01 | 1.04 / 1.00 / 1.02 |

</details>

<details>
<summary><b>L40S (sm89)</b>: forward / backward / training step per head dimension, length and block size</summary>

| head dimension | forward | backward | training step (fwd + bwd) | training step, geometric mean |
|---|---|---|---|---|
| 32 | 0.88 - 1.04 | 0.96 - 1.11 | 0.94 - 1.10 | 0.99 |
| 64 | 0.87 - 1.10 | 0.96 - 1.13 | 0.94 - 1.12 | 0.99 |
| 96 | 0.95 - 1.13 | 0.96 - 1.11 | 0.97 - 1.12 | 1.00 |
| 128 | 0.95 - 1.12 | 0.98 - 1.13 | 0.98 - 1.14 | 1.02 |
| 192 | 0.95 - 1.11 | 0.90 - 1.05 | 0.90 - 1.06 | 0.96 |
| 256 | 0.96 - 1.12 | 0.96 - 1.12 | 0.96 - 1.13 | 1.00 |

#### BF16, head dimension 128, 16 heads (L40S)

| | block size | forward ms (TFLOPS) | FA2 forward | rel. throughput | backward ms (TFLOPS) | FA2 backward | rel. throughput | fwd + bwd |
|---|---|---|---|---|---|---|---|---|
| L = 512, B = 32 | 4 | 0.786 (88) | 0.784 (88) | **1.00** | 3.748 (46) | 3.774 (46) | **1.00** | 1.01 |
|  | 16 | 0.786 (90) | 0.784 (88) | **1.02** | 3.751 (47) | 3.774 (46) | **1.02** | 1.03 |
|  | 64 | 0.785 (99) | 0.784 (88) | **1.12** | 3.738 (52) | 3.774 (46) | **1.13** | 1.14 |
| L = 1K, B = 16 | 4 | 1.208 (114) | 1.146 (120) | **0.95** | 4.123 (84) | 4.169 (82) | **1.02** | 1.00 |
|  | 16 | 1.177 (119) | 1.146 (120) | **0.99** | 4.124 (85) | 4.169 (82) | **1.04** | 1.02 |
|  | 64 | 1.154 (127) | 1.146 (120) | **1.06** | 4.089 (89) | 4.169 (82) | **1.09** | 1.08 |
| L = 2K, B = 8 | 4 | 2.147 (128) | 2.095 (131) | **0.98** | 5.615 (123) | 5.505 (125) | **0.98** | 0.98 |
|  | 16 | 2.149 (129) | 2.095 (131) | **0.98** | 5.609 (123) | 5.505 (125) | **0.98** | 0.99 |
|  | 64 | 2.114 (134) | 2.095 (131) | **1.02** | 5.518 (128) | 5.505 (125) | **1.02** | 1.03 |
| L = 4K, B = 4 | 4 | 3.735 (147) | 3.716 (148) | **0.99** | 9.416 (146) | 9.444 (146) | **1.00** | 1.00 |
|  | 16 | 3.751 (147) | 3.716 (148) | **0.99** | 9.407 (147) | 9.444 (146) | **1.01** | 1.00 |
|  | 64 | 3.682 (152) | 3.716 (148) | **1.03** | 9.206 (152) | 9.444 (146) | **1.04** | 1.04 |
| L = 8K, B = 2 | 4 | 6.912 (159) | 6.893 (160) | **0.99** | 16.760 (164) | 17.230 (160) | **1.02** | 1.02 |
|  | 16 | 6.912 (159) | 6.893 (160) | **0.99** | 16.774 (164) | 17.230 (160) | **1.02** | 1.01 |
|  | 64 | 6.878 (161) | 6.893 (160) | **1.01** | 16.727 (166) | 17.230 (160) | **1.04** | 1.03 |
| L = 16K, B = 1 | 4 | 13.225 (166) | 13.078 (168) | **0.99** | 32.269 (170) | 32.166 (171) | **0.99** | 0.99 |
|  | 16 | 13.174 (167) | 13.078 (168) | **0.99** | 32.092 (171) | 32.166 (171) | **1.00** | 1.00 |
|  | 64 | 13.159 (168) | 13.078 (168) | **1.00** | 32.130 (172) | 32.166 (171) | **1.01** | 1.01 |

#### Other head dimensions (BF16, L40S): relative throughput forward / backward / training step

| | L | bs=4: fwd / bwd / step | bs=16: fwd / bwd / step | bs=64: fwd / bwd / step |
|---|---|---|---|---|
| d = 32 | 512 | 0.94 / 1.00 / 0.99 | 0.96 / 1.03 / 1.01 | 1.04 / 1.11 / 1.10 |
|  | 1K | 0.88 / 0.96 / 0.94 | 0.89 / 0.97 / 0.95 | 0.94 / 1.01 / 0.99 |
|  | 2K | 0.95 / 0.96 / 0.96 | 0.95 / 0.96 / 0.96 | 0.97 / 0.98 / 0.98 |
|  | 4K | 0.97 / 1.01 / 1.00 | 0.97 / 1.01 / 0.99 | 0.97 / 1.02 / 1.01 |
|  | 8K | 0.98 / 0.99 / 0.99 | 0.99 / 0.99 / 0.99 | 0.99 / 1.00 / 1.00 |
|  | 16K | 0.99 / 1.01 / 1.00 | 0.99 / 1.02 / 1.01 | 0.99 / 1.02 / 1.01 |
| d = 64 | 512 | 1.00 / 1.00 / 1.01 | 1.01 / 1.03 / 1.03 | 1.10 / 1.13 / 1.12 |
|  | 1K | 0.87 / 0.99 / 0.96 | 0.88 / 1.00 / 0.97 | 0.91 / 1.04 / 1.01 |
|  | 2K | 0.90 / 0.96 / 0.94 | 0.91 / 0.96 / 0.95 | 0.93 / 1.00 / 0.98 |
|  | 4K | 0.93 / 0.97 / 0.96 | 0.93 / 0.97 / 0.96 | 0.95 / 0.99 / 0.98 |
|  | 8K | 0.95 / 0.99 / 0.98 | 0.95 / 0.99 / 0.98 | 0.96 / 1.01 / 1.00 |
|  | 16K | 0.95 / 0.99 / 0.98 | 0.95 / 1.00 / 0.99 | 0.96 / 1.01 / 0.99 |
| d = 96 | 512 | 1.00 / 1.00 / 0.99 | 1.03 / 1.00 / 1.02 | 1.13 / 1.11 / 1.12 |
|  | 1K | 0.95 / 0.96 / 0.97 | 0.96 / 0.97 / 0.98 | 1.03 / 1.03 / 1.03 |
|  | 2K | 0.98 / 0.96 / 0.97 | 0.98 / 0.96 / 0.97 | 1.03 / 1.00 / 1.01 |
|  | 4K | 1.00 / 0.99 / 0.99 | 1.00 / 0.99 / 0.99 | 1.02 / 1.01 / 1.01 |
|  | 8K | 1.00 / 1.00 / 1.00 | 1.01 / 1.00 / 1.01 | 1.02 / 1.01 / 1.02 |
|  | 16K | 0.99 / 1.00 / 0.99 | 1.00 / 1.01 / 1.00 | 1.01 / 1.01 / 1.00 |
| d = 192 | 512 | 1.00 / 0.95 / 0.94 | 1.03 / 0.95 / 0.97 | 1.11 / 1.05 / 1.06 |
|  | 1K | 0.95 / 0.90 / 0.90 | 0.97 / 0.91 / 0.92 | 1.01 / 0.97 / 0.98 |
|  | 2K | 0.98 / 0.90 / 0.91 | 0.98 / 0.90 / 0.91 | 1.01 / 0.93 / 0.96 |
|  | 4K | 1.01 / 0.91 / 0.93 | 1.01 / 0.91 / 0.93 | 1.02 / 0.94 / 0.96 |
|  | 8K | 1.00 / 0.98 / 0.98 | 1.00 / 0.98 / 0.98 | 1.01 / 0.98 / 0.98 |
|  | 16K | 1.00 / 0.95 / 0.96 | 1.00 / 0.95 / 0.96 | 1.00 / 0.97 / 0.97 |
| d = 256 | 512 | 1.00 / 1.00 / 1.00 | 1.02 / 1.02 / 1.03 | 1.12 / 1.12 / 1.13 |
|  | 1K | 0.96 / 0.99 / 0.98 | 0.97 / 1.00 / 1.00 | 1.07 / 1.07 / 1.07 |
|  | 2K | 0.99 / 0.97 / 0.97 | 1.00 / 0.97 / 0.98 | 1.04 / 1.02 / 1.02 |
|  | 4K | 0.99 / 0.96 / 0.96 | 1.00 / 0.96 / 0.97 | 1.03 / 0.98 / 0.99 |
|  | 8K | 0.99 / 0.98 / 0.98 | 0.99 / 0.98 / 0.99 | 1.01 / 0.99 / 1.00 |
|  | 16K | 1.00 / 0.98 / 0.99 | 1.00 / 0.99 / 0.99 | 1.01 / 1.00 / 1.00 |

#### FP16 and GQA (L40S)

GQA: 32 query heads, 8 KV heads, head dimension 128 (BF16).

| | L | bs=4: fwd / bwd / step | bs=16: fwd / bwd / step | bs=64: fwd / bwd / step |
|---|---|---|---|---|
| FP16, d = 128 | 512 | 0.98 / 1.00 / 1.01 | 0.99 / 1.02 / 1.03 | 1.13 / 1.13 / 1.13 |
|  | 1K | 0.96 / 1.01 / 1.00 | 0.97 / 1.02 / 1.02 | 1.06 / 1.09 / 1.08 |
|  | 2K | 0.98 / 0.98 / 0.98 | 0.98 / 0.98 / 0.99 | 1.02 / 1.03 / 1.02 |
|  | 4K | 0.99 / 1.01 / 1.00 | 1.00 / 1.01 / 1.00 | 1.02 / 1.01 / 1.02 |
|  | 8K | 1.00 / 1.03 / 1.02 | 1.00 / 1.03 / 1.02 | 1.01 / 1.04 / 1.03 |
|  | 16K | 1.00 / 1.01 / 1.00 | 1.01 / 1.01 / 1.01 | 1.00 / 1.01 / 1.01 |
| FP16, d = 64 | 512 | 0.99 / 1.00 / 1.00 | 1.01 / 1.03 / 1.03 | 1.10 / 1.13 / 1.12 |
|  | 1K | 0.88 / 0.99 / 0.96 | 0.88 / 1.00 / 0.96 | 0.92 / 1.05 / 1.01 |
|  | 2K | 0.92 / 0.96 / 0.95 | 0.92 / 0.97 / 0.95 | 0.94 / 1.00 / 0.98 |
|  | 4K | 0.94 / 1.00 / 0.99 | 0.94 / 1.01 / 0.99 | 0.95 / 1.02 / 1.00 |
|  | 8K | 0.95 / 1.00 / 0.98 | 0.95 / 1.00 / 0.98 | 0.95 / 1.01 / 0.99 |
|  | 16K | 0.96 / 1.01 / 0.99 | 0.96 / 1.01 / 0.99 | 0.96 / 1.01 / 1.00 |
| GQA 32 / 8 | 512 | 0.92 / 1.00 / 1.00 | 0.94 / 1.02 / 1.02 | 1.11 / 1.14 / 1.15 |
|  | 1K | 0.95 / 1.00 / 0.99 | 0.96 / 1.01 / 1.00 | 1.05 / 1.08 / 1.07 |
|  | 2K | 0.97 / 0.99 / 0.98 | 0.98 / 0.99 / 0.99 | 1.02 / 1.03 / 1.03 |
|  | 4K | 0.98 / 1.00 / 0.99 | 0.98 / 1.00 / 0.99 | 1.01 / 1.02 / 1.02 |
|  | 8K | 1.00 / 1.01 / 1.01 | 1.00 / 1.01 / 1.01 | 1.01 / 1.02 / 1.02 |
|  | 16K | 0.99 / 1.00 / 1.00 | 0.99 / 1.00 / 1.00 | 0.99 / 1.01 / 1.00 |

#### Varlen (BF16, head dimension 128, L40S)

Equal lengths: the varlen interface on samples of length `2L`. Random lengths: samples whose halves have random
lengths in `[L / 8, L]`, packed into 32K tokens (SFT-like packing); the baseline is FlashAttention-2's varlen causal
attention on the halves.

| | L | bs=4: fwd / bwd / step | bs=16: fwd / bwd / step | bs=64: fwd / bwd / step |
|---|---|---|---|---|
| equal lengths | 512 | 1.00 / 1.00 / 1.01 | 1.02 / 1.02 / 1.03 | 1.13 / 1.13 / 1.13 |
|  | 1K | 0.95 / 0.99 / 0.99 | 0.96 / 1.00 / 1.00 | 1.05 / 1.06 / 1.06 |
|  | 2K | 0.97 / 0.99 / 0.98 | 0.98 / 1.00 / 0.99 | 1.02 / 1.04 / 1.03 |
|  | 4K | 0.99 / 1.00 / 1.00 | 0.99 / 1.01 / 1.00 | 1.01 / 1.01 / 1.02 |
|  | 8K | 0.99 / 1.03 / 1.02 | 1.00 / 1.03 / 1.02 | 1.01 / 1.04 / 1.03 |
|  | 16K | 0.99 / 0.99 / 0.99 | 1.00 / 0.99 / 1.00 | 1.01 / 1.00 / 1.00 |
| random lengths |  [64, 512] | 1.00 / 1.03 / 1.01 | 1.03 / 1.06 / 1.04 | 1.15 / 1.19 / 1.17 |
|  |  [128, 1024] | 0.93 / 1.00 / 0.98 | 0.95 / 1.02 / 1.00 | 1.07 / 1.08 / 1.09 |
|  |  [256, 2048] | 0.98 / 0.97 / 0.97 | 0.98 / 0.98 / 0.98 | 1.03 / 1.03 / 1.03 |
|  |  [512, 4096] | 0.99 / 1.00 / 1.00 | 1.00 / 1.00 / 1.01 | 1.03 / 1.02 / 1.03 |
|  |  [1024, 8192] | 0.99 / 1.02 / 1.01 | 1.00 / 1.03 / 1.02 | 1.01 / 1.03 / 1.03 |
|  |  [2048, 16384] | 1.00 / 1.01 / 1.01 | 1.01 / 1.01 / 1.01 | 1.01 / 1.02 / 1.02 |

#### Dropout and deterministic backward (BF16, L40S)

Dropout 0.1 for both (FlashAttention-2 with `dropout_p=0.1`); deterministic: both with `deterministic=True`.
Head dimension 256 with dropout: forward only (the backward needs more than 99 KB of shared memory, as in FlashAttention-2).

| | L | bs=4: fwd / bwd / step | bs=16: fwd / bwd / step | bs=64: fwd / bwd / step |
|---|---|---|---|---|
| dropout, d = 128 | 512 | 0.98 / 1.00 / 1.01 | 1.00 / 1.02 / 1.03 | 1.10 / 1.13 / 1.13 |
|  | 1K | 0.93 / 1.01 / 0.99 | 0.93 / 1.02 / 1.00 | 1.00 / 1.09 / 1.06 |
|  | 2K | 0.95 / 0.99 / 0.98 | 0.96 / 0.99 / 0.98 | 0.99 / 1.03 / 1.03 |
|  | 4K | 0.99 / 1.00 / 1.00 | 0.99 / 1.01 / 1.00 | 1.01 / 1.01 / 1.02 |
|  | 8K | 1.01 / 1.01 / 1.01 | 1.02 / 1.02 / 1.02 | 1.02 / 1.03 / 1.03 |
|  | 16K | 1.01 / 1.00 / 1.00 | 1.01 / 1.00 / 1.00 | 1.01 / 1.01 / 1.01 |
| dropout, d = 64 | 512 | 1.01 / 1.00 / 1.01 | 1.03 / 1.03 / 1.03 | 1.13 / 1.13 / 1.13 |
|  | 1K | 0.92 / 0.99 / 0.97 | 0.93 / 1.00 / 0.98 | 0.97 / 1.05 / 1.03 |
|  | 2K | 0.97 / 0.94 / 0.96 | 0.98 / 0.95 / 0.96 | 0.99 / 0.98 / 0.99 |
|  | 4K | 0.98 / 1.01 / 1.00 | 0.99 / 1.01 / 1.00 | 1.00 / 1.02 / 1.01 |
|  | 8K | 0.99 / 0.97 / 0.98 | 0.99 / 0.98 / 0.98 | 1.00 / 0.99 / 1.00 |
|  | 16K | 0.99 / 0.98 / 0.98 | 0.99 / 0.98 / 0.99 | 0.99 / 0.99 / 0.99 |
| dropout, d = 256 | 512 | 1.00 / - / - | 1.02 / - / - | 1.12 / - / - |
|  | 1K | 0.97 / - / - | 0.97 / - / - | 1.06 / - / - |
|  | 2K | 0.98 / - / - | 0.99 / - / - | 1.04 / - / - |
|  | 4K | 0.99 / - / - | 0.99 / - / - | 1.01 / - / - |
|  | 8K | 1.00 / - / - | 1.00 / - / - | 1.01 / - / - |
|  | 16K | 0.99 / - / - | 0.99 / - / - | 1.00 / - / - |
| deterministic, d = 128 | 512 | 1.00 / 0.67 / 0.70 | 1.02 / 0.69 / 0.71 | 1.11 / 0.82 / 0.86 |
|  | 1K | 0.96 / 0.76 / 0.79 | 0.97 / 0.78 / 0.80 | 1.05 / 0.86 / 0.88 |
|  | 2K | 0.98 / 1.00 / 0.99 | 0.98 / 1.00 / 0.99 | 1.02 / 1.06 / 1.05 |
|  | 4K | 1.00 / 0.97 / 1.00 | 1.00 / 0.97 / 1.00 | 1.03 / 1.03 / 1.03 |
|  | 8K | 1.00 / 0.95 / 0.96 | 0.99 / 0.95 / 0.96 | 1.01 / 0.97 / 0.97 |
|  | 16K | 1.01 / 1.00 / 0.99 | 1.01 / 1.00 / 0.99 | 1.01 / 1.00 / 1.00 |

</details>

## How kernels are chosen (dispatch)

The routing is FlashAttention-2's (`run_mha_fwd`, `set_params_splitkv` and `mha_bwd` in `csrc/fda/flash_api.cpp`,
`run_mha_fwd_hdim*` / `run_mha_bwd_hdim*` in `csrc/fda/src/flash_{fwd,bwd}_launch_template.h`), with its granularity:

- **Architecture.** One binary for all sm8x GPUs: the kernels are compiled to sm80 SASS (+ compute_80 PTX, JIT-compiled
  on newer GPUs), as FlashAttention-2. GPUs of compute capability < 8 are rejected (`check_arch`).
- **Tile configuration.** One per head dimension, GPU class and dropout, in the format of FlashAttention-2's dispatch:
  sm80 (A100, A800, A30: 163 KB of shared memory per thread block) and sm86 / sm89 (`is_sm8x`: A10, A40, RTX A6000,
  RTX 30xx / 40xx, L4, L40(S): 99 KB), or FlashAttention-2's shared memory thresholds where it uses them. No rule
  depends on the GPU model or on the sequence length. The configurations are FlashAttention-2's (those of causal
  attention: the diffusion mask has the tile structure of causal attention on each half), except where ours with them
  is more than 3% slower than FlashAttention-2 and another configuration of its search space is at least 3% faster
  ([Tuning](#tuning); sm80 tuned on A100, sm8x on RTX A6000). The four changes:

  | head dimension | dropout | GPU class | FlashAttention-2 | here | ours with FlashAttention-2's vs FlashAttention-2 | gain |
  |---|---|---|---|---|---|---|
  | 32 | no | sm80 | fwd 128 x 128, 4 warps | fwd 64 x 64, 4 warps | 0.93 | +5.9% |
  | 64 | no | sm80 | fwd 128 x 128, 4 warps | fwd 128 x 64, 4 warps, Q in registers | 0.91 | +11.8% |
  | 128 | yes | sm80 | fwd 128 x 32, 4 warps | fwd 64 x 64, 4 warps | 0.97 | +3.3% |
  | 64 | yes | sm86 / sm89 | bwd 64 x 128, 8 warps, 2/4/4, V in registers | bwd 64 x 128, 8 warps, 2/4/2 | 0.92 | +4.0% |

  Everywhere else FlashAttention-2's configuration is kept: ours with it is within 3% of FlashAttention-2 or faster
  (the sm8x forward: 1.03-1.28x), or no configuration of the search space is 3% faster (e.g. the sm8x backward of head
  dimensions 32 / 64 without dropout, 0.92-0.93x FlashAttention-2). On sm8x, head dimensions 96, 192 and 256 have a
  single valid backward configuration (99 KB): for 192 it is FlashAttention-2's 64 x 64 with V in registers and no
  double buffering, which spills more in our kernel than in FlashAttention-2's (see [Performance](#performance)).
- **Split-KV forward.** As FlashAttention-2's `mha_fwd`: when the query tiles do not fill the GPU (batched inputs, no
  dropout), the key tiles are split with FlashAttention-2's `num_splits_heuristic` (the fewest splits within 85% of
  the best wave efficiency) on the equivalent causal problem (`2b * h * ceil(n / 64)` query tiles, `ceil(n / block_n)`
  key tiles), FlashAttention-2's split tiles (64 x 256 / 128 / 64, 4 warps) and combine kernel. As in
  FlashAttention-2, varlen inputs are not split (it only splits a varlen forward for a query length of 1).
- **Grids and block order.** FlashAttention-2's, on the `2b` halves of causal attention: forward
  `(2 * num_m_blocks, b, h)`, backward `(2 * num_n_blocks, b, h)`, the tiles of the noisy half first.
- **MQA / GQA backward.** FlashAttention-2's dK / dV per query head, summed over the query heads of a KV head.
- **Deterministic backward.** FlashAttention-2's `ceil(num_SMs / (b h))` thread blocks (and dQ accumulation buffers)
  per (batch, head).
- **Request shape.** The head dimension selects the kernel file (the smallest kernel `>= d`); dropout, softcapping,
  even sequence lengths and even head dimension select template variants, as in FlashAttention-2.

## Tuning

FlashAttention-2's workflow (alternative tile configurations measured against each other for every head dimension,
the best one hard-coded in the dispatch), automated:

- **Search space** (`csrc/fda/src/tuning.h`): every configuration in FlashAttention-2's launch templates (flash-attention
  2.8), used or left in comments, and those its comments report as measured: 19 forward and 25 backward
  configurations, each tried for every head dimension where it is valid. FlashAttention-2 2.8's backward kernel needs 8
  warps (its dQ accumulator copies have 256 threads), so its 4-warp backward configurations (left from
  FlashAttention-2 2.0) are not valid; on sm8x (99 KB) the backward of head dimensions 96, 192 and 256 has a single
  valid configuration, FlashAttention-2's.
- **Shapes**: FlashAttention-2's benchmark shapes (MHA, hidden size 2048: `nheads = 2048 // headdim`, FP16): for every
  shape `B x L` of FlashAttention-2 (16K tokens: `B = 16K / L`, L = 512 ... 16K), the diffusion attention on `B`
  samples of length `2L` (`[noisy L | clean L]`) against FlashAttention-2 causal attention on `2B x L` (the same visible
  pairs; each half has the tile structure of causal attention on `L` tokens); block sizes 4 and 64.
- **GPUs**: sm80 on A100, sm8x on RTX A6000 (sm86), as FlashAttention-2; sm89 (L40S) uses the sm8x configurations.
- **Rule**, per head dimension x GPU class x dropout, on the geometric mean of the throughput relative to
  FlashAttention-2 over the shapes: FlashAttention-2's configuration if ours with it is at most 3% slower than
  FlashAttention-2 (or faster); otherwise the whole search space is measured and its best configuration is used if it
  is at least 3% faster than FlashAttention-2's. Margins within 1.5% of the thresholds are measured again head to head
  (9 rounds).

```bash
# Tuning build: every candidate of csrc/fda/src/tuning.h in its own file (~8 minutes with 72 jobs, FP16 and BF16)
FDA_TUNING=1 FDA_DISABLE_SOFTCAP=1 python setup.py build_ext --inplace
# Ours with FlashAttention-2's configurations (the dispatch) vs FlashAttention-2, on its shapes, steady-state timing
python benchmarks/tune_configs.py --headdim 64 --direction fwd --min-ms 300 --candidates > a100/gap/fwd_hd64.txt
# More than 3% slower: every candidate (FDA_TUNE_FWD / FDA_TUNE_BWD select one at run time)
python benchmarks/tune_configs.py --headdim 64 --direction fwd --min-ms 300 --no-dispatch > a100/sweep/fwd_hd64.txt
python benchmarks/select_configs.py a100/sweep/fwd_hd64.txt --reference 0   # margins over FlashAttention-2's (c0)
# Close calls: head to head; with dropout: --dropout 0.1
python benchmarks/tune_configs.py --headdim 64 --direction fwd --min-ms 300 --candidates 0 5 --rounds 9 --no-dispatch
```

Every candidate is first checked against the reference implementation (with dropout, against the reference with the
dropout pattern returned by the kernel), then timed in interleaved rounds, each measurement paired with one of
FlashAttention-2 right before it: the median of the ratios is reported. With `--min-ms 300` every measurement runs
for about 300 ms after a warm-up of the same length: these power-capped GPUs change their clock with the power draw,
and short measurements favor the kernels that draw less power.

## How it works

The kernels are those of FlashAttention-2 (`compute_attn_1rowblock`, `compute_dq_dk_dv_1colblock` with the
seqk-parallel backward) with the causal mask replaced by the diffusion mask:

- **Tiles.** The two halves of a sample are tiled separately. A query tile of the clean half visits the clean key
  tiles up to its own (like causal attention); a query tile of the noisy half visits the clean key tiles before its
  blocks and the noisy key tiles of its blocks. Only the partially visible tiles (those crossing a block boundary, a
  few per query tile) are masked, with any block size; they are visited first (as FlashAttention-2 visits its
  diagonal first), so that only they need predicated loads. In the backward pass, a key tile visits the query tiles of
  both halves that see it. All the geometry is in `csrc/fda/src/diffusion.h`.
- **Forward.** When the scores of a tile take 64 registers per thread or more (head dimensions >= 64), the masked
  tiles are processed in two halves (two online-softmax updates on the same shared memory tile), which keeps the kernel below 255 registers
  without spills. With 128-row query tiles and block sizes dividing 16, each warp computes only the 16 keys of the
  noisy diagonal tiles that its rows can see (a quarter of the MMAs).
- **Backward.** FlashAttention-2's seqk-parallel kernel: one thread block per key tile and head, dQ accumulated in FP32
  with `atomicAdd` in a padded row layout where every half of every sample starts on a fresh tile, query tiles visited
  in decreasing order as FlashAttention-2. With MQA / GQA, dK / dV per query head summed over the query heads of a KV
  head afterwards (FlashAttention-2).
- **Grids.** FlashAttention-2's, on the equivalent causal problem (the `2b` halves of length `n`): forward
  `(2 * num_m_blocks, b, h)` and backward `(2 * num_n_blocks, b, h)`, the tiles of the noisy half first; the split-KV
  forward `(2 * num_m_blocks, num_splits, b * h)` followed by FlashAttention-2's combine kernel.
- **Dropout.** FlashAttention-2's Philox dropout on 16 x 32 blocks of the attention matrix, with the clean half at
  row / column `round_up(max n, 128)`: the same pattern in the forward and the backward, for batched and varlen inputs.
- **Determinism.** As in FlashAttention-2: `ceil(num_SMs / (b h))` thread blocks per (batch, head), each processing a
  fixed sequence of key tiles and accumulating their dQ in its own FP32 buffer; the buffers are summed in a fixed
  order.

## Limitations

- sm80 / sm86 / sm89 (sm90 runs the sm80 kernels through PTX; use the `main` branch there).
- Tuned as FlashAttention-2, with one configuration per GPU class: sm80 on A100, sm8x on RTX A6000 (sm86); the L40S
  (sm89) runs the sm8x configurations, which may not be its optimum (it is only used for tests and benchmarks), as
  GeForce Ampere / Ada cards (RTX 3090 / 4090: half the FP16 / BF16 MMA rate with FP32 accumulation).
- Routing and configurations are FlashAttention-2's (its search space, granularity, grids and heuristics; its
  configurations wherever ours with them are within 3% of FlashAttention-2), see
  [the dispatch](#how-kernels-are-chosen-dispatch). In particular the backward of head dimension 192 on sm8x runs
  FlashAttention-2's configuration, the only one of its search space that fits in 99 KB, which spills more in our
  kernel than in FlashAttention-2's: 0.54-0.74x FlashAttention-2 on RTX A6000 (6 MB of L2; a training step at
  0.63-0.82x), 0.90-1.05x on L40S.
- The backward at short sequences with small blocks (4, 16): down to 0.81x FlashAttention-2 at L <= 2K on A100 and
  RTX A6000 (the noisy diagonal key tiles are memory-bound); see [Performance](#performance).
- Deterministic backward at L <= 2K on L40S: down to 0.67x FlashAttention-2's deterministic backward (our thread
  blocks cover both halves of a sample: twice the working set of FlashAttention-2's).
- Self-attention on `[noisy | clean]` samples with equal halves (`2n` tokens).
- No FP8 (as FlashAttention-2).

## License

MIT (see `LICENSE`). The CUDA kernels are derived from FlashAttention-2 (BSD 3-Clause) and use CUTLASS (BSD 3-Clause);
see `NOTICE`.
