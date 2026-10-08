# Fast Diffusion Attention (FlashAttention-2 generation)

FlashAttention-2-class attention kernels for training block diffusion language models on NVIDIA Ampere and Ada GPUs
(sm80 / sm86 / sm89: A100, A800, A30, A10, A40, RTX A6000, RTX 30xx, L4, L40(S), RTX 40xx).

Block diffusion training (BD3-LM, SDAR, RCD, ...) runs the model on the concatenation of a noisy copy and a clean copy
of every sequence, with a block-structured attention mask. This library computes that attention with kernels derived
from [FlashAttention-2](https://github.com/Dao-AILab/flash-attention/tree/main/csrc/flash_attn) (CUTLASS / CuTe,
`mma.sync`, `ldmatrix`, `cp.async`, online softmax): the mask is never materialized, only the visible part of the
attention matrix is computed, and a training step runs at about the throughput of FlashAttention-2 on the same amount
of work or faster (geometric mean per head dimension: A800 0.94-1.00x, RTX A6000 0.96-1.30x, L40S 1.03-1.10x; see
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
224), block sizes from 1 to larger than `n`, odd `n`, MQA / GQA (with the GQA cases of the FlashAttention-3 tests,
and the grouped backward forced), dropout (the reference applies the dropout pattern returned in `S_dmask`, as in
FlashAttention-2), softcapping, varlen batches with very different and zero-length
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
pytest tests/ -n 8                           # full build: 4509 tests pass (+172 skipped) on RTX A6000 and L40S
FDA_EMULATE_ARCH=86 pytest tests/ -n 8       # the kernels of sm86 GPUs (99 KB of shared memory) on an sm80 GPU
FDA_TEST_HDIMS=128 pytest tests/             # only the head dimensions that were built
```

`FDA_EMULATE_ARCH=86` / `89` makes the dispatch select the kernels that an sm86 / sm89 GPU would run (all of them fit
an sm80 GPU), so that their numerics are tested too (the configurations an sm86 GPU cannot run are skipped, e.g.
head dimension 256 with dropout, as in FlashAttention-2). Natively, the suite passes on an RTX A6000 (sm86) and an
L40S (sm89): 4509 passed, 172 skipped; on an A800 (sm80): 4501 passed, 36 skipped, measured before the 144 dropout
cases of `test_long_sequence_tiles` were added (they run no other sm80 kernel). The tests of 2K tokens or more run one
at a time per GPU (`large_memory` fixture; their float32 references are computed sample by sample), so that several
workers can share a GPU of 46-48 GB.

## Performance

Measured with `benchmarks/benchmark_diffusion.py` with CUDA 13.0, PyTorch 2.14 and FlashAttention-2 2.8.4 on an A800
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

| head dimension (heads) | A800 | RTX A6000 | L40S |
|---|---|---|---|
| 32 (64) | 0.89 - 1.00 (**0.94**) | 0.94 - 1.24 (**1.00**) | 0.95 - 1.30 (**1.04**) |
| 64 (32) | 0.86 - 0.99 (**0.95**) | 0.99 - 1.19 (**1.04**) | 0.99 - 1.31 (**1.07**) |
| 96 (21) | 0.88 - 1.04 (**1.00**) | 1.00 - 1.20 (**1.05**) | 1.02 - 1.21 (**1.08**) |
| 128 (16) | 0.87 - 1.02 (**0.97**) | 0.96 - 1.14 (**1.05**) | 1.01 - 1.13 (**1.05**) |
| 192 (10) | 0.93 - 1.11 (**0.98**) | 1.10 - 1.68 (**1.30**) | 1.04 - 1.31 (**1.10**) |
| 256 (8) | 0.89 - 1.05 (**0.98**) | 0.90 - 1.08 (**0.96**) | 1.00 - 1.16 (**1.03**) |

Compared with the first release, the training step is faster where the retuned forward configurations, the block
order and the 32 x 64 backward tiles of head dimension 192 apply: on RTX A6000 head dimension 192 goes from 0.69 to
1.30 (geometric mean), 64 from 0.96 to 1.04 and 96 from 0.99 to 1.05; on L40S every head dimension gains 3-15%
(192: 0.96 to 1.10, 96: 1.00 to 1.08). With GQA the grouped backward adds 5-25% at short sequences (32 query / 8 KV
heads, head dimension 128, training step: A800 1.01, RTX A6000 1.06, L40S 1.10), and the deterministic backward runs
2 waves of thread blocks (on A800 9% faster than with FlashAttention-2's single wave: 1.03x FlashAttention-2's
deterministic training step). On sm80 (measured here on A800, in the first release on A100) the MHA results are those
of the first release within the noise: its sm80 configurations remained the best of the search space.

<details>
<summary><b>A800 80GB PCIe (sm80)</b>: forward / backward / training step per head dimension, length and block size</summary>

| head dimension | forward | backward | training step (fwd + bwd) | training step, geometric mean |
|---|---|---|---|---|
| 32 | 0.92 - 1.16 | 0.86 - 0.97 | 0.89 - 1.00 | 0.94 |
| 64 | 0.94 - 1.06 | 0.84 - 0.97 | 0.86 - 0.99 | 0.95 |
| 96 | 0.88 - 1.03 | 0.89 - 1.05 | 0.88 - 1.04 | 1.00 |
| 128 | 0.96 - 1.08 | 0.84 - 1.00 | 0.87 - 1.02 | 0.97 |
| 192 | 1.01 - 1.19 | 0.91 - 1.08 | 0.93 - 1.11 | 0.98 |
| 256 | 0.89 - 1.06 | 0.90 - 1.06 | 0.89 - 1.05 | 0.98 |

#### BF16, head dimension 128, 16 heads (A800)

| | block size | forward ms (TFLOPS) | FA2 forward | rel. throughput | backward ms (TFLOPS) | FA2 backward | rel. throughput | fwd + bwd |
|---|---|---|---|---|---|---|---|---|
| L = 512, B = 32 | 4 | 0.679 (102) | 0.651 (106) | **0.96** | 2.496 (69) | 2.094 (82) | **0.84** | 0.87 |
|  | 16 | 0.677 (105) | 0.651 (106) | **0.99** | 2.500 (71) | 2.094 (82) | **0.87** | 0.89 |
|  | 64 | 0.677 (114) | 0.651 (106) | **1.08** | 2.356 (82) | 2.094 (82) | **1.00** | 1.01 |
| L = 1K, B = 16 | 4 | 1.034 (133) | 1.029 (134) | **0.99** | 3.444 (100) | 3.054 (113) | **0.88** | 0.91 |
|  | 16 | 1.042 (134) | 1.029 (134) | **1.00** | 3.445 (101) | 3.054 (113) | **0.89** | 0.92 |
|  | 64 | 1.031 (142) | 1.029 (134) | **1.06** | 3.339 (109) | 3.054 (113) | **0.96** | 0.99 |
| L = 2K, B = 8 | 4 | 1.732 (159) | 1.773 (155) | **1.03** | 5.395 (128) | 5.012 (137) | **0.93** | 0.95 |
|  | 16 | 1.733 (160) | 1.773 (155) | **1.03** | 5.377 (129) | 5.012 (137) | **0.94** | 0.96 |
|  | 64 | 1.723 (165) | 1.773 (155) | **1.06** | 5.253 (135) | 5.012 (137) | **0.99** | 1.01 |
| L = 4K, B = 4 | 4 | 3.108 (177) | 3.242 (170) | **1.04** | 9.230 (149) | 8.923 (154) | **0.97** | 0.98 |
|  | 16 | 3.110 (177) | 3.242 (170) | **1.04** | 9.245 (149) | 8.923 (154) | **0.97** | 0.98 |
|  | 64 | 3.113 (179) | 3.242 (170) | **1.05** | 9.116 (153) | 8.923 (154) | **0.99** | 1.01 |
| L = 8K, B = 2 | 4 | 5.892 (187) | 6.164 (178) | **1.05** | 16.972 (162) | 16.712 (165) | **0.98** | 1.00 |
|  | 16 | 5.882 (187) | 6.164 (178) | **1.05** | 16.945 (163) | 16.712 (165) | **0.99** | 1.00 |
|  | 64 | 5.881 (188) | 6.164 (178) | **1.06** | 16.811 (165) | 16.712 (165) | **1.00** | 1.02 |
| L = 16K, B = 1 | 4 | 11.397 (193) | 12.082 (182) | **1.06** | 32.399 (170) | 32.013 (172) | **0.99** | 1.01 |
|  | 16 | 11.422 (193) | 12.082 (182) | **1.06** | 32.394 (170) | 32.013 (172) | **0.99** | 1.01 |
|  | 64 | 11.433 (193) | 12.082 (182) | **1.06** | 32.258 (171) | 32.013 (172) | **0.99** | 1.01 |

#### Other head dimensions (BF16, A800): relative throughput forward / backward / training step

| | L | bs=4: fwd / bwd / step | bs=16: fwd / bwd / step | bs=64: fwd / bwd / step |
|---|---|---|---|---|
| d = 32 | 512 | 0.96 / 0.86 / 0.89 | 0.97 / 0.88 / 0.90 | 1.16 / 0.97 / 1.00 |
|  | 1K | 0.94 / 0.90 / 0.90 | 0.95 / 0.90 / 0.91 | 1.05 / 0.95 / 0.97 |
|  | 2K | 0.95 / 0.92 / 0.93 | 0.94 / 0.92 / 0.92 | 1.00 / 0.95 / 0.96 |
|  | 4K | 0.93 / 0.94 / 0.93 | 0.94 / 0.95 / 0.94 | 0.96 / 0.96 / 0.95 |
|  | 8K | 0.92 / 0.96 / 0.95 | 0.92 / 0.96 / 0.95 | 0.94 / 0.97 / 0.96 |
|  | 16K | 0.93 / 0.96 / 0.95 | 0.92 / 0.96 / 0.94 | 0.93 / 0.96 / 0.96 |
| d = 64 | 512 | 0.94 / 0.84 / 0.86 | 0.96 / 0.86 / 0.88 | 1.06 / 0.95 / 0.97 |
|  | 1K | 0.98 / 0.88 / 0.91 | 0.99 / 0.89 / 0.92 | 1.05 / 0.93 / 0.97 |
|  | 2K | 1.01 / 0.92 / 0.94 | 1.01 / 0.93 / 0.95 | 1.04 / 0.95 / 0.97 |
|  | 4K | 1.02 / 0.94 / 0.97 | 1.03 / 0.95 / 0.97 | 1.04 / 0.96 / 0.98 |
|  | 8K | 1.02 / 0.96 / 0.98 | 1.03 / 0.96 / 0.98 | 1.04 / 0.97 / 0.99 |
|  | 16K | 1.03 / 0.97 / 0.99 | 1.03 / 0.97 / 0.99 | 1.04 / 0.97 / 0.99 |
| d = 96 | 512 | 0.88 / 0.89 / 0.88 | 0.91 / 0.90 / 0.90 | 1.01 / 1.04 / 1.03 |
|  | 1K | 0.95 / 0.94 / 0.94 | 0.96 / 0.95 / 0.95 | 1.02 / 1.03 / 1.02 |
|  | 2K | 0.99 / 0.98 / 0.99 | 0.99 / 0.99 / 0.99 | 1.01 / 1.04 / 1.03 |
|  | 4K | 1.01 / 1.01 / 1.01 | 1.01 / 1.01 / 1.01 | 1.02 / 1.04 / 1.03 |
|  | 8K | 1.02 / 1.03 / 1.03 | 1.02 / 1.03 / 1.03 | 1.02 / 1.05 / 1.04 |
|  | 16K | 1.03 / 1.03 / 1.03 | 1.03 / 1.03 / 1.03 | 1.03 / 1.03 / 1.03 |
| d = 192 | 512 | 1.01 / 0.91 / 0.93 | 1.03 / 0.92 / 0.95 | 1.19 / 1.08 / 1.11 |
|  | 1K | 1.02 / 0.92 / 0.95 | 1.04 / 0.93 / 0.96 | 1.14 / 1.02 / 1.05 |
|  | 2K | 1.03 / 0.93 / 0.96 | 1.03 / 0.94 / 0.96 | 1.08 / 0.98 / 1.00 |
|  | 4K | 1.09 / 0.94 / 0.98 | 1.09 / 0.94 / 0.97 | 1.08 / 0.97 / 0.99 |
|  | 8K | 1.11 / 0.95 / 0.98 | 1.11 / 0.95 / 0.98 | 1.11 / 0.96 / 0.99 |
|  | 16K | 1.12 / 0.94 / 0.98 | 1.12 / 0.94 / 0.99 | 1.12 / 0.95 / 0.99 |
| d = 256 | 512 | 0.89 / 0.90 / 0.89 | 0.91 / 0.91 / 0.91 | 1.00 / 1.06 / 1.05 |
|  | 1K | 0.95 / 0.93 / 0.93 | 0.95 / 0.94 / 0.94 | 1.00 / 1.03 / 1.02 |
|  | 2K | 1.00 / 0.95 / 0.96 | 1.00 / 0.95 / 0.96 | 1.03 / 1.00 / 1.01 |
|  | 4K | 1.02 / 0.95 / 0.97 | 1.03 / 0.96 / 0.97 | 1.04 / 0.98 / 0.99 |
|  | 8K | 1.05 / 0.96 / 0.99 | 1.05 / 0.96 / 0.99 | 1.05 / 0.98 / 1.00 |
|  | 16K | 1.06 / 0.98 / 1.00 | 1.06 / 0.98 / 1.00 | 1.06 / 0.99 / 1.00 |

#### FP16 and GQA (A800)

GQA: 32 query heads, 8 KV heads, head dimension 128 (BF16).

| | L | bs=4: fwd / bwd / step | bs=16: fwd / bwd / step | bs=64: fwd / bwd / step |
|---|---|---|---|---|
| FP16, d = 128 | 512 | 0.96 / 0.84 / 0.87 | 0.98 / 0.86 / 0.88 | 1.08 / 1.00 / 1.02 |
|  | 1K | 1.00 / 0.89 / 0.92 | 1.01 / 0.90 / 0.92 | 1.05 / 0.97 / 0.99 |
|  | 2K | 1.02 / 0.94 / 0.96 | 1.03 / 0.94 / 0.96 | 1.05 / 0.99 / 1.01 |
|  | 4K | 1.04 / 0.97 / 0.98 | 1.04 / 0.97 / 0.99 | 1.05 / 0.99 / 1.01 |
|  | 8K | 1.05 / 0.99 / 1.00 | 1.05 / 0.99 / 1.01 | 1.06 / 1.01 / 1.02 |
|  | 16K | 1.06 / 0.99 / 1.01 | 1.06 / 0.99 / 1.01 | 1.06 / 0.99 / 1.01 |
| FP16, d = 64 | 512 | 0.95 / 0.84 / 0.87 | 0.97 / 0.86 / 0.89 | 1.07 / 0.93 / 0.97 |
|  | 1K | 0.97 / 0.89 / 0.91 | 0.99 / 0.89 / 0.92 | 1.05 / 0.94 / 0.97 |
|  | 2K | 1.01 / 0.93 / 0.94 | 1.01 / 0.93 / 0.94 | 1.04 / 0.95 / 0.97 |
|  | 4K | 1.02 / 0.96 / 0.97 | 1.02 / 0.96 / 0.97 | 1.04 / 0.97 / 0.99 |
|  | 8K | 1.04 / 0.97 / 0.99 | 1.04 / 0.97 / 0.99 | 1.04 / 0.97 / 0.99 |
|  | 16K | 1.03 / 0.98 / 0.99 | 1.04 / 0.98 / 0.99 | 1.04 / 0.99 / 1.00 |
| GQA 32 / 8 | 512 | 1.00 / 1.04 / 1.03 | 1.03 / 1.07 / 1.06 | 1.11 / 1.22 / 1.19 |
|  | 1K | 1.01 / 0.98 / 0.99 | 1.02 / 0.99 / 1.00 | 1.06 / 1.08 / 1.08 |
|  | 2K | 1.02 / 0.95 / 0.97 | 1.02 / 0.95 / 0.97 | 1.04 / 1.00 / 1.01 |
|  | 4K | 1.03 / 0.97 / 0.98 | 1.03 / 0.97 / 0.98 | 1.04 / 0.99 / 1.00 |
|  | 8K | 1.03 / 0.98 / 0.99 | 1.03 / 0.98 / 0.99 | 1.03 / 0.99 / 1.00 |
|  | 16K | 1.03 / 0.99 / 1.00 | 1.03 / 0.99 / 1.00 | 1.04 / 0.99 / 1.01 |

#### Varlen (BF16, head dimension 128, A800)

Equal lengths: the varlen interface on samples of length `2L`. Random lengths: samples whose halves have random
lengths in `[L / 8, L]`, packed into 32K tokens (SFT-like packing); the baseline is FlashAttention-2's varlen causal
attention on the halves.

| | L | bs=4: fwd / bwd / step | bs=16: fwd / bwd / step | bs=64: fwd / bwd / step |
|---|---|---|---|---|
| equal lengths | 512 | 1.05 / 0.83 / 0.88 | 1.07 / 0.85 / 0.90 | 1.19 / 0.99 / 1.03 |
|  | 1K | 1.07 / 0.88 / 0.92 | 1.08 / 0.88 / 0.93 | 1.14 / 0.96 / 1.00 |
|  | 2K | 1.07 / 0.92 / 0.95 | 1.08 / 0.93 / 0.96 | 1.10 / 0.97 / 1.00 |
|  | 4K | 1.07 / 0.95 / 0.97 | 1.07 / 0.95 / 0.98 | 1.08 / 0.97 / 1.00 |
|  | 8K | 1.07 / 0.96 / 0.99 | 1.07 / 0.96 / 0.99 | 1.08 / 0.98 / 1.00 |
|  | 16K | 1.07 / 0.97 / 0.99 | 1.07 / 0.97 / 0.99 | 1.08 / 0.98 / 1.00 |
| random lengths |  [64, 512] | 1.07 / 0.85 / 0.91 | 1.10 / 0.89 / 0.93 | 1.25 / 1.05 / 1.10 |
|  |  [128, 1024] | 1.08 / 0.89 / 0.93 | 1.10 / 0.90 / 0.95 | 1.17 / 1.00 / 1.04 |
|  |  [256, 2048] | 1.08 / 0.90 / 0.94 | 1.09 / 0.91 / 0.96 | 1.13 / 0.96 / 1.00 |
|  |  [512, 4096] | 1.07 / 0.94 / 0.97 | 1.08 / 0.94 / 0.98 | 1.10 / 0.97 / 1.00 |
|  |  [1024, 8192] | 1.08 / 0.96 / 0.99 | 1.08 / 0.96 / 0.99 | 1.08 / 0.98 / 1.00 |
|  |  [2048, 16384] | 1.08 / 0.96 / 0.99 | 1.08 / 0.96 / 1.00 | 1.08 / 0.97 / 1.00 |

#### Dropout and deterministic backward (BF16, A800)

Dropout 0.1 for both (FlashAttention-2 with `dropout_p=0.1`); deterministic: both with `deterministic=True`.


| | L | bs=4: fwd / bwd / step | bs=16: fwd / bwd / step | bs=64: fwd / bwd / step |
|---|---|---|---|---|
| dropout, d = 128 | 512 | 0.94 / 0.85 / 0.87 | 0.96 / 0.87 / 0.89 | 1.12 / 1.01 / 1.03 |
|  | 1K | 0.96 / 0.90 / 0.91 | 0.98 / 0.91 / 0.93 | 1.07 / 0.98 / 1.00 |
|  | 2K | 0.97 / 0.94 / 0.95 | 0.99 / 0.95 / 0.96 | 1.04 / 0.99 / 1.01 |
|  | 4K | 0.98 / 0.97 / 0.97 | 0.98 / 0.98 / 0.98 | 1.01 / 1.01 / 1.00 |
|  | 8K | 0.99 / 0.99 / 0.99 | 0.99 / 0.99 / 0.99 | 1.01 / 1.01 / 1.01 |
|  | 16K | 1.00 / 0.99 / 1.00 | 1.00 / 0.99 / 1.00 | 1.00 / 1.00 / 1.00 |
| dropout, d = 64 | 512 | 0.86 / 0.86 / 0.86 | 0.89 / 0.87 / 0.88 | 0.98 / 0.96 / 0.96 |
|  | 1K | 0.92 / 0.89 / 0.90 | 0.93 / 0.90 / 0.91 | 0.98 / 0.95 / 0.96 |
|  | 2K | 0.96 / 0.94 / 0.94 | 0.97 / 0.94 / 0.95 | 0.99 / 0.96 / 0.97 |
|  | 4K | 0.98 / 0.97 / 0.97 | 0.99 / 0.98 / 0.98 | 1.01 / 0.98 / 0.99 |
|  | 8K | 1.00 / 0.98 / 0.98 | 1.01 / 0.99 / 0.99 | 1.01 / 0.99 / 1.00 |
|  | 16K | 1.01 / 0.99 / 1.00 | 1.02 / 0.99 / 1.00 | 1.01 / 0.99 / 1.00 |
| dropout, d = 256 | 512 | 0.92 / 0.91 / 0.91 | 0.94 / 0.92 / 0.93 | 1.03 / 1.08 / 1.07 |
|  | 1K | 0.97 / 0.94 / 0.95 | 0.98 / 0.95 / 0.96 | 1.02 / 1.04 / 1.04 |
|  | 2K | 1.01 / 0.95 / 0.97 | 1.01 / 0.96 / 0.98 | 1.04 / 1.01 / 1.02 |
|  | 4K | 1.02 / 0.96 / 0.97 | 1.02 / 0.96 / 0.97 | 1.04 / 0.98 / 1.00 |
|  | 8K | 1.03 / 0.96 / 0.98 | 1.03 / 0.97 / 0.98 | 1.04 / 0.98 / 1.00 |
|  | 16K | 1.04 / 0.98 / 0.99 | 1.04 / 0.98 / 0.99 | 1.05 / 0.99 / 1.00 |
| deterministic, d = 128 | 512 | 0.96 / 0.90 / 0.91 | 0.99 / 0.92 / 0.94 | 1.08 / 1.05 / 1.06 |
|  | 1K | 1.01 / 0.86 / 0.89 | 1.01 / 0.86 / 0.89 | 1.07 / 0.94 / 0.97 |
|  | 2K | 1.03 / 0.93 / 0.95 | 1.03 / 0.94 / 0.96 | 1.06 / 0.99 / 1.00 |
|  | 4K | 1.04 / 1.14 / 1.13 | 1.04 / 1.15 / 1.13 | 1.05 / 1.18 / 1.15 |
|  | 8K | 1.04 / 1.07 / 1.07 | 1.04 / 1.07 / 1.07 | 1.06 / 1.09 / 1.09 |
|  | 16K | 1.07 / 1.11 / 1.11 | 1.07 / 1.11 / 1.11 | 1.07 / 1.12 / 1.12 |

</details>

<details>
<summary><b>RTX A6000 (sm86)</b>: forward / backward / training step per head dimension, length and block size</summary>

| head dimension | forward | backward | training step (fwd + bwd) | training step, geometric mean |
|---|---|---|---|---|
| 32 | 1.00 - 2.54 | 0.85 - 1.00 | 0.94 - 1.24 | 1.00 |
| 64 | 1.23 - 2.00 | 0.83 - 0.97 | 0.99 - 1.19 | 1.04 |
| 96 | 1.18 - 1.97 | 0.88 - 1.02 | 1.00 - 1.20 | 1.05 |
| 128 | 1.03 - 1.55 | 0.85 - 1.08 | 0.96 - 1.14 | 1.05 |
| 192 | 1.08 - 1.30 | 1.06 - 1.82 | 1.10 - 1.68 | 1.30 |
| 256 | 1.00 - 1.30 | 0.85 - 1.05 | 0.90 - 1.08 | 0.96 |

#### BF16, head dimension 128, 16 heads (RTX A6000)

| | block size | forward ms (TFLOPS) | FA2 forward | rel. throughput | backward ms (TFLOPS) | FA2 backward | rel. throughput | fwd + bwd |
|---|---|---|---|---|---|---|---|---|
| L = 512, B = 32 | 4 | 1.087 (64) | 1.473 (47) | **1.36** | 4.305 (40) | 3.656 (47) | **0.85** | 0.96 |
|  | 16 | 1.085 (65) | 1.473 (47) | **1.38** | 4.295 (41) | 3.656 (47) | **0.87** | 0.97 |
|  | 64 | 1.065 (73) | 1.473 (47) | **1.55** | 4.005 (48) | 3.656 (47) | **1.02** | 1.14 |
| L = 1K, B = 16 | 4 | 1.651 (84) | 1.700 (81) | **1.04** | 5.700 (61) | 5.386 (64) | **0.95** | 0.97 |
|  | 16 | 1.658 (84) | 1.700 (81) | **1.04** | 5.704 (61) | 5.386 (64) | **0.95** | 0.97 |
|  | 64 | 1.574 (93) | 1.700 (81) | **1.15** | 5.419 (67) | 5.386 (64) | **1.05** | 1.08 |
| L = 2K, B = 8 | 4 | 2.935 (94) | 3.035 (91) | **1.03** | 8.676 (79) | 8.851 (78) | **1.01** | 1.02 |
|  | 16 | 2.939 (94) | 3.035 (91) | **1.03** | 8.675 (80) | 8.851 (78) | **1.03** | 1.02 |
|  | 64 | 2.858 (99) | 3.035 (91) | **1.09** | 8.408 (84) | 8.851 (78) | **1.08** | 1.08 |
| L = 4K, B = 4 | 4 | 5.458 (101) | 5.801 (95) | **1.06** | 15.039 (91) | 15.678 (88) | **1.03** | 1.05 |
|  | 16 | 5.458 (101) | 5.801 (95) | **1.06** | 15.040 (92) | 15.678 (88) | **1.05** | 1.05 |
|  | 64 | 5.357 (104) | 5.801 (95) | **1.09** | 14.739 (95) | 15.678 (88) | **1.08** | 1.08 |
| L = 8K, B = 2 | 4 | 10.514 (105) | 11.488 (96) | **1.09** | 27.612 (100) | 29.268 (94) | **1.06** | 1.07 |
|  | 16 | 10.453 (105) | 11.488 (96) | **1.09** | 27.715 (99) | 29.268 (94) | **1.05** | 1.06 |
|  | 64 | 10.391 (107) | 11.488 (96) | **1.11** | 27.315 (101) | 29.268 (94) | **1.07** | 1.09 |
| L = 16K, B = 1 | 4 | 20.485 (107) | 23.155 (95) | **1.13** | 53.857 (102) | 57.133 (96) | **1.06** | 1.08 |
|  | 16 | 20.359 (108) | 23.155 (95) | **1.14** | 54.008 (102) | 57.133 (96) | **1.06** | 1.08 |
|  | 64 | 20.398 (108) | 23.155 (95) | **1.14** | 53.559 (103) | 57.133 (96) | **1.07** | 1.09 |

#### Other head dimensions (BF16, RTX A6000): relative throughput forward / backward / training step

| | L | bs=4: fwd / bwd / step | bs=16: fwd / bwd / step | bs=64: fwd / bwd / step |
|---|---|---|---|---|
| d = 32 | 512 | 2.04 / 0.89 / 1.09 | 2.11 / 0.89 / 1.11 | 2.54 / 1.00 / 1.24 |
|  | 1K | 1.56 / 0.85 / 1.02 | 1.58 / 0.87 / 1.03 | 1.71 / 0.91 / 1.08 |
|  | 2K | 1.05 / 0.90 / 0.94 | 1.05 / 0.90 / 0.94 | 1.12 / 0.92 / 0.97 |
|  | 4K | 1.00 / 0.93 / 0.94 | 1.00 / 0.93 / 0.94 | 1.03 / 0.94 / 0.96 |
|  | 8K | 1.00 / 0.93 / 0.95 | 1.01 / 0.93 / 0.96 | 1.02 / 0.94 / 0.97 |
|  | 16K | 1.02 / 0.95 / 0.96 | 1.02 / 0.95 / 0.96 | 1.03 / 0.95 / 0.97 |
| d = 64 | 512 | 1.80 / 0.83 / 1.03 | 1.83 / 0.85 / 1.05 | 2.00 / 0.96 / 1.19 |
|  | 1K | 1.70 / 0.88 / 1.05 | 1.70 / 0.89 / 1.06 | 1.78 / 0.97 / 1.14 |
|  | 2K | 1.43 / 0.90 / 1.03 | 1.43 / 0.91 / 1.04 | 1.46 / 0.95 / 1.07 |
|  | 4K | 1.23 / 0.92 / 0.99 | 1.24 / 0.92 / 1.00 | 1.25 / 0.95 / 1.02 |
|  | 8K | 1.24 / 0.93 / 1.00 | 1.25 / 0.93 / 1.01 | 1.25 / 0.94 / 1.02 |
|  | 16K | 1.27 / 0.93 / 1.02 | 1.27 / 0.94 / 1.02 | 1.27 / 0.94 / 1.03 |
| d = 96 | 512 | 1.71 / 0.88 / 1.05 | 1.74 / 0.90 / 1.07 | 1.97 / 1.02 / 1.20 |
|  | 1K | 1.31 / 0.91 / 1.00 | 1.31 / 0.92 / 1.00 | 1.39 / 0.98 / 1.07 |
|  | 2K | 1.18 / 0.94 / 1.00 | 1.18 / 0.95 / 1.00 | 1.22 / 0.99 / 1.04 |
|  | 4K | 1.23 / 0.97 / 1.04 | 1.24 / 0.97 / 1.04 | 1.25 / 0.99 / 1.07 |
|  | 8K | 1.28 / 0.98 / 1.06 | 1.30 / 0.98 / 1.06 | 1.30 / 1.00 / 1.07 |
|  | 16K | 1.32 / 0.99 / 1.07 | 1.33 / 0.99 / 1.08 | 1.33 / 1.00 / 1.09 |
| d = 192 | 512 | 1.08 / 1.10 / 1.10 | 1.10 / 1.10 / 1.12 | 1.22 / 1.29 / 1.29 |
|  | 1K | 1.18 / 1.06 / 1.10 | 1.19 / 1.08 / 1.11 | 1.27 / 1.19 / 1.21 |
|  | 2K | 1.26 / 1.19 / 1.21 | 1.26 / 1.20 / 1.21 | 1.30 / 1.27 / 1.28 |
|  | 4K | 1.25 / 1.33 / 1.30 | 1.27 / 1.33 / 1.32 | 1.28 / 1.38 / 1.35 |
|  | 8K | 1.22 / 1.38 / 1.33 | 1.22 / 1.38 / 1.33 | 1.23 / 1.40 / 1.35 |
|  | 16K | 1.22 / 1.80 / 1.66 | 1.22 / 1.80 / 1.66 | 1.23 / 1.82 / 1.68 |
| d = 256 | 512 | 1.11 / 0.89 / 0.92 | 1.13 / 0.89 / 0.94 | 1.30 / 1.05 / 1.08 |
|  | 1K | 1.00 / 0.90 / 0.91 | 1.01 / 0.90 / 0.93 | 1.11 / 1.00 / 1.02 |
|  | 2K | 1.04 / 0.94 / 0.95 | 1.05 / 0.94 / 0.96 | 1.10 / 0.98 / 1.01 |
|  | 4K | 1.08 / 0.93 / 0.96 | 1.09 / 0.93 / 0.96 | 1.11 / 0.96 / 0.99 |
|  | 8K | 1.12 / 0.92 / 0.96 | 1.13 / 0.92 / 0.97 | 1.14 / 0.93 / 0.98 |
|  | 16K | 1.13 / 0.85 / 0.90 | 1.13 / 0.85 / 0.90 | 1.14 / 0.85 / 0.91 |

#### FP16 and GQA (RTX A6000)

GQA: 32 query heads, 8 KV heads, head dimension 128 (BF16).

| | L | bs=4: fwd / bwd / step | bs=16: fwd / bwd / step | bs=64: fwd / bwd / step |
|---|---|---|---|---|
| FP16, d = 128 | 512 | 1.34 / 0.87 / 0.96 | 1.38 / 0.89 / 0.98 | 1.51 / 1.04 / 1.13 |
|  | 1K | 1.04 / 0.95 / 0.97 | 1.05 / 0.97 / 0.98 | 1.15 / 1.06 / 1.07 |
|  | 2K | 1.05 / 1.01 / 1.03 | 1.05 / 1.03 / 1.03 | 1.10 / 1.08 / 1.09 |
|  | 4K | 1.08 / 1.05 / 1.05 | 1.08 / 1.05 / 1.05 | 1.11 / 1.08 / 1.09 |
|  | 8K | 1.12 / 1.07 / 1.08 | 1.12 / 1.05 / 1.08 | 1.13 / 1.09 / 1.09 |
|  | 16K | 1.15 / 1.05 / 1.08 | 1.15 / 1.05 / 1.08 | 1.16 / 1.06 / 1.10 |
| FP16, d = 64 | 512 | 1.77 / 0.85 / 1.03 | 1.83 / 0.87 / 1.06 | 1.97 / 0.98 / 1.18 |
|  | 1K | 1.66 / 0.88 / 1.04 | 1.68 / 0.89 / 1.06 | 1.74 / 0.97 / 1.14 |
|  | 2K | 1.39 / 0.90 / 1.02 | 1.41 / 0.91 / 1.03 | 1.43 / 0.95 / 1.07 |
|  | 4K | 1.25 / 0.92 / 1.01 | 1.25 / 0.92 / 1.01 | 1.26 / 0.95 / 1.03 |
|  | 8K | 1.25 / 0.93 / 1.01 | 1.26 / 0.93 / 1.02 | 1.26 / 0.95 / 1.03 |
|  | 16K | 1.26 / 0.94 / 1.02 | 1.27 / 0.94 / 1.03 | 1.27 / 0.95 / 1.03 |
| GQA 32 / 8 | 512 | 1.53 / 0.90 / 1.02 | 1.57 / 0.93 / 1.04 | 1.85 / 1.07 / 1.22 |
|  | 1K | 1.05 / 0.98 / 0.99 | 1.06 / 0.98 / 1.00 | 1.17 / 1.08 / 1.09 |
|  | 2K | 1.03 / 1.03 / 1.03 | 1.04 / 1.04 / 1.04 | 1.10 / 1.09 / 1.10 |
|  | 4K | 1.06 / 1.05 / 1.05 | 1.07 / 1.06 / 1.06 | 1.09 / 1.08 / 1.08 |
|  | 8K | 1.09 / 1.05 / 1.06 | 1.09 / 1.05 / 1.06 | 1.11 / 1.08 / 1.09 |
|  | 16K | 1.12 / 1.06 / 1.08 | 1.14 / 1.06 / 1.08 | 1.14 / 1.07 / 1.09 |

#### Varlen (BF16, head dimension 128, RTX A6000)

Equal lengths: the varlen interface on samples of length `2L`. Random lengths: samples whose halves have random
lengths in `[L / 8, L]`, packed into 32K tokens (SFT-like packing); the baseline is FlashAttention-2's varlen causal
attention on the halves.

| | L | bs=4: fwd / bwd / step | bs=16: fwd / bwd / step | bs=64: fwd / bwd / step |
|---|---|---|---|---|
| equal lengths | 512 | 1.29 / 0.85 / 0.93 | 1.33 / 0.87 / 0.96 | 1.51 / 1.02 / 1.12 |
|  | 1K | 1.02 / 0.94 / 0.95 | 1.04 / 0.94 / 0.96 | 1.14 / 1.03 / 1.05 |
|  | 2K | 1.04 / 1.00 / 1.01 | 1.04 / 1.00 / 1.01 | 1.10 / 1.06 / 1.06 |
|  | 4K | 1.09 / 1.01 / 1.04 | 1.09 / 1.01 / 1.04 | 1.12 / 1.04 / 1.07 |
|  | 8K | 1.10 / 1.02 / 1.05 | 1.10 / 1.02 / 1.04 | 1.11 / 1.04 / 1.06 |
|  | 16K | 1.15 / 1.02 / 1.06 | 1.15 / 1.02 / 1.06 | 1.15 / 1.03 / 1.07 |
| random lengths |  [64, 512] | 1.20 / 0.83 / 0.91 | 1.23 / 0.86 / 0.93 | 1.45 / 1.00 / 1.11 |
|  |  [128, 1024] | 0.97 / 0.93 / 0.93 | 0.99 / 0.94 / 0.95 | 1.10 / 1.06 / 1.07 |
|  |  [256, 2048] | 1.01 / 0.96 / 0.98 | 1.01 / 0.96 / 0.98 | 1.08 / 1.03 / 1.05 |
|  |  [512, 4096] | 1.07 / 1.01 / 1.02 | 1.07 / 1.01 / 1.03 | 1.10 / 1.06 / 1.06 |
|  |  [1024, 8192] | 1.09 / 1.02 / 1.04 | 1.11 / 1.02 / 1.04 | 1.12 / 1.04 / 1.06 |
|  |  [2048, 16384] | 1.13 / 0.99 / 1.03 | 1.14 / 0.99 / 1.03 | 1.14 / 1.01 / 1.04 |

#### Dropout and deterministic backward (BF16, RTX A6000)

Dropout 0.1 for both (FlashAttention-2 with `dropout_p=0.1`); deterministic: both with `deterministic=True`.
Head dimension 256 with dropout: forward only (the backward needs more than 99 KB of shared memory, as in FlashAttention-2).

| | L | bs=4: fwd / bwd / step | bs=16: fwd / bwd / step | bs=64: fwd / bwd / step |
|---|---|---|---|---|
| dropout, d = 128 | 512 | 1.24 / 0.91 / 0.98 | 1.27 / 0.93 / 1.00 | 1.45 / 1.09 / 1.16 |
|  | 1K | 1.25 / 0.98 / 1.05 | 1.25 / 1.00 / 1.05 | 1.38 / 1.10 / 1.16 |
|  | 2K | 1.23 / 1.05 / 1.10 | 1.23 / 1.05 / 1.10 | 1.27 / 1.11 / 1.16 |
|  | 4K | 1.05 / 1.06 / 1.06 | 1.06 / 1.06 / 1.06 | 1.08 / 1.10 / 1.09 |
|  | 8K | 1.09 / 1.07 / 1.08 | 1.09 / 1.07 / 1.07 | 1.11 / 1.09 / 1.10 |
|  | 16K | 1.12 / 1.07 / 1.08 | 1.12 / 1.07 / 1.08 | 1.12 / 1.08 / 1.09 |
| dropout, d = 64 | 512 | 0.98 / 0.87 / 0.89 | 1.02 / 0.87 / 0.91 | 1.09 / 1.00 / 1.02 |
|  | 1K | 0.97 / 0.89 / 0.91 | 0.97 / 0.90 / 0.92 | 1.03 / 0.98 / 1.00 |
|  | 2K | 1.01 / 0.93 / 0.95 | 1.01 / 0.93 / 0.96 | 1.04 / 0.99 / 1.00 |
|  | 4K | 1.03 / 0.94 / 0.97 | 1.05 / 0.94 / 0.97 | 1.05 / 0.98 / 0.99 |
|  | 8K | 1.06 / 0.96 / 0.98 | 1.06 / 0.96 / 0.98 | 1.07 / 0.98 / 1.01 |
|  | 16K | 1.08 / 0.97 / 0.99 | 1.08 / 0.97 / 0.99 | 1.08 / 0.97 / 0.99 |
| dropout, d = 256 | 512 | 1.11 / - / - | 1.14 / - / - | 1.34 / - / - |
|  | 1K | 1.03 / - / - | 1.04 / - / - | 1.15 / - / - |
|  | 2K | 1.09 / - / - | 1.09 / - / - | 1.14 / - / - |
|  | 4K | 1.16 / - / - | 1.17 / - / - | 1.19 / - / - |
|  | 8K | 1.19 / - / - | 1.20 / - / - | 1.21 / - / - |
|  | 16K | 1.21 / - / - | 1.21 / - / - | 1.22 / - / - |
| deterministic, d = 128 | 512 | 1.36 / 0.91 / 0.97 | 1.38 / 0.95 / 0.99 | 1.53 / 1.14 / 1.16 |
|  | 1K | 1.04 / 0.93 / 0.94 | 1.05 / 0.93 / 0.95 | 1.16 / 1.04 / 1.05 |
|  | 2K | 1.06 / 0.97 / 0.98 | 1.06 / 0.97 / 0.98 | 1.11 / 1.03 / 1.04 |
|  | 4K | 1.09 / 0.97 / 1.00 | 1.10 / 0.97 / 1.01 | 1.12 / 1.00 / 1.03 |
|  | 8K | 1.12 / 0.97 / 0.98 | 1.12 / 0.97 / 0.98 | 1.13 / 0.97 / 0.99 |
|  | 16K | 1.14 / 1.03 / 1.04 | 1.15 / 1.03 / 1.06 | 1.16 / 1.06 / 1.06 |

</details>

<details>
<summary><b>L40S (sm89)</b>: forward / backward / training step per head dimension, length and block size</summary>

| head dimension | forward | backward | training step (fwd + bwd) | training step, geometric mean |
|---|---|---|---|---|
| 32 | 0.91 - 1.21 | 0.95 - 1.31 | 0.95 - 1.30 | 1.04 |
| 64 | 1.06 - 1.41 | 0.96 - 1.29 | 0.99 - 1.31 | 1.07 |
| 96 | 1.07 - 1.34 | 0.96 - 1.19 | 1.02 - 1.21 | 1.08 |
| 128 | 0.97 - 1.25 | 0.96 - 1.13 | 1.01 - 1.13 | 1.05 |
| 192 | 0.93 - 1.31 | 1.05 - 1.32 | 1.04 - 1.31 | 1.10 |
| 256 | 0.93 - 1.12 | 0.97 - 1.17 | 1.00 - 1.16 | 1.03 |

#### BF16, head dimension 128, 16 heads (L40S)

| | block size | forward ms (TFLOPS) | FA2 forward | rel. throughput | backward ms (TFLOPS) | FA2 backward | rel. throughput | fwd + bwd |
|---|---|---|---|---|---|---|---|---|
| L = 512, B = 32 | 4 | 0.818 (85) | 0.787 (88) | **0.97** | 3.708 (47) | 3.765 (46) | **1.02** | 1.01 |
|  | 16 | 0.818 (87) | 0.787 (88) | **0.99** | 3.705 (48) | 3.765 (46) | **1.04** | 1.03 |
|  | 64 | 0.817 (95) | 0.787 (88) | **1.08** | 3.688 (52) | 3.765 (46) | **1.13** | 1.13 |
| L = 1K, B = 16 | 4 | 1.119 (123) | 1.185 (116) | **1.06** | 4.190 (82) | 4.178 (82) | **1.00** | 1.01 |
|  | 16 | 1.118 (125) | 1.185 (116) | **1.08** | 4.192 (83) | 4.178 (82) | **1.01** | 1.03 |
|  | 64 | 1.114 (131) | 1.185 (116) | **1.13** | 4.132 (88) | 4.178 (82) | **1.07** | 1.09 |
| L = 2K, B = 8 | 4 | 1.790 (154) | 2.171 (127) | **1.21** | 5.924 (116) | 5.699 (121) | **0.96** | 1.02 |
|  | 16 | 1.792 (155) | 2.171 (127) | **1.22** | 5.948 (116) | 5.699 (121) | **0.96** | 1.02 |
|  | 64 | 1.814 (156) | 2.171 (127) | **1.23** | 5.836 (121) | 5.699 (121) | **1.00** | 1.06 |
| L = 4K, B = 4 | 4 | 3.169 (174) | 3.885 (142) | **1.23** | 10.004 (138) | 10.026 (137) | **1.01** | 1.06 |
|  | 16 | 3.124 (177) | 3.885 (142) | **1.25** | 9.982 (138) | 10.026 (137) | **1.01** | 1.06 |
|  | 64 | 3.176 (176) | 3.885 (142) | **1.24** | 9.940 (140) | 10.026 (137) | **1.02** | 1.07 |
| L = 8K, B = 2 | 4 | 5.840 (188) | 7.195 (153) | **1.23** | 17.652 (156) | 17.792 (155) | **1.01** | 1.06 |
|  | 16 | 5.760 (191) | 7.195 (153) | **1.25** | 17.728 (155) | 17.792 (155) | **1.00** | 1.06 |
|  | 64 | 5.797 (191) | 7.195 (153) | **1.25** | 17.586 (158) | 17.792 (155) | **1.02** | 1.07 |
| L = 16K, B = 1 | 4 | 11.056 (199) | 13.750 (160) | **1.24** | 33.070 (166) | 33.051 (166) | **1.00** | 1.06 |
|  | 16 | 11.053 (199) | 13.750 (160) | **1.24** | 32.967 (167) | 33.051 (166) | **1.01** | 1.06 |
|  | 64 | 11.110 (199) | 13.750 (160) | **1.24** | 33.002 (167) | 33.051 (166) | **1.01** | 1.07 |

#### Other head dimensions (BF16, L40S): relative throughput forward / backward / training step

| | L | bs=4: fwd / bwd / step | bs=16: fwd / bwd / step | bs=64: fwd / bwd / step |
|---|---|---|---|---|
| d = 32 | 512 | 1.09 / 1.20 / 1.17 | 1.10 / 1.20 / 1.19 | 1.21 / 1.31 / 1.30 |
|  | 1K | 0.91 / 1.06 / 1.02 | 0.92 / 1.06 / 1.03 | 0.96 / 1.12 / 1.08 |
|  | 2K | 0.95 / 0.95 / 0.95 | 0.95 / 0.95 / 0.95 | 0.98 / 0.97 / 0.98 |
|  | 4K | 0.97 / 1.00 / 0.99 | 0.97 / 1.01 / 1.00 | 0.99 / 1.01 / 1.01 |
|  | 8K | 0.99 / 1.01 / 1.01 | 0.99 / 1.02 / 1.01 | 0.99 / 1.02 / 1.02 |
|  | 16K | 0.99 / 1.02 / 1.01 | 0.99 / 1.02 / 1.01 | 1.00 / 1.02 / 1.01 |
| d = 64 | 512 | 1.26 / 1.18 / 1.18 | 1.29 / 1.21 / 1.20 | 1.41 / 1.29 / 1.31 |
|  | 1K | 1.06 / 1.06 / 1.07 | 1.08 / 1.08 / 1.08 | 1.13 / 1.13 / 1.13 |
|  | 2K | 1.09 / 0.96 / 0.99 | 1.09 / 0.96 / 0.99 | 1.11 / 0.99 / 1.02 |
|  | 4K | 1.11 / 1.00 / 1.03 | 1.11 / 1.00 / 1.03 | 1.12 / 1.02 / 1.05 |
|  | 8K | 1.10 / 1.01 / 1.04 | 1.10 / 1.02 / 1.04 | 1.11 / 1.02 / 1.05 |
|  | 16K | 1.10 / 1.00 / 1.03 | 1.09 / 1.01 / 1.03 | 1.10 / 1.01 / 1.03 |
| d = 96 | 512 | 1.17 / 1.05 / 1.09 | 1.20 / 1.08 / 1.11 | 1.31 / 1.19 / 1.21 |
|  | 1K | 1.07 / 1.03 / 1.03 | 1.09 / 1.04 / 1.04 | 1.15 / 1.10 / 1.11 |
|  | 2K | 1.22 / 0.96 / 1.02 | 1.23 / 0.97 / 1.03 | 1.25 / 0.99 / 1.06 |
|  | 4K | 1.28 / 1.00 / 1.07 | 1.28 / 1.01 / 1.07 | 1.31 / 1.03 / 1.09 |
|  | 8K | 1.31 / 0.99 / 1.07 | 1.32 / 0.99 / 1.07 | 1.33 / 1.01 / 1.09 |
|  | 16K | 1.34 / 0.99 / 1.08 | 1.34 / 0.99 / 1.08 | 1.34 / 1.01 / 1.09 |
| d = 192 | 512 | 1.17 / 1.18 / 1.16 | 1.19 / 1.21 / 1.19 | 1.31 / 1.32 / 1.31 |
|  | 1K | 0.93 / 1.10 / 1.07 | 0.95 / 1.11 / 1.09 | 0.99 / 1.19 / 1.14 |
|  | 2K | 1.04 / 1.05 / 1.04 | 1.05 / 1.05 / 1.05 | 1.07 / 1.09 / 1.08 |
|  | 4K | 1.04 / 1.05 / 1.06 | 1.04 / 1.05 / 1.04 | 1.05 / 1.08 / 1.07 |
|  | 8K | 1.04 / 1.11 / 1.09 | 1.04 / 1.10 / 1.09 | 1.04 / 1.12 / 1.10 |
|  | 16K | 1.01 / 1.06 / 1.05 | 1.01 / 1.07 / 1.05 | 1.01 / 1.07 / 1.06 |
| d = 256 | 512 | 0.93 / 1.02 / 1.02 | 0.96 / 1.05 / 1.04 | 1.06 / 1.17 / 1.16 |
|  | 1K | 1.02 / 1.00 / 1.00 | 1.03 / 1.01 / 1.02 | 1.08 / 1.09 / 1.08 |
|  | 2K | 1.08 / 0.99 / 1.00 | 1.09 / 0.99 / 1.01 | 1.12 / 1.03 / 1.05 |
|  | 4K | 1.09 / 0.97 / 1.00 | 1.09 / 0.98 / 1.00 | 1.11 / 1.00 / 1.02 |
|  | 8K | 1.10 / 0.99 / 1.01 | 1.10 / 0.99 / 1.01 | 1.12 / 1.01 / 1.03 |
|  | 16K | 1.12 / 0.99 / 1.02 | 1.11 / 0.99 / 1.02 | 1.11 / 0.99 / 1.02 |

#### FP16 and GQA (L40S)

GQA: 32 query heads, 8 KV heads, head dimension 128 (BF16).

| | L | bs=4: fwd / bwd / step | bs=16: fwd / bwd / step | bs=64: fwd / bwd / step |
|---|---|---|---|---|
| FP16, d = 128 | 512 | 0.97 / 1.02 / 1.01 | 0.99 / 1.04 / 1.04 | 1.08 / 1.15 / 1.13 |
|  | 1K | 1.09 / 1.00 / 1.02 | 1.11 / 1.01 / 1.04 | 1.15 / 1.07 / 1.09 |
|  | 2K | 1.21 / 0.97 / 1.02 | 1.23 / 0.97 / 1.03 | 1.25 / 1.02 / 1.07 |
|  | 4K | 1.23 / 1.00 / 1.06 | 1.23 / 1.01 / 1.06 | 1.25 / 1.02 / 1.08 |
|  | 8K | 1.24 / 0.99 / 1.06 | 1.26 / 0.99 / 1.06 | 1.25 / 1.01 / 1.07 |
|  | 16K | 1.25 / 0.99 / 1.06 | 1.25 / 0.99 / 1.06 | 1.25 / 1.00 / 1.07 |
| FP16, d = 64 | 512 | 1.25 / 1.18 / 1.18 | 1.28 / 1.21 / 1.21 | 1.40 / 1.32 / 1.32 |
|  | 1K | 1.06 / 1.04 / 1.06 | 1.07 / 1.05 / 1.07 | 1.10 / 1.12 / 1.11 |
|  | 2K | 1.08 / 0.97 / 1.00 | 1.08 / 0.97 / 1.00 | 1.11 / 1.01 / 1.03 |
|  | 4K | 1.10 / 1.00 / 1.03 | 1.10 / 1.00 / 1.03 | 1.11 / 1.01 / 1.05 |
|  | 8K | 1.09 / 1.01 / 1.03 | 1.09 / 1.01 / 1.03 | 1.10 / 1.02 / 1.04 |
|  | 16K | 1.10 / 1.01 / 1.03 | 1.10 / 1.02 / 1.04 | 1.10 / 1.02 / 1.04 |
| GQA 32 / 8 | 512 | 1.01 / 1.17 / 1.14 | 1.04 / 1.20 / 1.17 | 1.14 / 1.32 / 1.28 |
|  | 1K | 1.17 / 1.06 / 1.08 | 1.19 / 1.08 / 1.10 | 1.24 / 1.14 / 1.16 |
|  | 2K | 1.26 / 1.03 / 1.07 | 1.27 / 1.03 / 1.08 | 1.28 / 1.06 / 1.11 |
|  | 4K | 1.24 / 1.00 / 1.06 | 1.25 / 1.01 / 1.07 | 1.24 / 1.02 / 1.07 |
|  | 8K | 1.27 / 1.01 / 1.07 | 1.28 / 1.01 / 1.07 | 1.26 / 1.03 / 1.08 |
|  | 16K | 1.24 / 0.99 / 1.05 | 1.24 / 1.00 / 1.06 | 1.24 / 1.01 / 1.06 |

#### Varlen (BF16, head dimension 128, L40S)

Equal lengths: the varlen interface on samples of length `2L`. Random lengths: samples whose halves have random
lengths in `[L / 8, L]`, packed into 32K tokens (SFT-like packing); the baseline is FlashAttention-2's varlen causal
attention on the halves.

| | L | bs=4: fwd / bwd / step | bs=16: fwd / bwd / step | bs=64: fwd / bwd / step |
|---|---|---|---|---|
| equal lengths | 512 | 0.97 / 1.02 / 1.00 | 0.99 / 1.04 / 1.03 | 1.09 / 1.13 / 1.13 |
|  | 1K | 1.04 / 0.98 / 0.99 | 1.06 / 0.99 / 1.00 | 1.13 / 1.05 / 1.06 |
|  | 2K | 1.24 / 0.94 / 1.01 | 1.24 / 0.95 / 1.02 | 1.27 / 0.98 / 1.05 |
|  | 4K | 1.23 / 0.99 / 1.04 | 1.25 / 0.99 / 1.05 | 1.24 / 1.02 / 1.07 |
|  | 8K | 1.24 / 1.00 / 1.06 | 1.25 / 1.00 / 1.07 | 1.25 / 1.02 / 1.08 |
|  | 16K | 1.25 / 0.99 / 1.06 | 1.25 / 0.99 / 1.05 | 1.27 / 1.01 / 1.07 |
| random lengths |  [64, 512] | 1.00 / 1.06 / 1.03 | 1.03 / 1.09 / 1.06 | 1.15 / 1.22 / 1.18 |
|  |  [128, 1024] | 0.99 / 1.02 / 1.01 | 1.00 / 1.03 / 1.02 | 1.09 / 1.10 / 1.10 |
|  |  [256, 2048] | 1.15 / 0.96 / 1.00 | 1.15 / 0.96 / 1.01 | 1.20 / 1.01 / 1.06 |
|  |  [512, 4096] | 1.15 / 0.97 / 1.01 | 1.17 / 0.98 / 1.02 | 1.18 / 1.02 / 1.06 |
|  |  [1024, 8192] | 1.22 / 1.01 / 1.06 | 1.23 / 1.01 / 1.07 | 1.24 / 1.02 / 1.08 |
|  |  [2048, 16384] | 1.21 / 1.01 / 1.05 | 1.22 / 1.01 / 1.06 | 1.23 / 1.02 / 1.07 |

#### Dropout and deterministic backward (BF16, L40S)

Dropout 0.1 for both (FlashAttention-2 with `dropout_p=0.1`); deterministic: both with `deterministic=True`.
Head dimension 256 with dropout: forward only (the backward needs more than 99 KB of shared memory, as in FlashAttention-2).

| | L | bs=4: fwd / bwd / step | bs=16: fwd / bwd / step | bs=64: fwd / bwd / step |
|---|---|---|---|---|
| dropout, d = 128 | 512 | 0.97 / 1.02 / 1.01 | 1.00 / 1.04 / 1.04 | 1.08 / 1.13 / 1.14 |
|  | 1K | 0.95 / 0.99 / 0.97 | 0.97 / 1.00 / 0.99 | 1.01 / 1.06 / 1.05 |
|  | 2K | 1.03 / 0.97 / 0.99 | 1.04 / 0.98 / 0.99 | 1.05 / 1.03 / 1.03 |
|  | 4K | 1.05 / 1.01 / 1.01 | 1.05 / 1.01 / 1.02 | 1.08 / 1.02 / 1.04 |
|  | 8K | 1.09 / 1.00 / 1.03 | 1.10 / 1.00 / 1.03 | 1.10 / 1.01 / 1.03 |
|  | 16K | 1.09 / 1.00 / 1.02 | 1.10 / 1.00 / 1.02 | 1.10 / 1.01 / 1.03 |
| dropout, d = 64 | 512 | 1.19 / 1.18 / 1.17 | 1.22 / 1.21 / 1.20 | 1.33 / 1.29 / 1.30 |
|  | 1K | 0.98 / 1.03 / 1.02 | 0.99 / 1.04 / 1.03 | 1.02 / 1.10 / 1.08 |
|  | 2K | 0.99 / 0.96 / 0.97 | 1.01 / 0.96 / 0.98 | 1.02 / 0.99 / 1.00 |
|  | 4K | 1.01 / 1.00 / 1.00 | 1.01 / 1.00 / 1.01 | 1.02 / 1.01 / 1.02 |
|  | 8K | 1.01 / 0.99 / 1.00 | 1.01 / 0.99 / 1.00 | 1.02 / 1.01 / 1.01 |
|  | 16K | 1.01 / 0.99 / 0.99 | 1.01 / 0.98 / 0.99 | 1.01 / 0.99 / 0.99 |
| dropout, d = 256 | 512 | 0.93 / - / - | 0.96 / - / - | 1.06 / - / - |
|  | 1K | 1.03 / - / - | 1.04 / - / - | 1.09 / - / - |
|  | 2K | 1.08 / - / - | 1.09 / - / - | 1.12 / - / - |
|  | 4K | 1.11 / - / - | 1.11 / - / - | 1.13 / - / - |
|  | 8K | 1.10 / - / - | 1.10 / - / - | 1.11 / - / - |
|  | 16K | 1.11 / - / - | 1.11 / - / - | 1.11 / - / - |
| deterministic, d = 128 | 512 | 0.97 / 0.64 / 0.67 | 0.99 / 0.64 / 0.69 | 1.08 / 0.79 / 0.82 |
|  | 1K | 1.02 / 0.78 / 0.81 | 1.02 / 0.78 / 0.81 | 1.09 / 0.89 / 0.90 |
|  | 2K | 1.18 / 1.00 / 1.00 | 1.19 / 1.00 / 1.00 | 1.22 / 1.06 / 1.05 |
|  | 4K | 1.24 / 0.97 / 1.01 | 1.24 / 1.00 / 1.01 | 1.25 / 1.03 / 1.04 |
|  | 8K | 1.24 / 0.97 / 0.99 | 1.25 / 0.97 / 1.00 | 1.26 / 1.00 / 1.01 |
|  | 16K | 1.23 / 1.05 / 1.06 | 1.23 / 1.05 / 1.05 | 1.24 / 1.05 / 1.06 |

</details>

## How kernels are chosen (dispatch)

The format is FlashAttention-2's (`run_mha_fwd`, `set_params_splitkv` and `mha_bwd` in `csrc/fda/flash_api.cpp`,
`run_mha_fwd_hdim*` / `run_mha_bwd_hdim*` in `csrc/fda/src/flash_{fwd,bwd}_launch_template.h`). The run-time
heuristics read only the hardware (SM count, L2 size, shared memory) and the shape of the problem:

- **Architecture.** One binary for all sm8x GPUs: the kernels are compiled to sm80 SASS (+ compute_80 PTX, JIT-compiled
  on newer GPUs), as FlashAttention-2. GPUs of compute capability < 8 are rejected (`check_arch`).
- **Tile configuration.** One per head dimension, GPU class and dropout, in the format of FlashAttention-2's dispatch:
  sm80 (A100, A800, A30: 163 KB of shared memory per thread block), sm86 (A10, A40, RTX A6000, RTX 30xx: 99 KB) and
  sm89 (L4, L40(S), RTX 40xx: 99 KB) — `is_sm8x` for both where they agree, `is_sm89` where they do not (forward
  only) — or FlashAttention-2's shared memory thresholds where it uses them. The configurations of the first release
  (FlashAttention-2's, with four exceptions), unless another configuration of the search space is at least 3% faster
  on the GPU of the class ([Tuning](#tuning); sm80: A800, sm86: RTX A6000, sm89: L40S). A few depend on the sequence
  length, where the split is at least 3% faster than the best single configuration. Changes since the first
  release (forward unless noted; "up to 2K": up to 2K tokens per half):

  | head dimension | dropout | GPU class | first release | here | gain |
  |---|---|---|---|---|---|
  | 32 | no | sm86 | 128 x 128, 4 warps | 64 x 64, 4 warps (as sm80) | RTX A6000 +3.5% |
  | 32 | yes | all | 128 x 128, 4 warps | 128 x 64, 4 warps, Q in registers | A800 +8.7%, RTX A6000 +8.3%, L40S +5.7% |
  | 64 | no | sm86, sm89 | 128 x 128, 4 warps | 128 x 64, 4 warps, Q in registers | RTX A6000 +20.4%, L40S +13.2% |
  | 96 | no | sm86, sm89 | 64 x 64, 4 warps | 128 x 64, 4 warps; sm89 above 2K: 256 x 64, 8 warps | RTX A6000 +9.6%, L40S +17.5% (+3.1% more above 2K) |
  | 96 | yes | sm86, sm89 | 64 x 64, 4 warps | 128 x 64, 4 warps | RTX A6000 +10.5%, L40S +13.9% |
  | 128 | no | sm89 | 64 x 64, 4 warps | 128 x 64, 4 warps | L40S +14.8% |
  | 128 | yes | sm86 | 128 x 32, 4 warps | 64 x 64, 4 warps up to 1K, 128 x 32 above | RTX A6000 +3.5% |
  | 128 | yes | sm89 | 128 x 32, 4 warps | 128 x 64, 8 warps up to 2K, 256 x 64, 8 warps above | L40S +5.0% |
  | 192 | no | sm80 | 128 x 64, 8 warps | 64 x 64, 4 warps up to 2K, 128 x 128, 8 warps above | A800 +9.1% |
  | 192 | yes | sm86, sm89 | 64 x 64, 4 warps | 128 x 64, 8 warps | RTX A6000 +8.2%, L40S +15.0% |
  | 256 | no | sm89 | 64 x 64, 4 warps | 128 x 32, 8 warps | L40S +4.7% |
  | 256 | yes | sm86, sm89 | 64 x 64, 4 warps | 128 x 32, 8 warps | RTX A6000 +4.2%, L40S +6.9% |
  | 192 | no | sm86, sm89 | bwd 64 x 64, 8 warps, V in registers, no double buffering | bwd 32 x 64, 8 warps | RTX A6000 0.6-0.8x -> 1.1-1.4x FlashAttention-2, L40S +19% |

  The four exceptions of the first release remain (sm80 forward: head dimension 32 64 x 64, 64 128 x 64 with Q in
  registers, 128 with dropout 64 x 64; sm86 / sm89 backward of head dimension 64 with dropout: 64 x 128 with the
  atom layout 2 / 4 / 2). Every other backward configuration is kept: no configuration of the search space is 3%
  faster (at most +1%). The 32 x 64 backward tiles of head dimension 192 are outside FlashAttention-2's search space
  (FlashAttention-2's 64 x 64 with V in registers, the only one of its space that fits in 99 KB, spills more in our
  kernel than in FlashAttention-2's); they cannot do dropout, which keeps FlashAttention-2's configuration.
- **Split-KV forward.** As FlashAttention-2's `mha_fwd`: when the query tiles do not fill the GPU (batched inputs, no
  dropout), the key tiles are split with FlashAttention-2's `num_splits_heuristic` (the fewest splits within 85% of
  the best wave efficiency) on the equivalent causal problem (`2b * h * ceil(n / 64)` query tiles, `ceil(n / block_n)`
  key tiles), FlashAttention-2's split tiles (64 x 256 / 128 / 64, 4 warps) and combine kernel. As in
  FlashAttention-2, varlen inputs are not split (it only splits a varlen forward for a query length of 1).
- **MQA / GQA backward.** A thread block sums the dK / dV of `k` query heads of a KV head (`bwd_heads_per_cta`), so
  that a KV tile is loaded once for them and their dK / dV are not written per query head and summed afterwards
  (FlashAttention-2's dk_expanded, `k = 1`). `k` is the largest divisor of `h / h_k` up to 4 on sm80 (5 for head
  dimension 192) and 2 on sm86 / sm89 (4 for head dimensions 32 and 256, 1 for 96) such that the data of the `k`
  heads (Q, dO and the FP32 dQ accumulators of a sample: `16 n d` bytes each) is at most 0.3x the L2, with at least 2
  waves of thread blocks; `k = h / h_k` writes the final dK / dV, a smaller `k` partial sums of `k` heads. Never with
  dropout, softcapping or in deterministic mode. On FlashAttention-2's shapes with FlashAttention-3's head counts
  (`nheads = 2048 / d`, `nheads_kv = nheads / 4` or 1), the backward at 512 - 1K tokens per half is on average 8.3%
  (A800), 6.5% (L40S) and 2.8% (RTX A6000: 6 MB of L2, head dimensions 32 / 64 only) faster than with one query head
  per thread block, up to 23%, 17% and 22%; neutral for long sequences.
- **Block order.** The grids are FlashAttention-2's: forward `(2 * num_m_blocks, b, h)`, backward
  `(2 * num_n_blocks, b, h / k)`. The thread blocks run in groups of (batch, head) pairs of about 4 waves (2 for
  backward head dimensions > 128), the most expensive tiles of a group first (forward: the last query tiles of both
  halves; backward: the clean key tiles); in the backward, the Q / dO / dQ accumulators of a group must fit in twice
  the L2. Against FlashAttention-2's order (the tiles of the noisy half first, one (batch, head) after the other):
  A800 +0.7%, RTX A6000 +2.7%, L40S +2.8% (geometric means), up to +23% for short sequences.
- **Deterministic backward.** About 2 waves of thread blocks per launch (FlashAttention-2: `ceil(num_SMs / (b h))`
  per (batch, head), 1 wave), at most one per key tile, each accumulating its dQ in its own FP32 buffer: A800 +9%,
  L40S +2%, RTX A6000 +0.2% over 1 wave.
- **Request shape.** The head dimension selects the kernel file (the smallest kernel `>= d`); dropout, softcapping,
  even sequence lengths and even head dimension select template variants, as in FlashAttention-2.

## Tuning

FlashAttention-2's workflow (alternative tile configurations measured against each other for every head dimension,
the best one hard-coded in the dispatch), automated:

- **Search space** (`csrc/fda/src/tuning.h`): every configuration in FlashAttention-2's launch templates (flash-attention
  2.8), used or left in comments, and those its comments report as measured: 19 forward and 25 backward
  configurations; plus 9 backward configurations of ours (64 x 32 and 32 x 64 tiles that fit 99 KB without V in
  registers, other atom layouts and double buffering choices). Each is tried for every head dimension where it is
  valid. FlashAttention-2 2.8's backward kernel needs 8 warps (its dQ accumulator copies have 256 threads), so its
  4-warp backward configurations (left from FlashAttention-2 2.0) are not valid.
- **Shapes**: FlashAttention-2's benchmark shapes (MHA, hidden size 2048: `nheads = 2048 // headdim`, FP16): for every
  shape `B x L` of FlashAttention-2 (16K tokens: `B = 16K / L`, L = 512 ... 16K), the diffusion attention on `B`
  samples of length `2L` (`[noisy L | clean L]`) against FlashAttention-2 causal attention on `2B x L` (the same visible
  pairs; each half has the tile structure of causal attention on `L` tokens); block sizes 4 and 64. With and without
  dropout (0.1), as FlashAttention-2. GQA (4 query heads per KV head) gives the same choices.
- **GPUs**: sm80 on A800, sm86 on RTX A6000, sm89 on L40S.
- **Rule**, per head dimension x GPU class x dropout, on the geometric mean of the throughput relative to
  FlashAttention-2 over the shapes (and the GPUs of the class, every GPU weighing the same, if it has several): the
  configuration of the dispatch unless another is at least 3% faster (and no GPU of the class more than 1% slower); a
  configuration per range of sequence lengths only if the split is at least 3% faster than the best single
  configuration. Margins within 1.5% of the threshold are measured again head to head (9 rounds, twice when the first
  run is within 0.5% of it; the sweeps have about 2% of noise, the dispatch against itself).
  `benchmarks/select_configs.py` applies the rule.

```bash
# Tuning build: every candidate of csrc/fda/src/tuning.h in its own file (~10 minutes with 80 jobs, FP16 and BF16)
FDA_TUNING=1 FDA_DISABLE_SOFTCAP=1 python setup.py build_ext --inplace
# Every candidate and the dispatch vs FlashAttention-2, on its shapes, steady-state timing, on every GPU of the class
# (FDA_TUNE_FWD / FDA_TUNE_BWD select a candidate at run time); with dropout: --dropout 0.1
python benchmarks/tune_configs.py --headdim 64 --direction fwd --min-ms 300 > a6000/fwd_hd64.txt
# The decision for the class (several files: one per GPU of the class)
python benchmarks/select_configs.py a6000/fwd_hd64.txt
# Close calls: head to head
python benchmarks/tune_configs.py --headdim 64 --direction fwd --min-ms 300 --candidates 5 6 --rounds 9
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
  in decreasing order as FlashAttention-2. With MQA / GQA, a thread block processes its key tile for several query
  heads of the KV head and sums their dK / dV: it iterates over the query tiles, and for each over the heads, so that
  the thread blocks of the key tiles of a KV head read the same query tiles at the same time (head by head, they
  drift apart and thrash the L2: 1-15% slower, up to 40% with many heads per thread block).
- **Grids.** FlashAttention-2's, on the equivalent causal problem (the `2b` halves of length `n`): forward
  `(2 * num_m_blocks, b, h)` and backward `(2 * num_n_blocks, b, h / k)`; the split-KV forward
  `(2 * num_m_blocks, num_splits, b * h)` followed by FlashAttention-2's combine kernel. The linear block index is
  mapped to (tile, batch, head) by groups of heads, the most expensive tiles of a group first (see
  [the dispatch](#how-kernels-are-chosen-dispatch)).
- **Dropout.** FlashAttention-2's Philox dropout on 16 x 32 blocks of the attention matrix, with the clean half at
  row / column `round_up(max n, 128)`: the same pattern in the forward and the backward, for batched and varlen inputs.
- **Determinism.** As in FlashAttention-2, every thread block of a (batch, head) processes a fixed sequence of key
  tiles and accumulates their dQ in its own FP32 buffer; the buffers are summed in a fixed order. About 2 waves of
  thread blocks (FlashAttention-2: 1, `ceil(num_SMs / (b h))` per (batch, head)): ours cover both halves of a sample.

## Limitations

- sm80 / sm86 / sm89 (sm90 runs the sm80 kernels through PTX; use the `main` branch there).
- One configuration per GPU class, each tuned on a single GPU: sm80 on A800, sm86 on RTX A6000, sm89 on L40S. Other
  GPUs of a class may prefer other configurations, sm89 especially: the RTX 4090 (128 SMs, 72 MB of L2) and the L4
  differ from the L40S (142 SMs, 96 MB). GeForce Ampere / Ada cards (RTX 3090 / 4090: half the FP16 / BF16 MMA rate
  with FP32 accumulation) are not tested.
- The backward of head dimension 192 with dropout on sm86 / sm89 keeps FlashAttention-2's configuration (the 32 x 64
  tiles cannot do dropout), which spills more in our kernel: 0.57-0.72x FlashAttention-2 on RTX A6000 (6 MB of L2),
  0.93-1.17x on L40S.
- The backward at short sequences with small blocks (4, 16): down to 0.83x FlashAttention-2 at L = 512 on A800 and
  RTX A6000 (the noisy diagonal key tiles are memory-bound); see [Performance](#performance).
- Deterministic backward at L <= 1K on L40S: down to 0.64x FlashAttention-2's deterministic backward (a thread block
  covers both halves of a sample: twice the working set of FlashAttention-2's).
- The grouped GQA backward runs only without dropout and softcapping and not in deterministic mode (otherwise
  FlashAttention-2's dK / dV per query head); with little L2 (RTX A6000: 6 MB) it applies only to small head
  dimensions and short sequences.
- Self-attention on `[noisy | clean]` samples with equal halves (`2n` tokens).
- No FP8 (as FlashAttention-2).

## License

MIT (see `LICENSE`). The CUDA kernels are derived from FlashAttention-2 (BSD 3-Clause) and use CUTLASS (BSD 3-Clause);
see `NOTICE`.
