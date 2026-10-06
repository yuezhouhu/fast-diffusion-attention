# Fast Diffusion Attention

FlashAttention-3-class attention kernels for training block diffusion language models on NVIDIA Hopper GPUs.

Block diffusion training (BD3-LM, SDAR, RCD, ...) runs the model on the concatenation of a noisy copy and a clean copy
of every sequence, with a block-structured attention mask. This library computes that attention with Hopper kernels
derived from [FlashAttention-3](https://github.com/Dao-AILab/flash-attention/tree/main/hopper) (CUTLASS / CuTe, TMA,
WGMMA, warp specialization, persistent scheduling): the mask is never materialized, only the visible part of the
attention matrix is computed, and a training step runs at the throughput of FlashAttention-3 on the same amount of
work (0.96-1.10x, see [Performance](#performance)).

- Forward and backward in FP16 / BF16 (training), FP8 e4m3 forward (inference)
- Batched interface (pretraining) and varlen interface (packed samples of different lengths, SFT)
- Any block size, head dimensions up to 256, MQA / GQA, softcapping, deterministic backward, `torch.compile`
- Interface modeled after FlashAttention-3 (`flash_attn_func` / `flash_attn_varlen_func`)

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

Requirements: Hopper GPU (sm90: H100 / H800 / H200 / H20), CUDA >= 12.3 (tested with 13.0), PyTorch >= 2.4 (tested
with 2.12), Linux.

```bash
git clone --recursive https://github.com/yuezhouhu/fast-diffusion-attention.git
cd fast-diffusion-attention
pip install --no-build-isolation -v .
```

The full build compiles 50 kernel files (about 7 minutes on 20 cores with `MAX_JOBS=16`). Environment variables to build only
what you need:

| Variable | Effect |
| --- | --- |
| `FDA_HDIMS="64 128"` | head dimensions to build (out of 64 96 128 192 256; a head dimension `d` uses the smallest kernel `>= d`) |
| `FDA_DISABLE_FP16=1` | no FP16 kernels (BF16 only) |
| `FDA_DISABLE_FP8=1` | no FP8 forward kernels |
| `FDA_DISABLE_SOFTCAP=1` | no softcapping kernels |
| `FDA_DISABLE_BACKWARD=1` | forward only |
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

### FP8 forward (inference)

```python
q8, k8, v8 = (x.detach().to(torch.float8_e4m3fn) for x in (q, k, v))
descale = torch.ones(B, H_kv, device="cuda", dtype=torch.float32)   # per (sample, KV head)
out = diffusion_attn_func(q8, k8, v8, block_size=4, q_descale=descale, k_descale=descale, v_descale=descale)  # BF16
```

### API

```python
diffusion_attn_func(q, k, v, block_size, softmax_scale=None, q_descale=None, k_descale=None, v_descale=None,
                    softcap=0.0, deterministic=False, sm_margin=0, return_attn_probs=False)
diffusion_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, block_size,
                           softmax_scale=None, q_descale=None, k_descale=None, v_descale=None,
                           softcap=0.0, deterministic=False, sm_margin=0, return_attn_probs=False)
```

| Argument | |
| --- | --- |
| `q`, `k`, `v` | `(batch, 2n, nheads, headdim)` / `(batch, 2n, nheads_kv, headdim)` (batched), `(total, nheads, headdim)` / `(total, nheads_kv, headdim)` (varlen). FP16, BF16 or `float8_e4m3fn` (forward only). `nheads` must be a multiple of `nheads_kv`; `headdim <= 256`, a multiple of 8 (16 for FP8). The last dimension must be contiguous. |
| `block_size` | size of the diffusion blocks, any positive integer |
| `cu_seqlens_q` | `(batch + 1,)` int32 cumulative sample lengths; every length is even (`2 n_i`). Self-attention only: `cu_seqlens_k` is `None` or equal to `cu_seqlens_q`, `max_seqlen_k` is `None` or equal to `max_seqlen_q` |
| `softmax_scale` | scale of `QK^T`, default `1 / sqrt(headdim)` |
| `q_descale`, `k_descale`, `v_descale` | `(batch, nheads_kv)` float32 descaling factors of FP8 inputs |
| `softcap` | `> 0` enables `softcap * tanh(scores / softcap)` |
| `deterministic` | bitwise reproducible backward (slightly slower) |
| `sm_margin` | number of SMs to leave free, e.g. for communication kernels |
| `return_attn_probs` | also return the log-sum-exp of the scores, `(batch, nheads, 2n)` or `(nheads, total)` |

The output has the shape and dtype of `q` (BF16 for FP8 inputs). Both functions are differentiable (FP16 / BF16) and
work with `torch.compile` (they are registered as custom ops with fake implementations).

## Accuracy

`tests/test_diffusion_attn.py` follows the methodology of the FlashAttention test suites: for the output, the
log-sum-exp and dQ / dK / dV, the error of the kernel with respect to a float32 reference (computed from the same
low-precision inputs) must be at most twice the error of a straightforward PyTorch implementation in the same
low precision, plus a few ulps. It covers both dtypes, head dimensions 64-256 (including 80, 112, 160, 224), block
sizes from 1 to larger than `n`, odd `n`, GQA / MQA, softcapping, varlen batches with very different and zero-length
samples, non-contiguous inputs, custom softmax scales, deterministic mode and `torch.compile`.

Beyond the max-error bound, `test_no_systematic_error` checks over many random problems that the errors have no
systematic component: the relative bias and the relative scale error (least-squares slope of the kernel result against
the reference, minus 1) of the output and of every gradient average to zero within 4 standard errors, and are on
average no larger than twice those of PyTorch in the same precision. FP8 is tested as in FlashAttention-3: the error
must be at most twice that of a BF16 PyTorch implementation that quantizes P to FP8 like the kernel, and the errors
must be unbiased.

`test_accuracy_matches_fa3` compares directly with FlashAttention-3: with `block_size = 1` the clean half of a sample is
causal attention on the clean tokens, and with a zero output gradient on the noisy half its gradients are those of that
causal attention. On this subproblem the errors of the output, the LSE and dQ / dK / dV must match those of
FlashAttention-3 (for all dtypes and head dimensions; in practice the forward results are bitwise identical to FA3's
for most configurations). Deterministic mode is checked to be bitwise reproducible, and the varlen kernels to be
bitwise identical to the batched ones on equal-length samples.

```bash
pytest tests/                               # 1100 tests pass on the full build (+20 skipped)
FDA_TEST_HDIMS=128 pytest tests/            # only the head dimensions that were built
```

## Performance

Measured with `benchmarks/benchmark_diffusion.py` on an H800 PCIe (114 SMs, 2.0 TB/s HBM, 350 W power cap) with
CUDA 13.0, PyTorch 2.12 and FlashAttention-3 3.0.0 (flash-attention commit 3451a2a, `hopper/`). Batches of 16K tokens,
32 query heads, 8 KV heads, head dimension 128 unless noted. Times cover the whole forward / backward call (including
the pre- and post-processing kernels of the backward). TFLOPS count only the visible (query, key) pairs:
`4 * pairs * headdim * nheads` for the forward, 2.5x that for the backward.

The baseline is FlashAttention-3 *causal* attention on the two halves of every sample (a batch of `2B` sequences of
length `n`). It has the same number of visible pairs and the same tile structure as the diffusion mask, so the ratio
of the times is the ratio of the throughputs (causal attention on the `2n` tokens does twice the work). Variants are
measured in interleaved rounds and the medians are reported: the power-capped GPU changes its clock with the load.

**Summary.** Relative throughput = TFLOPS on the visible pairs relative to FlashAttention-3 (> 1: faster). The
backward pass, 75-80% of the attention time of a training step, is on par with FlashAttention-3 (0.95-1.18x its
throughput); the forward pass reaches 0.87-1.06x (0.89-0.98x for BF16 with head dimension 128). A whole training step
(forward + backward, BF16, head dimension 128) runs at 0.96-1.10x the throughput of FlashAttention-3. The remaining
forward gap comes from the noisy query tiles, which visit one more key tile than causal attention (their noisy
diagonal tile, at half width when the block size divides 64): about 5% more work at L = 2K and 1% at 16K, plus a few
percent of lower efficiency per iteration at long sequences.

#### BF16, head dimension 128

| | block size | forward ms (TFLOPS) | FA3 forward | rel. throughput | backward ms (TFLOPS) | FA3 backward | rel. throughput |
|---|---|---|---|---|---|---|---|
| L = 2K, B = 8 | 4 | 0.447 (309) | 0.400 (344) | **0.90** | 1.941 (178) | 1.976 (174) | **1.02** |
|  | 16 | 0.457 (306) | 0.400 (344) | **0.89** | 1.937 (180) | 1.976 (174) | **1.03** |
|  | 64 | 0.447 (327) | 0.400 (344) | **0.95** | 1.842 (198) | 1.976 (174) | **1.14** |
| L = 4K, B = 4 | 4 | 0.742 (371) | 0.684 (402) | **0.92** | 2.814 (245) | 2.840 (242) | **1.01** |
|  | 16 | 0.746 (371) | 0.684 (402) | **0.92** | 2.832 (245) | 2.840 (242) | **1.01** |
|  | 64 | 0.749 (378) | 0.684 (402) | **0.94** | 2.754 (257) | 2.840 (242) | **1.06** |
| L = 8K, B = 2 | 4 | 1.341 (410) | 1.270 (433) | **0.95** | 4.616 (298) | 4.515 (305) | **0.98** |
|  | 16 | 1.340 (412) | 1.270 (433) | **0.95** | 4.604 (300) | 4.515 (305) | **0.98** |
|  | 64 | 1.348 (414) | 1.270 (433) | **0.96** | 4.499 (310) | 4.515 (305) | **1.02** |
| L = 16K, B = 1 | 4 | 2.673 (412) | 2.589 (425) | **0.97** | 8.180 (336) | 7.853 (350) | **0.96** |
|  | 16 | 2.646 (416) | 2.589 (425) | **0.98** | 8.087 (341) | 7.853 (350) | **0.97** |
|  | 64 | 2.699 (411) | 2.589 (425) | **0.97** | 8.042 (344) | 7.853 (350) | **0.98** |

#### FP8 forward, head dimension 128

| | block size | forward ms (TFLOPS) | FA3 forward | rel. throughput |
|---|---|---|---|---|
| L = 2K, B = 8 | 4 | 0.416 (332) | 0.368 (374) | **0.89** |
|  | 64 | 0.411 (356) | 0.368 (374) | **0.95** |
| L = 4K, B = 4 | 4 | 0.650 (424) | 0.602 (457) | **0.93** |
|  | 64 | 0.653 (434) | 0.602 (457) | **0.95** |
| L = 8K, B = 2 | 4 | 1.128 (488) | 1.036 (531) | **0.92** |
|  | 64 | 1.141 (489) | 1.036 (531) | **0.92** |
| L = 16K, B = 1 | 4 | 2.064 (533) | 1.929 (570) | **0.94** |
|  | 64 | 2.011 (551) | 1.929 (570) | **0.97** |

#### Varlen, samples of random lengths (BF16, head dimension 128)

| | block size | forward ms (TFLOPS) | FA3 forward | rel. throughput | backward ms (TFLOPS) | FA3 backward | rel. throughput |
|---|---|---|---|---|---|---|---|
| lengths in [256, 2048] | 4 | 0.443 (216) | 0.421 (226) | **0.96** | 1.695 (141) | 1.779 (134) | **1.05** |
|  | 64 | 0.442 (233) | 0.421 (226) | **1.03** | 1.631 (158) | 1.779 (134) | **1.18** |
| lengths in [512, 4096] | 4 | 0.648 (311) | 0.632 (318) | **0.98** | 2.298 (219) | 2.256 (223) | **0.98** |
|  | 64 | 0.646 (323) | 0.632 (318) | **1.02** | 2.247 (232) | 2.256 (223) | **1.04** |
| lengths in [1024, 8192] | 4 | 0.520 (324) | 0.496 (339) | **0.96** | 1.780 (236) | 1.698 (248) | **0.95** |
|  | 64 | 0.519 (333) | 0.496 (339) | **0.98** | 1.750 (247) | 1.698 (248) | **1.00** |
| lengths in [2048, 16384] | 4 | 1.627 (397) | 1.582 (408) | **0.97** | 5.130 (315) | 4.889 (330) | **0.95** |
|  | 64 | 1.667 (391) | 1.582 (408) | **0.96** | 5.105 (319) | 4.889 (330) | **0.97** |

#### Head dimensions 64 and 256 (BF16)

| | block size | forward ms (TFLOPS) | FA3 forward | rel. throughput | backward ms (TFLOPS) | FA3 backward | rel. throughput |
|---|---|---|---|---|---|---|---|
| d = 64, L = 2K, B = 8 | 4 | 0.337 (205) | 0.341 (201) | **1.02** | 1.116 (154) | 1.127 (153) | **1.01** |
|  | 64 | 0.350 (208) | 0.341 (201) | **1.03** | 1.137 (161) | 1.127 (153) | **1.05** |
| d = 64, L = 4K, B = 4 | 4 | 0.576 (239) | 0.502 (274) | **0.87** | 1.605 (214) | 1.539 (223) | **0.96** |
|  | 64 | 0.594 (238) | 0.502 (274) | **0.87** | 1.640 (216) | 1.539 (223) | **0.97** |
| d = 64, L = 8K, B = 2 | 4 | 1.020 (270) | 0.915 (301) | **0.90** | 2.592 (265) | 2.463 (279) | **0.95** |
|  | 64 | 1.036 (269) | 0.915 (301) | **0.89** | 2.599 (269) | 2.463 (279) | **0.96** |
| d = 64, L = 16K, B = 1 | 4 | 1.782 (309) | 1.693 (325) | **0.95** | 4.544 (303) | 4.379 (314) | **0.96** |
|  | 64 | 1.785 (310) | 1.693 (325) | **0.95** | 4.537 (305) | 4.379 (314) | **0.97** |
| d = 256, L = 4K, B = 4 | 4 | 1.390 (396) | 1.297 (424) | **0.93** | 6.759 (204) | 6.672 (206) | **0.99** |
|  | 64 | 1.392 (407) | 1.297 (424) | **0.96** | 6.628 (214) | 6.672 (206) | **1.04** |
| d = 256, L = 16K, B = 1 | 4 | 5.553 (396) | 5.832 (377) | **1.05** | 22.022 (250) | 21.643 (254) | **0.98** |
|  | 64 | 5.555 (399) | 5.832 (377) | **1.06** | 21.445 (258) | 21.643 (254) | **1.02** |

#### Large blocks (BF16, head dimension 128)

| | block size | forward ms (TFLOPS) | FA3 forward | rel. throughput | backward ms (TFLOPS) | FA3 backward | rel. throughput |
|---|---|---|---|---|---|---|---|
| L = 4K, B = 4 | 256 | 0.759 (407) | 0.684 (402) | **1.01** | 3.038 (254) | 2.877 (239) | **1.06** |
|  | 1024 | 0.989 (417) | 0.684 (402) | **1.04** | 3.688 (280) | 2.877 (239) | **1.17** |
| L = 16K, B = 1 | 256 | 2.925 (388) | 2.802 (392) | **0.99** | 9.148 (310) | 8.670 (317) | **0.98** |
|  | 1024 | 3.328 (372) | 2.802 (392) | **0.95** | 9.914 (312) | 8.670 (317) | **0.98** |

## How it works

The kernels are those of FlashAttention-3 for Hopper with the causal mask replaced by the diffusion mask:

- **Tiles.** The two halves of a sample are tiled separately. A query tile of the clean half visits the clean key tiles
  up to its own (like causal attention); a query tile of the noisy half visits the clean key tiles before its blocks
  and the noisy key tiles of its blocks. Only the partially visible tiles (those crossing a block boundary, at most a
  few per query tile) are masked, with large blocks too. In the backward pass, a key tile visits the query tiles of
  both halves that see it.
- **Forward.** Persistent warp-specialized kernel (1 producer warpgroup issuing TMA loads, 2 consumer warpgroups
  running WGMMA with softmax overlapped with the GEMMs), longest-processing-time-first scheduling across batches,
  heads and halves. When the block size divides 64, the noisy diagonal tile of a 128-row query tile is computed at
  half width (each consumer warpgroup only needs the 64 keys of its own rows). For head dimension 64 (and FP8 with
  head dimension 128) the tile size depends on the length of the halves: smaller tiles for short halves, where the
  fixed cost of the noisy diagonal per query tile matters, the larger tiles of FA3 for long ones.
- **Backward.** The main kernel follows FA3 (one CTA per key tile and head, dQ accumulated in FP32 with TMA reduce-add,
  dK / dV of GQA heads summed in FP32). When the block size divides 64, the block-diagonal noisy-noisy part runs in a
  separate kernel that processes all the query heads of a KV head per tile (pipelined with `cp.async`), writes the
  initial value of dQ of the noisy queries and the complete dK / dV of the noisy keys; the main kernel then only
  iterates over the clean key tiles.
- **Varlen.** Samples are scheduled longest first; per-sample offsets are aligned to tiles in the FP32 accumulators.
- **Determinism.** As in FA3, the dQ (and GQA dK / dV) accumulations are ordered with semaphores.

## Limitations

- Hopper (sm90a) only.
- Self-attention on `[noisy | clean]` samples with equal halves (`2n` tokens).
- No FP8 backward and no deterministic backward for head dimension 256 (as FlashAttention-3), no dropout.

## License

MIT (see `LICENSE`). The CUDA kernels are derived from FlashAttention-3 (BSD 3-Clause) and use CUTLASS (BSD 3-Clause);
see `NOTICE`.
