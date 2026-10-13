# Performance on Hopper

[README](../README.md) · [Blog post](blog.md) · Ampere / Ada GPUs: [`fa2` branch](https://github.com/yuezhouhu/fast-diffusion-attention/blob/fa2/docs/performance.md)

Every measurement of the Hopper kernels (`main` branch) against FlashAttention-3. The charts of the README are drawn
from the tables of this page by `benchmarks/plot_figures.py`.

## Setup

Measured with `benchmarks/benchmark_diffusion.py` on an H800 PCIe (114 SMs, 2.0 TB/s HBM, 350 W power cap) with
CUDA 13.0, PyTorch 2.12 and FlashAttention-3 3.0.0 (flash-attention commit 3451a2a, `hopper/`). Batches of 16K tokens,
32 query heads, 8 KV heads, head dimension 128 unless noted. Times cover the whole forward / backward call (including
the pre- and post-processing kernels of the backward). TFLOPS count only the visible (query, key) pairs:
`4 * pairs * headdim * nheads` for the forward, 2.5x that for the backward.

The baseline is FlashAttention-3 *causal* attention on the two halves of every sample (a batch of `2B` sequences of
length `n`). It has the same number of visible pairs as the diffusion mask (up to the full diagonal blocks:
`n^2 + n * block_size` against `n^2 + n` per sample) and the same tile structure, so the throughputs on the visible
pairs compare the kernels on equal work (causal attention on the `2n` tokens does twice the work). Variants are
measured in interleaved rounds and the medians are reported: the power-capped GPU changes its clock with the load.

<p>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/equal_work_dark.png">
  <img alt="The diffusion mask has the visible pairs of causal attention on the two halves of the sample" src="../assets/equal_work.png">
</picture>
</p>

**Relative throughput**: TFLOPS of the diffusion kernels divided by those of FlashAttention-3 (> 1: faster).
**fwd + bwd**: relative throughput of a training step (forward and backward together).

## Summary

The backward pass, 75-80% of the attention time of a training step, is on par with FlashAttention-3 (0.95-1.18x its
throughput); the forward pass reaches 0.87-1.06x (0.89-0.98x for BF16 with head dimension 128). A whole training step
runs at 0.96-1.10x the throughput of FlashAttention-3 with BF16 and head dimension 128, and at 0.94-1.15x over all the
configurations below. The remaining forward gap comes from the noisy query tiles, which visit one more key tile than
causal attention (their noisy diagonal tile, at half width when the block size divides 64): about 5% more work at
L = 2K and 1% at 16K, plus a few percent of lower efficiency per iteration at long sequences.

## BF16, head dimension 128

<p>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/perf_h800_bf16_dark.png">
  <img alt="Forward, backward and training-step TFLOPS on H800, BF16, head dimension 128" src="../assets/perf_h800_bf16.png">
</picture>
</p>

| | block size | forward ms (TFLOPS) | FA3 forward | rel. throughput | backward ms (TFLOPS) | FA3 backward | rel. throughput | fwd + bwd |
|---|---|---|---|---|---|---|---|---|
| L = 2K, B = 8 | 4 | 0.447 (309) | 0.400 (344) | **0.90** | 1.941 (178) | 1.976 (174) | **1.02** | 1.00 |
|  | 16 | 0.457 (306) | 0.400 (344) | **0.89** | 1.937 (180) | 1.976 (174) | **1.03** | 1.01 |
|  | 64 | 0.447 (327) | 0.400 (344) | **0.95** | 1.842 (198) | 1.976 (174) | **1.14** | 1.10 |
| L = 4K, B = 4 | 4 | 0.742 (371) | 0.684 (402) | **0.92** | 2.814 (245) | 2.840 (242) | **1.01** | 0.99 |
|  | 16 | 0.746 (371) | 0.684 (402) | **0.92** | 2.832 (245) | 2.840 (242) | **1.01** | 0.99 |
|  | 64 | 0.749 (378) | 0.684 (402) | **0.94** | 2.754 (257) | 2.840 (242) | **1.06** | 1.04 |
| L = 8K, B = 2 | 4 | 1.341 (410) | 1.270 (433) | **0.95** | 4.616 (298) | 4.515 (305) | **0.98** | 0.97 |
|  | 16 | 1.340 (412) | 1.270 (433) | **0.95** | 4.604 (300) | 4.515 (305) | **0.98** | 0.98 |
|  | 64 | 1.348 (414) | 1.270 (433) | **0.96** | 4.499 (310) | 4.515 (305) | **1.02** | 1.00 |
| L = 16K, B = 1 | 4 | 2.673 (412) | 2.589 (425) | **0.97** | 8.180 (336) | 7.853 (350) | **0.96** | 0.96 |
|  | 16 | 2.646 (416) | 2.589 (425) | **0.98** | 8.087 (341) | 7.853 (350) | **0.97** | 0.97 |
|  | 64 | 2.699 (411) | 2.589 (425) | **0.97** | 8.042 (344) | 7.853 (350) | **0.98** | 0.98 |

## Other configurations

<p>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/perf_h800_more_dark.png">
  <img alt="TFLOPS on H800 for FP8, varlen batches and head dimensions 64 and 256" src="../assets/perf_h800_more.png">
</picture>
</p>

### FP8 forward, head dimension 128

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

### Varlen, samples of random lengths (BF16, head dimension 128)

| | block size | forward ms (TFLOPS) | FA3 forward | rel. throughput | backward ms (TFLOPS) | FA3 backward | rel. throughput | fwd + bwd |
|---|---|---|---|---|---|---|---|---|
| lengths in [256, 2048] | 4 | 0.443 (216) | 0.421 (226) | **0.96** | 1.695 (141) | 1.779 (134) | **1.05** | 1.03 |
|  | 64 | 0.442 (233) | 0.421 (226) | **1.03** | 1.631 (158) | 1.779 (134) | **1.18** | 1.15 |
| lengths in [512, 4096] | 4 | 0.648 (311) | 0.632 (318) | **0.98** | 2.298 (219) | 2.256 (223) | **0.98** | 0.98 |
|  | 64 | 0.646 (323) | 0.632 (318) | **1.02** | 2.247 (232) | 2.256 (223) | **1.04** | 1.03 |
| lengths in [1024, 8192] | 4 | 0.520 (324) | 0.496 (339) | **0.96** | 1.780 (236) | 1.698 (248) | **0.95** | 0.95 |
|  | 64 | 0.519 (333) | 0.496 (339) | **0.98** | 1.750 (247) | 1.698 (248) | **1.00** | 0.99 |
| lengths in [2048, 16384] | 4 | 1.627 (397) | 1.582 (408) | **0.97** | 5.130 (315) | 4.889 (330) | **0.95** | 0.96 |
|  | 64 | 1.667 (391) | 1.582 (408) | **0.96** | 5.105 (319) | 4.889 (330) | **0.97** | 0.96 |

### Head dimensions 64 and 256 (BF16)

| | block size | forward ms (TFLOPS) | FA3 forward | rel. throughput | backward ms (TFLOPS) | FA3 backward | rel. throughput | fwd + bwd |
|---|---|---|---|---|---|---|---|---|
| d = 64, L = 2K, B = 8 | 4 | 0.337 (205) | 0.341 (201) | **1.02** | 1.116 (154) | 1.127 (153) | **1.01** | 1.01 |
|  | 64 | 0.350 (208) | 0.341 (201) | **1.03** | 1.137 (161) | 1.127 (153) | **1.05** | 1.05 |
| d = 64, L = 4K, B = 4 | 4 | 0.576 (239) | 0.502 (274) | **0.87** | 1.605 (214) | 1.539 (223) | **0.96** | 0.94 |
|  | 64 | 0.594 (238) | 0.502 (274) | **0.87** | 1.640 (216) | 1.539 (223) | **0.97** | 0.94 |
| d = 64, L = 8K, B = 2 | 4 | 1.020 (270) | 0.915 (301) | **0.90** | 2.592 (265) | 2.463 (279) | **0.95** | 0.94 |
|  | 64 | 1.036 (269) | 0.915 (301) | **0.89** | 2.599 (269) | 2.463 (279) | **0.96** | 0.94 |
| d = 64, L = 16K, B = 1 | 4 | 1.782 (309) | 1.693 (325) | **0.95** | 4.544 (303) | 4.379 (314) | **0.96** | 0.96 |
|  | 64 | 1.785 (310) | 1.693 (325) | **0.95** | 4.537 (305) | 4.379 (314) | **0.97** | 0.97 |
| d = 256, L = 4K, B = 4 | 4 | 1.390 (396) | 1.297 (424) | **0.93** | 6.759 (204) | 6.672 (206) | **0.99** | 0.98 |
|  | 64 | 1.392 (407) | 1.297 (424) | **0.96** | 6.628 (214) | 6.672 (206) | **1.04** | 1.02 |
| d = 256, L = 16K, B = 1 | 4 | 5.553 (396) | 5.832 (377) | **1.05** | 22.022 (250) | 21.643 (254) | **0.98** | 1.00 |
|  | 64 | 5.555 (399) | 5.832 (377) | **1.06** | 21.445 (258) | 21.643 (254) | **1.02** | 1.03 |

### Large blocks (BF16, head dimension 128)

| | block size | forward ms (TFLOPS) | FA3 forward | rel. throughput | backward ms (TFLOPS) | FA3 backward | rel. throughput | fwd + bwd |
|---|---|---|---|---|---|---|---|---|
| L = 4K, B = 4 | 256 | 0.759 (407) | 0.684 (402) | **1.01** | 3.038 (254) | 2.877 (239) | **1.06** | 1.05 |
|  | 1024 | 0.989 (417) | 0.684 (402) | **1.04** | 3.688 (280) | 2.877 (239) | **1.17** | 1.14 |
| L = 16K, B = 1 | 256 | 2.925 (388) | 2.802 (392) | **0.99** | 9.148 (310) | 8.670 (317) | **0.98** | 0.98 |
|  | 1024 | 3.328 (372) | 2.802 (392) | **0.95** | 9.914 (312) | 8.670 (317) | **0.98** | 0.97 |

## Ampere and Ada GPUs (`fa2` branch)

The [`fa2` branch](https://github.com/yuezhouhu/fast-diffusion-attention/tree/fa2) has the FlashAttention-2 generation
of the kernels, for sm80 / sm86 / sm89 GPUs. Throughput of a training step (forward + backward, BF16) relative to
FlashAttention-2 on FlashAttention-2's benchmark shapes (MHA with hidden size 2048, L = 512 ... 16K, block sizes 4, 16
and 64): range and (geometric mean). Every measurement:
[`docs/performance.md` of the `fa2` branch](https://github.com/yuezhouhu/fast-diffusion-attention/blob/fa2/docs/performance.md).

<p>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/perf_ampere_ada_dark.png">
  <img alt="Training-step throughput relative to FlashAttention-2 on A800, RTX A6000 and L40S" src="../assets/perf_ampere_ada.png">
</picture>
</p>

| head dimension (heads) | A800 | RTX A6000 | L40S |
|---|---|---|---|
| 32 (64) | 0.89 - 1.00 (**0.94**) | 0.94 - 1.24 (**1.00**) | 0.95 - 1.30 (**1.04**) |
| 64 (32) | 0.86 - 0.99 (**0.95**) | 0.99 - 1.19 (**1.04**) | 0.99 - 1.31 (**1.07**) |
| 96 (21) | 0.88 - 1.04 (**1.00**) | 1.00 - 1.20 (**1.05**) | 1.02 - 1.21 (**1.08**) |
| 128 (16) | 0.87 - 1.02 (**0.97**) | 0.96 - 1.14 (**1.05**) | 1.01 - 1.13 (**1.05**) |
| 192 (10) | 0.93 - 1.11 (**0.98**) | 1.10 - 1.68 (**1.30**) | 1.04 - 1.31 (**1.10**) |
| 256 (8) | 0.89 - 1.05 (**0.98**) | 0.90 - 1.08 (**0.96**) | 1.00 - 1.16 (**1.03**) |
