# Dispatch and tuning

[README](../README.md) · [Performance](performance.md)

How the `fa2` branch picks a kernel configuration for every head dimension, GPU class and problem shape, and how these
choices were tuned on an A800 (sm80), an RTX A6000 (sm86) and an L40S (sm89).

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
