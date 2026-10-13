# CLAUDE.md

Ampere / Ada (sm80, sm86, sm89) attention kernels for block diffusion LM training, derived from the FlashAttention-2
kernels (`csrc/flash_attn` of flash-attention 2.8, CUTLASS / CuTe, `mma.sync` + `cp.async`). Branch `fa2` of
fast-diffusion-attention (`main` has the FlashAttention-3 generation for Hopper). Python package
`fast_diffusion_attention` (FA2-style interface in `diffusion_attn_interface.py`), extension module `fda_2_cuda`,
CUDA sources in `csrc/fda`, CUTLASS submodule in `csrc/cutlass` (same commit as FlashAttention-2).

## Commands

```bash
git submodule update --init csrc/cutlass
# Full build (36 kernel files: fwd, fwd_split, bwd; ~1.5 min with 36 jobs); builds the extension in place
MAX_JOBS=36 NVCC_THREADS=2 python setup.py build_ext --inplace
# Development build: hdim 128, BF16 only (~1.5 min)
FDA_HDIMS=128 FDA_DISABLE_FP16=1 MAX_JOBS=8 python setup.py build_ext --inplace
FDA_RESOURCE_USAGE=1 ...   # ptxas register / spill report per kernel
FDA_CUDA_ARCHS="80;86;89"  # extra SASS (default "80": the sm80 SASS runs on all sm8x GPUs, as FlashAttention-2)

python -m pytest tests/ -n 4                         # full build (FDA_TEST_GPUS=0,1,2,3 spreads the workers; it
                                                     # replaces CUDA_VISIBLE_DEVICES: machine indices)
FDA_TEST_HDIMS=128 FDA_TEST_DISABLE_FP16=1 python -m pytest tests/   # dev build
FDA_EMULATE_ARCH=86 python -m pytest tests/          # the kernels that sm86 / sm89 GPUs would use, on an A100

# Defaults: FlashAttention-2's benchmark shapes B x L (MHA, hidden 2048: nheads = 2048 // headdim, 16K tokens,
# L = 512-16K) as B samples of 2L ([noisy L | clean L]) vs FlashAttention-2 causal on 2B x L
python benchmarks/benchmark_diffusion.py [--headdim 128] [--block-sizes 4 64] [--heads 32 --kv-heads 8] [--dropout 0.1]
    [--varlen [--mixed]] [--deterministic] [--no-bwd] --min-ms 300
python benchmarks/plot_figures.py   # figures of the README / docs (assets/*.png, light + dark) from docs/performance.md
# Tuning (FlashAttention-2 workflow, automated): the candidates of csrc/fda/src/tuning.h (FlashAttention-2's search
# space, every configuration of its launch templates: 19 forward, 25 backward; plus 9 backward configurations of ours)
# vs the dispatch and FlashAttention-2, on its shapes, FP16. One generated file per candidate (csrc/fda/src/tuning/,
# not committed): ~10 min with 80 jobs (FP16 + BF16).
FDA_TUNING=1 FDA_DISABLE_SOFTCAP=1 MAX_JOBS=80 NVCC_THREADS=1 python setup.py build_ext --inplace
# 1. Every candidate and the dispatch, on the GPU of every class (sm80: A800, sm86: RTX A6000, sm89: L40S), with and
#    without dropout (--dropout 0.1); ~15 min per head dimension and direction
python benchmarks/tune_configs.py --headdim 64 --direction fwd --min-ms 300 > a6000/fwd_hd64.txt
# 2. Decision (the rules of the dispatch): the dispatch unless a candidate is >= 3% faster on the class (geometric
#    mean; with several GPUs per class every GPU weighs the same and none may be more than 1% slower); a split by
#    sequence length only if >= 3% faster than the best single choice
python benchmarks/select_configs.py a6000/fwd_hd64.txt
# 3. Margins within 1.5% of the threshold: head to head with 9 rounds (the sweeps have ~2% of noise)
python benchmarks/tune_configs.py --headdim 64 --direction fwd --min-ms 300 --candidates 5 6 --rounds 9
```

The measurements are in `docs/performance.md`, the dispatch and tuning notes in `docs/tuning.md` (the README has the
summary and the charts): after new measurements, update the tables and regenerate the figures (needs matplotlib; uses
the Inter font if installed, `--font-dir`).

Experiment knobs (environment, read at run time): `FDA_FWD_NUM_SPLITS` (number of splits of the split-KV forward,
tests), `FDA_BWD_HEADS_PER_CTA` (GQA grouping of the backward: the largest divisor of h / h_k not above the value),
`FDA_SCHED_HEADS_FWD` / `FDA_SCHED_HEADS_BWD` ((batch, head) pairs per group of the thread block order),
`FDA_DET_WAVES` (thread blocks of the deterministic backward, in waves), `FDA_TUNE_FWD` / `FDA_TUNE_BWD` (tuning
builds), `FDA_EMULATE_ARCH` (sm86 / sm89 dispatch on an A100). For
compile-time variants, A/B them in one process: build both into one extension behind a template flag selected by an
environment variable (e.g. reuse the `Return_softmax` slot of the forward kernel with dropout disabled), never compare
separate builds from separate runs.

GPUs 4-6 of the development machine are reserved for other work: use `CUDA_VISIBLE_DEVICES` in 0-3. The GPU (A100
PCIe, 300 W cap) changes its clock with the power draw and FlashAttention-2's own times vary by up to 4% between
runs: compare variants in interleaved rounds in one process (the benchmarks do), never from separate runs. Profiling:
`nsys profile` + `nsys stats --report cuda_gpu_kern_sum`; ncu needs the performance counters, i.e. sudo (allowed on the
development machine and on b2).

sm89 machine (`ssh b2`, 8x L40S, NFS-shared home; tests, benchmarks, and the sm89 tuning):
only GPUs 0-3 (set `CUDA_DEVICE_ORDER=PCI_BUS_ID`); build in a
clone on the local disk (`/data/local/<user>`, with `csrc/cutlass` linked to the NFS checkout), not in the shared
repo directory. The L40S clock follows the power draw within tens of milliseconds: always time with `--min-ms 300`
(tune_configs.py, benchmark_diffusion.py); without it, tuning sweeps contradict each other by up to 20%. Compare
candidates within one run (the ratio vs FlashAttention-2 still drifts by a few % between runs). Logins can take 1-2
minutes: give ssh commands long timeouts and run long jobs with nohup.

sm86 machine (RTX A6000, 300 W cap; access: ask the user): only GPU 7 (`CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=7`, and `FDA_TEST_GPUS=7` for the tests: the other GPUs run other jobs), environment in
`/tmp/<user>-fda` (venv with the same torch and flash-attn builds as here). No performance counters there (no sudo):
nsys only. Power-capped like the L40S: time with `--min-ms 300`.

sm80 tuning machine (A800 80GB PCIe, one GPU, rented; access: ask the user, never store its credentials): work under
`/root/autodl-tmp/fda` (venv with the same torch, CUDA 13.0 in `/usr/local/cuda`). Uploads through its ssh gateway
run at ~50 KB/s: build there from a source tarball rather than uploading tuning builds. The container is limited to
120 GB of memory (`MAX_JOBS=24` for tuning builds; more are OOM-killed). Its glibc is 2.35: an extension built on a
newer system must not need glibc >= 2.36 (`std::atoi` / `strtol` compiled with glibc 2.38 headers reference
`__isoc23_strtol`: environment variables are parsed with `parse_env_int`, hardware_info.h).

## Layout and semantics

- A sample is `[noisy n | clean n]` (length `2n`, so seqlen / every varlen length must be even). Blocks of
  `block_size` in each half. Noisy block: itself + previous clean blocks. Clean block: itself + previous clean blocks.
  Fully bidirectional within a block. Reference: `fast_diffusion_attention/reference.py`.
- `csrc/fda/src/diffusion.h` holds all the diffusion-specific geometry: `DiffusionBlockInfo` (FA2's `BlockInfo`:
  offsets of the halves, padded accumulator rows), `DiffusionGeometry` (visible key intervals), `DiffusionFwdTiles`
  (key tiles of a query tile, masked ones first), `DiffusionBwdTiles` (query tiles of a key tile, masked ones first),
  `apply_diffusion_mask` (FA2 accumulator layout).
- The kernels are FA2's `compute_attn_1rowblock` (and its split-KV variant) / `compute_dq_dk_dv_1colblock`
  (seqk-parallel) with the tile loops driven by these structs. Each half is tiled separately (tiles never straddle the
  two halves). Grids are FA2's on the equivalent causal problem (the 2b halves of length n): forward
  `(2 * num_m_blocks, b, h)`, backward `(2 * num_n_blocks, b, h / bwd_heads_per_cta)`. Block order (`tile_of_block`):
  groups of `sched_heads` (batch, head) pairs of about 4 waves of thread blocks (2 for backward head dimensions > 128;
  `get_sched_heads`, hardware_info.h), the most expensive tiles of the group first (forward: the last query tiles of
  both halves; backward: the clean key tiles). The backward groups also keep the Q / dO / dQaccum of their heads within
  twice the L2. Measured against FA2's order (the first release): A800 +0.7%, RTX A6000 +2.7%, L40S +2.8% (geometric
  means; up to +23% for short sequences, worst -9%: forward hdim 192 / 256 at 512 tokens). The deterministic backward
  keeps FA2's assignment (thread block `x` of a (batch, head) processes the key tiles `x, x + nsplits, ...` of the
  enumeration, the clean key tiles first).
- FP32 accumulators of the backward (dQaccum, LSE copy, dPsum) use a padded row layout where every half of every
  sample is followed by `kAccumPad` (128) rows: `DiffusionBlockInfo::padded_row`, `total_padded = total + 256 b`
  (`flash_api.cpp`, and the fake ops of `diffusion_attn_interface.py`).

## Invariants that are easy to break

- dQaccum is written by FA2's atomicAdd layout (`GmemTiledCopydQaccumAtomicAdd` over the `TiledMmadQ` fragments),
  i.e. in a permuted order that only the kernels with the same traits read back: the preprocess, main and convert
  kernels of a backward launch must use the same `Kernel_traits` (`run_flash_bwd_seqk_parallel` does).
- `kAccumPad` must be at least the largest backward `kBlockM` (static_assert in the backward kernel).
- Dropout: the coordinates of the 16 x 32 random blocks put the clean half at row / column `seqlen_half_rounded =
  round_up(max n, 128)`, in the forward and backward (and in the returned `S_dmask`, `(b, h, 2 hr, 2 hr)`).
- GQA in the backward: a thread block sums the dK / dV of `bwd_heads_per_cta` = k query heads of a KV head
  (`Is_grouped` kernels, only without dropout and softcapping; k = 1 in deterministic mode). k = h / h_k writes the
  final dK / dV; 1 < k < h / h_k partial sums in the input dtype (dk_expanded of h / k heads, summed afterwards); k = 1
  is FA2's dk_expanded. The choice (`bwd_heads_per_cta`, flash_api.cpp): the largest divisor of h / h_k up to a
  maximum per head dimension and kernel class, with the data of the k heads (16 n d bytes each) within 0.3x the L2 and
  at least 2 waves of thread blocks. The iterations run over (query tile, head), the heads inner: head by head, the
  thread blocks of a KV head drift apart and thrash the L2 (1-15% slower, up to 40%). Partial sums round sums of k
  heads: where the per-head gradients are exact (n = 1: all probabilities are 1) a forced partial grouping is 1 ulp
  off where FA2 is exact (the tests with `FDA_BWD_HEADS_PER_CTA=2` fail there; the choice never groups such problems).
- The grouped and ungrouped backward kernels are as register-bound as the forward: an extra compile-time branch
  elsewhere in the kernel made the hdim 256 sm80 grouped kernel 10% slower on A800. A/B every backward kernel change
  with and without GQA grouping, on all three GPUs.
- Split-KV forward (FA2's, batched without dropout): `set_params_splitkv` (flash_api.cpp) computes FA2's
  `num_splits_heuristic` with the `block_n` of `run_mha_fwd_splitkv_dispatch`: they must agree. Partial results in
  `(num_splits, b, h, seqlen[, d_rounded])` FP32 buffers; the combine kernel assumes the batched LSE layout
  `(b, h, seqlen)`. Only the 64-row split kernels exist (no `kDiagSubtiles`); empty splits write 0 / -inf.
- The forward step lambda must stay force-inlined (`__attribute__((always_inline))`): otherwise the accumulators it
  captures by reference go to local memory (multi-KB stack frames, check with `FDA_RESOURCE_USAGE=1`).
- Register pressure decides the speed of the forward, and ptxas reacts chaotically to small changes: always check
  `FDA_RESOURCE_USAGE=1` (spills, and the register count vs occupancy) and A/B-time every kernel change.
  - 128-row tiles (2 thread blocks per SM at <= 255 registers): spills of loop-carried values (e.g. the softmax row
    sums) in the unmasked iterations cost 5-9%. Hence the masked iterations in two halves (`kSplitMasked`: scores of
    64 registers per thread or more, hdim >= 64) and the 16-key diagonal sub-tiles with the softmax statistics
    initialized before the loop instead of a special first iteration (`kDiagSubtiles`: 128-row tiles only).
  - 64-row tiles: the occupancy is set by the register count (hdim 32: 128 registers for 4 thread blocks per SM,
    hdim 96: <= 168 for 3). Any extra code path in these kernels (the diagonal sub-tiles, removing the special first
    iteration) can push them over a boundary: -10% to -13% measured. `__launch_bounds__` does not fix it (forcing
    128 registers spills and is still 8% slower; even `__launch_bounds__(128, 1)` changes the allocation), so the
    64-row kernels keep FlashAttention-2's loop structure.
  - Measured and rejected: skipping masked MMA atoms (branches or predicated HMMA) or row groups in the masked tiles
    of the forward (slower), a separate launch for the noisy key tiles of the backward with 64 x 64 tiles (slower,
    latency-bound short thread blocks), precomputed per-row block starts or opaque row offsets for the mask (spills
    move around, no gain), interleaving the query tiles of the two halves in the backward (A100 neutral, L40S / RTX
    A6000 -7% to +12% depending on the length), keeping only the next run start of `DiffusionBwdTiles` in the
    backward loop (fewer static spills, more local memory traffic: check dynamic traffic with ncu, not SASS counts).
- Routing keeps FlashAttention-2's format: the dispatch functions in FlashAttention-2's format, one configuration per
  head dimension x GPU class (sm80, sm86, sm89: `is_sm8x` where sm86 and sm89 agree, `is_sm89` where they do not, or
  FlashAttention-2's shared memory thresholds) x dropout, from the search space of tuning.h on FlashAttention-2's
  shapes: the configuration of the first release (FlashAttention-2's, with four exceptions) unless another is >= 3%
  faster on the GPU of the class (sm80: A800, sm86: RTX A6000, sm89: L40S; `benchmarks/select_configs.py`, which also
  takes several GPUs per class). A configuration may depend on the sequence length only where a split is >= 3%
  faster than the best single configuration (sm80 forward hdim 192, sm86 forward hdim 128 with dropout, sm89 forward
  hdim 96 and hdim 128 with dropout; the thresholds, 1K / 2K tokens per half, are covered by
  `test_long_sequence_tiles`). Margins within 0.5% of 3% get two head-to-head runs. sm89 is tuned on a single card
  (L40S): other sm89 GPUs (RTX 4090, L4) may prefer other configurations. No rule per GPU model: the run-time
  heuristics only read the SM count, the L2 size and the shared memory (GQA grouping, block order, split-KV,
  deterministic splits). The tuning hook is in the generated `run_mha_fwd_` / `run_mha_bwd_` (tuning builds only), not
  in the dispatch.
- Diagonal sub-tiles with dropout use `Dropout::apply_dropout_half` (one half of the 32-column random block): the
  dropout pattern of an element must not depend on the tiling (the backward regenerates it with its own tiles).
- The backward's noisy diagonal key tiles are memory-bound (one masked iteration per thread block); their cost
  depends on the L2 reuse of Q / dO, i.e. on the block order (groups of heads whose data fits in twice the L2).
- Backward configurations with V in registers spill (255 registers); on GPUs with 6 MB of L2 the local memory
  traffic of FlashAttention-2's hdim 192 sm8x configuration makes it slower than FlashAttention-2 on RTX A6000 (ncu on
  b2: 11-15 GB of local traffic per call): the sm8x dispatch uses 32 x 64 tiles without dropout (RTX A6000 0.6-0.8x ->
  1.1-1.4x FlashAttention-2, also the best configuration on L40S); with dropout FA2's configuration (the 32-row tiles
  have no dropout support).
- Tuning: a forward candidate replaces the kernel of `run_mha_fwd_`, not the split-KV kernel, which the batched
  forward picks for small problems: `tune_configs.py` sets `FDA_FWD_NUM_SPLITS=1` with a forward candidate (otherwise
  its correctness check tests the split kernel).
- `S_dmask` (debug output) is written only inside the halves: tiles larger than 128 (e.g. 256 x 64) would otherwise
  overwrite the other half, which starts at row / column `seqlen_half_rounded` (a multiple of 128).
- Deterministic backward: the number of dQ accumulation buffers, `get_deterministic_splits` (hardware_info.h: 2 waves
  of thread blocks, at most the number of key tiles), is used both for the allocation (flash_api.cpp) and the grid
  (flash_bwd_launch_template.h): they must agree. Thread block `x` processes the key tiles `x, x + nsplits, ...` of
  both halves of its (batch, head). 2 waves vs FA2's 1 (`ceil(num_SMs / (b h))`): A800 +9%, L40S +2%, RTX A6000 +0.2%
  (4 waves: A800 +2.5%, L40S -3.9%). Re-tune (all three GPUs, steady state) when changing kernels.

## Tests

`tests/test_diffusion_attn.py`: errors vs a float32 reference must be <= 2x those of a PyTorch implementation in the
same precision (FlashAttention methodology) for out, LSE, dQ, dK, dV; dropout through the S_dmask pattern; bias /
scale-error statistics (`test_no_systematic_error`); `test_accuracy_matches_fa2` (with `block_size = 1` the clean half
is causal attention: errors must match FlashAttention-2's); bitwise determinism; varlen == batched (without split-KV);
`test_split_kv` (forced numbers of splits, including empty splits). `FDA_FWD_NUM_SPLITS=<k>` forces the split-KV forward
in every batched test without dropout. GQA: `test_gqa_fa3` / `test_gqa_fa3_varlen` (FlashAttention-3's GQA cases:
6 query heads, 6 / 2 / 1 KV heads, its sequence lengths and batch sizes, softcapping), `test_gqa_grouped` (the grouped
backward forced on some of them: on RTX A6000 the default choice rarely groups); `FDA_BWD_HEADS_PER_CTA=<k>` forces the
grouping in every test (failures with partial grouping at n = 1 are expected, see above). Large problems: the float32
references are computed sample by sample beyond `_REF_CHUNK_ELEMENTS`, and the `large_memory` fixture runs the tests
of >= 2K tokens one at a time per GPU (46-48 GB GPUs with several workers per GPU).
