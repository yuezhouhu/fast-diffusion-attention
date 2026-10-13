# CLAUDE.md

Hopper (sm90a) attention kernels for block diffusion LM training, derived from the FlashAttention-3 `hopper/` kernels
(CUTLASS / CuTe). Python package `fast_diffusion_attention`, CUDA sources in `csrc/fda`, CUTLASS submodule in
`csrc/cutlass`.

## Commands

```bash
git submodule update --init csrc/cutlass
# Full build (50 kernel files, ~6 min with 16 jobs); builds the extension in place
MAX_JOBS=16 NVCC_THREADS=1 python setup.py build_ext --inplace
# Development build: hdim 128, BF16, no FP8 / softcap (~2 min)
FDA_HDIMS=128 FDA_DISABLE_FP16=1 FDA_DISABLE_FP8=1 FDA_DISABLE_SOFTCAP=1 MAX_JOBS=4 python setup.py build_ext --inplace
FDA_RESOURCE_USAGE=1 ...   # ptxas register / spill report per kernel (look for spills in the hot kernels)

python -m pytest tests/                                  # full build (python -m: in-place build importable)
FDA_TEST_HDIMS=128 FDA_TEST_DISABLE_FP16=1 FDA_TEST_DISABLE_FP8=1 FDA_TEST_DISABLE_SOFTCAP=1 python -m pytest tests/  # dev build

python benchmarks/benchmark_diffusion.py [--block-sizes 4 64] [--dtype fp8] [--varlen [--mixed]] [--no-bwd]
python benchmarks/plot_figures.py   # figures of the README / docs (assets/*.png, light + dark) from docs/performance.md
```

The measurements are in `docs/performance.md` (the README has the summary and the charts, `docs/blog.md` the
blog post): after new measurements, update its tables and regenerate the figures (needs matplotlib; uses the Inter
font if installed, `--font-dir`).

The GPU of the development machine (H800 PCIe, 350 W cap) changes its clock with the power draw: compare variants
in interleaved rounds (the benchmark does), never from separate runs. For kernel-level times use
`nsys profile` + `nsys stats --report cuda_gpu_kern_sum` (ncu is not permitted on that machine).

## Layout and semantics

- A sample is `[noisy n | clean n]` (length `2n`, so seqlen / every varlen length must be even). Blocks of
  `block_size` in each half. Noisy block: itself + previous clean blocks. Clean block: itself + previous clean blocks.
  Fully bidirectional within a block. Reference: `fast_diffusion_attention/reference.py` (`diffusion_mask`,
  `attention_ref`).
- `csrc/fda/diffusion.h` holds all the mask geometry: `DiffusionSeqlenInfo` (offsets of the halves, padded
  accumulator offsets), `DiffusionGeometry` (visible key / query intervals), `DiffusionFwdTiles` (key tiles of a
  query tile), `DiffusionBwdTiles` (query tiles of a key tile), `DiffusionMask` (per-element masking).
- Tile coordinates are 4-tuples `(m_block, head, batch, half)`, half 0 = noisy, 1 = clean. Each half is tiled
  separately (tiles never straddle the two halves).
- FP32 accumulators (dQaccum, LSE_log2, dPsum; dKaccum / dVaccum for GQA) use a padded layout where every half of
  every sample starts at a multiple of the tile size: `offset_padded<kBlock>(bidb, half)`. Their sizes are computed in
  `flash_api.cpp` (`total_q_padded`, `total_k_padded`).

## Invariants that are easy to break

- Backward `kBlockM` / `kBlockN` per head dimension are hard-coded in three places that must agree:
  `run_mha_bwd_hdim*` in `flash_bwd_launch_template.h`, `mha_bwd` in `flash_api.cpp`, `_bwd_block_mn` in
  `interface.py` (fake op for `torch.compile`).
- When `64 % block_size == 0` (and backward `kBlockM` is 64 or 128), the noisy-noisy block-diagonal part of the
  backward runs in `FlashAttnBwdNoisyDiag` (`flash_bwd_noisy_diag_kernel.h`) and the main kernel skips the noisy
  key tiles (`noisy_key_tiles = false`). That kernel writes the *initial* value of dQaccum of the noisy rows in the
  exact layout the main kernel's dQ MMA fragments + R2S copy produce (it is templated on the main kernel's
  `TiledMmadQ`), and complete dK / dV of the noisy keys. Consequently the preprocess kernel skips zeroing dQaccum of
  the noisy half and the GQA dK / dV postprocess skips the noisy half.
- The forward uses a half-width (64-key) diagonal tile for noisy query tiles when `64 % block_size == 0`
  (`HalfDiag` in `mainloop_fwd_sm90_tma_gmma_ws.hpp`, 128 x 128 BF16 / FP16 tiles only).
- Forward tile sizes can depend on the sequence length (`tile_size_fwd_sm90(..., long_seq)` and
  `fwd_long_seq_threshold` in `tile_size.h`, dispatched in `run_mha_fwd_`): hdim 64 FP16 / BF16 uses 128 x 128 for
  halves <= 2048 tokens (192 x 128 above), FP8 hdim 128 uses 128 x 128 for halves <= 4096 tokens (128 x 224 above).
  Tests must cover both sides of the threshold (`test_long_sequence_tiles*`).
- Deterministic backward orders the dQ accumulation with semaphores (`dq_rank` in the backward mainloop) and the GQA
  dK / dV accumulation in `epilogue_bwd.hpp`; there is no deterministic variant for hdim 256 (`mha_bwd` raises).
- Avoid register spills in the consumer warpgroups of the main kernels (check with `FDA_RESOURCE_USAGE=1`); a spill
  in the backward hot loop costs ~10%. Register split (2 MMA warpgroups): 24 for the producer / 240 for the MMA
  warpgroups, except 40 / 232 for the varlen kernels (`LargeLoadRegs`; not FP8 forward, not softcap backward hdim
  96 / 128). Zero spills is not the goal: 40 / 232 removes the producer spills of the non-varlen kernels but makes them
  2-5% slower (measured). Decide splits with A/B timings (`ab_bench`-style, ratio to FA3 in the same process), not
  with spill counts alone; re-check after any change to the tile enumeration in `diffusion.h`.

## Tests

`tests/test_diffusion_attn.py`: errors vs a float32 reference must be <= 2x those of a PyTorch implementation in the
same precision (FlashAttention methodology), plus bias / scale-error tests (`test_no_systematic_error`), FP8 tests
following FA3 (P quantized to FP8 in the PyTorch baseline), bitwise determinism, varlen == batched, and
`test_accuracy_matches_fa3` (with `block_size = 1` the clean half is causal attention: errors must match FA3's).
