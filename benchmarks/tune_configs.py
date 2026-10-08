# Copyright (c) 2026, fast-diffusion-attention authors.
"""Measures the candidate kernel configurations of csrc/fda/src/tuning.h on the current GPU (tuning builds).

    FDA_TUNING=1 FDA_HDIMS=128 FDA_DISABLE_FP16=1 FDA_DISABLE_DROPOUT=1 FDA_DISABLE_SOFTCAP=1 \
        python setup.py build_ext --inplace
    python benchmarks/tune_configs.py --headdim 128 --direction bwd [--candidates 0 1 4] [--seqlens 2048 8192]

The shapes are those of FlashAttention-2's benchmarks (benchmarks/benchmark_flash_attention.py), FP16, MHA, hidden
size 2048 (nheads = 2048 // headdim): for every shape B x L of FlashAttention-2 (16K tokens: B = 16K / L, L = 512 -
16K), the diffusion attention on B samples of length 2L ([noisy L | clean L]) against FlashAttention-2 causal attention
on 2B sequences of length L (the same visible pairs; each half has the tile structure of FlashAttention-2's causal
attention on L tokens). Block sizes 4 (divides 16: diagonal sub-tiles of the 128-row forward tiles) and 64 (a multiple
of the tiles).

The configurations with dropout are tuned separately, as in FlashAttention-2 (build without FDA_DISABLE_DROPOUT=1,
--dropout 0.1: FlashAttention-2 also runs with dropout).

For every shape, the time of every candidate is measured in interleaved rounds, each measurement paired with one of
FlashAttention-2 causal attention on the two halves of the samples right before it (same visible pairs, see
benchmark_diffusion.py): the median over the rounds of the throughput ratios of the pairs is printed (> 1: faster).
Pairing cancels the slow drifts of the clock of power-capped GPUs (L40S: +-5% between unpaired measurements of the
same kernel). On GPUs whose clock follows the power draw within tens of milliseconds (L40S, 350 W: 1.3-1.7 GHz
instead of 2.5 GHz under these kernels), use --min-ms 300: every measurement then runs at the steady clock of its
kernel; short measurements favor the kernels that draw less power and contradict each other between runs. The default
configuration of the dispatch is "dispatch".
Every candidate is first checked against the reference implementation. The best candidates are then hard-coded in
the dispatch (flash_fwd_launch_template.h / flash_bwd_launch_template.h), as in the FlashAttention-2 workflow.
"""

import argparse
import os
import random
import statistics
import sys
from pathlib import Path

import torch

try:
    import fast_diffusion_attention  # noqa: F401
except ImportError:  # extension built in place (python setup.py build_ext --inplace)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fast_diffusion_attention import diffusion_attn_func
from fast_diffusion_attention.reference import attention_ref, num_visible_pairs

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None


MIN_MS = 0.0  # --min-ms


def cuda_time(fn, iters=10, warmup=2):
    """Time of one call (ms). With --min-ms: warm up for about MIN_MS and time a window of about MIN_MS, i.e. measure
    at the steady clock of the kernel under the power cap."""
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    if MIN_MS > 0:
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        warmup = iters = max(1, int(MIN_MS / max(start.elapsed_time(end), 1e-3)))
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def set_candidate(direction, i):
    """Candidate i of the direction (None: the dispatch). The forward candidates are configurations of the kernel
    without split-KV (FlashAttention-2's run_mha_fwd_): selecting one also disables the split-KV forward, which the
    batched forward chooses when its query tiles do not fill the GPU (e.g. the small problem of check_candidate)."""
    var = "FDA_TUNE_FWD" if direction == "fwd" else "FDA_TUNE_BWD"
    if i is None:
        os.environ.pop(var, None)
        if direction == "fwd":
            os.environ.pop("FDA_FWD_NUM_SPLITS", None)
    else:
        os.environ[var] = str(i)
        if direction == "fwd":
            os.environ["FDA_FWD_NUM_SPLITS"] = "1"


def dropout_keep_mask(S_dmask, n):
    """(batch, nheads, 2 hr, 2 hr) S_dmask of the kernel -> (batch, nheads, 2n, 2n) dropout pattern (True = kept): the
    clean half starts at hr (as in tests/test_diffusion_attn.py)."""
    hr = S_dmask.shape[-1] // 2
    idx = torch.cat([torch.arange(n), hr + torch.arange(n)]).to(S_dmask.device)
    return S_dmask[:, :, idx][:, :, :, idx] >= 0


def check_candidate(direction, i, d, dtype, dropout_p=0.0):
    """Correctness of a candidate on a small problem (errors at most 2x those of PyTorch in the same precision). With
    dropout, the reference uses the dropout pattern returned by the kernel (S_dmask), and the kernel without S_dmask
    (the one timed) must give the same output."""
    set_candidate(direction, i)
    try:
        n = 333
        torch.random.manual_seed(0)
        q, k, v = [torch.randn(2, 2 * n, h, d, device="cuda", dtype=dtype, requires_grad=True) for h in (4, 2, 2)]
        g = torch.randn_like(q)
        ref_kw = {}
        if dropout_p > 0:
            torch.random.manual_seed(1)
            out_plain = diffusion_attn_func(q, k, v, 7, dropout_p=dropout_p)
            torch.random.manual_seed(1)
            out, _, S_dmask = diffusion_attn_func(q, k, v, 7, dropout_p=dropout_p, return_attn_probs=True)
            if not torch.equal(out_plain, out):
                return "the outputs with and without S_dmask differ"
            ref_kw = dict(dropout_p=dropout_p, dropout_mask=dropout_keep_mask(S_dmask, n))
        else:
            out = diffusion_attn_func(q, k, v, 7)
        out_ref, _, _ = attention_ref(q, k, v, 7, **ref_kw)
        out_pt, _, _ = attention_ref(q, k, v, 7, upcast=False, reorder_ops=True, **ref_kw)
        res = [(out, out_ref, out_pt)]
        if direction == "bwd":
            grads = torch.autograd.grad(out, (q, k, v), g)
            grads_ref = torch.autograd.grad(out_ref, (q, k, v), g)
            grads_pt = torch.autograd.grad(out_pt, (q, k, v), g)
            res += list(zip(grads, grads_ref, grads_pt))
        for x, r, p in res:
            err, err_pt = (x - r).abs().max().item(), (p - r).abs().max().item()
            if not err <= 2 * err_pt + 1e-4:
                return f"wrong results ({err} vs {err_pt})"
        return None
    except RuntimeError as e:
        return str(e).splitlines()[0]
    finally:
        set_candidate(direction, None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument("--direction", choices=["fwd", "bwd"], default="fwd")
    parser.add_argument("--dtype", default="fp16", choices=["bf16", "fp16"], help="FlashAttention-2's benchmark: FP16")
    parser.add_argument("--candidates", type=int, nargs="*", default=None)
    parser.add_argument("--seqlens", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384],
                        help="FlashAttention-2's sequence lengths L: samples of length 2L")
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[4, 64])
    parser.add_argument("--batch-tokens", type=int, default=16384, help="FlashAttention-2's tokens per batch: B = batch_tokens // L")
    parser.add_argument("--hidden", type=int, default=2048, help="nheads = hidden // headdim (FlashAttention-2)")
    parser.add_argument("--heads", type=int, default=None, help="default: hidden // headdim")
    parser.add_argument("--kv-heads", type=int, default=None, help="default: heads (MHA)")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.0, help="dropout probability (FlashAttention-2 too)")
    parser.add_argument("--min-ms", type=float, default=0.0,
                        help="warm up and measure for at least this long (ms): steady clock on power-capped GPUs")
    parser.add_argument("--no-dispatch", action="store_true", help="do not measure the configuration of the dispatch")
    args = parser.parse_args()
    global MIN_MS
    MIN_MS = args.min_ms
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    D = args.headdim
    H = args.heads if args.heads is not None else args.hidden // D
    Hk = args.kv_heads if args.kv_heads is not None else H
    num_candidates = 19 if args.direction == "fwd" else 34
    candidates = args.candidates if args.candidates is not None else list(range(num_candidates))
    p = args.dropout
    print(f"{torch.cuda.get_device_name()}: {args.direction}, headdim {D}, {args.dtype}, nheads {H}/{Hk}"
          + (f", dropout {p}" if p > 0 else ""))
    valid = []
    for i in candidates:
        err = check_candidate(args.direction, i, D, dtype, p)
        print(f"  candidate {i}: {'ok' if err is None else err}")
        if err is None:
            valid.append(i)
    names = ([] if args.no_dispatch else ["dispatch"]) + [f"c{i}" for i in valid]
    print(f"{'shape':>18s} " + " ".join(f"{n:>8s}" for n in names))
    for L in args.seqlens:
        # FlashAttention-2's shape B x L: B samples [noisy L | clean L] of the diffusion attention, 2B x L of causal
        # attention for FlashAttention-2
        n = L
        B = max(1, args.batch_tokens // L)
        for bs in args.block_sizes:
            q, k, v = [torch.randn(B, 2 * n, h, D, device="cuda", dtype=dtype, requires_grad=True) for h in (H, Hk, Hk)]
            g = torch.randn_like(q)
            fns = {}
            for name, i in ([] if args.no_dispatch else [("dispatch", None)]) + [(f"c{i}", i) for i in valid]:
                if args.direction == "fwd":
                    fn = (lambda: diffusion_attn_func(q, k, v, bs, dropout_p=p))
                else:
                    set_candidate("fwd", None)
                    out = diffusion_attn_func(q, k, v, bs, dropout_p=p)
                    fn = (lambda out=out: torch.autograd.grad(out, (q, k, v), g, retain_graph=True))
                fns[name] = (fn, i)
            base = None
            if flash_attn_func is not None:
                qc, kc, vc = [x.detach().view(2 * B, n, *x.shape[2:]).requires_grad_() for x in (q, k, v)]
                if args.direction == "fwd":
                    base = lambda: flash_attn_func(qc, kc, vc, dropout_p=p, causal=True)
                else:
                    out_fa2 = flash_attn_func(qc, kc, vc, dropout_p=p, causal=True)
                    gc = g.view(2 * B, n, H, D)
                    base = lambda: torch.autograd.grad(out_fa2, (qc, kc, vc), gc, retain_graph=True)
            pairs = B * num_visible_pairs(n, bs)
            pairs_fa2 = B * 2 * (n * (n + 1) // 2)
            # Paired measurements: (reference, candidate) back to back, the reference being FlashAttention-2 (or the
            # dispatch without FlashAttention-2)
            ref = base if base is not None else next(iter(fns.values()))[0]
            ratios = {name: [] for name in fns}
            rng = random.Random(0)
            for _ in range(args.rounds):
                order = list(fns.items())
                rng.shuffle(order)
                for name, (fn, i) in order:
                    set_candidate(args.direction, None)
                    t_ref = cuda_time(ref)
                    set_candidate(args.direction, i)
                    t = cuda_time(fn)
                    ratios[name].append(t_ref / t)
                set_candidate(args.direction, None)
            scale = pairs / pairs_fa2 if base is not None else 1.0
            rel = {name: scale * statistics.median(ratios[name]) for name in fns}
            best = max(rel, key=rel.get)
            row = f"{f'L={L} bs={bs}':>18s} " + " ".join(
                f"{rel[nm]:7.3f}{'*' if nm == best else ' '}" for nm in names)
            print(row, flush=True)
    print("relative throughput vs FlashAttention-2 causal on the halves (* best)" if flash_attn_func is not None
          else "time relative to the dispatch (* best)")


if __name__ == "__main__":
    main()
