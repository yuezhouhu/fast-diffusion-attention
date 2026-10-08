# Copyright (c) 2026, fast-diffusion-attention authors.
"""Throughput of the diffusion attention vs FlashAttention-2.

A sample of length 2n is [noisy (n tokens) | clean (n tokens)]. The diffusion mask has about n^2 visible
(query, key) pairs: as many as causal attention on the two halves separately, half of causal attention on the 2n
tokens. TFLOPS are always computed from the visible pairs (4 * pairs * headdim * nheads for the forward, 2.5x that
for the backward), i.e. masked-out work is never counted.

Baselines (if FlashAttention-2 is installed):
  * FA2 causal 2x(n): causal attention on the two halves of length n, as a batch of 2B sequences of length n
                      (same amount of work and same tile structure as the diffusion mask)
  * FA2 causal(L):    causal attention on the 2n tokens of a sample (2x the work of the diffusion mask, for scale)

Times are measured with CUDA events over back-to-back iterations, for the whole forward / backward including all the
auxiliary kernels (backward pre- and post-processing, GQA reduction of dK / dV). The GPU clock varies with the power
draw (e.g. on power-capped PCIe cards), so all the variants of a shape are measured in interleaved rounds and the
median over the rounds is reported.

The default shapes are those of FlashAttention-2's benchmarks (benchmarks/benchmark_flash_attention.py): MHA, hidden
size 2048 (nheads = 2048 // headdim), and for every shape B x L of FlashAttention-2 (16K tokens: B = 16K / L,
L = 512 - 16K) the diffusion attention on B samples of length 2L ([noisy L | clean L]) against FlashAttention-2 causal
attention on 2B sequences of length L. The rows are labelled with FlashAttention-2's L and B.

Usage: python benchmarks/benchmark_diffusion.py [--seqlens 512 1024 2048 4096 8192 16384] [--block-sizes 4 16 64]
           [--headdim 128] [--heads 32 --kv-heads 8] [--dtype bf16] [--dropout 0.1] [--varlen [--mixed]]
           [--deterministic] [--no-bwd]
"""

import argparse
import random
import statistics
import sys
from pathlib import Path

import torch

try:
    import fast_diffusion_attention  # noqa: F401
except ImportError:  # extension built in place (python setup.py build_ext --inplace)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fast_diffusion_attention import diffusion_attn_func, diffusion_attn_varlen_func
from fast_diffusion_attention.reference import num_visible_pairs

try:
    import flash_attn
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    flash_attn = None


MIN_MS = 0.0  # --min-ms


def cuda_time(fn, iters=20, warmup=5):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    if MIN_MS > 0:
        # Warm up for about MIN_MS and time a window of about MIN_MS: steady clock under the power cap
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
    return start.elapsed_time(end) / iters  # ms


def make_fwd_bwd(f, q, k, v, bwd):
    def fwd():
        return f(q, k, v)
    if not bwd:
        return fwd, None
    out = f(q, k, v)
    g = torch.randn_like(out)

    def bwd_fn():
        return torch.autograd.grad(out, (q, k, v), g, retain_graph=True)
    return fwd, bwd_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqlens", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384],
                        help="FlashAttention-2's sequence lengths L: samples [noisy L | clean L] of length 2L")
    parser.add_argument("--batch-tokens", type=int, default=16384,
                        help="FlashAttention-2's tokens per batch: batch size = max(1, batch_tokens // L)")
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument("--hidden", type=int, default=2048, help="nheads = hidden // headdim (FlashAttention-2)")
    parser.add_argument("--heads", type=int, default=None, help="default: hidden // headdim")
    parser.add_argument("--kv-heads", type=int, default=None, help="default: heads (MHA)")
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--varlen", action="store_true", help="use the varlen interface (packed samples)")
    parser.add_argument("--mixed", action="store_true",
                        help="varlen with samples whose halves have random lengths L_i in [L / 8, L] (SFT-like "
                             "packing), packed into 2 x batch-tokens tokens, instead of samples of length 2L")
    parser.add_argument("--deterministic", action="store_true",
                        help="deterministic backward (FlashAttention-2's deterministic=True for the baselines too)")
    parser.add_argument("--no-bwd", action="store_true")
    parser.add_argument("--no-full", action="store_true", help="do not measure FA2 causal(L)")
    parser.add_argument("--rounds", type=int, default=5, help="interleaved measurement rounds (median reported)")
    parser.add_argument("--min-ms", type=float, default=0.0,
                        help="warm up and measure every variant for at least this long (ms), i.e. at its steady "
                             "clock: needed on GPUs whose clock follows the power draw quickly (e.g. L40S: 300)")
    args = parser.parse_args()
    if args.heads is None:
        args.heads = args.hidden // args.headdim
    if args.kv_heads is None:
        args.kv_heads = args.heads
    global MIN_MS
    MIN_MS = args.min_ms
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    bwd = not args.no_bwd
    H, Hk, D = args.heads, args.kv_heads, args.headdim
    p = args.dropout
    det = args.deterministic
    print(f"{torch.cuda.get_device_name()}, nheads {H}/{Hk}, headdim {D}, {args.dtype}"
          f"{', dropout %g' % p if p > 0 else ''}{', varlen' if args.varlen else ''}"
          f"{', deterministic' if det else ''}"
          f"{', FA2 ' + flash_attn.__version__ if flash_attn is not None else ''}; "
          f"time in ms (TFLOPS on visible (q, k) pairs)")
    assert not args.mixed or args.varlen, "--mixed requires --varlen"
    for L in args.seqlens:
        # FlashAttention-2's shape B x L: B samples [noisy L | clean L] of length S = 2L for the diffusion attention,
        # causal attention on 2B sequences of length n = L for FlashAttention-2
        n = L
        S = 2 * L
        B = max(1, args.batch_tokens // L)
        if args.mixed:
            variants = mixed_varlen_variants(args, S, 2 * args.batch_tokens, dtype, bwd)
            fns = {name: make_fwd_bwd(f, *qkv, bwd) for name, (f, _, qkv) in variants.items()}
            variants = {name: (f, pairs) for name, (f, pairs, _) in variants.items()}
            report(args, f"L_i in [{L // 8}, {L}]", variants, fns, bwd)
            continue
        q, k, v = [torch.randn(B, S, h, D, device="cuda", dtype=dtype).requires_grad_(bwd) for h in (H, Hk, Hk)]
        variants = {}
        if flash_attn is not None:
            # FA2 causal attention on the two halves: (B, 2n) -> (2B, n)
            if not args.varlen:
                def fa2_halves(q, k, v):
                    return flash_attn_func(q.view(2 * B, n, *q.shape[2:]), k.view(2 * B, n, *k.shape[2:]),
                                           v.view(2 * B, n, *v.shape[2:]), dropout_p=p, causal=True,
                                           deterministic=det)
            else:
                cu_h = torch.arange(0, B * S + 1, n, device="cuda", dtype=torch.int32)

                def fa2_halves(q, k, v):
                    return flash_attn_varlen_func(q.view(-1, *q.shape[2:]), k.view(-1, *k.shape[2:]),
                                                  v.view(-1, *v.shape[2:]), cu_h, cu_h, n, n, dropout_p=p, causal=True,
                                                  deterministic=det)
            variants["FA2 causal 2x(n)"] = (fa2_halves, B * 2 * (n * (n + 1) // 2))

            if not args.no_full and not args.varlen:
                def fa2_full(q, k, v):
                    return flash_attn_func(q, k, v, dropout_p=p, causal=True, deterministic=det)
                variants["FA2 causal(L)"] = (fa2_full, B * S * (S + 1) // 2)
        for bs in args.block_sizes:
            if not args.varlen:
                def f(q, k, v, bs=bs):
                    return diffusion_attn_func(q, k, v, bs, dropout_p=p, deterministic=det)
            else:
                cu = torch.arange(0, B * S + 1, S, device="cuda", dtype=torch.int32)

                def f(q, k, v, bs=bs, cu=cu):
                    return diffusion_attn_varlen_func(q.view(-1, *q.shape[2:]), k.view(-1, *k.shape[2:]),
                                                      v.view(-1, *v.shape[2:]), cu, cu, S, S, bs, dropout_p=p,
                                                      deterministic=det)
            variants[f"diffusion bs={bs}"] = (f, B * num_visible_pairs(n, bs))
        fns = {name: make_fwd_bwd(f, q, k, v, bwd) for name, (f, _) in variants.items()}
        report(args, f"L = {L}, batch {B}", variants, fns, bwd)


def mixed_varlen_variants(args, S, tokens, dtype, bwd):
    """Samples of random even lengths in [S / 8, S] (halves of lengths in [S / 16, S / 2]), packed into `tokens`
    tokens."""
    H, Hk, D = args.heads, args.kv_heads, args.headdim
    p = args.dropout
    det = args.deterministic
    gen = random.Random(S)
    ns = []
    while 2 * (sum(ns) + S // 2) <= tokens:
        ns.append(gen.randint(S // 16, S // 2))
    total = 2 * sum(ns)
    cu = torch.tensor([0] + [2 * x for x in ns], device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
    q, k, v = [torch.randn(total, h, D, device="cuda", dtype=dtype).requires_grad_(bwd) for h in (H, Hk, Hk)]
    variants = {}
    if flash_attn is not None:
        # FA2 varlen causal attention on the halves: every sample is two sequences of length n_i
        cu_halves = torch.tensor([0] + [x for x in ns for _ in range(2)], device="cuda",
                                 dtype=torch.int32).cumsum(0, dtype=torch.int32)

        def fa2_halves(q, k, v):
            return flash_attn_varlen_func(q, k, v, cu_halves, cu_halves, max(ns), max(ns), dropout_p=p, causal=True,
                                          deterministic=det)
        variants["FA2 causal 2x(n)"] = (fa2_halves, sum(2 * (x * (x + 1) // 2) for x in ns), (q, k, v))
    for bs in args.block_sizes:
        def f(q, k, v, bs=bs):
            return diffusion_attn_varlen_func(q, k, v, cu, cu, 2 * max(ns), 2 * max(ns), bs, dropout_p=p,
                                              deterministic=det)
        variants[f"diffusion bs={bs}"] = (f, sum(num_visible_pairs(x, bs) for x in ns), (q, k, v))
    print(f"{len(ns)} samples, {total} tokens")
    return variants


def report(args, title, variants, fns, bwd):
    D, H = args.headdim, args.heads
    times = {name: ([], []) for name in variants}
    rng = random.Random(0)
    for _ in range(args.rounds):
        # Random order in every round: the clock of power-capped GPUs depends on the preceding work
        order = list(fns.items())
        rng.shuffle(order)
        for name, (fwd, bwd_fn) in order:
            times[name][0].append(cuda_time(fwd))
            if bwd_fn is not None:
                times[name][1].append(cuda_time(bwd_fn))
    print(title)
    base = "FA2 causal 2x(n)" if "FA2 causal 2x(n)" in variants else None
    for name, (_, pairs) in variants.items():
        flops = 4 * pairs * D * H
        t_fwd = statistics.median(times[name][0])
        tf_fwd = flops / t_fwd * 1e-9
        s = f"  {name:20s} fwd {t_fwd:7.3f} ({tf_fwd:4.0f})"
        if bwd:
            t_bwd = statistics.median(times[name][1])
            s += f"   bwd {t_bwd:7.3f} ({2.5 * flops / t_bwd * 1e-9:4.0f})"
        if base is not None and name.startswith("diffusion"):
            # Relative throughput (TFLOPS on visible pairs) w.r.t. FA2 causal on the halves
            b_pairs = variants[base][1]
            b_fwd = 4 * b_pairs * D * H / statistics.median(times[base][0]) * 1e-9
            s += f"   rel fwd {tf_fwd / b_fwd:.2f}"
            if bwd:
                b_bwd = 2.5 * 4 * b_pairs * D * H / statistics.median(times[base][1]) * 1e-9
                s += f" bwd {2.5 * flops / t_bwd * 1e-9 / b_bwd:.2f}"
        print(s)


if __name__ == "__main__":
    main()
