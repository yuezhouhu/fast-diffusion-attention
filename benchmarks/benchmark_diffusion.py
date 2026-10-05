# Copyright (c) 2026, fast-diffusion-attention authors.
"""Throughput of the diffusion attention vs FlashAttention-3.

A sample of length L = 2n is [noisy (n tokens) | clean (n tokens)]. The diffusion mask has about n^2 visible
(query, key) pairs: as many as causal attention on the two halves separately, half of causal attention on the L
tokens. TFLOPS are always computed from the visible pairs (4 * pairs * headdim * nheads for the forward, 2.5x that
for the backward), i.e. masked-out work is never counted.

Baselines (if FlashAttention-3 is installed):
  * FA3 causal 2x(n): causal attention on the two halves of length n, as a batch of 2B sequences of length n
                      (same amount of work and same tile structure as the diffusion mask)
  * FA3 causal(L):    causal attention on the L tokens (2x the work of the diffusion mask, shown for scale)

Times are measured with CUDA events over back-to-back iterations, for the whole forward / backward including all the
auxiliary kernels (backward pre- and post-processing). The GPU clock varies with the power draw (e.g. on power-capped
PCIe cards), so all the variants of a shape are measured in interleaved rounds and the median over the rounds is
reported.

Usage: python benchmarks/benchmark_diffusion.py [--seqlens 2048 4096 8192 16384] [--block-sizes 4 16 64] [--dtype fp8]
"""

import argparse
import random
import statistics

import torch

from fast_diffusion_attention import diffusion_attn_func, diffusion_attn_varlen_func
from fast_diffusion_attention.reference import num_visible_pairs

try:
    from flash_attn_3 import flash_attn_interface as fa3
except ImportError:
    try:
        import flash_attn_interface as fa3
    except ImportError:
        fa3 = None


def cuda_time(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
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
    parser.add_argument("--seqlens", type=int, nargs="+", default=[2048, 4096, 8192, 16384],
                        help="length L = 2n of a [noisy | clean] sample")
    parser.add_argument("--batch-tokens", type=int, default=16384, help="batch size = max(1, batch_tokens // L)")
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--headdim", type=int, default=128)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp8"])
    parser.add_argument("--varlen", action="store_true", help="use the varlen interface (packed samples)")
    parser.add_argument("--no-bwd", action="store_true")
    parser.add_argument("--rounds", type=int, default=5, help="interleaved measurement rounds (median reported)")
    args = parser.parse_args()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp8": torch.float8_e4m3fn}[args.dtype]
    bwd = not args.no_bwd and args.dtype != "fp8"
    H, Hk, D = args.heads, args.kv_heads, args.headdim
    print(f"nheads {H}/{Hk}, headdim {D}, {args.dtype}{', varlen' if args.varlen else ''}; "
          f"time in ms (TFLOPS on visible (q, k) pairs)")
    for L in args.seqlens:
        assert L % 2 == 0
        n = L // 2
        B = max(1, args.batch_tokens // L)
        q, k, v = [torch.randn(B, L, h, D, device="cuda", dtype=torch.bfloat16).to(dtype).requires_grad_(bwd)
                   for h in (H, Hk, Hk)]
        variants = {}
        if fa3 is not None:
            # FA3 causal attention on the two halves: (B, 2n) -> (2B, n)
            def fa3_halves(q, k, v):
                return fa3.flash_attn_func(q.view(2 * B, n, *q.shape[2:]), k.view(2 * B, n, *k.shape[2:]),
                                           v.view(2 * B, n, *v.shape[2:]), causal=True)
            variants["FA3 causal 2x(n)"] = (fa3_halves, B * 2 * (n * (n + 1) // 2))

            def fa3_full(q, k, v):
                return fa3.flash_attn_func(q, k, v, causal=True)
            variants["FA3 causal(L)"] = (fa3_full, B * L * (L + 1) // 2)
        for bs in args.block_sizes:
            if not args.varlen:
                def f(q, k, v, bs=bs):
                    return diffusion_attn_func(q, k, v, bs)
            else:
                cu = torch.arange(0, B * L + 1, L, device="cuda", dtype=torch.int32)

                def f(q, k, v, bs=bs, cu=cu):
                    return diffusion_attn_varlen_func(q.view(-1, *q.shape[2:]), k.view(-1, *k.shape[2:]),
                                                      v.view(-1, *v.shape[2:]), cu, cu, L, L, bs)
            variants[f"diffusion bs={bs}"] = (f, B * num_visible_pairs(n, bs))
        fns = {name: make_fwd_bwd(f, q, k, v, bwd) for name, (f, _) in variants.items()}
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
        print(f"L = {L}, batch {B}")
        for name, (_, pairs) in variants.items():
            flops = 4 * pairs * D * H
            t_fwd = statistics.median(times[name][0])
            s = f"  {name:20s} fwd {t_fwd:7.3f} ({flops / t_fwd * 1e-9:4.0f})"
            if bwd:
                t_bwd = statistics.median(times[name][1])
                s += f"   bwd {t_bwd:7.3f} ({2.5 * flops / t_bwd * 1e-9:4.0f})"
            print(s)


if __name__ == "__main__":
    main()
