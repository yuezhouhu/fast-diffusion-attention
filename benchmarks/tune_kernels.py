#!/usr/bin/env python3
"""
Manual kernel tuning for diff_attn_with_noisy_input on current GPU.

Sweeps (BLOCK_M, BLOCK_N, num_warps, num_stages) for each kernel
(forward, dQ, dKdV) and head_dim (64, 128).

Results are keyed by (compute_capability, head_dim) and can be
hardcoded into diff_attn_noisy.py.

Usage:
    python benchmarks/tune_kernels.py
"""

import torch
import math
import gc

from fast_diffusion_attention.diff_attn_noisy import (
    _diff_noisy_attn_fwd_kernel,
    _diff_noisy_attn_bwd_dq_kernel,
    _diff_noisy_attn_bwd_dkdv_kernel,
)

DEVICE = "cuda"
STAIRCASE_SIZE = 4
LOG2S = 2
# Test across representative seqlens — sum of times as metric
TEST_SEQLENS = [1024, 2048, 4096]


def median_time(fn, warmup=5, repeat=20):
    """Return median execution time in ms.  inf on failure."""
    try:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
    except Exception:
        return float("inf")
    evts = [(torch.cuda.Event(enable_timing=True),
             torch.cuda.Event(enable_timing=True)) for _ in range(repeat)]
    for s, e in evts:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    ts = sorted(s.elapsed_time(e) for s, e in evts)
    return ts[len(ts) // 2]


def make_data(seqlen, n_heads, n_kv_heads, head_dim):
    """Create single-doc test tensors.  Returns dict or None."""
    S = STAIRCASE_SIZE
    L1 = max((seqlen // 3 // S) * S, S)
    L0 = max(((seqlen - 2 * L1) // S) * S, S)
    T = L0 + 2 * L1
    q = torch.randn(T, n_heads, head_dim, device=DEVICE, dtype=torch.float16)
    k = torch.randn(T, n_kv_heads, head_dim, device=DEVICE, dtype=torch.float16)
    v = torch.randn(T, n_kv_heads, head_dim, device=DEVICE, dtype=torch.float16)
    cu = torch.tensor([0, T], dtype=torch.int32, device=DEVICE)
    l0 = torch.tensor([L0], dtype=torch.int32, device=DEVICE)
    l1 = torch.tensor([L1], dtype=torch.int32, device=DEVICE)
    return dict(q=q, k=k, v=v, cu=cu, l0=l0, l1=l1, T=T, L0=L0, L1=L1)


def run_fwd_and_get_bwd_inputs(data, n_heads, n_kv_heads, head_dim):
    """Run forward with default config, return (o, lse, do, D) for backward tuning."""
    d = data
    n_kv_groups = n_heads // n_kv_heads
    scale = (1.0 / math.sqrt(head_dim)) * 1.44269504
    T = d["T"]
    BM, BN = 128, 64
    o = torch.empty_like(d["q"])
    lse = torch.empty(T, n_heads, dtype=torch.float32, device=DEVICE)
    grid = ((T + BM - 1) // BM, n_heads, 1)
    _diff_noisy_attn_fwd_kernel[grid](
        d["q"], d["k"], d["v"], o, lse, d["cu"], d["l0"], d["l1"],
        d["q"].stride(0), d["q"].stride(1), d["q"].stride(2),
        d["k"].stride(0), d["k"].stride(1), d["k"].stride(2),
        d["v"].stride(0), d["v"].stride(1), d["v"].stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        n_heads=n_heads, sm_scale=scale,
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_D=head_dim,
        LOG2S=LOG2S, N_KV_GROUPS=n_kv_groups,
        num_warps=8, num_stages=2,
    )
    do = torch.randn_like(o)
    D = (do.float() * o.float()).sum(dim=-1)
    return o, lse, do, D


# ────────────────────────────── Forward Tuning ──────────────────────────────

def tune_fwd(head_dim, n_heads, n_kv_heads):
    n_kv_groups = n_heads // n_kv_heads
    scale = (1.0 / math.sqrt(head_dim)) * 1.44269504

    search = [
        (BM, BN, W, S)
        for BM in [64, 128]
        for BN in [32, 64, 128]
        for W in [4, 8]
        for S in [1, 2, 3, 4]
    ]

    print(f"\n{'='*90}")
    print(f"FORWARD  head_dim={head_dim}  n_heads={n_heads}  n_kv_heads={n_kv_heads}")
    print(f"{'='*90}")
    print(f"{'BM':>4} {'BN':>4} {'W':>2} {'S':>2} | ", end="")
    for sl in TEST_SEQLENS:
        print(f"  {sl:>5}", end="")
    print(f" | {'total':>8}")
    print("-" * 60)

    best, best_t = None, float("inf")
    for BM, BN, W, S in search:
        total = 0.0
        ok = True
        per_seq = []
        for seqlen in TEST_SEQLENS:
            d = make_data(seqlen, n_heads, n_kv_heads, head_dim)
            T = d["T"]
            o = torch.empty_like(d["q"])
            lse = torch.empty(T, n_heads, dtype=torch.float32, device=DEVICE)
            grid = ((T + BM - 1) // BM, n_heads, 1)

            def run(q=d["q"], k=d["k"], v=d["v"], o=o, lse=lse,
                    cu=d["cu"], l0=d["l0"], l1=d["l1"], grid=grid):
                _diff_noisy_attn_fwd_kernel[grid](
                    q, k, v, o, lse, cu, l0, l1,
                    q.stride(0), q.stride(1), q.stride(2),
                    k.stride(0), k.stride(1), k.stride(2),
                    v.stride(0), v.stride(1), v.stride(2),
                    o.stride(0), o.stride(1), o.stride(2),
                    n_heads=n_heads, sm_scale=scale,
                    BLOCK_M=BM, BLOCK_N=BN, BLOCK_D=head_dim,
                    LOG2S=LOG2S, N_KV_GROUPS=n_kv_groups,
                    num_warps=W, num_stages=S,
                )

            t = median_time(run)
            del o, lse, d; gc.collect(); torch.cuda.empty_cache()
            if t == float("inf"):
                ok = False; break
            per_seq.append(t)
            total += t

        if not ok:
            continue
        tag = " ***" if total < best_t else ""
        print(f"{BM:>4} {BN:>4} {W:>2} {S:>2} | ", end="")
        for t in per_seq:
            print(f"  {t:>5.2f}", end="")
        print(f" | {total:>8.3f}{tag}")
        if total < best_t:
            best_t = total; best = (BM, BN, W, S)

    print(f"\nBEST FWD d={head_dim}: BLOCK_M={best[0]} BLOCK_N={best[1]} "
          f"num_warps={best[2]} num_stages={best[3]}  ({best_t:.3f}ms)")
    return best


# ────────────────────────────── dQ Tuning ──────────────────────────────────

def tune_dq(head_dim, n_heads, n_kv_heads):
    n_kv_groups = n_heads // n_kv_heads
    scale = (1.0 / math.sqrt(head_dim)) * 1.44269504

    search = [
        (BM, BN, W, S)
        for BM in [32, 64, 128]
        for BN in [32, 64, 128]
        for W in [2, 4, 8]
        for S in [1, 2, 3]
    ]

    print(f"\n{'='*90}")
    print(f"DQ BACKWARD  head_dim={head_dim}  n_heads={n_heads}  n_kv_heads={n_kv_heads}")
    print(f"{'='*90}")
    print(f"{'BM':>4} {'BN':>4} {'W':>2} {'S':>2} | ", end="")
    for sl in TEST_SEQLENS:
        print(f"  {sl:>5}", end="")
    print(f" | {'total':>8}")
    print("-" * 60)

    best, best_t = None, float("inf")
    for BM, BN, W, S in search:
        total = 0.0
        ok = True
        per_seq = []
        for seqlen in TEST_SEQLENS:
            d = make_data(seqlen, n_heads, n_kv_heads, head_dim)
            T = d["T"]
            o, lse, do, D_buf = run_fwd_and_get_bwd_inputs(d, n_heads, n_kv_heads, head_dim)
            dq = torch.empty_like(d["q"])
            grid = ((T + BM - 1) // BM, n_heads, 1)

            def run(q=d["q"], k=d["k"], v=d["v"], o=o, do=do, dq=dq,
                    lse=lse, D=D_buf, cu=d["cu"], l0=d["l0"], l1=d["l1"], grid=grid):
                _diff_noisy_attn_bwd_dq_kernel[grid](
                    q, k, v, o, do, dq, lse, D, cu, l0, l1,
                    q.stride(0), q.stride(1), q.stride(2),
                    k.stride(0), k.stride(1), k.stride(2),
                    v.stride(0), v.stride(1), v.stride(2),
                    o.stride(0), o.stride(1), o.stride(2),
                    n_heads=n_heads, sm_scale=scale,
                    BLOCK_M=BM, BLOCK_N=BN, BLOCK_D=head_dim,
                    LOG2S=LOG2S, N_KV_GROUPS=n_kv_groups,
                    num_warps=W, num_stages=S,
                )

            t = median_time(run)
            del o, lse, do, D_buf, dq, d; gc.collect(); torch.cuda.empty_cache()
            if t == float("inf"):
                ok = False; break
            per_seq.append(t)
            total += t

        if not ok:
            continue
        tag = " ***" if total < best_t else ""
        print(f"{BM:>4} {BN:>4} {W:>2} {S:>2} | ", end="")
        for t in per_seq:
            print(f"  {t:>5.2f}", end="")
        print(f" | {total:>8.3f}{tag}")
        if total < best_t:
            best_t = total; best = (BM, BN, W, S)

    print(f"\nBEST DQ d={head_dim}: BLOCK_M={best[0]} BLOCK_N={best[1]} "
          f"num_warps={best[2]} num_stages={best[3]}  ({best_t:.3f}ms)")
    return best


# ────────────────────────────── dKdV Tuning ────────────────────────────────

def tune_dkdv(head_dim, n_heads, n_kv_heads):
    n_kv_groups = n_heads // n_kv_heads
    scale = (1.0 / math.sqrt(head_dim)) * 1.44269504

    search = [
        (BM, BN, W, S)
        for BM in [32, 64, 128]
        for BN in [32, 64, 128]
        for W in [2, 4, 8]
        for S in [1, 2, 3]
    ]

    print(f"\n{'='*90}")
    print(f"DKDV BACKWARD  head_dim={head_dim}  n_heads={n_heads}  n_kv_heads={n_kv_heads}")
    print(f"{'='*90}")
    print(f"{'BM':>4} {'BN':>4} {'W':>2} {'S':>2} | ", end="")
    for sl in TEST_SEQLENS:
        print(f"  {sl:>5}", end="")
    print(f" | {'total':>8}")
    print("-" * 60)

    best, best_t = None, float("inf")
    for BM, BN, W, S in search:
        total = 0.0
        ok = True
        per_seq = []
        for seqlen in TEST_SEQLENS:
            d = make_data(seqlen, n_heads, n_kv_heads, head_dim)
            T = d["T"]
            o, lse, do, D_buf = run_fwd_and_get_bwd_inputs(d, n_heads, n_kv_heads, head_dim)
            dk_buf = torch.zeros(T, n_heads, head_dim, dtype=d["k"].dtype, device=DEVICE)
            dv_buf = torch.zeros(T, n_heads, head_dim, dtype=d["v"].dtype, device=DEVICE)
            grid = ((T + BN - 1) // BN, n_heads, 1)

            def run(q=d["q"], k=d["k"], v=d["v"], o=o, do=do,
                    dk=dk_buf, dv=dv_buf, lse=lse, D=D_buf,
                    cu=d["cu"], l0=d["l0"], l1=d["l1"], grid=grid):
                _diff_noisy_attn_bwd_dkdv_kernel[grid](
                    q, k, v, o, do, dk, dv, lse, D, cu, l0, l1,
                    q.stride(0), q.stride(1), q.stride(2),
                    k.stride(0), k.stride(1), k.stride(2),
                    v.stride(0), v.stride(1), v.stride(2),
                    o.stride(0), o.stride(1), o.stride(2),
                    dk.stride(0), dk.stride(1), dk.stride(2),
                    n_heads=n_heads, sm_scale=scale,
                    BLOCK_M=BM, BLOCK_N=BN, BLOCK_D=head_dim,
                    LOG2S=LOG2S, N_KV_GROUPS=n_kv_groups,
                    num_warps=W, num_stages=S,
                )

            t = median_time(run)
            del o, lse, do, D_buf, dk_buf, dv_buf, d
            gc.collect(); torch.cuda.empty_cache()
            if t == float("inf"):
                ok = False; break
            per_seq.append(t)
            total += t

        if not ok:
            continue
        tag = " ***" if total < best_t else ""
        print(f"{BM:>4} {BN:>4} {W:>2} {S:>2} | ", end="")
        for t in per_seq:
            print(f"  {t:>5.2f}", end="")
        print(f" | {total:>8.3f}{tag}")
        if total < best_t:
            best_t = total; best = (BM, BN, W, S)

    print(f"\nBEST DKDV d={head_dim}: BLOCK_M={best[0]} BLOCK_N={best[1]} "
          f"num_warps={best[2]} num_stages={best[3]}  ({best_t:.3f}ms)")
    return best


# ────────────────────────────── Main ───────────────────────────────────────

def main():
    cc = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    print(f"GPU: {name}  Compute Capability: {cc[0]}.{cc[1]}")

    results = {}

    for head_dim, n_heads, n_kv_heads, label in [
        (64, 16, 16, "MHA d=64"),
        (128, 16, 8, "GQA d=128"),
    ]:
        print(f"\n{'#'*90}")
        print(f"# {label}: head_dim={head_dim}, n_heads={n_heads}, n_kv_heads={n_kv_heads}")
        print(f"{'#'*90}")

        fwd = tune_fwd(head_dim, n_heads, n_kv_heads)
        dq = tune_dq(head_dim, n_heads, n_kv_heads)
        dkdv = tune_dkdv(head_dim, n_heads, n_kv_heads)
        results[(head_dim, n_heads, n_kv_heads)] = dict(fwd=fwd, dq=dq, dkdv=dkdv)

    # ── Print summary ──
    print(f"\n\n{'='*90}")
    print("SUMMARY — Optimal configs for hardcoding")
    print(f"GPU: {name}  CC: {cc[0]}.{cc[1]}")
    print(f"{'='*90}")
    for (hd, nh, nkv), cfg in results.items():
        print(f"\nhead_dim={hd}, n_heads={nh}, n_kv_heads={nkv}:")
        for kernel, (BM, BN, W, S) in cfg.items():
            print(f"  {kernel:>4}: BLOCK_M={BM:>3}, BLOCK_N={BN:>3}, num_warps={W}, num_stages={S}")

    # ── Print as Python dict for copy-paste ──
    print(f"\n\n# Python dict for diff_attn_noisy.py:")
    print(f"# Key: (cc_major, head_dim)")
    print(f"_TUNED_CONFIGS = {{")
    for (hd, nh, nkv), cfg in results.items():
        fwd = cfg["fwd"]
        dq = cfg["dq"]
        dkdv = cfg["dkdv"]
        print(f"    ({cc[0]}, {hd}): {{")
        print(f"        'fwd':  ({fwd[0]:>3}, {fwd[1]:>3}, {fwd[2]}, {fwd[3]}),  # BLOCK_M, BLOCK_N, num_warps, num_stages")
        print(f"        'dq':   ({dq[0]:>3}, {dq[1]:>3}, {dq[2]}, {dq[3]}),")
        print(f"        'dkdv': ({dkdv[0]:>3}, {dkdv[1]:>3}, {dkdv[2]}, {dkdv[3]}),")
        print(f"    }},")
    print(f"}}")


if __name__ == "__main__":
    main()
