"""Quick targeted tuning of backward kernels for multi-doc scenarios."""
import torch
import math
import gc
from fast_diffusion_attention.diff_attn_noisy import (
    _diff_noisy_attn_fwd_kernel, _diff_noisy_attn_bwd_dq_kernel,
    _diff_noisy_attn_bwd_dkdv_kernel, _compute_delta_kernel,
)


def bench_fn(fn, warmup=10, repeat=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(repeat)]
    for s, e in events:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    times = sorted([s.elapsed_time(e) for s, e in events])
    return times[len(times) // 2]


def test_config(kernel_name, BM, BN, NW, NS, head_dim, n_heads, n_kv_heads, S=4):
    device = "cuda"
    n_kv_groups = n_heads // n_kv_heads
    log2s = int(math.log2(S))
    scale = 1.0 / math.sqrt(head_dim)
    sm_scale = scale * 1.44269504
    BLOCK_D = head_dim

    # Test across representative multi-doc scenarios
    scenarios = [(1024, 32), (2048, 16), (4096, 8), (8192, 4)]
    total_time = 0.0

    for doc_seqlen, num_docs in scenarios:
        L1 = (doc_seqlen // 3 // S) * S
        L0 = doc_seqlen - 2 * L1
        L0 = (L0 // S) * S
        actual = L0 + 2 * L1
        total = actual * num_docs

        q = torch.randn(total, n_heads, head_dim, device=device, dtype=torch.float16).contiguous()
        k = torch.randn(total, n_kv_heads, head_dim, device=device, dtype=torch.float16).contiguous()
        v = torch.randn_like(k)
        o = torch.randn_like(q)
        do_t = torch.randn_like(q)
        cu = torch.tensor([actual * i for i in range(num_docs + 1)], dtype=torch.int32, device=device)
        l0a = torch.full((num_docs,), L0, dtype=torch.int32, device=device)
        l1a = torch.full((num_docs,), L1, dtype=torch.int32, device=device)

        # Forward for lse
        lse = torch.empty(total, n_heads, dtype=torch.float32, device=device)
        _diff_noisy_attn_fwd_kernel[((actual + 128 - 1) // 128, n_heads, num_docs)](
            q, k, v, o, lse, cu, l0a, l1a,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            n_heads=n_heads, sm_scale=sm_scale,
            BLOCK_M=128, BLOCK_N=32, BLOCK_D=BLOCK_D, LOG2S=log2s,
            N_KV_GROUPS=n_kv_groups, num_warps=4, num_stages=1,
        )
        D = torch.empty(total, n_heads, dtype=torch.float32, device=device)
        _compute_delta_kernel[(total * n_heads,)](o, do_t, D, BLOCK_D=BLOCK_D, num_warps=max(1, BLOCK_D // 32))
        torch.cuda.synchronize()

        try:
            if kernel_name == "dq":
                dq = torch.empty_like(q)
                num_tiles = (actual + BM - 1) // BM
                def run(dq=dq):
                    _diff_noisy_attn_bwd_dq_kernel[(num_tiles, n_heads, num_docs)](
                        q, k, v, o, do_t, dq, lse, D, cu, l0a, l1a,
                        q.stride(0), q.stride(1), q.stride(2),
                        k.stride(0), k.stride(1), k.stride(2),
                        v.stride(0), v.stride(1), v.stride(2),
                        o.stride(0), o.stride(1), o.stride(2),
                        n_heads=n_heads, sm_scale=sm_scale, scale=scale,
                        BLOCK_M=BM, BLOCK_N=BN, BLOCK_D=BLOCK_D, LOG2S=log2s,
                        N_KV_GROUPS=n_kv_groups, num_warps=NW, num_stages=NS,
                    )
            else:
                dk_buf = torch.empty(total, n_heads, BLOCK_D, dtype=k.dtype, device=device)
                dv_buf = torch.empty(total, n_heads, BLOCK_D, dtype=v.dtype, device=device)
                num_tiles = (actual + BN - 1) // BN
                def run(dk_buf=dk_buf, dv_buf=dv_buf):
                    _diff_noisy_attn_bwd_dkdv_kernel[(num_tiles, n_heads, num_docs)](
                        q, k, v, o, do_t, dk_buf, dv_buf, lse, D, cu, l0a, l1a,
                        q.stride(0), q.stride(1), q.stride(2),
                        k.stride(0), k.stride(1), k.stride(2),
                        v.stride(0), v.stride(1), v.stride(2),
                        o.stride(0), o.stride(1), o.stride(2),
                        dk_buf.stride(0), dk_buf.stride(1), dk_buf.stride(2),
                        n_heads=n_heads, sm_scale=sm_scale, scale=scale,
                        BLOCK_M=BM, BLOCK_N=BN, BLOCK_D=BLOCK_D, LOG2S=log2s,
                        N_KV_GROUPS=n_kv_groups, num_warps=NW, num_stages=NS,
                    )
            t = bench_fn(run)
            total_time += t
        except Exception as e:
            return None

        del q, k, v, o, do_t, lse, D
        gc.collect()
        torch.cuda.empty_cache()

    return total_time


def main():
    print(f"GPU: {torch.cuda.get_device_name()}")

    # Targeted configs — only the most promising ones
    dq_configs = [
        (32, 32, 2, 1), (32, 32, 4, 1),
        (32, 64, 2, 1), (32, 64, 4, 1), (32, 64, 4, 2),
        (64, 32, 4, 1), (64, 32, 8, 1),
        (64, 64, 4, 1), (64, 64, 8, 1),
        (128, 32, 4, 1), (128, 64, 4, 1), (128, 64, 8, 1),
    ]
    dkdv_configs = [
        (32, 32, 2, 1), (32, 32, 4, 1), (32, 32, 4, 3),
        (32, 64, 4, 1), (32, 64, 4, 2),
        (64, 32, 2, 1), (64, 32, 4, 1), (64, 32, 8, 1),
        (64, 64, 4, 1), (64, 64, 8, 1),
        (128, 32, 4, 1), (128, 64, 4, 1),
    ]

    for head_dim, nh, nkv, desc in [(64, 16, 16, "MHA d=64"), (128, 16, 8, "GQA 2x d=128")]:
        for kernel_name, configs in [("dq", dq_configs), ("dkdv", dkdv_configs)]:
            print(f"\n{'='*60}")
            print(f"{desc}: Tuning {kernel_name}")
            print(f"{'='*60}")
            best_config = None
            best_time = float('inf')
            for cfg in configs:
                t = test_config(kernel_name, *cfg, head_dim, nh, nkv)
                if t is not None:
                    tag = " <-- BEST" if t < best_time else ""
                    print(f"  BM={cfg[0]:>3} BN={cfg[1]:>3} W={cfg[2]} S={cfg[3]}: {t:8.3f} ms{tag}")
                    if t < best_time:
                        best_time = t
                        best_config = cfg
                else:
                    print(f"  BM={cfg[0]:>3} BN={cfg[1]:>3} W={cfg[2]} S={cfg[3]}: FAILED")
            print(f"BEST {kernel_name}: {best_config} = {best_time:.3f} ms")


if __name__ == "__main__":
    main()
