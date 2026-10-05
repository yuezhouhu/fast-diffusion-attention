# Copyright (c) 2026, fast-diffusion-attention authors.
"""Correctness tests of the diffusion attention kernels.

Methodology (as in the FlashAttention test suites): the error of the kernel w.r.t. a float32 reference must be at most
twice the error of a straightforward PyTorch implementation in the same low precision, plus a few ulps, for the
output, the LSE and dQ / dK / dV. Systematic errors (bias, scale error) are tested separately over many random
problems (test_no_systematic_error).

    pytest tests/test_diffusion_attn.py
    FDA_TEST_HDIMS=128 pytest tests/test_diffusion_attn.py   # only the head dimensions that were built
"""

import math
import os

import pytest
import torch

from fast_diffusion_attention import diffusion_attn_func, diffusion_attn_varlen_func
from fast_diffusion_attention.reference import attention_ref, attention_varlen_ref, diffusion_mask, num_visible_pairs

DEVICE = "cuda"
HDIMS_BUILT = [int(h) for h in os.getenv("FDA_TEST_HDIMS", "64,96,128,192,256").split(",")]
DISABLE_FP16 = os.getenv("FDA_TEST_DISABLE_FP16", "0") == "1"
DISABLE_FP8 = os.getenv("FDA_TEST_DISABLE_FP8", "0") == "1"
DISABLE_SOFTCAP = os.getenv("FDA_TEST_DISABLE_SOFTCAP", "0") == "1"


def _rounded_hdim(d):
    for h in (64, 96, 128, 192, 256):
        if d <= h:
            return h
    return 256


def hdims(*ds):
    return [d for d in ds if _rounded_hdim(d) in HDIMS_BUILT]


DTYPES = [torch.bfloat16] + ([] if DISABLE_FP16 else [torch.float16])
SOFTCAPS = [0.0] + ([] if DISABLE_SOFTCAP else [15.0])


def block_diff_mask(q_idx, kv_idx, block_size, n, use_regular_causal=True):
    """Copy of `block_diff_mask` of the I-DLM training code (training/model/*/modeling_sdar.py)."""
    x0_flag_q = q_idx >= n
    x0_flag_kv = kv_idx >= n
    block_q = torch.where(x0_flag_q == 1, (q_idx - n) // block_size, q_idx // block_size)
    block_kv = torch.where(x0_flag_kv == 1, (kv_idx - n) // block_size, kv_idx // block_size)
    if use_regular_causal:
        block_diagonal = (block_q == block_kv) & (x0_flag_q == 0) & (x0_flag_kv == 0) & (q_idx >= kv_idx)
    else:
        block_diagonal = (block_q == block_kv) & (x0_flag_q == x0_flag_kv)
    offset_block_causal = (block_q > block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 0)
    if use_regular_causal:
        causal_mask = (q_idx >= kv_idx) & (x0_flag_kv == 1) & (x0_flag_q == 1)
    else:
        causal_mask = (block_q >= block_kv) & (x0_flag_kv == 1) & (x0_flag_q == 1)
    return block_diagonal | offset_block_causal | causal_mask


@pytest.mark.parametrize("n", [1, 2, 5, 16, 33, 100])
@pytest.mark.parametrize("block_size", [1, 2, 3, 4, 7, 16, 200])
def test_mask_matches_idlm(n, block_size):
    idx = torch.arange(2 * n)
    ref = block_diff_mask(idx[:, None], idx[None, :], block_size, n, use_regular_causal=False)
    assert torch.equal(diffusion_mask(n, block_size), ref)
    assert num_visible_pairs(n, block_size) == int(ref.sum())


def _errors_ok(name, out, out_ref, out_pt, rtol=2.0, atol=None):
    err = (out.float() - out_ref.float()).abs().max().item()
    err_pt = (out_pt.float() - out_ref.float()).abs().max().item()
    if atol is None:
        # A few ulps of the reference values
        atol = 2 * (out_ref.float() + 0.3 - 0.3 - out_ref.float()).abs().max().item() + 1e-5
    assert err <= rtol * err_pt + atol, f"{name}: max error {err} > {rtol} * {err_pt} (PyTorch) + {atol}"


def _check_fwd_bwd(q, k, v, block_size, softcap=0.0, deterministic=False, cu_seqlens=None, max_seqlen=None):
    g = torch.randn(q.shape, device=q.device, dtype=q.dtype)
    if cu_seqlens is None:
        out, lse = diffusion_attn_func(q, k, v, block_size, softcap=softcap, deterministic=deterministic,
                                       return_attn_probs=True)
        ref_fn = lambda **kw: attention_ref(q, k, v, block_size, softcap=softcap, **kw)
    else:
        out, lse = diffusion_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, block_size,
                                              softcap=softcap, deterministic=deterministic, return_attn_probs=True)
        ref_fn = lambda **kw: attention_varlen_ref(q, k, v, cu_seqlens, block_size, softcap=softcap, **kw)
    out_ref, lse_ref = ref_fn()
    out_pt, lse_pt = ref_fn(upcast=False, reorder_ops=True)
    _errors_ok("out", out, out_ref, out_pt)
    assert (lse - lse_ref).abs().max().item() <= 2 * (lse_pt - lse_ref).abs().max().item() + 1e-4
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)
    dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q, k, v), g)
    dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q, k, v), g)
    for name, x, r, p in [("dq", dq, dq_ref, dq_pt), ("dk", dk, dk_ref, dk_pt), ("dv", dv, dv_ref, dv_pt)]:
        _errors_ok(name, x, r, p)
    return dq, dk, dv


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("d", hdims(64, 96, 128, 192, 256))
@pytest.mark.parametrize("softcap", SOFTCAPS)
@pytest.mark.parametrize("nheads, nheads_k", [(4, 4), (6, 2), (4, 1)])
@pytest.mark.parametrize(
    "batch, n, block_size",
    [(2, 128, 4), (1, 256, 32), (3, 100, 3), (2, 333, 16), (1, 517, 7), (1, 64, 64), (1, 300, 128),
     (2, 191, 200), (1, 1, 1), (2, 5, 2), (1, 1024, 1), (1, 777, 64)],
)
def test_diffusion_attn(batch, n, block_size, nheads, nheads_k, softcap, d, dtype):
    torch.random.manual_seed(n * 1000 + block_size)
    q = torch.randn(batch, 2 * n, nheads, d, device=DEVICE, dtype=dtype, requires_grad=True)
    k = torch.randn(batch, 2 * n, nheads_k, d, device=DEVICE, dtype=dtype, requires_grad=True)
    v = torch.randn(batch, 2 * n, nheads_k, d, device=DEVICE, dtype=dtype, requires_grad=True)
    _check_fwd_bwd(q, k, v, block_size, softcap=softcap)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("d", hdims(64, 80, 112, 128, 160, 224, 256))
@pytest.mark.parametrize("block_size", [1, 4, 37])
def test_diffusion_attn_headdims(block_size, d, dtype):
    """Head dimensions that are not one of the kernel sizes are padded inside the kernel."""
    torch.random.manual_seed(d)
    batch, n, nheads, nheads_k = 2, 211, 4, 2
    q = torch.randn(batch, 2 * n, nheads, d, device=DEVICE, dtype=dtype, requires_grad=True)
    k = torch.randn(batch, 2 * n, nheads_k, d, device=DEVICE, dtype=dtype, requires_grad=True)
    v = torch.randn(batch, 2 * n, nheads_k, d, device=DEVICE, dtype=dtype, requires_grad=True)
    _check_fwd_bwd(q, k, v, block_size)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("d", hdims(64, 128, 256))
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("nheads, nheads_k", [(4, 4), (6, 2)])
@pytest.mark.parametrize("block_size", [1, 4, 13, 64, 160])
def test_diffusion_attn_varlen(block_size, nheads, nheads_k, deterministic, d, dtype):
    torch.random.manual_seed(block_size)
    ns = [17, 300, 1, 128, 513, 64, 2, 255]
    seqlens = [2 * x for x in ns]
    cu_seqlens = torch.tensor([0] + list(torch.tensor(seqlens).cumsum(0)), device=DEVICE, dtype=torch.int32)
    total = sum(seqlens)
    q = torch.randn(total, nheads, d, device=DEVICE, dtype=dtype, requires_grad=True)
    k = torch.randn(total, nheads_k, d, device=DEVICE, dtype=dtype, requires_grad=True)
    v = torch.randn(total, nheads_k, d, device=DEVICE, dtype=dtype, requires_grad=True)
    _check_fwd_bwd(q, k, v, block_size, deterministic=deterministic, cu_seqlens=cu_seqlens, max_seqlen=max(seqlens))


@pytest.mark.parametrize("d", hdims(64, 128))
def test_varlen_matches_batched(d):
    """Packing samples of equal length gives exactly the result of the batched kernel (deterministic backward, so
    that the order of the dQ (and GQA dK / dV) accumulations is the same)."""
    torch.random.manual_seed(0)
    batch, n, nheads, nheads_k, block_size = 3, 200, 4, 2, 8
    q = torch.randn(batch, 2 * n, nheads, d, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(batch, 2 * n, nheads_k, d, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(batch, 2 * n, nheads_k, d, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn_like(q)
    out, lse = diffusion_attn_func(q, k, v, block_size, deterministic=True, return_attn_probs=True)
    grads = torch.autograd.grad(out, (q, k, v), g)
    cu = torch.arange(0, batch * 2 * n + 1, 2 * n, device=DEVICE, dtype=torch.int32)
    qv, kv, vv = [x.detach().reshape(-1, *x.shape[2:]).requires_grad_() for x in (q, k, v)]
    out_v, lse_v = diffusion_attn_varlen_func(qv, kv, vv, cu, cu, 2 * n, 2 * n, block_size, deterministic=True,
                                              return_attn_probs=True)
    grads_v = torch.autograd.grad(out_v, (qv, kv, vv), g.reshape(-1, *g.shape[2:]))
    assert torch.equal(out.reshape(-1, *out.shape[2:]), out_v)
    assert torch.equal(lse.permute(1, 0, 2).reshape(nheads, -1), lse_v)
    for x, y in zip(grads, grads_v):
        assert torch.equal(x.reshape(-1, *x.shape[2:]), y)


@pytest.mark.parametrize("d", hdims(64, 128, 192))
@pytest.mark.parametrize("nheads, nheads_k", [(4, 4), (8, 2)])
@pytest.mark.parametrize("varlen", [False, True])
def test_deterministic(varlen, nheads, nheads_k, d):
    torch.random.manual_seed(1)
    batch, n, block_size = 2, 700, 4
    q = torch.randn(batch, 2 * n, nheads, d, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(batch, 2 * n, nheads_k, d, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(batch, 2 * n, nheads_k, d, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    g = torch.randn_like(q)

    def run():
        if not varlen:
            out = diffusion_attn_func(q, k, v, block_size, deterministic=True)
            return torch.autograd.grad(out, (q, k, v), g)
        cu = torch.arange(0, batch * 2 * n + 1, 2 * n, device=DEVICE, dtype=torch.int32)
        flat = [x.reshape(-1, *x.shape[2:]) for x in (q, k, v)]
        out = diffusion_attn_varlen_func(*flat, cu, cu, 2 * n, 2 * n, block_size, deterministic=True)
        return torch.autograd.grad(out, (q, k, v), g.reshape(-1, *g.shape[2:]))

    ref = run()
    for _ in range(5):
        for x, y in zip(run(), ref):
            assert torch.equal(x, y)


@pytest.mark.skipif(DISABLE_FP8, reason="FP8 kernels not built")
@pytest.mark.parametrize("d", hdims(64, 96, 128, 192, 256))
@pytest.mark.parametrize("nheads, nheads_k", [(4, 4), (6, 2)])
@pytest.mark.parametrize("use_descale", [False, True])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("batch, n, block_size", [(2, 300, 4), (1, 517, 32), (2, 64, 1), (1, 400, 160)])
def test_diffusion_attn_fp8(batch, n, block_size, varlen, use_descale, nheads, nheads_k, d):
    """FP8 (e4m3) forward. As in FlashAttention-3, P is quantized to FP8 inside the kernel, which dominates the
    error: the error must be within 16x the error of a BF16 PyTorch implementation, and unbiased."""
    torch.random.manual_seed(n + d)
    qf, kf, vf = [torch.randn(batch, 2 * n, h, d, device=DEVICE) for h in (nheads, nheads_k, nheads_k)]
    if use_descale:
        descales = [torch.rand(batch, nheads_k, device=DEVICE) + 0.5 for _ in range(3)]
        q8, k8, v8 = [(x / s.repeat_interleave(x.shape[2] // nheads_k, dim=1)[:, None, :, None]).to(torch.float8_e4m3fn)
                      for x, s in zip((qf, kf, vf), descales)]
    else:
        descales = [None] * 3
        q8, k8, v8 = [x.to(torch.float8_e4m3fn) for x in (qf, kf, vf)]
    # Dequantized inputs, the reference is computed on them
    deq = []
    for x, s in zip((q8, k8, v8), descales):
        xf = x.float()
        if s is not None:
            xf = xf * s.repeat_interleave(x.shape[2] // s.shape[1], dim=1)[:, None, :, None]
        deq.append(xf)
    if not varlen:
        out, lse = diffusion_attn_func(q8, k8, v8, block_size, q_descale=descales[0], k_descale=descales[1],
                                       v_descale=descales[2], return_attn_probs=True)
    else:
        cu = torch.arange(0, batch * 2 * n + 1, 2 * n, device=DEVICE, dtype=torch.int32)
        flat = [x.reshape(-1, *x.shape[2:]) for x in (q8, k8, v8)]
        out, lse = diffusion_attn_varlen_func(*flat, cu, cu, 2 * n, 2 * n, block_size, q_descale=descales[0],
                                              k_descale=descales[1], v_descale=descales[2], return_attn_probs=True)
        out = out.reshape(batch, 2 * n, nheads, d)
        lse = lse.reshape(nheads, batch, 2 * n).permute(1, 0, 2)
    assert out.dtype == torch.bfloat16
    out_ref, lse_ref = attention_ref(*deq, block_size)
    out_pt, _ = attention_ref(*[x.to(torch.bfloat16) for x in deq], block_size, upcast=False)
    err = (out.float() - out_ref.float()).abs().max().item()
    err_pt = (out_pt.float() - out_ref.float()).abs().max().item()
    assert err <= 16 * err_pt + 1e-3, f"FP8 max error {err} vs BF16 PyTorch {err_pt}"
    # LSE is computed in FP32 from FP8 scores (no P quantization)
    assert (lse - lse_ref).abs().max().item() < 1e-2
    # Unbiased: the mean error is small compared to the mean absolute error
    diff = (out.float() - out_ref.float())
    assert diff.mean().abs().item() <= 0.1 * diff.abs().mean().item() + 1e-5


def test_softmax_scale():
    torch.random.manual_seed(0)
    q, k, v = [torch.randn(1, 400, 2, 128, device=DEVICE, dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    for scale in [0.05, 0.3]:
        out = diffusion_attn_func(q, k, v, 8, softmax_scale=scale)
        out_ref, _ = attention_ref(q, k, v, 8, softmax_scale=scale)
        out_pt, _ = attention_ref(q, k, v, 8, softmax_scale=scale, upcast=False, reorder_ops=True)
        _errors_ok("out", out, out_ref, out_pt)


def test_non_contiguous_inputs():
    """q, k, v slices of a packed qkv tensor (strided batch / row / head dimensions)."""
    torch.random.manual_seed(0)
    batch, n, nheads, d = 2, 150, 4, 128
    qkv = torch.randn(batch, 2 * n, 3, nheads, d, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    q, k, v = qkv.unbind(2)
    _check_fwd_bwd(q, k, v, 4)


@pytest.mark.parametrize("d", hdims(128))
def test_no_systematic_error(d):
    """Over many random problems, the mean bias and the mean relative scale error (slope of a least-squares fit of the
    kernel result against the reference) of the output and of every gradient must be zero within 4 standard errors,
    and their magnitude must not exceed that of the PyTorch implementation in the same precision."""
    torch.random.manual_seed(123)
    stats = {name: {"fda": [], "pt": []} for name in ("out", "dq", "dk", "dv")}
    for trial in range(24):
        batch, n, nheads, nheads_k = 1, int(torch.randint(50, 400, (1,))), 4, 2
        block_size = int(torch.randint(1, 40, (1,)))
        q = torch.randn(batch, 2 * n, nheads, d, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(batch, 2 * n, nheads_k, d, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(batch, 2 * n, nheads_k, d, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
        g = torch.randn_like(q)
        out = diffusion_attn_func(q, k, v, block_size)
        res = {"fda": [out, *torch.autograd.grad(out, (q, k, v), g)]}
        out_ref, _ = attention_ref(q, k, v, block_size)
        ref = [out_ref, *torch.autograd.grad(out_ref, (q, k, v), g)]
        out_pt, _ = attention_ref(q, k, v, block_size, upcast=False, reorder_ops=True)
        res["pt"] = [out_pt, *torch.autograd.grad(out_pt, (q, k, v), g)]
        for impl in ("fda", "pt"):
            for name, x, r in zip(("out", "dq", "dk", "dv"), res[impl], ref):
                x, r = x.double().flatten(), r.double().flatten()
                bias = (x - r).mean().item() / r.abs().mean().item()
                slope = ((x * r).sum() / (r * r).sum()).item() - 1.0
                stats[name][impl].append((bias, slope))
    for name, s in stats.items():
        for k_idx, what in enumerate(("bias", "scale error")):
            vals = torch.tensor([b[k_idx] for b in s["fda"]], dtype=torch.float64)
            vals_pt = torch.tensor([b[k_idx] for b in s["pt"]], dtype=torch.float64)
            mean, sem = vals.mean().item(), vals.std().item() / math.sqrt(len(vals))
            assert abs(mean) <= 4 * sem + 1e-7, f"{name} {what}: mean {mean} not zero (standard error {sem})"
            assert vals.abs().mean().item() <= 2 * vals_pt.abs().mean().item() + 1e-6, (
                f"{name} {what}: {vals.abs().mean().item()} vs PyTorch {vals_pt.abs().mean().item()}")


def test_zero_length_varlen_sample():
    torch.random.manual_seed(0)
    seqlens = [0, 200, 0, 64]
    cu = torch.tensor([0] + list(torch.tensor(seqlens).cumsum(0)), device=DEVICE, dtype=torch.int32)
    total = sum(seqlens)
    q, k, v = [torch.randn(total, 4, 128, device=DEVICE, dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    _check_fwd_bwd(q, k, v, 4, cu_seqlens=cu, max_seqlen=max(seqlens))


def test_torch_compile():
    torch.random.manual_seed(0)
    q, k, v = [torch.randn(2, 256, 4, 128, device=DEVICE, dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    g = torch.randn(2, 256, 4, 128, device=DEVICE, dtype=torch.bfloat16)

    def f(q, k, v):
        return diffusion_attn_func(q, k, v, 16)

    out = f(q, k, v)
    grads = torch.autograd.grad(out, (q, k, v), g)
    f_c = torch.compile(f, fullgraph=True)
    out_c = f_c(q, k, v)
    grads_c = torch.autograd.grad(out_c, (q, k, v), g)
    assert torch.equal(out, out_c)
    for x, y in zip(grads, grads_c):
        torch.testing.assert_close(x, y, rtol=0, atol=1e-2)


def test_errors():
    q = torch.randn(1, 15, 2, 128, device=DEVICE, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="even"):
        diffusion_attn_func(q, q, q, 4)
    q = torch.randn(1, 16, 2, 128, device=DEVICE, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="block_size"):
        diffusion_attn_func(q, q, q, 0)
    with pytest.raises(RuntimeError):
        diffusion_attn_func(q.float(), q.float(), q.float(), 4)
