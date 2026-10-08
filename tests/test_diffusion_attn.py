# Copyright (c) 2026, fast-diffusion-attention authors.
"""Correctness tests of the diffusion attention kernels (FlashAttention-2 generation).

Methodology (as in the FlashAttention-2 test suite, tests/test_flash_attn.py): the error of the kernel w.r.t. a
float32 reference (computed from the same low-precision inputs) must be at most twice the error of a straightforward
PyTorch implementation in the same low precision, plus a few ulps, for the output, the LSE and dQ / dK / dV. Dropout
is tested as in FlashAttention-2: the dropout pattern returned by the kernel (S_dmask, return_attn_probs=True) is
applied to the reference, and the dropout fraction must be close to dropout_p. Systematic errors (bias, scale error)
are tested separately over many random problems (test_no_systematic_error), and with block_size = 1 the clean half is
causal attention whose errors must match those of FlashAttention-2 (test_accuracy_matches_fa2).

    pytest tests/test_diffusion_attn.py
    FDA_TEST_HDIMS=128 FDA_TEST_DISABLE_FP16=1 pytest tests/test_diffusion_attn.py   # development builds
    FDA_EMULATE_ARCH=86 pytest tests/test_diffusion_attn.py   # the kernels selected on sm86 / sm89 GPUs
"""

import math
import os

import pytest
import torch

from fast_diffusion_attention import (
    diffusion_attn_func,
    diffusion_attn_kvpacked_func,
    diffusion_attn_qkvpacked_func,
    diffusion_attn_varlen_func,
    diffusion_attn_varlen_kvpacked_func,
    diffusion_attn_varlen_qkvpacked_func,
)
from fast_diffusion_attention.reference import attention_ref, attention_varlen_ref, diffusion_mask, num_visible_pairs

try:
    from flash_attn import flash_attn_func as fa2_func
except ImportError:
    fa2_func = None

DEVICE = "cuda"
HDIMS_BUILT = [int(h) for h in os.getenv("FDA_TEST_HDIMS", "32,64,96,128,192,256").split(",")]
DISABLE_FP16 = os.getenv("FDA_TEST_DISABLE_FP16", "0") == "1"
DISABLE_DROPOUT = os.getenv("FDA_TEST_DISABLE_DROPOUT", "0") == "1"
DISABLE_SOFTCAP = os.getenv("FDA_TEST_DISABLE_SOFTCAP", "0") == "1"
# sm86 / sm89 have 99 KB of shared memory per thread block: no dropout in the backward with head dimension 256
SMEM_LIMITED = os.getenv("FDA_EMULATE_ARCH", "") in ("86", "89") or (
    torch.cuda.get_device_capability() in [(8, 6), (8, 9)])


def _rounded_hdim(d):
    return (d + 31) // 32 * 32 if d <= 128 else (d + 63) // 64 * 64


def hdims(*ds):
    return [d for d in ds if _rounded_hdim(d) in HDIMS_BUILT]


DTYPES = [torch.bfloat16] + ([] if DISABLE_FP16 else [torch.float16])
DROPOUTS = [0.0] + ([] if DISABLE_DROPOUT else [0.17])
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


def s_dmask_to_dense(S_dmask, n):
    """(batch, nheads, 2 hr, 2 hr) S_dmask of the kernel -> (batch, nheads, 2n, 2n): the clean half starts at hr."""
    hr = S_dmask.shape[-1] // 2
    idx = torch.cat([torch.arange(n), hr + torch.arange(n)]).to(S_dmask.device)
    return S_dmask[:, :, idx][:, :, :, idx]


def dropout_mask_of(S_dmask, n, block_size):
    """Dropout pattern (True = kept) from S_dmask, and the dropout fraction over the visible positions."""
    S = s_dmask_to_dense(S_dmask, n)
    mask = diffusion_mask(n, block_size, device=S.device)
    keep = S >= 0
    visible = mask.expand_as(keep)
    fraction = 1.0 - (keep & visible).sum().item() / max(visible.sum().item(), 1)
    return keep, fraction


# Attention matrices of more elements than this: the references are computed sample by sample, each with its backward
# right away. The samples are independent, so this gives the same values with the memory of one sample (the long
# sequences of FlashAttention-3's cases would not fit in 46-48 GB otherwise).
_REF_CHUNK_ELEMENTS = 2 ** 28


def _ref_results(q, k, v, g, block_size, cu_seqlens=None, dropout_mask=None, **kw):
    """(out, lse, dq, dk, dv) of attention_ref (batched inputs) or attention_varlen_ref (packed samples, cu_seqlens),
    with the keyword arguments kw of the reference."""
    if cu_seqlens is None:
        b, S, h = q.shape[0], q.shape[1], q.shape[2]
        if b * h * S * S <= _REF_CHUNK_ELEMENTS:
            out, lse, _ = attention_ref(q, k, v, block_size, dropout_mask=dropout_mask, **kw)
            return (out, lse) + torch.autograd.grad(out, (q, k, v), g)
        parts = []
        for i in range(b):
            qi, ki, vi = [x[i:i + 1].detach().requires_grad_() for x in (q, k, v)]
            mi = None if dropout_mask is None else dropout_mask[i:i + 1]
            out, lse, _ = attention_ref(qi, ki, vi, block_size, dropout_mask=mi, **kw)
            parts.append((out.detach(), lse.detach()) + torch.autograd.grad(out, (qi, ki, vi), g[i:i + 1]))
        return tuple(torch.cat(t, dim=0) for t in zip(*parts))
    cu = cu_seqlens.tolist()
    h = q.shape[1]
    if h * sum((cu[i + 1] - cu[i]) ** 2 for i in range(len(cu) - 1)) <= _REF_CHUNK_ELEMENTS:
        out, lse = attention_varlen_ref(q, k, v, cu_seqlens, block_size, dropout_mask=dropout_mask, **kw)
        return (out, lse) + torch.autograd.grad(out, (q, k, v), g)
    parts = []
    for i in range(len(cu) - 1):
        s, e = cu[i], cu[i + 1]
        qi, ki, vi = [x[s:e].detach().requires_grad_() for x in (q, k, v)]
        mi = None if dropout_mask is None else [dropout_mask[i]]
        cu_i = torch.tensor([0, e - s], device=cu_seqlens.device, dtype=cu_seqlens.dtype)
        out, lse = attention_varlen_ref(qi, ki, vi, cu_i, block_size, dropout_mask=mi, **kw)
        parts.append((out.detach(), lse.detach()) + torch.autograd.grad(out, (qi, ki, vi), g[s:e]))
    outs, lses, dqs, dks, dvs = zip(*parts)
    return (torch.cat(outs, dim=0), torch.cat(lses, dim=-1), torch.cat(dqs, dim=0), torch.cat(dks, dim=0),
            torch.cat(dvs, dim=0))


def _check(q, k, v, block_size, dropout_p=0.0, softcap=0.0, deterministic=False, cu_seqlens=None, max_seqlen=None,
           packing=None, check_lse=True):
    """Runs the kernel (batched if cu_seqlens is None, else varlen; packing: None / 'kv' / 'qkv') and compares the
    output, the LSE and the gradients with the reference."""
    g = torch.randn(q.shape, device=q.device, dtype=q.dtype)
    kw = dict(dropout_p=dropout_p, softcap=softcap, deterministic=deterministic, return_attn_probs=True)
    if cu_seqlens is None:
        if packing == "qkv":
            qkv = torch.stack([q, k, v], dim=2).detach().requires_grad_()
            out, lse, S_dmask = diffusion_attn_qkvpacked_func(qkv, block_size, **kw)
            inputs = (qkv,)
        elif packing == "kv":
            kv = torch.stack([k, v], dim=2).detach().requires_grad_()
            out, lse, S_dmask = diffusion_attn_kvpacked_func(q, kv, block_size, **kw)
            inputs = (q, kv)
        else:
            out, lse, S_dmask = diffusion_attn_func(q, k, v, block_size, **kw)
            inputs = (q, k, v)
        n = q.shape[1] // 2
        dropout_mask = None
        if dropout_p > 0:
            dropout_mask, fraction = dropout_mask_of(S_dmask, n, block_size)
            assert abs(fraction - dropout_p) <= 0.01 + 2.0 / math.sqrt(max(num_visible_pairs(n, block_size), 1)), \
                f"dropout fraction {fraction} vs {dropout_p}"
        ref_mask = dropout_mask
    else:
        if packing == "qkv":
            qkv = torch.stack([q, k, v], dim=1).detach().requires_grad_()
            out, lse, S_dmask = diffusion_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, block_size, **kw)
            inputs = (qkv,)
        elif packing == "kv":
            kv = torch.stack([k, v], dim=1).detach().requires_grad_()
            out, lse, S_dmask = diffusion_attn_varlen_kvpacked_func(q, kv, cu_seqlens, cu_seqlens, max_seqlen,
                                                                    max_seqlen, block_size, **kw)
            inputs = (q, kv)
        else:
            out, lse, S_dmask = diffusion_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
                                                           block_size, **kw)
            inputs = (q, k, v)
        dropout_masks = None
        if dropout_p > 0:
            cu = cu_seqlens.tolist()
            dropout_masks = []
            for i in range(len(cu) - 1):
                n_i = (cu[i + 1] - cu[i]) // 2
                keep, _ = dropout_mask_of(S_dmask[i:i + 1], n_i, block_size)
                dropout_masks.append(keep)
        ref_mask = dropout_masks
    # The references: float32, and a plain PyTorch implementation in the input precision (output and gradients)
    ref_kw = dict(cu_seqlens=cu_seqlens, dropout_mask=ref_mask, softcap=softcap, dropout_p=dropout_p)
    out_ref, lse_ref, dq_ref, dk_ref, dv_ref = _ref_results(q, k, v, g, block_size, **ref_kw)
    out_pt, lse_pt, dq_pt, dk_pt, dv_pt = _ref_results(q, k, v, g, block_size, upcast=False, reorder_ops=True, **ref_kw)
    _errors_ok("out", out, out_ref, out_pt)
    if check_lse:
        assert (lse - lse_ref).abs().max().item() <= 2 * (lse_pt - lse_ref).abs().max().item() + 1e-4
    grads = torch.autograd.grad(out, inputs, g)
    if packing == "qkv":
        dim = 2 if cu_seqlens is None else 1
        dq, dk, dv = grads[0].unbind(dim)
    elif packing == "kv":
        dim = 2 if cu_seqlens is None else 1
        dq = grads[0]
        dk, dv = grads[1].unbind(dim)
    else:
        dq, dk, dv = grads
    for name, x, r, p in [("dq", dq, dq_ref, dq_pt), ("dk", dk, dk_ref, dk_pt), ("dv", dv, dv_ref, dv_pt)]:
        _errors_ok(name, x, r, p)
    return out, (dq, dk, dv)


def _skip_unsupported(d, dropout_p):
    if dropout_p > 0 and SMEM_LIMITED and _rounded_hdim(d) == 256:
        pytest.skip("no dropout in the backward with head dimension 256 on sm86 / sm89 (as FlashAttention-2)")


def _rand(shape, dtype):
    return torch.randn(*shape, device=DEVICE, dtype=dtype, requires_grad=True)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("d", hdims(32, 64, 96, 128, 192, 256))
@pytest.mark.parametrize("dropout_p", DROPOUTS)
@pytest.mark.parametrize("nheads, nheads_k", [(4, 4), (6, 2), (4, 1)])
@pytest.mark.parametrize(
    "batch, n, block_size",
    [(2, 128, 4), (1, 256, 32), (3, 100, 3), (2, 333, 16), (1, 517, 7), (1, 64, 64), (1, 300, 128),
     (2, 191, 200), (1, 1, 1), (2, 5, 2), (1, 1024, 1), (1, 777, 64)],
)
def test_diffusion_attn(batch, n, block_size, nheads, nheads_k, dropout_p, d, dtype):
    _skip_unsupported(d, dropout_p)
    if n == 1 and dropout_p > 0:
        # Every query sees a single key: dQ = dK = 0 exactly, and the kernel (like FlashAttention-2) computes
        # dS = P (dP - rowsum(dO * O)) from the output O rounded to the input precision, i.e. pure rounding noise,
        # while the PyTorch softmax backward gives exactly 0: the error ratio is meaningless.
        pytest.skip("degenerate problem with dropout")
    torch.random.manual_seed(n * 1000 + block_size)
    q = _rand((batch, 2 * n, nheads, d), dtype)
    k = _rand((batch, 2 * n, nheads_k, d), dtype)
    v = _rand((batch, 2 * n, nheads_k, d), dtype)
    _check(q, k, v, block_size, dropout_p=dropout_p)


@pytest.mark.skipif(DISABLE_SOFTCAP, reason="softcap kernels not built")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("d", hdims(64, 128, 256))
@pytest.mark.parametrize("batch, n, block_size", [(2, 128, 4), (1, 333, 16), (2, 191, 200), (1, 1024, 1)])
def test_diffusion_attn_softcap(batch, n, block_size, d, dtype):
    torch.random.manual_seed(n + block_size)
    q, k, v = [_rand((batch, 2 * n, h, d), dtype) for h in (4, 2, 2)]
    _check(q, k, v, block_size, softcap=15.0)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("d", hdims(40, 59, 80, 111, 160, 224))
@pytest.mark.parametrize("dropout_p", DROPOUTS)
@pytest.mark.parametrize("block_size", [1, 4, 37])
def test_diffusion_attn_headdims(block_size, dropout_p, d, dtype):
    """Head dimensions that are not one of the kernel sizes: padded inside the kernel (multiples of 8) or by the
    Python interface (as FlashAttention-2)."""
    _skip_unsupported(d, dropout_p)
    torch.random.manual_seed(d)
    batch, n, nheads, nheads_k = 2, 211, 4, 2
    q = _rand((batch, 2 * n, nheads, d), dtype)
    k = _rand((batch, 2 * n, nheads_k, d), dtype)
    v = _rand((batch, 2 * n, nheads_k, d), dtype)
    _check(q, k, v, block_size, dropout_p=dropout_p)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("d", hdims(64, 128))
@pytest.mark.parametrize("packing", ["kv", "qkv"])
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("dropout_p", DROPOUTS)
def test_packed(dropout_p, varlen, packing, d, dtype):
    torch.random.manual_seed(3)
    nheads_k = 2 if packing == "kv" else 4
    if not varlen:
        q, k, v = [_rand((2, 2 * 150, h, d), dtype) for h in (4, nheads_k, nheads_k)]
        _check(q, k, v, 8, dropout_p=dropout_p, packing=packing)
    else:
        ns = [17, 150, 64]
        cu = torch.tensor([0] + [2 * x for x in ns], device=DEVICE).cumsum(0).to(torch.int32)
        q, k, v = [_rand((int(cu[-1]), h, d), dtype) for h in (4, nheads_k, nheads_k)]
        _check(q, k, v, 8, dropout_p=dropout_p, cu_seqlens=cu, max_seqlen=2 * max(ns), packing=packing)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("d", hdims(32, 64, 128, 256))
@pytest.mark.parametrize("dropout_p", DROPOUTS)
@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("nheads, nheads_k", [(4, 4), (6, 2)])
@pytest.mark.parametrize("block_size", [1, 4, 13, 64, 160])
def test_diffusion_attn_varlen(block_size, nheads, nheads_k, deterministic, dropout_p, d, dtype):
    _skip_unsupported(d, dropout_p)
    torch.random.manual_seed(block_size)
    ns = [17, 300, 1, 128, 513, 64, 2, 255]
    seqlens = [2 * x for x in ns]
    cu_seqlens = torch.tensor([0] + list(torch.tensor(seqlens).cumsum(0)), device=DEVICE, dtype=torch.int32)
    total = sum(seqlens)
    q = _rand((total, nheads, d), dtype)
    k = _rand((total, nheads_k, d), dtype)
    v = _rand((total, nheads_k, d), dtype)
    _check(q, k, v, block_size, dropout_p=dropout_p, deterministic=deterministic, cu_seqlens=cu_seqlens,
           max_seqlen=max(seqlens))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("d", hdims(32, 64, 96, 128, 192, 256))
@pytest.mark.parametrize("dropout_p", DROPOUTS)
@pytest.mark.parametrize("varlen", [False, True])
@pytest.mark.parametrize("block_size", [4, 7])
@pytest.mark.parametrize("n", [600, 1100, 2100])
def test_long_sequence_tiles(n, block_size, varlen, dropout_p, d, dtype):
    """Long halves (several 128-row query tiles, partial last tiles), with block sizes that divide 16 (16-key
    diagonal sub-tiles of the 128-row forward tiles) and that do not, batched and varlen, with and without dropout:
    both sides of the length-dependent configurations of the dispatch (1K and 2K tokens per half)."""
    _skip_unsupported(d, dropout_p)
    torch.random.manual_seed(n + d + block_size)
    nheads, nheads_k = 2, 1
    ns = [n] if not varlen else [300, n, 17]
    cu = torch.tensor([0] + [2 * x for x in ns], device=DEVICE).cumsum(0).to(torch.int32)
    q, k, v = [_rand((int(cu[-1]), h, d), dtype) for h in (nheads, nheads_k, nheads_k)]
    if not varlen:
        _check(*[x.detach().unflatten(0, (1, -1)).requires_grad_() for x in (q, k, v)], block_size, dropout_p=dropout_p)
    else:
        _check(q, k, v, block_size, dropout_p=dropout_p, cu_seqlens=cu, max_seqlen=2 * max(ns))


def test_zero_length_varlen_sample():
    torch.random.manual_seed(0)
    seqlens = [0, 200, 0, 64]
    cu = torch.tensor([0] + list(torch.tensor(seqlens).cumsum(0)), device=DEVICE, dtype=torch.int32)
    total = sum(seqlens)
    q, k, v = [_rand((total, 4, 128), torch.bfloat16) for _ in range(3)]
    _check(q, k, v, 4, cu_seqlens=cu, max_seqlen=max(seqlens))


@pytest.mark.parametrize("d", hdims(64, 128))
def test_varlen_matches_batched(d, monkeypatch):
    """Packing samples of equal length gives exactly the result of the batched kernel (deterministic backward, so
    that the order of the dQ accumulations is the same; without split-KV in the batched forward, which the varlen
    forward does not use, as in FlashAttention-2)."""
    monkeypatch.setenv("FDA_FWD_NUM_SPLITS", "1")
    torch.random.manual_seed(0)
    batch, n, nheads, nheads_k, block_size = 3, 200, 4, 2, 8
    q = _rand((batch, 2 * n, nheads, d), torch.bfloat16)
    k = _rand((batch, 2 * n, nheads_k, d), torch.bfloat16)
    v = _rand((batch, 2 * n, nheads_k, d), torch.bfloat16)
    g = torch.randn_like(q)
    out, lse, _ = diffusion_attn_func(q, k, v, block_size, deterministic=True, return_attn_probs=True)
    grads = torch.autograd.grad(out, (q, k, v), g)
    cu = torch.arange(0, batch * 2 * n + 1, 2 * n, device=DEVICE, dtype=torch.int32)
    qv, kv, vv = [x.detach().reshape(-1, *x.shape[2:]).requires_grad_() for x in (q, k, v)]
    out_v, lse_v, _ = diffusion_attn_varlen_func(qv, kv, vv, cu, cu, 2 * n, 2 * n, block_size, deterministic=True,
                                                 return_attn_probs=True)
    grads_v = torch.autograd.grad(out_v, (qv, kv, vv), g.reshape(-1, *g.shape[2:]))
    assert torch.equal(out.reshape(-1, *out.shape[2:]), out_v)
    assert torch.equal(lse.permute(1, 0, 2).reshape(nheads, -1), lse_v)
    for x, y in zip(grads, grads_v):
        assert torch.equal(x.reshape(-1, *x.shape[2:]), y)


# MQA / GQA cases of FlashAttention-3's tests (hopper/test_flash_attn.py, test_flash_attn_output and
# test_flash_attn_varlen_output): mha_type mha / mqa / gqa with nheads = 6 and nheads_kv = 6 / 1 / 2, batch size 9 (2
# beyond 2048 tokens), BF16 / FP16, softcap 0 / 15, all the head dimensions, its equal query / key lengths L as the
# lengths of the halves (samples of 2L tokens), its random varlen lengths (generate_random_padding_mask, mode
# "random": [max(1, L - 20), L]). Block sizes 1 (the clean half is causal attention, FlashAttention-3's causal), 4
# (divides 16) and 64. The GQA-grouped backward is exercised by running the suite with FDA_BWD_HEADS_PER_CTA.
def _fa3_nheads(mha_type):
    nheads = 6
    return nheads, nheads if mha_type == "mha" else (2 if mha_type == "gqa" else 1)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("softcap", [0.0] + ([15.0] if not DISABLE_SOFTCAP else []))
@pytest.mark.parametrize("d", hdims(32, 64, 96, 128, 192, 256))
@pytest.mark.parametrize("block_size", [1, 4, 64])
@pytest.mark.parametrize("L", [1, 256, 1024, 4096, 4224])
def test_gqa_fa3(L, block_size, d, softcap, mha_type, dtype, request):
    if L >= 2048:
        request.getfixturevalue("large_memory")
    torch.random.manual_seed(0)
    batch_size = 9 if L <= 2048 else 2
    nheads, nheads_kv = _fa3_nheads(mha_type)
    q = _rand((batch_size, 2 * L, nheads, d), dtype)
    k = _rand((batch_size, 2 * L, nheads_kv, d), dtype)
    v = _rand((batch_size, 2 * L, nheads_kv, d), dtype)
    _check(q, k, v, block_size, softcap=softcap)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("softcap", [0.0] + ([15.0] if not DISABLE_SOFTCAP else []))
@pytest.mark.parametrize("d", hdims(32, 64, 96, 128, 192, 256))
@pytest.mark.parametrize("block_size", [1, 4, 64])
@pytest.mark.parametrize("L", [1, 128, 256, 1024, 2048, 4096])
def test_gqa_fa3_varlen(L, block_size, d, softcap, mha_type, dtype, request):
    if L >= 2048:
        request.getfixturevalue("large_memory")
    torch.random.manual_seed(L + d + block_size)
    batch_size = 9 if L <= 2048 else 2
    nheads, nheads_kv = _fa3_nheads(mha_type)
    ns = torch.randint(max(1, L - 20), L + 1, (batch_size,)).tolist()
    seqlens = [2 * x for x in ns]
    cu_seqlens = torch.tensor([0] + list(torch.tensor(seqlens).cumsum(0)), device=DEVICE, dtype=torch.int32)
    total = sum(seqlens)
    q = _rand((total, nheads, d), dtype)
    k = _rand((total, nheads_kv, d), dtype)
    v = _rand((total, nheads_kv, d), dtype)
    _check(q, k, v, block_size, softcap=softcap, cu_seqlens=cu_seqlens, max_seqlen=max(seqlens))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("heads_per_cta", [2, 64])
@pytest.mark.parametrize("mha_type", ["mqa", "gqa"])
@pytest.mark.parametrize("d", hdims(32, 64, 96, 128, 192, 256))
@pytest.mark.parametrize("block_size", [1, 4, 64])
@pytest.mark.parametrize("L, varlen", [(256, False), (1024, False), (256, True)])
def test_gqa_grouped(L, varlen, block_size, d, mha_type, heads_per_cta, dtype, monkeypatch):
    """The GQA-grouped backward (thread blocks summing the dK / dV of several query heads of a KV head,
    bwd_heads_per_cta in flash_api.cpp), which the default choice uses only for some GPUs and shapes, forced on cases
    of test_gqa_fa3 / test_gqa_fa3_varlen: 64, all the query heads of a KV head (final dK / dV); 2, partial sums of 2
    query heads, summed afterwards (h / h_k = 6; with h / h_k = 3, the kernel is the ungrouped one)."""
    monkeypatch.setenv("FDA_BWD_HEADS_PER_CTA", str(heads_per_cta))
    torch.random.manual_seed(L + d + block_size)
    batch_size = 9
    nheads, nheads_kv = _fa3_nheads(mha_type)
    if not varlen:
        q = _rand((batch_size, 2 * L, nheads, d), dtype)
        k = _rand((batch_size, 2 * L, nheads_kv, d), dtype)
        v = _rand((batch_size, 2 * L, nheads_kv, d), dtype)
        _check(q, k, v, block_size)
        return
    ns = torch.randint(max(1, L - 20), L + 1, (batch_size,)).tolist()
    seqlens = [2 * x for x in ns]
    cu_seqlens = torch.tensor([0] + list(torch.tensor(seqlens).cumsum(0)), device=DEVICE, dtype=torch.int32)
    total = sum(seqlens)
    q = _rand((total, nheads, d), dtype)
    k = _rand((total, nheads_kv, d), dtype)
    v = _rand((total, nheads_kv, d), dtype)
    _check(q, k, v, block_size, cu_seqlens=cu_seqlens, max_seqlen=max(seqlens))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("d", hdims(32, 64, 96, 128, 192, 256))
@pytest.mark.parametrize("num_splits", [2, 3, 7, 64])
@pytest.mark.parametrize("batch, n, block_size, nheads, nheads_k, softcap",
                         [(1, 77, 4, 2, 2, 0.0), (2, 512, 16, 4, 2, 0.0), (1, 1000, 7, 2, 1, 0.0), (1, 300, 64, 2, 2, 30.0)])
def test_split_kv(batch, n, block_size, nheads, nheads_k, softcap, num_splits, d, dtype, monkeypatch):
    """The split-KV forward (FlashAttention-2's, chosen by the batched forward without dropout when its query tiles do
    not fill the GPU) with forced numbers of splits, including more splits than key tiles (empty splits)."""
    if softcap > 0 and DISABLE_SOFTCAP:
        pytest.skip("softcap kernels not built")
    monkeypatch.setenv("FDA_FWD_NUM_SPLITS", str(num_splits))
    torch.random.manual_seed(0)
    q = _rand((batch, 2 * n, nheads, d), dtype)
    k = _rand((batch, 2 * n, nheads_k, d), dtype)
    v = _rand((batch, 2 * n, nheads_k, d), dtype)
    _check(q, k, v, block_size, softcap=softcap)


@pytest.mark.parametrize("d", hdims(64, 128, 192, 256))
@pytest.mark.parametrize("nheads, nheads_k", [(4, 4), (8, 2)])
@pytest.mark.parametrize("varlen", [False, True])
def test_deterministic(varlen, nheads, nheads_k, d):
    torch.random.manual_seed(1)
    batch, n, block_size = 2, 700, 4
    q = _rand((batch, 2 * n, nheads, d), torch.bfloat16)
    k = _rand((batch, 2 * n, nheads_k, d), torch.bfloat16)
    v = _rand((batch, 2 * n, nheads_k, d), torch.bfloat16)
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


@pytest.mark.skipif(DISABLE_DROPOUT, reason="dropout kernels not built")
def test_dropout_rng():
    """Dropout: the same RNG state gives the same pattern, consecutive calls give different patterns, and the
    gradients are those of the pattern of the forward pass (tested in _check)."""
    torch.random.manual_seed(0)
    q, k, v = [_rand((2, 2 * 256, 4, 128), torch.bfloat16) for _ in range(3)]
    state = torch.cuda.get_rng_state()
    out1, _, S1 = diffusion_attn_func(q, k, v, 16, dropout_p=0.3, return_attn_probs=True)
    torch.cuda.set_rng_state(state)
    out2, _, S2 = diffusion_attn_func(q, k, v, 16, dropout_p=0.3, return_attn_probs=True)
    out3, _, S3 = diffusion_attn_func(q, k, v, 16, dropout_p=0.3, return_attn_probs=True)
    assert torch.equal(out1, out2)
    mask = diffusion_mask(256, 16, device=DEVICE)
    k1, k3 = s_dmask_to_dense(S1, 256) >= 0, s_dmask_to_dense(S3, 256) >= 0
    assert (k1 != k3)[:, :, mask].float().mean().item() > 0.2
    out_nodrop = diffusion_attn_func(q, k, v, 16, dropout_p=0.3, return_attn_probs=False)
    assert out_nodrop.shape == out1.shape


def test_softmax_scale():
    torch.random.manual_seed(0)
    q, k, v = [_rand((1, 400, 2, 128), torch.bfloat16) for _ in range(3)]
    for scale in [0.05, 0.3]:
        out = diffusion_attn_func(q, k, v, 8, softmax_scale=scale)
        out_ref, _, _ = attention_ref(q, k, v, 8, softmax_scale=scale)
        out_pt, _, _ = attention_ref(q, k, v, 8, softmax_scale=scale, upcast=False, reorder_ops=True)
        _errors_ok("out", out, out_ref, out_pt)


def test_non_contiguous_inputs():
    """q, k, v slices of a packed qkv tensor (strided batch / row / head dimensions)."""
    torch.random.manual_seed(0)
    batch, n, nheads, d = 2, 150, 4, 128
    qkv = torch.randn(batch, 2 * n, 3, nheads, d, device=DEVICE, dtype=torch.bfloat16, requires_grad=True)
    q, k, v = qkv.unbind(2)
    _check(q, k, v, 4)


@pytest.mark.parametrize("d", hdims(64, 128))
def test_no_systematic_error(d):
    """Over many random problems, the mean bias and the mean relative scale error (slope of a least-squares fit of the
    kernel result against the reference) of the output and of every gradient must be zero within 4 standard errors,
    and their magnitude must not exceed that of the PyTorch implementation in the same precision."""
    torch.random.manual_seed(123)
    stats = {name: {"fda": [], "pt": []} for name in ("out", "dq", "dk", "dv")}
    for trial in range(24):
        batch, n, nheads, nheads_k = 1, int(torch.randint(50, 400, (1,))), 4, 2
        block_size = int(torch.randint(1, 40, (1,)))
        q = _rand((batch, 2 * n, nheads, d), torch.bfloat16)
        k = _rand((batch, 2 * n, nheads_k, d), torch.bfloat16)
        v = _rand((batch, 2 * n, nheads_k, d), torch.bfloat16)
        g = torch.randn_like(q)
        out = diffusion_attn_func(q, k, v, block_size)
        res = {"fda": [out, *torch.autograd.grad(out, (q, k, v), g)]}
        out_ref, _, _ = attention_ref(q, k, v, block_size)
        ref = [out_ref, *torch.autograd.grad(out_ref, (q, k, v), g)]
        out_pt, _, _ = attention_ref(q, k, v, block_size, upcast=False, reorder_ops=True)
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


@pytest.mark.skipif(fa2_func is None, reason="FlashAttention-2 (flash_attn) not installed")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("d", hdims(32, 64, 96, 128, 192, 256))
def test_accuracy_matches_fa2(d, dtype):
    """With block_size = 1, the clean half is causal attention on the clean tokens, and with a zero output gradient
    on the noisy half, the gradients of the clean half are those of that causal attention. On this subproblem, the
    errors w.r.t. float32 of the output, the LSE and dQ / dK / dV must match those of FlashAttention-2 (mean error at
    most 10% larger, max error at most 50% larger plus a few ulps)."""
    torch.random.manual_seed(d)
    batch, n, nheads, nheads_k = 2, 777, 6, 2
    q, k, v = [_rand((batch, 2 * n, h, d), dtype) for h in (nheads, nheads_k, nheads_k)]
    out, lse, _ = diffusion_attn_func(q, k, v, 1, return_attn_probs=True)
    qc, kc, vc = [x[:, n:].detach().clone().requires_grad_() for x in (q, k, v)]
    out_fa2, lse_fa2, _ = fa2_func(qc, kc, vc, causal=True, return_attn_probs=True)
    out_ref, lse_ref, _ = attention_ref(q, k, v, 1)
    results = {"out": (out[:, n:], out_fa2, out_ref[:, n:]), "lse": (lse[:, :, n:], lse_fa2, lse_ref[:, :, n:])}
    g = torch.randn_like(out)
    g[:, :n] = 0
    grads = torch.autograd.grad(out, (q, k, v), g)
    grads_fa2 = torch.autograd.grad(out_fa2, (qc, kc, vc), g[:, n:])
    grads_ref = torch.autograd.grad(out_ref, (q, k, v), g)
    for name, x, y, r in zip(("dq", "dk", "dv"), grads, grads_fa2, grads_ref):
        # dS = P (dP - dPsum) = 0 on the noisy queries, the only ones that see the noisy keys
        assert torch.all(x[:, :n] == 0), f"{name} of the noisy half is not zero"
        results[name] = (x[:, n:], y, r[:, n:])
    for name, (x, y, r) in results.items():
        x, y, r = x.float(), y.float(), r.float()
        atol = 2 * (r + 0.3 - 0.3 - r).abs().max().item() + 1e-6
        err_mean, err_mean_fa2 = (x - r).abs().mean().item(), (y - r).abs().mean().item()
        err_max, err_max_fa2 = (x - r).abs().max().item(), (y - r).abs().max().item()
        assert err_mean <= 1.1 * err_mean_fa2 + 1e-7, f"{name}: mean error {err_mean} vs FA2 {err_mean_fa2}"
        assert err_max <= 1.5 * err_max_fa2 + atol, f"{name}: max error {err_max} vs FA2 {err_max_fa2}"


def test_torch_compile():
    torch.random.manual_seed(0)
    q, k, v = [_rand((2, 256, 4, 128), torch.bfloat16) for _ in range(3)]
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


@pytest.mark.parametrize("d", hdims(128))
def test_torch_compile_varlen(d):
    torch.random.manual_seed(0)
    cu = torch.tensor([0, 100, 356], device=DEVICE, dtype=torch.int32)
    q, k, v = [_rand((356, 4, d), torch.bfloat16) for _ in range(3)]

    def f(q, k, v):
        return diffusion_attn_varlen_func(q, k, v, cu, cu, 256, 256, 8)

    out = f(q, k, v)
    out_c = torch.compile(f, fullgraph=True)(q, k, v)
    assert torch.equal(out, out_c)
    out_c.sum().backward()


def test_errors():
    q = torch.randn(1, 15, 2, 128, device=DEVICE, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="even"):
        diffusion_attn_func(q, q, q, 4)
    q = torch.randn(1, 16, 2, 128, device=DEVICE, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="block_size"):
        diffusion_attn_func(q, q, q, 0)
    with pytest.raises(RuntimeError):
        diffusion_attn_func(q.float(), q.float(), q.float(), 4)
    with pytest.raises(RuntimeError, match="dropout"):
        diffusion_attn_func(q, q, q, 4, dropout_p=0.1, softcap=10.0)
