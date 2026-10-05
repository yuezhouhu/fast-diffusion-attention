# Copyright (c) 2026, fast-diffusion-attention authors.
"""Reference (PyTorch) implementation of the diffusion attention.

Every sample is laid out as ``[noisy (n tokens) | clean (n tokens)]``. The positions of each half are split in
blocks of ``block_size`` tokens, counted from the start of the half. With i, j positions of the noisy half and c, c'
positions of the clean half (block(x) = x // block_size):

* noisy i sees noisy j iff block(j) == block(i)   (its own block, bidirectional)
* noisy i sees clean c' iff block(c') < block(i)  (all the previous clean blocks)
* clean c sees clean c' iff block(c') <= block(c) (block-causal)
* clean c never sees the noisy half.
"""

import math
from typing import Optional

import torch


def diffusion_mask(n: int, block_size: int, device=None) -> torch.Tensor:
    """Boolean (2n, 2n) mask of one sample, True where the query (row) attends to the key (column)."""
    idx = torch.arange(2 * n, device=device)
    clean = idx >= n
    blk = torch.where(clean, idx - n, idx) // block_size
    q_clean, k_clean = clean[:, None], clean[None, :]
    q_blk, k_blk = blk[:, None], blk[None, :]
    noisy_noisy = ~q_clean & ~k_clean & (q_blk == k_blk)
    noisy_clean = ~q_clean & k_clean & (k_blk < q_blk)
    clean_clean = q_clean & k_clean & (k_blk <= q_blk)
    return noisy_noisy | noisy_clean | clean_clean


def num_visible_pairs(n: int, block_size: int) -> int:
    """Number of (query, key) pairs visible in a sample with halves of n tokens (exact, without the mask)."""
    num_blocks = (n + block_size - 1) // block_size
    sizes = [min(block_size, n - b * block_size) for b in range(num_blocks)]
    total, prefix = 0, 0
    for s in sizes:
        total += s * s            # noisy block attends to itself
        total += s * prefix       # noisy block attends to the previous clean blocks
        total += s * (prefix + s)  # clean block attends to the clean prefix and itself
        prefix += s
    return total


def attention_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int,
    softmax_scale: Optional[float] = None,
    softcap: float = 0.0,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    upcast: bool = True,
    reorder_ops: bool = False,
    intermediate_dtype: Optional[torch.dtype] = None,
):
    """Diffusion attention on batched inputs.

    q: (batch, 2n, nheads, d), k / v: (batch, 2n, nheads_k, d). Returns (out, lse) with out: (batch, 2n, nheads, d)
    in the dtype of q (bf16 for fp8 inputs) and lse: (batch, nheads, 2n) in float32.

    upcast=True computes everything in float32 (the reference). upcast=False mimics a straightforward implementation
    in the input precision (used to measure the error of a "plain PyTorch" implementation).
    """
    dtype_og = q.dtype if q.dtype != torch.float8_e4m3fn else torch.bfloat16
    batch, seqlen, nheads, d = q.shape
    assert seqlen % 2 == 0, "the sequence length must be even"
    nheads_k = k.shape[2]
    g = nheads // nheads_k
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    if q_descale is not None:
        q = (q.float() * q_descale.repeat_interleave(g, dim=1)[:, None, :, None]).to(q.dtype)
    if k_descale is not None:
        k = (k.float() * k_descale[:, None, :, None]).to(k.dtype)
    if v_descale is not None:
        v = (v.float() * v_descale[:, None, :, None]).to(v.dtype)
    if q.dtype == torch.float8_e4m3fn:
        q, k, v = q.to(dtype_og), k.to(dtype_og), v.to(dtype_og)
    k = k.repeat_interleave(g, dim=2)
    v = v.repeat_interleave(g, dim=2)
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(d)
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q * softmax_scale, k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
    if softcap > 0:
        scores = torch.tanh(scores / softcap) * softcap
    mask = diffusion_mask(seqlen // 2, block_size, device=q.device)
    scores = scores.masked_fill(~mask, float("-inf"))
    lse = torch.logsumexp(scores.float(), dim=-1)
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    if intermediate_dtype is not None:
        attention = attention.to(intermediate_dtype).to(attention.dtype)
    out = torch.einsum("bhts,bshd->bthd", attention, v)
    return out.to(dtype=dtype_og), lse


def attention_varlen_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    block_size: int,
    **kwargs,
):
    """Diffusion attention on packed samples: q (total, nheads, d), k / v (total, nheads_k, d), sample i occupies
    the rows [cu_seqlens[i], cu_seqlens[i + 1]). Returns (out, lse) with lse: (nheads, total)."""
    cu = cu_seqlens.tolist()
    outs, lses = [], []
    for i in range(len(cu) - 1):
        s, e = cu[i], cu[i + 1]
        o, l = attention_ref(q[None, s:e], k[None, s:e], v[None, s:e], block_size, **kwargs)
        outs.append(o[0])
        lses.append(l[0])
    return torch.cat(outs, dim=0), torch.cat(lses, dim=-1)
