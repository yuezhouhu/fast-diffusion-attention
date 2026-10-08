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
    dropout_p: float = 0.0,
    dropout_mask: Optional[torch.Tensor] = None,
    upcast: bool = True,
    reorder_ops: bool = False,
):
    """Diffusion attention on batched inputs (attention_ref of the FlashAttention-2 tests, with the diffusion mask).

    q: (batch, 2n, nheads, d), k / v: (batch, 2n, nheads_k, d). Returns (out, lse, attention) with out: (batch, 2n,
    nheads, d) in the dtype of q, lse: (batch, nheads, 2n) in float32 and attention: (batch, nheads, 2n, 2n), the
    attention probabilities (before dropout).

    dropout_mask: (batch, nheads, 2n, 2n) bool, True where the probabilities are kept; they are scaled by
    1 / (1 - dropout_p).
    upcast=True computes everything in float32 (the reference). upcast=False mimics a straightforward implementation
    in the input precision (used to measure the error of a "plain PyTorch" implementation); reorder_ops=True scales k
    instead of q (as in the FlashAttention tests).
    """
    dtype_og = q.dtype
    batch, seqlen, nheads, d = q.shape
    assert seqlen % 2 == 0, "the sequence length must be even"
    nheads_k = k.shape[2]
    g = nheads // nheads_k
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
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
    dropout_scaling = 1.0 / (1 - dropout_p)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    out = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    return out.to(dtype=dtype_og), lse, attention.to(dtype=dtype_og)


def attention_varlen_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    block_size: int,
    dropout_mask: Optional[list] = None,
    **kwargs,
):
    """Diffusion attention on packed samples: q (total, nheads, d), k / v (total, nheads_k, d), sample i occupies
    the rows [cu_seqlens[i], cu_seqlens[i + 1]). Returns (out, lse) with lse: (nheads, total).
    dropout_mask: optional list of the (1, nheads, 2 n_i, 2 n_i) dropout masks of the samples."""
    cu = cu_seqlens.tolist()
    outs, lses = [], []
    for i in range(len(cu) - 1):
        s, e = cu[i], cu[i + 1]
        mask_i = None if dropout_mask is None else dropout_mask[i]
        o, l, _ = attention_ref(q[None, s:e], k[None, s:e], v[None, s:e], block_size, dropout_mask=mask_i, **kwargs)
        outs.append(o[0])
        lses.append(l[0])
    return torch.cat(outs, dim=0), torch.cat(lses, dim=-1)
