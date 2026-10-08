# Copyright (c) 2023, Tri Dao.
# Copyright (c) 2026, fast-diffusion-attention authors.
"""Python interface of the diffusion attention kernels, modeled after the FlashAttention-2 interface
(flash_attn/flash_attn_interface.py).

Every sample is laid out as [noisy half (n tokens) | clean half (n tokens)]. The tokens of each half are split into
blocks of `block_size` tokens; a noisy block attends to itself (bidirectionally) and to all the previous clean blocks,
a clean block attends to itself (bidirectionally) and to all the previous clean blocks.
"""

from typing import Optional, Tuple

import torch

# isort: off
# We need to import the CUDA kernels after importing torch
import fda_2_cuda as fda_gpu
# isort: on


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def round_multiple(x, m):
    return (x + m - 1) // m * m


# Rows of padding after every half of every sample in the FP32 accumulators of the backward pass (flash.h)
_ACCUM_PAD = 128


@torch.library.custom_op("fda::_diffusion_attn_forward", mutates_args=(), device_types="cuda")
def _diffusion_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int,
    dropout_p: float,
    softmax_scale: float,
    softcap: float,
    return_softmax: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, softmax_lse, S_dmask, rng_state = fda_gpu.fwd(
        q, k, v, None, block_size, dropout_p, softmax_scale, softcap, return_softmax,
    )
    return out, softmax_lse, S_dmask, rng_state


@torch.library.register_fake("fda::_diffusion_attn_forward")
def _diffusion_attn_forward_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int,
    dropout_p: float,
    softmax_scale: float,
    softcap: float,
    return_softmax: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    batch_size, seqlen, num_heads, head_size = q.shape
    out = torch.empty_like(q)
    softmax_lse = torch.empty((batch_size, num_heads, seqlen), dtype=torch.float32, device=q.device, layout=q.layout)
    p = torch.empty((0,), dtype=q.dtype, device=q.device, layout=q.layout)
    if return_softmax:
        hr = round_multiple(seqlen // 2, 128)
        p = torch.empty((batch_size, num_heads, 2 * hr, 2 * hr), dtype=q.dtype, device=q.device, layout=q.layout)
    rng_state = torch.empty((2,), dtype=torch.int64, device=q.device)
    return out, softmax_lse, p, rng_state


@torch.library.custom_op("fda::_diffusion_attn_varlen_forward", mutates_args=(), device_types="cuda")
def _diffusion_attn_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    block_size: int,
    dropout_p: float,
    softmax_scale: float,
    softcap: float = 0.0,
    return_softmax: bool = False,
    zero_tensors: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, softmax_lse, S_dmask, rng_state = fda_gpu.varlen_fwd(
        q, k, v, None, cu_seqlens, max_seqlen, block_size, dropout_p, softmax_scale, zero_tensors, softcap,
        return_softmax,
    )
    return out, softmax_lse, S_dmask, rng_state


@torch.library.register_fake("fda::_diffusion_attn_varlen_forward")
def _diffusion_attn_varlen_forward_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    block_size: int,
    dropout_p: float,
    softmax_scale: float,
    softcap: float = 0.0,
    return_softmax: bool = False,
    zero_tensors: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    batch_size = cu_seqlens.numel() - 1
    total, num_heads, _ = q.shape
    out = torch.empty_like(q)
    softmax_lse = torch.empty((num_heads, total), dtype=torch.float32, device=q.device, layout=q.layout)
    p = torch.empty((0,), dtype=q.dtype, device=q.device, layout=q.layout)
    if return_softmax:
        hr = round_multiple(round_multiple(max_seqlen, 2) // 2, 128)
        p = torch.empty((batch_size, num_heads, 2 * hr, 2 * hr), dtype=q.dtype, device=q.device, layout=q.layout)
    rng_state = torch.empty((2,), dtype=torch.int64, device=q.device)
    return out, softmax_lse, p, rng_state


@torch.library.custom_op("fda::_diffusion_attn_backward", mutates_args=("dq", "dk", "dv"), device_types="cuda")
def _diffusion_attn_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    block_size: int,
    dropout_p: float,
    softmax_scale: float,
    softcap: float,
    deterministic: bool,
    rng_state: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # dq, dk, dv are allocated by us so they should already be contiguous
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    dq, dk, dv, softmax_d = fda_gpu.bwd(
        dout, q, k, v, out, softmax_lse, dq, dk, dv, block_size, dropout_p, softmax_scale, softcap, deterministic,
        rng_state,
    )
    return softmax_d


@torch.library.register_fake("fda::_diffusion_attn_backward")
def _diffusion_attn_backward_fake(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    block_size: int,
    dropout_p: float,
    softmax_scale: float,
    softcap: float,
    deterministic: bool,
    rng_state: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch_size, seqlen, num_heads, _ = q.shape
    total_padded = batch_size * seqlen + 2 * _ACCUM_PAD * batch_size
    return torch.empty((num_heads, total_padded), device=q.device, dtype=torch.float32)


@torch.library.custom_op("fda::_diffusion_attn_varlen_backward", mutates_args=("dq", "dk", "dv"), device_types="cuda")
def _diffusion_attn_varlen_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    block_size: int,
    dropout_p: float,
    softmax_scale: float,
    softcap: float,
    deterministic: bool,
    rng_state: Optional[torch.Tensor] = None,
    zero_tensors: bool = False,
) -> torch.Tensor:
    # dq, dk, dv are allocated by us so they should already be contiguous
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    dq, dk, dv, softmax_d = fda_gpu.varlen_bwd(
        dout, q, k, v, out, softmax_lse, dq, dk, dv, cu_seqlens, max_seqlen, block_size, dropout_p, softmax_scale,
        zero_tensors, softcap, deterministic, rng_state,
    )
    return softmax_d


@torch.library.register_fake("fda::_diffusion_attn_varlen_backward")
def _diffusion_attn_varlen_backward_fake(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    block_size: int,
    dropout_p: float,
    softmax_scale: float,
    softcap: float,
    deterministic: bool,
    rng_state: Optional[torch.Tensor] = None,
    zero_tensors: bool = False,
) -> torch.Tensor:
    batch_size = cu_seqlens.numel() - 1
    total, num_heads, _ = q.shape
    total_padded = total + 2 * _ACCUM_PAD * batch_size
    return torch.empty((num_heads, total_padded), device=q.device, dtype=torch.float32)


_wrapped_diffusion_attn_forward = torch.ops.fda._diffusion_attn_forward
_wrapped_diffusion_attn_varlen_forward = torch.ops.fda._diffusion_attn_varlen_forward
_wrapped_diffusion_attn_backward = torch.ops.fda._diffusion_attn_backward
_wrapped_diffusion_attn_varlen_backward = torch.ops.fda._diffusion_attn_varlen_backward


def _pad_head_dim(*xs):
    head_size_og = xs[0].size(-1)
    if head_size_og % 8 == 0:
        return xs
    return [torch.nn.functional.pad(x, [0, 8 - head_size_og % 8]) for x in xs]


def _check_self_attention(cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k):
    if cu_seqlens_k is not None and cu_seqlens_k is not cu_seqlens_q:
        assert cu_seqlens_k.shape == cu_seqlens_q.shape, "cu_seqlens_k must be equal to cu_seqlens_q (self-attention)"
    if max_seqlen_k is not None:
        assert max_seqlen_k == max_seqlen_q, "max_seqlen_k must be equal to max_seqlen_q (self-attention)"


class DiffusionAttnQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, block_size, dropout_p, softmax_scale, softcap, deterministic, return_softmax,
                is_grad_enabled):
        is_grad = is_grad_enabled and qkv.requires_grad
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        q, k, v = qkv[:, :, 0].detach(), qkv[:, :, 1].detach(), qkv[:, :, 2].detach()
        head_size_og = q.size(3)
        q, k, v = _pad_head_dim(q, k, v)
        out_padded, softmax_lse, S_dmask, rng_state = _wrapped_diffusion_attn_forward(
            q, k, v, block_size, dropout_p, softmax_scale, softcap=softcap,
            return_softmax=return_softmax and dropout_p > 0,
        )
        if is_grad:
            ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
            ctx.block_size = block_size
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.softcap = softcap
            ctx.deterministic = deterministic
        out = out_padded[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
        qkv_shape = q.shape[:-2] + (3, *q.shape[-2:])
        dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
        (dout_padded,) = _pad_head_dim(dout)
        _wrapped_diffusion_attn_backward(
            dout_padded, q, k, v, out, softmax_lse, dqkv[:, :, 0], dqkv[:, :, 1], dqkv[:, :, 2],
            ctx.block_size, ctx.dropout_p, ctx.softmax_scale, ctx.softcap, ctx.deterministic, rng_state=rng_state,
        )
        dqkv = dqkv[..., : dout.shape[-1]]  # We could have padded the head dimension
        return dqkv, None, None, None, None, None, None, None


class DiffusionAttnVarlenQKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qkv, cu_seqlens, max_seqlen, block_size, dropout_p, softmax_scale, softcap, deterministic,
                return_softmax, is_grad_enabled):
        is_grad = is_grad_enabled and qkv.requires_grad
        if softmax_scale is None:
            softmax_scale = qkv.shape[-1] ** (-0.5)
        q, k, v = qkv[:, 0].detach(), qkv[:, 1].detach(), qkv[:, 2].detach()
        head_size_og = q.size(2)
        q, k, v = _pad_head_dim(q, k, v)
        out_padded, softmax_lse, S_dmask, rng_state = _wrapped_diffusion_attn_varlen_forward(
            q, k, v, cu_seqlens, max_seqlen, block_size, dropout_p, softmax_scale, softcap=softcap,
            return_softmax=return_softmax and dropout_p > 0,
        )
        if is_grad:
            ctx.save_for_backward(q, k, v, out_padded, softmax_lse, cu_seqlens, rng_state)
            ctx.max_seqlen = max_seqlen
            ctx.block_size = block_size
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.softcap = softcap
            ctx.deterministic = deterministic
        out = out_padded[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens, rng_state = ctx.saved_tensors
        qkv_shape = q.shape[:-2] + (3, *q.shape[-2:])
        dqkv = torch.empty(qkv_shape, dtype=q.dtype, device=q.device)
        (dout_padded,) = _pad_head_dim(dout)
        _wrapped_diffusion_attn_varlen_backward(
            dout_padded, q, k, v, out, softmax_lse, dqkv[:, 0], dqkv[:, 1], dqkv[:, 2], cu_seqlens, ctx.max_seqlen,
            ctx.block_size, ctx.dropout_p, ctx.softmax_scale, ctx.softcap, ctx.deterministic, rng_state=rng_state,
        )
        dqkv = dqkv[..., : dout.shape[-1]]  # We could have padded the head dimension
        return dqkv, None, None, None, None, None, None, None, None, None


class DiffusionAttnKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, block_size, dropout_p, softmax_scale, softcap, deterministic, return_softmax,
                is_grad_enabled):
        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, kv])
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        k, v = kv[:, :, 0].detach(), kv[:, :, 1].detach()
        head_size_og = q.size(3)
        q, k, v = _pad_head_dim(q, k, v)
        out_padded, softmax_lse, S_dmask, rng_state = _wrapped_diffusion_attn_forward(
            q, k, v, block_size, dropout_p, softmax_scale, softcap=softcap,
            return_softmax=return_softmax and dropout_p > 0,
        )
        if is_grad:
            ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
            ctx.block_size = block_size
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.softcap = softcap
            ctx.deterministic = deterministic
        out = out_padded[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
        dq = torch.empty_like(q)
        kv_shape = k.shape[:-2] + (2, *k.shape[-2:])
        dkv = torch.empty(kv_shape, dtype=k.dtype, device=k.device)
        (dout_padded,) = _pad_head_dim(dout)
        _wrapped_diffusion_attn_backward(
            dout_padded, q, k, v, out, softmax_lse, dq, dkv[:, :, 0], dkv[:, :, 1],
            ctx.block_size, ctx.dropout_p, ctx.softmax_scale, ctx.softcap, ctx.deterministic, rng_state=rng_state,
        )
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dkv = dkv[..., : dout.shape[-1]]
        return dq, dkv, None, None, None, None, None, None, None


class DiffusionAttnVarlenKVPackedFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, cu_seqlens, max_seqlen, block_size, dropout_p, softmax_scale, softcap, deterministic,
                return_softmax, is_grad_enabled):
        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, kv])
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        k, v = kv[:, 0].detach(), kv[:, 1].detach()
        head_size_og = q.size(2)
        q, k, v = _pad_head_dim(q, k, v)
        out_padded, softmax_lse, S_dmask, rng_state = _wrapped_diffusion_attn_varlen_forward(
            q, k, v, cu_seqlens, max_seqlen, block_size, dropout_p, softmax_scale, softcap=softcap,
            return_softmax=return_softmax and dropout_p > 0,
        )
        if is_grad:
            ctx.save_for_backward(q, k, v, out_padded, softmax_lse, cu_seqlens, rng_state)
            ctx.max_seqlen = max_seqlen
            ctx.block_size = block_size
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.softcap = softcap
            ctx.deterministic = deterministic
        out = out_padded[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens, rng_state = ctx.saved_tensors
        dq = torch.empty_like(q)
        kv_shape = k.shape[:-2] + (2, *k.shape[-2:])
        dkv = torch.empty(kv_shape, dtype=k.dtype, device=k.device)
        (dout_padded,) = _pad_head_dim(dout)
        _wrapped_diffusion_attn_varlen_backward(
            dout_padded, q, k, v, out, softmax_lse, dq, dkv[:, 0], dkv[:, 1], cu_seqlens, ctx.max_seqlen,
            ctx.block_size, ctx.dropout_p, ctx.softmax_scale, ctx.softcap, ctx.deterministic, rng_state=rng_state,
        )
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dkv = dkv[..., : dout.shape[-1]]
        return dq, dkv, None, None, None, None, None, None, None, None, None


class DiffusionAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, block_size, dropout_p, softmax_scale, softcap, deterministic, return_softmax,
                is_grad_enabled):
        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, k, v])
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_size_og = q.size(3)
        q, k, v = _pad_head_dim(q, k, v)
        out_padded, softmax_lse, S_dmask, rng_state = _wrapped_diffusion_attn_forward(
            q, k, v, block_size, dropout_p, softmax_scale, softcap=softcap,
            return_softmax=return_softmax and dropout_p > 0,
        )
        if is_grad:
            ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
            ctx.block_size = block_size
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.softcap = softcap
            ctx.deterministic = deterministic
        out = out_padded[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        (dout_padded,) = _pad_head_dim(dout)
        _wrapped_diffusion_attn_backward(
            dout_padded, q, k, v, out, softmax_lse, dq, dk, dv,
            ctx.block_size, ctx.dropout_p, ctx.softmax_scale, ctx.softcap, ctx.deterministic, rng_state=rng_state,
        )
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : dout.shape[-1]]
        dv = dv[..., : dout.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None


class DiffusionAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens, max_seqlen, block_size, dropout_p, softmax_scale, softcap, deterministic,
                return_softmax, is_grad_enabled):
        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, k, v])
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_size_og = q.size(2)
        q, k, v = _pad_head_dim(q, k, v)
        out_padded, softmax_lse, S_dmask, rng_state = _wrapped_diffusion_attn_varlen_forward(
            q, k, v, cu_seqlens, max_seqlen, block_size, dropout_p, softmax_scale, softcap=softcap,
            return_softmax=return_softmax and dropout_p > 0,
        )
        if is_grad:
            ctx.save_for_backward(q, k, v, out_padded, softmax_lse, cu_seqlens, rng_state)
            ctx.max_seqlen = max_seqlen
            ctx.block_size = block_size
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.softcap = softcap
            ctx.deterministic = deterministic
        out = out_padded[..., :head_size_og]
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens, rng_state = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        (dout_padded,) = _pad_head_dim(dout)
        _wrapped_diffusion_attn_varlen_backward(
            dout_padded, q, k, v, out, softmax_lse, dq, dk, dv, cu_seqlens, ctx.max_seqlen,
            ctx.block_size, ctx.dropout_p, ctx.softmax_scale, ctx.softcap, ctx.deterministic, rng_state=rng_state,
        )
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : dout.shape[-1]]
        dv = dv[..., : dout.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None


def diffusion_attn_qkvpacked_func(
    qkv,
    block_size,
    dropout_p=0.0,
    softmax_scale=None,
    softcap=0.0,  # <=0.0 means deactivate
    deterministic=False,
    return_attn_probs=False,
):
    """dropout_p should be set to 0.0 during evaluation
    If Q, K, V are already stacked into 1 tensor, this function will be faster than
    calling diffusion_attn_func on Q, K, V since the backward pass avoids explicit concatenation
    of the gradients of Q, K, V.
    For multi-query and grouped-query attention (MQA/GQA), please see
    diffusion_attn_kvpacked_func and diffusion_attn_func.
    Every sample is laid out as [noisy half (n tokens) | clean half (n tokens)] (2n tokens). The tokens of each half
    are split into blocks of `block_size` tokens, counted from the start of the half (the last block may be shorter):
    a noisy block attends to itself (bidirectionally) and to all the previous clean blocks, a clean block attends to
    itself (bidirectionally) and to all the previous clean blocks. Clean tokens never attend to noisy tokens.

    Arguments:
        qkv: (batch_size, 2n, 3, nheads, headdim)
        block_size: int. Size of the diffusion blocks (any positive integer).
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        softcap: float. Anything > 0 activates softcapping attention.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, 2n, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, 2n). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, 2 hr, 2 hr), hr = round_up(n, 128): the
            output of softmax (possibly with different scaling), the clean half starts at row / column hr. Only
            computed if dropout_p > 0; it also encodes the dropout pattern (negative means that location was
            dropped, nonnegative means it was kept).
    """
    return DiffusionAttnQKVPackedFunc.apply(
        qkv, block_size, dropout_p, softmax_scale, softcap, deterministic, return_attn_probs, torch.is_grad_enabled(),
    )


def diffusion_attn_kvpacked_func(
    q,
    kv,
    block_size,
    dropout_p=0.0,
    softmax_scale=None,
    softcap=0.0,  # 0.0 means deactivated
    deterministic=False,
    return_attn_probs=False,
):
    """dropout_p should be set to 0.0 during evaluation
    If K, V are already stacked into 1 tensor, this function will be faster than
    calling diffusion_attn_func on Q, K, V since the backward pass avoids explicit concatenation
    of the gradients of K, V.
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.
    Every sample is laid out as [noisy half (n tokens) | clean half (n tokens)] (2n tokens). The tokens of each half
    are split into blocks of `block_size` tokens, counted from the start of the half (the last block may be shorter):
    a noisy block attends to itself (bidirectionally) and to all the previous clean blocks, a clean block attends to
    itself (bidirectionally) and to all the previous clean blocks. Clean tokens never attend to noisy tokens.

    Arguments:
        q: (batch_size, 2n, nheads, headdim)
        kv: (batch_size, 2n, 2, nheads_k, headdim)
        Other arguments: see diffusion_attn_qkvpacked_func.
    Return:
        See diffusion_attn_qkvpacked_func.
    """
    return DiffusionAttnKVPackedFunc.apply(
        q, kv, block_size, dropout_p, softmax_scale, softcap, deterministic, return_attn_probs,
        torch.is_grad_enabled(),
    )


def diffusion_attn_func(
    q,
    k,
    v,
    block_size,
    dropout_p=0.0,
    softmax_scale=None,
    softcap=0.0,  # 0.0 means deactivated
    deterministic=False,
    return_attn_probs=False,
):
    """dropout_p should be set to 0.0 during evaluation
    Diffusion attention on a batch of samples of equal length (pretraining).
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.
    Every sample is laid out as [noisy half (n tokens) | clean half (n tokens)] (2n tokens). The tokens of each half
    are split into blocks of `block_size` tokens, counted from the start of the half (the last block may be shorter):
    a noisy block attends to itself (bidirectionally) and to all the previous clean blocks, a clean block attends to
    itself (bidirectionally) and to all the previous clean blocks. Clean tokens never attend to noisy tokens.

    Arguments:
        q: (batch_size, 2n, nheads, headdim)
        k: (batch_size, 2n, nheads_k, headdim)
        v: (batch_size, 2n, nheads_k, headdim)
        block_size: int. Size of the diffusion blocks (any positive integer).
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        softcap: float. Anything > 0 activates softcapping attention.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, 2n, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, 2n). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, 2 hr, 2 hr), hr = round_up(n, 128): the
            output of softmax (possibly with different scaling), the clean half starts at row / column hr. Only
            computed if dropout_p > 0; it also encodes the dropout pattern (negative means that location was
            dropped, nonnegative means it was kept).
    """
    return DiffusionAttnFunc.apply(
        q, k, v, block_size, dropout_p, softmax_scale, softcap, deterministic, return_attn_probs,
        torch.is_grad_enabled(),
    )


def diffusion_attn_varlen_qkvpacked_func(
    qkv,
    cu_seqlens,
    max_seqlen,
    block_size,
    dropout_p=0.0,
    softmax_scale=None,
    softcap=0.0,  # 0.0 means deactivated
    deterministic=False,
    return_attn_probs=False,
):
    """dropout_p should be set to 0.0 during evaluation
    If Q, K, V are already stacked into 1 tensor, this function will be faster than
    calling diffusion_attn_varlen_func on Q, K, V since the backward pass avoids explicit concatenation
    of the gradients of Q, K, V.
    For multi-query and grouped-query attention (MQA/GQA), please see
    diffusion_attn_varlen_kvpacked_func and diffusion_attn_varlen_func.
    Every sample is laid out as [noisy half (n tokens) | clean half (n tokens)] (2n tokens). The tokens of each half
    are split into blocks of `block_size` tokens, counted from the start of the half (the last block may be shorter):
    a noisy block attends to itself (bidirectionally) and to all the previous clean blocks, a clean block attends to
    itself (bidirectionally) and to all the previous clean blocks. Clean tokens never attend to noisy tokens.

    Sample i occupies the rows [cu_seqlens[i], cu_seqlens[i + 1]) and has an even length 2 n_i. Samples never attend
    to each other (SFT with packing).

    Arguments:
        qkv: (total, 3, nheads, headdim), where total = total number of tokens in the batch.
        cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into qkv.
        max_seqlen: int. Maximum sequence length in the batch (2 max n_i).
        Other arguments: see diffusion_attn_func.
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (nheads, total). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, 2 hr, 2 hr), hr = round_up(max_seqlen / 2,
            128). The output of softmax (possibly with different scaling), see diffusion_attn_func.
    """
    return DiffusionAttnVarlenQKVPackedFunc.apply(
        qkv, cu_seqlens, max_seqlen, block_size, dropout_p, softmax_scale, softcap, deterministic, return_attn_probs,
        torch.is_grad_enabled(),
    )


def diffusion_attn_varlen_kvpacked_func(
    q,
    kv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    block_size,
    dropout_p=0.0,
    softmax_scale=None,
    softcap=0.0,  # 0.0 means deactivated
    deterministic=False,
    return_attn_probs=False,
):
    """dropout_p should be set to 0.0 during evaluation
    If K, V are already stacked into 1 tensor, this function will be faster than
    calling diffusion_attn_varlen_func on Q, K, V since the backward pass avoids explicit concatenation
    of the gradients of K, V.
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    Every sample is laid out as [noisy half (n tokens) | clean half (n tokens)] (2n tokens). The tokens of each half
    are split into blocks of `block_size` tokens, counted from the start of the half (the last block may be shorter):
    a noisy block attends to itself (bidirectionally) and to all the previous clean blocks, a clean block attends to
    itself (bidirectionally) and to all the previous clean blocks. Clean tokens never attend to noisy tokens.

    Self-attention: cu_seqlens_k (max_seqlen_k) must be None or equal to cu_seqlens_q (max_seqlen_q).

    Arguments:
        q: (total, nheads, headdim), where total = total number of tokens in the batch.
        kv: (total, 2, nheads_k, headdim)
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q and kv.
        cu_seqlens_k: None or cu_seqlens_q.
        max_seqlen_q: int. Maximum sequence length in the batch (2 max n_i).
        max_seqlen_k: None or max_seqlen_q.
        Other arguments: see diffusion_attn_func.
    Return:
        See diffusion_attn_varlen_qkvpacked_func.
    """
    _check_self_attention(cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
    return DiffusionAttnVarlenKVPackedFunc.apply(
        q, kv, cu_seqlens_q, max_seqlen_q, block_size, dropout_p, softmax_scale, softcap, deterministic,
        return_attn_probs, torch.is_grad_enabled(),
    )


def diffusion_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    block_size,
    dropout_p=0.0,
    softmax_scale=None,
    softcap=0.0,  # 0.0 means deactivated
    deterministic=False,
    return_attn_probs=False,
):
    """dropout_p should be set to 0.0 during evaluation
    Diffusion attention on packed samples of variable lengths (SFT with packing).
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    Every sample is laid out as [noisy half (n tokens) | clean half (n tokens)] (2n tokens). The tokens of each half
    are split into blocks of `block_size` tokens, counted from the start of the half (the last block may be shorter):
    a noisy block attends to itself (bidirectionally) and to all the previous clean blocks, a clean block attends to
    itself (bidirectionally) and to all the previous clean blocks. Clean tokens never attend to noisy tokens.

    Sample i occupies the rows [cu_seqlens_q[i], cu_seqlens_q[i + 1]) of q, k and v and has an even length 2 n_i.
    Samples never attend to each other. Self-attention: cu_seqlens_k (max_seqlen_k) must be None or equal to
    cu_seqlens_q (max_seqlen_q).

    Arguments:
        q: (total, nheads, headdim), where total = total number of tokens in the batch.
        k: (total, nheads_k, headdim)
        v: (total, nheads_k, headdim)
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q, k and v.
        cu_seqlens_k: None or cu_seqlens_q.
        max_seqlen_q: int. Maximum sequence length in the batch (2 max n_i).
        max_seqlen_k: None or max_seqlen_q.
        Other arguments: see diffusion_attn_func.
    Return:
        See diffusion_attn_varlen_qkvpacked_func.
    """
    _check_self_attention(cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)
    return DiffusionAttnVarlenFunc.apply(
        q, k, v, cu_seqlens_q, max_seqlen_q, block_size, dropout_p, softmax_scale, softcap, deterministic,
        return_attn_probs, torch.is_grad_enabled(),
    )
