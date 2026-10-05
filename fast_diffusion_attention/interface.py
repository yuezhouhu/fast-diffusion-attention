# Copyright (c) 2023, Tri Dao.
# Copyright (c) 2026, fast-diffusion-attention authors.
"""Python interface of the diffusion attention kernels, modeled after the FlashAttention-3 interface."""

from typing import Optional, Tuple

import torch

# isort: off
# We need to import the CUDA kernels after importing torch
import fast_diffusion_attention._C  # noqa: F401  Registers the `fda` operators with PyTorch
# isort: on

fda_gpu = torch.ops.fda


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def round_multiple(x, m):
    return (x + m - 1) // m * m


def _round_up_headdim(head_size: int) -> int:
    for h in (64, 96, 128, 192, 256):
        if head_size <= h:
            return h
    return 256


def _bwd_block_mn(head_size: int, softcap: float) -> Tuple[int, int]:
    # Must match the backward kernel configs (flash_api.cpp / flash_bwd_launch_template.h)
    hdim = _round_up_headdim(head_size)
    block_m = (96 if softcap > 0.0 else 128) if hdim <= 64 else 64
    block_n = 128 if hdim <= 128 else (96 if hdim <= 192 else 80)
    return block_m, block_n


@torch.library.custom_op("fda::_diffusion_attn_forward", mutates_args=(), device_types="cuda")
def _diffusion_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    softcap: float = 0.0,
    sm_margin: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    cu_seqlens = maybe_contiguous(cu_seqlens)
    out, softmax_lse = fda_gpu.fwd(
        q, k, v, None, cu_seqlens, max_seqlen, block_size,
        q_descale, k_descale, v_descale, softmax_scale, softcap, sm_margin,
    )
    return out, softmax_lse


@torch.library.register_fake("fda::_diffusion_attn_forward")
def _diffusion_attn_forward_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    softcap: float = 0.0,
    sm_margin: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    out_dtype = torch.bfloat16 if q.dtype == torch.float8_e4m3fn else q.dtype
    out = torch.empty(q.shape, dtype=out_dtype, device=q.device)
    if cu_seqlens is not None:
        total, num_heads, _ = q.shape
        softmax_lse = torch.empty((num_heads, total), dtype=torch.float32, device=q.device)
    else:
        batch_size, seqlen, num_heads, _ = q.shape
        softmax_lse = torch.empty((batch_size, num_heads, seqlen), dtype=torch.float32, device=q.device)
    return out, softmax_lse


@torch.library.custom_op("fda::_diffusion_attn_backward", mutates_args=("dq", "dk", "dv"), device_types="cuda")
def _diffusion_attn_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    block_size: int,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    softcap: float = 0.0,
    deterministic: bool = False,
    sm_margin: int = 0,
) -> torch.Tensor:
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    _, _, _, softmax_d = fda_gpu.bwd(
        dout, q, k, v, out, softmax_lse, dq, dk, dv, cu_seqlens, max_seqlen, block_size,
        softmax_scale, softcap, deterministic, sm_margin,
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
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    block_size: int,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    softmax_scale: Optional[float] = None,
    softcap: float = 0.0,
    deterministic: bool = False,
    sm_margin: int = 0,
) -> torch.Tensor:
    if cu_seqlens is not None:
        batch_size = cu_seqlens.shape[0] - 1
        total = q.shape[0]
    else:
        batch_size = q.shape[0]
        total = q.shape[0] * q.shape[1]
    num_heads = q.shape[-2]
    block_m, _ = _bwd_block_mn(q.shape[-1], softcap)
    total_q_padded = round_multiple(total + 2 * batch_size * block_m, block_m)
    return torch.empty((num_heads, total_q_padded), dtype=torch.float32, device=q.device)


class DiffusionAttnFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, block_size, softmax_scale, q_descale, k_descale, v_descale, softcap,
                deterministic, sm_margin):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        out, softmax_lse = _diffusion_attn_forward(
            q, k, v, block_size, None, None, q_descale, k_descale, v_descale, softmax_scale, softcap, sm_margin,
        )
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.block_size = block_size
        ctx.softmax_scale = softmax_scale
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.sm_margin = sm_margin
        return out, softmax_lse

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        assert q.dtype in (torch.float16, torch.bfloat16), "The backward pass only supports fp16 and bf16"
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _diffusion_attn_backward(
            dout, q, k, v, out, softmax_lse, dq, dk, dv, ctx.block_size, None, None,
            ctx.softmax_scale, ctx.softcap, ctx.deterministic, ctx.sm_margin,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None


class DiffusionAttnVarlenFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens, max_seqlen, block_size, softmax_scale, q_descale, k_descale,
                v_descale, softcap, deterministic, sm_margin):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        out, softmax_lse = _diffusion_attn_forward(
            q, k, v, block_size, cu_seqlens, max_seqlen, q_descale, k_descale, v_descale, softmax_scale,
            softcap, sm_margin,
        )
        ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens)
        ctx.max_seqlen = max_seqlen
        ctx.block_size = block_size
        ctx.softmax_scale = softmax_scale
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.sm_margin = sm_margin
        return out, softmax_lse

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens = ctx.saved_tensors
        assert q.dtype in (torch.float16, torch.bfloat16), "The backward pass only supports fp16 and bf16"
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _diffusion_attn_backward(
            dout, q, k, v, out, softmax_lse, dq, dk, dv, ctx.block_size, cu_seqlens, ctx.max_seqlen,
            ctx.softmax_scale, ctx.softcap, ctx.deterministic, ctx.sm_margin,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None


def diffusion_attn_func(
    q,
    k,
    v,
    block_size,
    softmax_scale=None,
    q_descale=None, k_descale=None, v_descale=None,
    softcap=0.0,
    deterministic=False,
    sm_margin=0,
    return_attn_probs=False,
):
    """Diffusion attention on a batch of samples of equal length (pretraining).

    Every sample is laid out as [noisy half (n tokens) | clean half (n tokens)]. The tokens of each half are split
    in blocks of `block_size` tokens; a noisy block attends to itself (bidirectionally) and to all the previous
    clean blocks, a clean block attends to itself (bidirectionally) and to all the previous clean blocks.

    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads than Q.
    Note that the number of heads in Q must be divisible by the number of heads in KV.

    Arguments:
        q: (batch_size, 2n, nheads, headdim)
        k: (batch_size, 2n, nheads_k, headdim)
        v: (batch_size, 2n, nheads_k, headdim)
        block_size: int, size of the diffusion blocks (any positive integer).
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        q_descale, k_descale, v_descale: (batch_size, nheads_k) float32 descaling factors of FP8 inputs.
        softcap: float. Anything > 0 activates softcapping attention.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory.
        sm_margin: int. Number of SMs to leave free (e.g. for communication kernels).
        return_attn_probs: bool. Whether to also return the logsumexp of the attention scores.
    Return:
        out: (batch_size, 2n, nheads, headdim). For FP8 inputs, out is BF16.
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, 2n). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
    """
    out, softmax_lse = DiffusionAttnFunc.apply(
        q, k, v, block_size, softmax_scale, q_descale, k_descale, v_descale, softcap, deterministic, sm_margin,
    )
    return (out, softmax_lse) if return_attn_probs else out


def diffusion_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    block_size,
    softmax_scale=None,
    q_descale=None, k_descale=None, v_descale=None,
    softcap=0.0,
    deterministic=False,
    sm_margin=0,
    return_attn_probs=False,
):
    """Diffusion attention on packed samples of variable lengths (SFT with packing).

    Sample i occupies the rows [cu_seqlens_q[i], cu_seqlens_q[i + 1]) of q, k and v and is laid out as
    [noisy half (n_i tokens) | clean half (n_i tokens)]: every sample has an even length. Samples never attend to
    each other. See `diffusion_attn_func` for the mask.

    Arguments:
        q: (total, nheads, headdim)
        k: (total, nheads_k, headdim)
        v: (total, nheads_k, headdim)
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths of the samples.
        cu_seqlens_k: must be None or equal to cu_seqlens_q (self-attention).
        max_seqlen_q: int. Maximum length of a sample (2 max n_i).
        max_seqlen_k: must be None or equal to max_seqlen_q.
        Other arguments: see `diffusion_attn_func`.
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (nheads, total).
    """
    if cu_seqlens_k is not None and cu_seqlens_k is not cu_seqlens_q:
        assert cu_seqlens_k.shape == cu_seqlens_q.shape, "cu_seqlens_k must be equal to cu_seqlens_q"
    if max_seqlen_k is not None:
        assert max_seqlen_k == max_seqlen_q, "max_seqlen_k must be equal to max_seqlen_q"
    out, softmax_lse = DiffusionAttnVarlenFunc.apply(
        q, k, v, cu_seqlens_q, max_seqlen_q, block_size, softmax_scale, q_descale, k_descale, v_descale,
        softcap, deterministic, sm_margin,
    )
    return (out, softmax_lse) if return_attn_probs else out
