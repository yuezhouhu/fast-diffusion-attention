# Copyright (c) 2026, fast-diffusion-attention authors.
"""FlashAttention-2 class attention kernels for diffusion language models (sm80 / sm86 / sm89)."""

from fast_diffusion_attention.version import __version__
from fast_diffusion_attention.diffusion_attn_interface import (
    diffusion_attn_func,
    diffusion_attn_kvpacked_func,
    diffusion_attn_qkvpacked_func,
    diffusion_attn_varlen_func,
    diffusion_attn_varlen_kvpacked_func,
    diffusion_attn_varlen_qkvpacked_func,
)

__all__ = [
    "__version__",
    "diffusion_attn_func",
    "diffusion_attn_kvpacked_func",
    "diffusion_attn_qkvpacked_func",
    "diffusion_attn_varlen_func",
    "diffusion_attn_varlen_kvpacked_func",
    "diffusion_attn_varlen_qkvpacked_func",
]
