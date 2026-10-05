# Copyright (c) 2026, fast-diffusion-attention authors.
"""FlashAttention-3 class Hopper attention kernels for diffusion language models."""

from fast_diffusion_attention.version import __version__
from fast_diffusion_attention.interface import diffusion_attn_func, diffusion_attn_varlen_func

__all__ = ["__version__", "diffusion_attn_func", "diffusion_attn_varlen_func"]
