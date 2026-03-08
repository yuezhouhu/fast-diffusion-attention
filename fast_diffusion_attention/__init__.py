"""Fast Diffusion Attention — GPU-accelerated attention kernels for diffusion LM training."""

from .diff_attn_noisy import diff_attn_with_noisy_input
from .block_prefill_attention import diff_attn_varlen_v2

__all__ = [
    "diff_attn_with_noisy_input",
    "diff_attn_varlen_v2",
]
