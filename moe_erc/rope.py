"""Parameter-free rotary position encoding for full multi-head attention."""

import math

import torch
from torch import nn


class RotaryPositionalEmbedding(nn.Module):
    """Rotate adjacent coordinate pairs of Q and K shaped [B, H, S, head_dim].

    Pair m uses angle position * base**(-2*m/head_dim). Positions start at 0
    on every forward; no KV cache or offsets are implemented. Frequencies are
    rebuilt on the input device in FP32 (FP64 for double inputs), so module-wide
    BF16 conversion cannot round a stored inverse-frequency buffer.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 512, base: float = 10000.0) -> None:
        super().__init__()
        if type(head_dim) is not int or head_dim <= 0 or head_dim % 2:
            raise ValueError("head_dim must be a positive even integer")
        if type(max_seq_len) is not int or max_seq_len <= 0:
            raise ValueError("max_seq_len must be a positive integer")
        if (isinstance(base, bool) or not isinstance(base, (int, float))
                or not math.isfinite(base) or base <= 0):
            raise ValueError("base must be finite and positive")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if q.ndim != 4 or k.shape != q.shape or q.shape[-1] != self.head_dim:
            raise ValueError("Q and K must have matching [B, H, S, head_dim] shapes")
        if not 1 <= q.shape[-2] <= self.max_seq_len:
            raise ValueError(f"sequence length must be in [1, {self.max_seq_len}]")
        if q.device != k.device or q.dtype != k.dtype:
            raise ValueError("Q and K must have the same device and dtype")
        if not q.is_floating_point():
            raise TypeError("RoPE expects floating-point Q and K")
        dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=q.device.type, enabled=False):
            exponent = torch.arange(0, self.head_dim, 2, device=q.device, dtype=dtype) / self.head_dim
            inv_freq = self.base ** (-exponent)
            positions = torch.arange(q.shape[-2], device=q.device, dtype=dtype)
            angles = positions[:, None] * inv_freq[None, :]
            cos, sin = angles.cos(), angles.sin()

            def rotate(x: torch.Tensor) -> torch.Tensor:
                value = x.to(dtype)
                even, odd = value[..., 0::2], value[..., 1::2]
                rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
                return rotated.flatten(-2).to(x.dtype)

            return rotate(q), rotate(k)
