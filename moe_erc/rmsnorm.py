"""Scale-only RMS normalization over the final hidden dimension."""

import math

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Compute x / sqrt(mean(x**2) + eps) * weight, without mean centering.

    FP16/BF16 inputs are normalized and scaled in FP32, then cast back.
    FP64 inputs retain their precision. There is no learned bias.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        if type(d_model) is not int or d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        if (isinstance(eps, bool) or not isinstance(eps, (int, float))
                or not math.isfinite(eps) or eps <= 0):
            raise ValueError("eps must be finite and positive")
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 1 or x.shape[-1] != self.d_model:
            raise ValueError(f"expected last dimension {self.d_model}")
        if not x.is_floating_point():
            raise TypeError("RMSNorm expects floating-point input")
        compute_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            value = x.to(compute_dtype)
            normalized = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + self.eps)
            output = normalized * self.weight.to(compute_dtype)
        return output.to(x.dtype)
