"""Learned full-softmax router with unrenormalized Top-K selection."""

from typing import NamedTuple

import torch
from torch import nn

from .config import ModelConfig


class RouterOutput(NamedTuple):
    logits: torch.Tensor
    probabilities: torch.Tensor
    topk_indices: torch.Tensor
    topk_weights: torch.Tensor


class MoERouter(nn.Module):
    """Accept [B,T,d] or flattened [N,d]; route on the final dimension.

    R is exposed as router.weight [n_experts,d_model]. Softmax uses FP32 for
    FP16/BF16/FP32 logits and preserves FP64 for numerical reference tests.
    Top-K ties follow torch.topk's ordering; no random routing is introduced.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.projection = nn.Linear(config.d_model, config.n_experts, bias=False)

    @property
    def weight(self) -> nn.Parameter:
        """Original learnable R, without copying or detaching."""
        return self.projection.weight

    def forward(self, x: torch.Tensor) -> RouterOutput:
        if x.ndim not in (2, 3) or x.shape[-1] != self.config.d_model:
            raise ValueError(f"expected [N,{self.config.d_model}] or [B,T,{self.config.d_model}]")
        if not x.is_floating_point():
            raise TypeError("router expects floating-point input")
        logits = self.projection(x)
        dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
        probabilities = torch.softmax(logits, dim=-1, dtype=dtype)
        weights, indices = torch.topk(probabilities, k=self.config.top_k, dim=-1)
        return RouterOutput(logits, probabilities, indices, weights)
