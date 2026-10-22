"""Causal MHA building block; no residual, normalization, MoE or KV cache."""

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig
from .rope import RotaryPositionalEmbedding


class CausalSelfAttention(nn.Module):
    """Bias-free Q/K/V/output projections with RoPE and causal PyTorch SDPA."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.rope = RotaryPositionalEmbedding(config.head_dim, config.max_seq_len, config.rope_theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.config.d_model:
            raise ValueError(f"expected [batch, seq_len, {self.config.d_model}]")
        if not 1 <= x.shape[1] <= self.config.max_seq_len:
            raise ValueError(f"sequence length must be in [1, {self.config.max_seq_len}]")
        if not x.is_floating_point():
            raise TypeError("attention expects floating-point input")
        batch, seq_len, width = x.shape

        def split_heads(projection: nn.Linear) -> torch.Tensor:
            return projection(x).reshape(batch, seq_len, self.config.n_heads, self.config.head_dim).transpose(1, 2)

        q, k = self.rope(split_heads(self.q_proj), split_heads(self.k_proj))
        v = split_heads(self.v_proj)
        attended = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.config.dropout if self.training else 0.0,
        )
        merged = attended.transpose(1, 2).contiguous().view(batch, seq_len, width)
        return self.out_proj(merged)
