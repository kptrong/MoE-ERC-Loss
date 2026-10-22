"""AoE expert: factorize only the SwiGLU gate, with reusable routing activations."""

import torch
from torch import nn
from torch.nn import functional as F

from .config import AoEConfig


class AoEExpert(nn.Module):
    def __init__(self, config: AoEConfig) -> None:
        super().__init__()
        self.gate_down_proj = nn.Linear(config.d_model, config.aoe_rank, bias=False)
        self.gate_up_proj = nn.Linear(config.aoe_rank, config.expert_hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.expert_hidden_dim, bias=False)
        self.down_proj = nn.Linear(config.expert_hidden_dim, config.d_model, bias=False)

    def forward(self, x: torch.Tensor, *, gate_down_activation: torch.Tensor | None = None) -> torch.Tensor:
        """If supplied, reuse z=[...,rank] from routing without detaching it.

        PyTorch stores gate-down weight [rank,d] and gate-up weight [D,rank].
        down_proj is the final expert output projection, not the gate factor.
        """
        z = self.gate_down_proj(x) if gate_down_activation is None else gate_down_activation
        if z.shape != (*x.shape[:-1], self.gate_down_proj.out_features) or z.device != x.device:
            raise ValueError("cached gate-down activation shape/device does not match input")
        return self.down_proj(F.silu(self.gate_up_proj(z)) * self.up_proj(x))
