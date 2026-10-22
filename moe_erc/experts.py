"""SwiGLU experts with directly accessible gate projection parameters."""

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


class SwiGLUExpert(nn.Module):
    """down_proj(SiLU(gate_proj(x)) * up_proj(x)); no bias or normalization.

    gate_proj.weight is [expert_hidden_dim, d_model], the transpose of the
    paper's row-vector convention W_gate [d_model, expert_hidden_dim].
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.expert_hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.expert_hidden_dim, bias=False)
        self.down_proj = nn.Linear(config.expert_hidden_dim, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
