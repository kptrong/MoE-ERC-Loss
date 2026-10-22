"""Pre-Norm decoder block reusing attention, RMSNorm and sparse MoE."""

import torch
from torch import nn

from .attention import CausalSelfAttention
from .config import ModelConfig
from .moe import RoutingDiagnostics, SparseMoE
from .rmsnorm import RMSNorm


class MoEDecoderBlock(nn.Module):
    """x + attention(norm_1(x)), then h + moe(norm_2(h))."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm_1 = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attention = CausalSelfAttention(config)
        self.norm_2 = RMSNorm(config.d_model, config.rms_norm_eps)
        self.moe = SparseMoE(config)

    def forward(
        self, x: torch.Tensor, *, return_routing_info: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, RoutingDiagnostics]:
        h = x + self.attention(self.norm_1(x))
        normalized = self.norm_2(h)
        if return_routing_info:
            moe_output, diagnostics = self.moe(normalized, return_diagnostics=True)
            return h + moe_output, diagnostics
        return h + self.moe(normalized)
