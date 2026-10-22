"""AoE: all experts project to rank r, only Top-K finish their SwiGLU forward."""

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import nn

from .aoe_expert import AoEExpert
from .config import AoEConfig
from .decoder import MoEDecoderBlock
from .moe import RoutingDiagnostics


class AoERoutingOutput(NamedTuple):
    gate_down_activations: torch.Tensor  # [N,n_experts,rank], attached
    scores: torch.Tensor                 # [N,n_experts], L2 before SiLU
    probabilities: torch.Tensor
    topk_indices: torch.Tensor           # [N,K]
    topk_weights: torch.Tensor           # full-softmax probabilities, not renormalized


@dataclass(frozen=True)
class AoERoutingDiagnostics(RoutingDiagnostics):
    scores: torch.Tensor                 # [B,T,n_experts], detached, opt-in only
    probabilities: torch.Tensor          # [B,T,n_experts], detached, opt-in only
    mean_activation_norms: torch.Tensor  # [n_experts]


class SparseAoE(nn.Module):
    """No independent router weights, no capacity/drop policy or auxiliary loss.

    Unselected experts still receive gate-down gradients through softmax routing.
    Their gate-up/up/output projections are not executed. All gate-down tensors
    remain attached for both routing and selected-expert computation.
    """

    def __init__(self, config: AoEConfig) -> None:
        super().__init__()
        if not isinstance(config, AoEConfig):
            raise TypeError("SparseAoE requires AoEConfig")
        self.config = config
        self.experts = nn.ModuleList(AoEExpert(config) for _ in range(config.n_experts))

    def compute_routing(self, flat: torch.Tensor) -> AoERoutingOutput:
        """Compute all gate-down projections exactly once, with no activation."""
        if flat.ndim != 2 or flat.shape[-1] != self.config.d_model or not flat.shape[0]:
            raise ValueError(f"expected nonempty [N,{self.config.d_model}]")
        if not flat.is_floating_point():
            raise TypeError("AoE expects floating-point input")
        activations = torch.stack([expert.gate_down_proj(flat) for expert in self.experts], dim=1)
        dtype = torch.float64 if activations.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=flat.device.type, enabled=False):
            scores = torch.linalg.vector_norm(activations.to(dtype), ord=2, dim=-1)
            probabilities = scores.softmax(dim=-1)
        weights, indices = probabilities.topk(self.config.top_k, dim=-1)
        return AoERoutingOutput(activations, scores, probabilities, indices, weights)

    def forward(self, x: torch.Tensor, *, return_diagnostics: bool = False):
        if x.ndim != 3 or x.shape[-1] != self.config.d_model or not x.shape[0] or not x.shape[1]:
            raise ValueError(f"expected nonempty [B,T,{self.config.d_model}]")
        flat = x.reshape(-1, self.config.d_model)
        routing = self.compute_routing(flat)
        dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        accumulated = torch.zeros(flat.shape, device=x.device, dtype=dtype)
        for index, expert in enumerate(self.experts):
            token_ids, slots = torch.where(routing.topk_indices == index)
            if not token_ids.numel():
                continue
            selected_x = flat.index_select(0, token_ids)
            selected_z = routing.gate_down_activations[:, index].index_select(0, token_ids)
            expert_output = expert(selected_x, gate_down_activation=selected_z)
            weighted = expert_output.to(dtype) * routing.topk_weights[token_ids, slots].to(dtype).unsqueeze(-1)
            accumulated.index_add_(0, token_ids, weighted)
        output = accumulated.to(x.dtype).reshape_as(x)
        if not return_diagnostics:
            return output
        with torch.no_grad():
            counts = torch.bincount(routing.topk_indices.reshape(-1), minlength=self.config.n_experts)
            diagnostic = AoERoutingDiagnostics(
                token_counts=counts,
                assignment_fractions=counts.float() / (flat.shape[0] * self.config.top_k),
                mean_probabilities=routing.probabilities.detach().mean(dim=0),
                topk_indices=routing.topk_indices.detach().reshape(*x.shape[:2], self.config.top_k),
                scores=routing.scores.detach().reshape(*x.shape[:2], self.config.n_experts),
                probabilities=routing.probabilities.detach().reshape(*x.shape[:2], self.config.n_experts),
                mean_activation_norms=routing.scores.detach().mean(dim=0),
            )
        return output, diagnostic


class AoEDecoderBlock(MoEDecoderBlock):
    """Identical shared Pre-Norm/residual forward; replace only the FFN sublayer."""

    expert_layer_cls = SparseAoE
