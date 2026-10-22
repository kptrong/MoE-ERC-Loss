"""Sparse token dispatch to independent SwiGLU experts; no auxiliary losses."""

from dataclasses import dataclass

import torch
from torch import nn

from .config import ModelConfig
from .experts import SwiGLUExpert
from .router import MoERouter


@dataclass(frozen=True)
class RoutingDiagnostics:
    """Detached observations, returned only on request; no tensors are cached.

    token_counts[e] counts token-expert assignments (each token has K).
    assignment_fractions[e] = token_counts[e] / (B*T*K), summing to one.
    mean_probabilities[e] averages the full-softmax probability over B*T tokens.
    topk_indices has shape [B,T,K]. These detached observations are for logging,
    not for constructing a future differentiable load-balancing loss.
    """

    token_counts: torch.Tensor
    assignment_fractions: torch.Tensor
    mean_probabilities: torch.Tensor
    topk_indices: torch.Tensor


class SparseMoE(nn.Module):
    """Sum selected expert outputs weighted by full-softmax probabilities.

    No token dropping or capacity limit. Only assigned tokens reach an expert;
    experts with no assignments are not called and may have None gradients.
    FP16/BF16 outputs are accumulated in FP32, then cast to the input dtype.
    Normalization and residual connections belong to the future decoder block.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.router = MoERouter(config)
        self.experts = nn.ModuleList(SwiGLUExpert(config) for _ in range(config.n_experts))

    def get_gate_weights(self) -> torch.Tensor:
        """Stack gates as [n_experts, expert_hidden_dim, d_model].

        This is a differentiable stack of existing parameters, not a new learned
        parameter. ERC can access R via self.router.weight and each gate via
        self.experts[i].gate_proj.weight. No detach is applied here.
        """
        return torch.stack([expert.gate_proj.weight for expert in self.experts])

    def forward(
        self, x: torch.Tensor, *, return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, RoutingDiagnostics]:
        if x.ndim != 3 or x.shape[-1] != self.config.d_model:
            raise ValueError(f"expected [B,T,{self.config.d_model}]")
        if x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError("batch size and sequence length must be positive")
        if not x.is_floating_point():
            raise TypeError("SparseMoE expects floating-point input")
        flat = x.reshape(-1, self.config.d_model)
        routing = self.router(flat)
        accumulation_dtype = (
            torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        )
        accumulated = torch.zeros(flat.shape, device=x.device, dtype=accumulation_dtype)
        for expert_id, expert in enumerate(self.experts):
            token_ids, slots = torch.where(routing.topk_indices == expert_id)
            if token_ids.numel() == 0:
                continue
            expert_output = expert(flat.index_select(0, token_ids))
            weights = routing.topk_weights[token_ids, slots].to(accumulation_dtype)
            weighted = expert_output.to(accumulation_dtype) * weights.unsqueeze(-1)
            accumulated.index_add_(0, token_ids, weighted)
        output = accumulated.to(x.dtype).reshape_as(x)
        if not return_diagnostics:
            return output
        with torch.no_grad():
            counts = torch.bincount(
                routing.topk_indices.reshape(-1), minlength=self.config.n_experts,
            )
            diagnostics = RoutingDiagnostics(
                token_counts=counts,
                assignment_fractions=counts.float() / (flat.shape[0] * self.config.top_k),
                mean_probabilities=routing.probabilities.detach().mean(dim=0),
                topk_indices=routing.topk_indices.detach().reshape(
                    x.shape[0], x.shape[1], self.config.top_k,
                ),
            )
        return output, diagnostics
