"""Analytical counts only; independent of PyTorch and GPU availability."""

from dataclasses import dataclass

from .config import AoEConfig, ModelConfig


@dataclass(frozen=True)
class ParameterCounts:
    token_embeddings: int
    attention_projections: int
    swiglu_experts: int
    router_weights: int
    normalization: int

    @property
    def total(self) -> int:
        return (
            self.token_embeddings + self.attention_projections
            + self.swiglu_experts + self.router_weights + self.normalization
        )


def count_parameters(config: ModelConfig, model_type: str = "vanilla_moe") -> ParameterCounts:
    """Count unique trainable scalars, not per-token activated parameters.

    Assumes two scale-only RMSNorms per block plus one final RMSNorm.
    Q, K, V and output projections are full multi-head attention matrices.
    RoPE has no learned parameters. The LM head shares the embedding matrix.
    swiglu_experts includes the factorized gate for AoE; router_weights is zero.
    With ModelConfig and model_type='aoe', rank defaults to 171 just as in the
    factory. Pass AoEConfig to select an explicit rank for small tests.
    """
    d, layers = config.d_model, config.n_layers
    if model_type == "vanilla_moe":
        per_expert = 3 * d * config.expert_hidden_dim
        router_weights = layers * config.n_experts * d
    elif model_type == "aoe":
        rank = config.aoe_rank if isinstance(config, AoEConfig) else 171
        per_expert = d * rank + rank * config.expert_hidden_dim + 2 * d * config.expert_hidden_dim
        router_weights = 0
    else:
        raise ValueError("model_type must be 'vanilla_moe' or 'aoe'")
    return ParameterCounts(
        token_embeddings=config.vocab_size * d,
        attention_projections=layers * 4 * d * d,
        swiglu_experts=layers * config.n_experts * per_expert,
        router_weights=router_weights,
        normalization=(2 * layers + 1) * d,
    )
