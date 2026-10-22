"""Analytical counts only; independent of PyTorch and GPU availability."""

from dataclasses import dataclass

from .config import ModelConfig


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


def count_parameters(config: ModelConfig) -> ParameterCounts:
    """Count unique trainable scalars, not per-token activated parameters.

    Assumes two scale-only RMSNorms per block plus one final RMSNorm.
    Q, K, V and output projections are full multi-head attention matrices.
    RoPE has no learned parameters. The LM head shares the embedding matrix.
    """
    d, layers = config.d_model, config.n_layers
    return ParameterCounts(
        token_embeddings=config.vocab_size * d,
        attention_projections=layers * 4 * d * d,
        swiglu_experts=layers * config.n_experts * 3 * d * config.expert_hidden_dim,
        router_weights=layers * config.n_experts * d,
        normalization=(2 * layers + 1) * d,
    )
