"""Validated architecture settings; no tensors or learned parameters are created."""

from dataclasses import dataclass, fields
import math


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 16_000
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    n_experts: int = 16
    top_k: int = 2
    expert_hidden_dim: int = 256
    max_seq_len: int = 512
    dropout: float = 0.0
    linear_bias: bool = False
    tie_embeddings: bool = True
    attention_type: str = "multi_head_causal"
    positional_embedding: str = "rope"
    normalization: str = "rmsnorm"
    expert_activation: str = "swiglu"
    shared_experts: int = 0
    # Nonlearned numerical settings, explicit project defaults rather than
    # hyperparameters claimed to reproduce the paper's large-scale recipe.
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        positive_integers = (
            "vocab_size", "d_model", "n_layers", "n_heads", "n_experts",
            "top_k", "expert_hidden_dim", "max_seq_len",
        )
        for name in positive_integers:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.head_dim % 2:
            raise ValueError("RoPE requires an even head_dim")
        if self.top_k > self.n_experts:
            raise ValueError("top_k must not exceed n_experts")
        if (isinstance(self.dropout, bool)
                or not isinstance(self.dropout, (int, float))
                or not math.isfinite(self.dropout)
                or not 0 <= self.dropout < 1):
            raise ValueError("dropout must be finite and in [0, 1)")
        for name in ("rope_theta", "rms_norm_eps"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if self.linear_bias is not False:
            raise ValueError("linear_bias must be False for this architecture")
        if self.tie_embeddings is not True:
            raise ValueError("tie_embeddings must be True for this architecture")
        if type(self.shared_experts) is not int or self.shared_experts != 0:
            raise ValueError("shared_experts must be 0 for this architecture")
        supported = {
            "attention_type": "multi_head_causal",
            "positional_embedding": "rope",
            "normalization": "rmsnorm",
            "expert_activation": "swiglu",
        }
        for name, expected in supported.items():
            if getattr(self, name) != expected:
                raise ValueError(f"{name} must be {expected!r}")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict:
        """Return serializable settings suitable for an experiment record."""
        return {field.name: getattr(self, field.name) for field in fields(self)}
