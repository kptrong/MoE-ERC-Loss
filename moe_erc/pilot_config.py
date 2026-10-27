"""Approved pilot defaults, exact sequence accounting and LR schedule."""

from dataclasses import asdict, dataclass
import math

from .config import ERCConfig

METHODS = ("vanilla_moe", "aoe", "vanilla_moe_erc")
DATASET_FP = "92acf38d5017658cd71df8449d8edc553d663cd32fcad4350bb55b7c8cc6e144"
TOKENIZER_SHA = "bfdf3501c3037ef2692e039e11b593e9d89b9143caa213ab16c8381e86831797"
PILOT_SHA = "812ea85c57344598adf4c18268fcca656da8661bdab400cdfd6542b9cc9fbe98"


@dataclass(frozen=True)
class PilotConfig:
    """CLI uses these defaults. Overrides allow explicit tiny local fixtures."""

    sequence_length: int = 512
    microbatch_size: int = 2
    accumulation_steps: int = 16
    validation_microbatch_size: int = 2
    seed: int = 42
    peak_lr: float = 4e-4
    min_lr: float = 4e-5
    warmup_updates: int = 64
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    clip_norm: float = 1.0
    lb_coefficient: float = 0.01
    bf16: bool = True
    evaluation_interval_tokens: int = 5_000_000
    diagnostic_sequences: int = 32
    console_interval_updates: int = 50
    protocol_version: str = "pilot-v1"

    def __post_init__(self):
        for name in ("sequence_length", "microbatch_size", "accumulation_steps",
                     "validation_microbatch_size", "warmup_updates", "evaluation_interval_tokens",
                     "diagnostic_sequences", "console_interval_updates"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.sequence_length < 2 or type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("invalid sequence length or seed")
        for name in ("peak_lr", "min_lr", "beta1", "beta2", "weight_decay", "clip_norm"):
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"{name} must be finite")
        if not 0 < self.min_lr <= self.peak_lr or self.clip_norm <= 0 or self.weight_decay < 0:
            raise ValueError("invalid LR, clipping or decay")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("invalid AdamW betas")
        if type(self.bf16) is not bool or isinstance(self.lb_coefficient, bool) or self.lb_coefficient != 0.01:
            raise ValueError("invalid precision or frozen LB coefficient")
        if self.protocol_version != "pilot-v1":
            raise ValueError("unsupported pilot version")

    @property
    def effective_sequences(self):
        return self.microbatch_size * self.accumulation_steps

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != set(cls().to_dict()):
            raise ValueError("pilot config schema mismatch")
        return cls(**value)


def erc_config_for(method):
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    return ERCConfig(enabled=method == "vanilla_moe_erc")


@dataclass(frozen=True)
class PilotPlan:
    indices: tuple[int, ...]
    config: PilotConfig

    def __post_init__(self):
        if (not isinstance(self.indices, tuple) or not self.indices
                or any(type(i) is not int or i < 0 for i in self.indices)
                or len(set(self.indices)) != len(self.indices)):
            raise ValueError("indices must be an immutable nonempty tuple of unique nonnegative IDs")

    @property
    def total_updates(self):
        return math.ceil(len(self.indices) / self.config.effective_sequences)

    @property
    def total_input_tokens(self):
        return len(self.indices) * self.config.sequence_length

    @property
    def total_targets(self):
        return len(self.indices) * (self.config.sequence_length - 1)

    @property
    def milestones(self):
        # A threshold first reached at the final update is the endpoint evaluation,
        # not an additional 50M milestone beside 50,000,384.
        positions_per_update = self.config.effective_sequences * self.config.sequence_length
        intermediate = (m for m in range(self.config.evaluation_interval_tokens,
                                         self.total_input_tokens,
                                         self.config.evaluation_interval_tokens)
                        if math.ceil(m / positions_per_update) < self.total_updates)
        return (0, *intermediate, self.total_input_tokens)

    def update_indices(self, completed_updates):
        if type(completed_updates) is not int or not 0 <= completed_updates <= self.total_updates:
            raise ValueError("invalid completed update count")
        cursor = min(completed_updates * self.config.effective_sequences, len(self.indices))
        return self.indices[cursor:cursor + self.config.effective_sequences]


def learning_rate(update, total_updates, config):
    """LR USED by one-based update s; final update uses min_lr exactly."""
    if not 1 <= update <= total_updates or total_updates <= config.warmup_updates:
        raise ValueError("schedule requires valid update and total_updates > warmup_updates")
    if update <= config.warmup_updates:
        return config.peak_lr * update / config.warmup_updates
    p = (update - config.warmup_updates) / (total_updates - config.warmup_updates)
    return config.min_lr + (config.peak_lr - config.min_lr) * (1 + math.cos(math.pi * p)) / 2
