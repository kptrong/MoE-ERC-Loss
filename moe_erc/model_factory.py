"""Architecture selection for a future shared training/evaluation pipeline."""

from .config import AoEConfig, ModelConfig
from .model import AoELanguageModel, MoELanguageModel


def create_model(model_type: str = "vanilla_moe", config: ModelConfig | None = None) -> MoELanguageModel:
    if model_type == "vanilla_moe":
        return MoELanguageModel(config if config is not None else ModelConfig())
    if model_type == "aoe":
        if config is None:
            config = AoEConfig()
        elif not isinstance(config, AoEConfig):
            config = AoEConfig(**config.to_dict())
        return AoELanguageModel(config)
    raise ValueError("model_type must be 'vanilla_moe' or 'aoe'; ERC is not a separate architecture")
