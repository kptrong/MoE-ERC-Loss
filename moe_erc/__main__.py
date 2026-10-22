"""Run with python -m moe_erc from the repository root."""

from dataclasses import asdict
import argparse

from .config import AoEConfig, ModelConfig
from .parameter_count import count_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description="Analytical Vanilla MoE / AoE parameter budget")
    parser.add_argument("--model-type", choices=("vanilla_moe", "aoe"), default="vanilla_moe")
    args = parser.parse_args()
    config = AoEConfig() if args.model_type == "aoe" else ModelConfig()
    counts = count_parameters(config, args.model_type)
    print(f"Analytical parameter budget ({args.model_type})")
    for name, value in asdict(counts).items():
        print(f"{name:24s} {value:>12,}")
    print(f"{'total':24s} {counts.total:>12,}")
    print("Tied LM head: 0 additional parameters; RoPE: 0 learned parameters.")


if __name__ == "__main__":
    main()
