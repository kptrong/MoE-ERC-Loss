"""Run with python -m moe_erc from the repository root."""

from dataclasses import asdict

from .config import ModelConfig
from .parameter_count import count_parameters


def main() -> None:
    config = ModelConfig()
    counts = count_parameters(config)
    print("Analytical parameter budget (default ModelConfig)")
    for name, value in asdict(counts).items():
        print(f"{name:24s} {value:>12,}")
    print(f"{'total':24s} {counts.total:>12,}")
    print("Tied LM head: 0 additional parameters; RoPE: 0 learned parameters.")


if __name__ == "__main__":
    main()
