"""One forward/backward on synthetic IDs; no optimizer, dataset or training."""

import argparse
from contextlib import nullcontext

import torch

from .config import ModelConfig
from .model import MoELanguageModel
from .parameter_count import count_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--bf16", action="store_true", help="CUDA BF16 autocast")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if args.bf16 and (args.device != "cuda" or not torch.cuda.is_bf16_supported()):
        parser.error("--bf16 requires a BF16-capable CUDA device")
    torch.manual_seed(123)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    config = ModelConfig()
    model = MoELanguageModel(config).to(args.device)
    actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected = count_parameters(config).total
    if actual != expected or actual != 52_288_000:
        raise AssertionError(f"parameter mismatch: actual={actual}, expected={expected}")
    assert model.lm_head.weight is model.token_embedding.weight
    input_ids = torch.randint(config.vocab_size, (1, 16), device=args.device)
    context = torch.autocast("cuda", dtype=torch.bfloat16) if args.bf16 else nullcontext()
    with context:
        result = model(input_ids, labels=input_ids)
    assert result.logits.shape == (1, 16, 16000)
    assert torch.isfinite(result.logits).all().item()
    assert torch.isfinite(result.loss).item()
    result.loss.backward()
    received = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all().item():
                raise AssertionError(f"nonfinite gradient: {name}")
            received += 1
    # Unselected experts may legitimately have None gradients in sparse dispatch.
    for name, parameter in (
        ("embedding", model.token_embedding.weight),
        *((f"layer {i} attention", layer.attention.q_proj.weight) for i, layer in enumerate(model.layers)),
        *((f"layer {i} router", layer.moe.router.weight) for i, layer in enumerate(model.layers)),
    ):
        assert parameter.grad is not None, f"missing gradient: {name}"
        assert parameter.grad.abs().sum().item() > 0, f"zero gradient: {name}"
    print(f"PyTorch: {torch.__version__}; device: {args.device}; BF16 autocast: {args.bf16}")
    print(f"Trainable parameters: {actual:,}; analytical: {expected:,}")
    print(f"Tied embedding/LM head: {model.lm_head.weight is model.token_embedding.weight}")
    print(f"Logits: {tuple(result.logits.shape)}; dtype: {result.logits.dtype}")
    print(f"Causal LM loss: {result.loss.item():.6f}; dtype: {result.loss.dtype}")
    print(f"Backward: PASS; finite gradients in {received} unique parameter tensors")


if __name__ == "__main__":
    main()
