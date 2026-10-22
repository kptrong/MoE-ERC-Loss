"""Compare technical execution of untrained Vanilla/AoE on identical IDs."""

import argparse
from contextlib import nullcontext
import gc

import torch

from moe_erc import AoEConfig, ModelConfig, count_parameters
from moe_erc.model_factory import create_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--bf16", action="store_true", help="CUDA BF16 autocast")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if args.bf16 and (args.device != "cuda" or not torch.cuda.is_bf16_supported()):
        parser.error("--bf16 requires BF16-capable CUDA")
    torch.set_num_threads(min(2, torch.get_num_threads()))
    torch.manual_seed(123)
    ids = torch.randint(16000, (1, 16), device=args.device)
    print(f"PyTorch {torch.__version__}; device={args.device}; BF16={args.bf16}")
    totals = {}
    for kind, config in (("vanilla_moe", ModelConfig()), ("aoe", AoEConfig())):
        # Sequential construction avoids retaining two full models/gradient sets.
        torch.manual_seed(456)
        model = create_model(kind, config).to(args.device)
        actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert actual == count_parameters(config, kind).total
        assert actual == (52_263_424 if kind == "aoe" else 52_288_000)
        assert model.lm_head.weight is model.token_embedding.weight
        context = torch.autocast("cuda", dtype=torch.bfloat16) if args.bf16 else nullcontext()
        with context:
            result = model(ids, labels=ids, return_routing_info=True)
        assert result.logits.shape == (1, 16, 16000)
        assert torch.isfinite(result.logits).all().item() and torch.isfinite(result.loss).item()
        result.loss.backward()
        assert model.token_embedding.weight.grad is not None
        for parameter in model.parameters():
            if parameter.grad is not None:
                assert torch.isfinite(parameter.grad).all().item()
        totals[kind] = actual
        print(f"{kind}: params={actual:,}, logits={tuple(result.logits.shape)}, loss={result.loss.item():.6f}, backward=PASS")
        for info in result.routing_info:
            diagnostic = info.diagnostics
            print(f"  layer {info.layer_index}: counts={diagnostic.token_counts.tolist()}, "
                  f"mean_p={diagnostic.mean_probabilities.mean().item():.6f}")
            if kind == "aoe":
                print(f"    mean L2 gate-down norm={diagnostic.mean_activation_norms.mean().item():.6f}")
        del parameter, model, result, info, diagnostic
        gc.collect()
        if args.device == "cuda":
            torch.cuda.empty_cache()
    print(f"AoE - Vanilla parameters: {totals['aoe'] - totals['vanilla_moe']:,}")
    print("Technical smoke only; these losses are not a model-quality comparison.")


if __name__ == "__main__":
    main()
