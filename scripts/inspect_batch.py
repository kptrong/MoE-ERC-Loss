"""Inspect a saved memmapped batch, optionally run one model forward/backward."""

import argparse

import torch

from moe_erc.config import ModelConfig
from moe_erc.dataset import PackedTokenDataset, make_dataloader
from moe_erc.model import MoELanguageModel
from moe_erc.tokenizer_utils import FrozenTokenizer
from ._common import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify-checksums", action="store_true")
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--model-check", action="store_true", help="Run default 52M LM forward/backward")
    parser.add_argument("--tiny-model", action="store_true", help="Explicit test-only small LM for fixture tokenizer")
    args = parser.parse_args()
    tokenizer = FrozenTokenizer(args.tokenizer)
    dataset = PackedTokenDataset(args.dataset, args.split, tokenizer=tokenizer,
                                  verify_checksums=args.verify_checksums)
    try:
        batch = next(iter(make_dataloader(dataset, batch_size=args.batch_size,
                                          seed=args.seed, num_workers=args.num_workers)))
        for name, tensor in batch.items():
            print(name, tuple(tensor.shape), tensor.dtype, "range:", tensor.min().item(), tensor.max().item())
        if args.decode:
            print(tokenizer.decode(batch["input_ids"][0].tolist()))
        if args.model_check or args.tiny_model:
            config = (ModelConfig(vocab_size=tokenizer.vocab_size, d_model=16, n_layers=2,
                                  n_heads=4, n_experts=4, expert_hidden_dim=8)
                      if args.tiny_model else ModelConfig())
            tokenizer.validate_model(config, allow_test=args.tiny_model)
            torch.manual_seed(args.seed)
            torch.set_num_threads(min(4, torch.get_num_threads()))
            model = MoELanguageModel(config)
            result = model(**batch)
            if not torch.isfinite(result.loss).item():
                raise ValueError("model loss is not finite")
            result.loss.backward()
            if not all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None):
                raise ValueError("model gradients are not finite")
            print("logits:", tuple(result.logits.shape), "loss:", result.loss.item(), "backward: PASS")
    finally:
        dataset.close()


if __name__ == "__main__":
    run(main)
