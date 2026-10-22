"""Offline synthetic data -> test tokenizer -> packed batch -> tiny LM backward."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import torch

from .config import ModelConfig
from .data_processing import PreprocessingConfig, audit_document_splits, prepare_dataset
from .dataset import PackedTokenDataset, make_dataloader
from .dolma_reader import canonical_text
from .model import MoELanguageModel
from .tokenizer_utils import SplitConfig, train_tokenizer


def synthetic_documents(split: SplitConfig, *, train_count: int = 12, validation_count: int = 6):
    """Search for texts already assigned to each split, never reassign a document.

    This is a deliberately balanced test fixture, not a sampled Dolma mixture.
    The requested production split remains unchanged (normally 99/1).
    """
    result = []
    remaining = {"train": train_count, "validation": validation_count}
    for candidate in range(100_000):
        text = (f"Research document {candidate}. Experts route language tokens accurately. "
                "Numbers 0123456789; Unicode: tiếng Việt, 日本語.\n") * 40
        text = canonical_text(text)
        assigned = split.assign(hashlib.sha256(text.encode()).hexdigest())
        if remaining[assigned] > 0:
            result.append({"text": text})
            remaining[assigned] -= 1
        if not any(remaining.values()):
            return result
    raise ValueError("could not build the requested synthetic split fixture")


def run_smoke(root: str | Path) -> dict:
    torch.manual_seed(123)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    source = root / "synthetic.jsonl"
    if source.exists():
        raise FileExistsError(source)
    split = SplitConfig()
    documents = synthetic_documents(split)
    with source.open("w", encoding="utf-8", newline="\n") as stream:
        for document in documents:
            stream.write(json.dumps(document, ensure_ascii=False) + "\n")
    tokenizer = train_tokenizer([str(source)], root / "tokenizer.json", vocab_size=320,
                                max_documents=12, test_only=True, split=split)
    state = prepare_dataset(
        [str(source)], tokenizer, root / "packed", allow_test_tokenizer=True,
        config=PreprocessingConfig(train_max_tokens=4096, validation_max_tokens=2048, split=split),
        provenance={"repository": "synthetic", "version": "offline-fixture-v1"},
    )
    dataset = PackedTokenDataset(root / "packed", "train", tokenizer=tokenizer, verify_checksums=True)
    try:
        batch = next(iter(make_dataloader(dataset, batch_size=2, seed=123)))
        config = ModelConfig(vocab_size=tokenizer.vocab_size, d_model=16, n_layers=2,
                             n_heads=4, n_experts=4, expert_hidden_dim=8)
        tokenizer.validate_model(config, allow_test=True)
        model = MoELanguageModel(config)
        output = model(**batch)
        if not torch.isfinite(output.loss).item():
            raise AssertionError("nonfinite smoke loss")
        output.loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all().item():
                raise AssertionError(f"nonfinite gradient: {name}")
        if model.token_embedding.weight.grad is None:
            raise AssertionError("missing embedding gradient")
        return {
            "offline": True, "test_tokenizer": True, "vocab_size": tokenizer.vocab_size,
            "batch_shapes": {k: list(v.shape) for k, v in batch.items()},
            "batch_dtypes": {k: str(v.dtype) for k, v in batch.items()},
            "token_range": [batch["input_ids"].min().item(), batch["input_ids"].max().item()],
            "sample_first_ids": batch["input_ids"][0, :16].tolist(),
            "labels_equal_inputs": torch.equal(batch["input_ids"], batch["labels"]),
            "logits_shape": list(output.logits.shape), "loss": output.loss.item(),
            "backward": "PASS", "split_audit": audit_document_splits(root / "packed"),
            "packed_tokens": {s: state["splits"][s]["tokens"] for s in ("train", "validation")},
        }
    finally:
        dataset.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", help="Keep synthetic artifacts in a new/empty directory; default uses temp files")
    args = parser.parse_args()
    torch.set_num_threads(min(2, torch.get_num_threads()))
    if args.output_dir:
        report = run_smoke(args.output_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="moe-data-smoke-") as root:
            report = run_smoke(root)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
