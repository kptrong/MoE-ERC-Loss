"""Split, tokenize and pack a frozen corpus; explicit opt-in for remote shards."""

import argparse

from moe_erc.data_processing import PreprocessingConfig, prepare_dataset
from moe_erc.tokenizer_utils import FrozenTokenizer
from ._common import run, source_arguments, source_options, split_options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source_arguments(parser)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-max-tokens", type=int, default=1_000_000)
    parser.add_argument("--validation-max-tokens", type=int, default=100_000)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--resume", action="store_true", help="Resume same plan at last committed shard")
    parser.add_argument("--max-shards-per-run", type=int, help="Stop cleanly at a shard boundary")
    parser.add_argument("--allow-test-tokenizer", action="store_true")
    args = parser.parse_args()
    sources, provenance, downloads = source_options(args)
    state = prepare_dataset(
        sources, FrozenTokenizer(args.tokenizer), args.output_dir,
        config=PreprocessingConfig(args.train_max_tokens, args.validation_max_tokens,
                                   args.sequence_length, split_options(args)),
        provenance=provenance, downloads=downloads, resume=args.resume,
        max_shards_per_run=args.max_shards_per_run, allow_test_tokenizer=args.allow_test_tokenizer,
    )
    print("Complete:", state["complete"], "committed shards:", state["next_shard"])
    for split, info in state["splits"].items():
        print(split, {k: v for k, v in info.items() if k not in ("pending", "segments")})


if __name__ == "__main__":
    run(main)
