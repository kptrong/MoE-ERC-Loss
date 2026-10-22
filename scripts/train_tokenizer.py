"""Train a frozen byte-level BPE on training documents, never validation."""

import argparse

from moe_erc.tokenizer_utils import train_tokenizer
from ._common import run, source_arguments, source_options, split_options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source_arguments(parser)
    parser.add_argument("--output", required=True, help="New tokenizer .json path")
    parser.add_argument("--vocab-size", type=int, default=16000)
    parser.add_argument("--max-documents", type=int, default=100_000, help="Fixed unique train-only sample size")
    parser.add_argument("--test-only", action="store_true", help="Allow a small fixture vocabulary; never production")
    args = parser.parse_args()
    sources, provenance, downloads = source_options(args)
    tokenizer = train_tokenizer(
        sources, args.output, split=split_options(args), vocab_size=args.vocab_size,
        max_documents=args.max_documents, test_only=args.test_only,
        provenance=provenance, downloads=downloads,
    )
    print("Tokenizer:", tokenizer.path)
    print("Vocabulary:", tokenizer.vocab_size, "special IDs:", tokenizer.special_ids)
    print("SHA-256:", tokenizer.fingerprint)


if __name__ == "__main__":
    run(main)
