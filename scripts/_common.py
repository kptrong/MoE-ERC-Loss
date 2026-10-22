"""Shared CLI options for local/explicitly-enabled remote source plans."""

import json
import logging
from pathlib import Path

from moe_erc.dolma_reader import DownloadOptions
from moe_erc.tokenizer_utils import SplitConfig


def source_arguments(parser):
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--inputs", nargs="+", help="Ordered local JSONL/.json.gz files")
    group.add_argument("--selection", help="Saved JSON plan from inspect_dolma")
    parser.add_argument("--seed", type=int, default=42, help="Stable document-split seed")
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--allow-download", action="store_true", help="Explicitly permit remote shard downloads")
    parser.add_argument("--cache-dir", default="data/raw")
    parser.add_argument("--max-download-bytes", type=int, default=1_000_000_000)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--retries", type=int, default=2)


def source_options(args):
    if args.selection:
        provenance = json.loads(Path(args.selection).read_text(encoding="utf-8"))
        sources = provenance["selected_urls"]
    else:
        sources = list(args.inputs)
        provenance = {"repository": "local", "version": "local-jsonl", "manifest_source": None,
                      "manifest_sha256": None, "selected_urls": sources}
    return sources, provenance, DownloadOptions(
        allow_download=args.allow_download, cache_dir=args.cache_dir,
        max_bytes=args.max_download_bytes, timeout=args.timeout, retries=args.retries,
    )


def split_options(args):
    return SplitConfig(seed=args.seed, validation_fraction=args.validation_fraction)


def run(main):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        main()
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        logging.error("%s", exc)
        raise SystemExit(1) from exc
