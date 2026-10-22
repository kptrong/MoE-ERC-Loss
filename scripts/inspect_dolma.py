"""Inspect a manifest and save a seeded subset; default is no shard download."""

import argparse
from pathlib import Path

from moe_erc.dolma_reader import (
    MANIFEST_URL, DownloadOptions, atomic_json, compressed_size, read_manifest,
    resolve_source, select_shards,
)
from ._common import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=MANIFEST_URL, help="Official URL or local manifest fixture")
    parser.add_argument("--max-shards", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-contains", help="Explicit substring filter on URL path; not mixture balancing")
    parser.add_argument("--check-size", action="store_true", help="Issue HEAD requests for compressed Content-Length")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Default: inspect only")
    mode.add_argument("--download", action="store_true", help="Explicitly download the selected shards")
    parser.add_argument("--save-selection")
    parser.add_argument("--cache-dir", default="data/raw")
    parser.add_argument("--max-download-bytes", type=int, default=1_000_000_000)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--retries", type=int, default=2)
    args = parser.parse_args()
    if args.save_selection and Path(args.save_selection).exists():
        raise FileExistsError("refusing to overwrite a frozen shard selection")
    manifest = read_manifest(args.manifest, timeout=args.timeout, retries=args.retries)
    selected = select_shards(manifest.pop("urls"), max_shards=args.max_shards,
                             seed=args.seed, source_contains=args.source_contains)
    plan = {**manifest, "selection_seed": args.seed, "source_contains": args.source_contains,
            "selection_algorithm": "sorted-unique, seeded-random-shuffle-v1",
            "selected_urls": selected}
    for url in selected:
        size = compressed_size(url, timeout=args.timeout) if args.check_size else None
        print(f"{url}\n  compressed bytes: {size if size is not None else 'unknown'}")
    print("Mode:", "explicit download" if args.download else "dry-run; no shard content downloaded")
    if args.save_selection:
        path = Path(args.save_selection)
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, plan)
    if args.download:
        options = DownloadOptions(True, args.cache_dir, args.max_download_bytes, args.timeout, args.retries)
        for source in selected:
            print("Cached:", resolve_source(source, options))


if __name__ == "__main__":
    run(main)
