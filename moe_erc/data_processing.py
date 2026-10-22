"""Streaming packing with immutable datasets and recoverable shard commits."""

from dataclasses import asdict, dataclass
from contextlib import closing
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import sqlite3

import numpy as np

from .dolma_reader import DownloadOptions, atomic_json, file_sha256, iter_documents, resolve_source
from .tokenizer_utils import FrozenTokenizer, SPLIT_ALGORITHM, SplitConfig, chain_hash


LOGGER = logging.getLogger(__name__)
SPLITS = ("train", "validation")


@dataclass(frozen=True)
class PreprocessingConfig:
    train_max_tokens: int = 1_000_000
    validation_max_tokens: int = 100_000
    sequence_length: int = 512
    split: SplitConfig = SplitConfig()

    def __post_init__(self):
        for name in ("train_max_tokens", "validation_max_tokens", "sequence_length"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.sequence_length > 512:
            raise ValueError("sequence_length exceeds the model's maximum 512")


class _Packer:
    def __init__(self, path: Path, length: int, pending: list[int]):
        self.path = path
        self.length = length
        self.pending = list(pending)
        self.stream = path.open("wb")
        self.sequences = 0

    def feed(self, ids: list[int]):
        combined = self.pending + ids
        size = len(combined) // self.length * self.length
        if size:
            np.asarray(combined[:size], dtype="<u2").tofile(self.stream)
            self.sequences += size // self.length
        self.pending = combined[size:]

    def close(self):
        if not self.stream.closed:
            self.stream.flush()
            os.fsync(self.stream.fileno())
            self.stream.close()


def _empty_split_state():
    return {
        "documents": 0, "encoded_tokens": 0, "accepted_tokens": 0,
        "budget_discarded_tokens": 0, "documents_skipped_budget": 0,
        "content_fingerprint": "00" * 32, "pending": [], "segments": [],
    }


def _json_digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def prepare_dataset(
    sources: list[str], tokenizer: FrozenTokenizer, output_dir: str | Path, *,
    config: PreprocessingConfig = PreprocessingConfig(), provenance: dict | None = None,
    downloads: DownloadOptions | None = None, resume: bool = False,
    max_shards_per_run: int | None = None, allow_test_tokenizer: bool = False,
) -> dict:
    """Checkpoint at shard boundaries; tails carry over into the next shard.

    New budgets require a new output directory, reusing the same tokenizer and
    frozen ordered source plan. Validation selection never depends on train
    budget: scanning continues until both budgets are filled or sources end.
    A partial final document may be truncated to its split's exact token cap.
    """
    if not sources or len(set(sources)) != len(sources):
        raise ValueError("require a nonempty source plan with no duplicate paths/URLs")
    if max_shards_per_run is not None and max_shards_per_run <= 0:
        raise ValueError("max_shards_per_run must be positive")
    if tokenizer.metadata["test_only"] and not allow_test_tokenizer:
        raise ValueError("test-only tokenizer requires explicit allow_test_tokenizer=True")
    if not 260 <= tokenizer.vocab_size <= 65536:
        raise ValueError("uint16 packing requires vocabulary in [260,65536]")
    if not tokenizer.metadata["test_only"] and tokenizer.vocab_size != 16000:
        raise ValueError("production vocabulary must be exactly 16000")
    if tokenizer.metadata["split"] != asdict(config.split):
        raise ValueError("tokenizer training split differs from preprocessing split")
    options = downloads or DownloadOptions()
    root = Path(output_dir)
    plan = {
        "sources": list(sources), "tokenizer_sha256": tokenizer.fingerprint,
        "config": asdict(config), "provenance": provenance or {},
    }
    checkpoint_path = root / "checkpoint.json"
    if resume:
        state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if state["plan_sha256"] != _json_digest(plan):
            raise ValueError("resume configuration, source plan or tokenizer differs from checkpoint")
        if not (root / "documents.sqlite").is_file():
            raise ValueError("resume deduplication database is missing")
    else:
        root.mkdir(parents=True, exist_ok=False)
        state = {
            "format_version": 1, "plan": plan, "plan_sha256": _json_digest(plan),
            "tokenizer": {"file": str(tokenizer.path), "sha256": tokenizer.fingerprint,
                          "vocab_size": tokenizer.vocab_size, "special_ids": tokenizer.special_ids,
                          "test_only": tokenizer.metadata["test_only"]},
            "dtype": "<u2", "sequence_length": config.sequence_length,
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "tokenizers": tokenizer.metadata["tokenizers_version"]},
            "split_algorithm": SPLIT_ALGORITHM, "next_shard": 0, "complete": False,
            "reader_stats": {}, "duplicates": 0, "unique_documents_seen": 0,
            "processed_sources": [], "splits": {s: _empty_split_state() for s in SPLITS},
        }
    with closing(sqlite3.connect(str(root / "documents.sqlite"))) as seen:
        seen.execute("CREATE TABLE IF NOT EXISTS documents (hash TEXT PRIMARY KEY, split TEXT NOT NULL, shard INTEGER NOT NULL)")
        if resume:
            # SQLite may have committed one shard before its JSON checkpoint.
            # Roll back those rows; uncommitted binary segments will be rebuilt.
            seen.execute("DELETE FROM documents WHERE shard >= ?", (state["next_shard"],))
            seen.commit()
            if seen.execute("SELECT COUNT(*) FROM documents").fetchone()[0] != state["unique_documents_seen"]:
                raise ValueError("deduplication database/checkpoint mismatch")
            for entry in state["processed_sources"]:
                path = resolve_source(entry["source"], options)
                if file_sha256(path) != entry["sha256"]:
                    raise ValueError(f"processed source changed: {entry['source']}")
            _validate_segments(root, state, checksums=True)
        else:
            seen.commit()
            atomic_json(checkpoint_path, state)
        if state["complete"]:
            validate_dataset(root, tokenizer=tokenizer, checksums=True)
            return json.loads((root / "dataset.json").read_text(encoding="utf-8"))
        limits = {"train": config.train_max_tokens, "validation": config.validation_max_tokens}
        processed_this_run = 0
        while (state["next_shard"] < len(sources)
               and not all(state["splits"][s]["accepted_tokens"] >= limits[s] for s in SPLITS)):
            index = state["next_shard"]
            source = sources[index]
            path = resolve_source(source, options)
            source_hash = file_sha256(path)
            writers = {s: _Packer(root / f"{s}-{index:06d}.bin.part", config.sequence_length,
                                  state["splits"][s]["pending"]) for s in SPLITS}
            try:
                for document in iter_documents(path, stats=state["reader_stats"]):
                    split = config.split.assign(document.content_hash)
                    inserted = seen.execute("INSERT OR IGNORE INTO documents VALUES (?,?,?)",
                                            (document.content_hash, split, index)).rowcount
                    if not inserted:
                        state["duplicates"] += 1
                        continue
                    state["unique_documents_seen"] += 1
                    info = state["splits"][split]
                    remaining = limits[split] - info["accepted_tokens"]
                    if remaining == 0:
                        info["documents_skipped_budget"] += 1
                        continue
                    ids = tokenizer.encode(document.text) + [tokenizer.eos_id]
                    accepted = ids[:remaining]
                    info["documents"] += 1
                    info["encoded_tokens"] += len(ids)
                    info["accepted_tokens"] += len(accepted)
                    info["budget_discarded_tokens"] += len(ids) - len(accepted)
                    info["content_fingerprint"] = chain_hash(info["content_fingerprint"], document.content_hash)
                    writers[split].feed(accepted)
                    if all(state["splits"][s]["accepted_tokens"] >= limits[s] for s in SPLITS):
                        break
                if file_sha256(path) != source_hash:
                    raise ValueError(f"source changed during processing: {source}")
                for split, writer in writers.items():
                    writer.close()
                    final = root / f"{split}-{index:06d}.bin"
                    writer.path.replace(final)
                    state["splits"][split]["pending"] = writer.pending
                    state["splits"][split]["segments"].append({
                        "file": final.name, "sequences": writer.sequences,
                        "bytes": final.stat().st_size, "sha256": file_sha256(final),
                    })
                seen.commit()
                state["processed_sources"].append({"source": source, "sha256": source_hash})
                state["next_shard"] += 1
                # Complete metadata is written separately after this checkpoint.
                atomic_json(checkpoint_path, state)
            finally:
                for writer in writers.values():
                    writer.close()
            processed_this_run += 1
            LOGGER.info("Committed shard %s/%s: %s", state["next_shard"], len(sources), source)
            enough = all(state["splits"][s]["accepted_tokens"] >= limits[s] for s in SPLITS)
            if enough or state["next_shard"] == len(sources):
                break
            if max_shards_per_run is not None and processed_this_run >= max_shards_per_run:
                return state
        # Derive final counts without discarding tails at intermediate shards.
        for split in SPLITS:
            info = state["splits"][split]
            info["sequences"] = sum(segment["sequences"] for segment in info["segments"])
            info["tokens"] = info["sequences"] * config.sequence_length
            info["tail_discarded_tokens"] = len(info["pending"])
            info["discarded_tokens"] = info["tail_discarded_tokens"] + info["budget_discarded_tokens"]
            info["budget_reached"] = info["accepted_tokens"] == limits[split]
            if not info["documents"]:
                LOGGER.warning("No %s documents; no document was reassigned to force a split", split)
            if not info["budget_reached"]:
                LOGGER.warning("%s exhausted sources at %s/%s accepted tokens (%s packed)",
                               split, info["accepted_tokens"], limits[split], info["tokens"])
        state["complete"] = True
        # dataset.json first: if interrupted here, resume finalizes from the last
        # checkpoint rather than trusting an incomplete or orphaned output.
        atomic_json(root / "dataset.json", state)
        atomic_json(checkpoint_path, state)
    return state


def _validate_segments(root: Path, metadata: dict, *, checksums: bool):
    length = metadata["sequence_length"]
    if not 1 <= length <= 512 or metadata["dtype"] != "<u2":
        raise ValueError("unsupported dataset sequence length or dtype")
    for split in SPLITS:
        for entry in metadata["splits"][split]["segments"]:
            path = root / entry["file"]
            if path.resolve().parent != root.resolve():
                raise ValueError("segment path must stay within dataset directory")
            expected = entry["sequences"] * length * 2
            if expected < 0 or entry["bytes"] != expected or path.stat().st_size != expected:
                raise ValueError(f"segment size mismatch: {path}")
            if checksums and file_sha256(path) != entry["sha256"]:
                raise ValueError(f"segment checksum mismatch: {path}")
            if checksums and expected:
                view = np.memmap(path, dtype="<u2", mode="r")
                for start in range(0, len(view), 1_000_000):
                    if int(view[start:start + 1_000_000].max()) >= metadata["tokenizer"]["vocab_size"]:
                        raise ValueError(f"out-of-vocabulary token in {path}")
                del view


def validate_dataset(root: str | Path, *, tokenizer: FrozenTokenizer | None = None, checksums: bool = True) -> dict:
    root = Path(root)
    metadata = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    if not metadata["complete"] or metadata["format_version"] != 1:
        raise ValueError("dataset is incomplete or has an unsupported format")
    if _json_digest(metadata["plan"]) != metadata["plan_sha256"]:
        raise ValueError("dataset plan fingerprint mismatch")
    if metadata["plan"]["tokenizer_sha256"] != metadata["tokenizer"]["sha256"]:
        raise ValueError("dataset plan/tokenizer fingerprint mismatch")
    if tokenizer is not None and (
        metadata["tokenizer"]["sha256"] != tokenizer.fingerprint
        or metadata["tokenizer"]["vocab_size"] != tokenizer.vocab_size
        or metadata["tokenizer"]["special_ids"] != tokenizer.special_ids
    ):
        raise ValueError("dataset/tokenizer mismatch")
    _validate_segments(root, metadata, checksums=checksums)
    for split in SPLITS:
        info = metadata["splits"][split]
        if info["sequences"] != sum(s["sequences"] for s in info["segments"]):
            raise ValueError("dataset sequence count mismatch")
        if info["tokens"] != info["sequences"] * metadata["sequence_length"]:
            raise ValueError("dataset token count mismatch")
        if info["encoded_tokens"] != info["tokens"] + info["discarded_tokens"]:
            raise ValueError("dataset discarded-token accounting mismatch")
    return metadata


def audit_document_splits(root: str | Path) -> dict:
    """Offline audit of all seen content hashes using the recorded split policy."""
    root = Path(root)
    metadata = json.loads((root / "checkpoint.json").read_text(encoding="utf-8"))
    split = SplitConfig(**metadata["plan"]["config"]["split"])
    counts = {s: 0 for s in SPLITS}
    connection = sqlite3.connect(f"{(root / 'documents.sqlite').resolve().as_uri()}?mode=ro", uri=True)
    try:
        for content_hash, assigned in connection.execute("SELECT hash, split FROM documents"):
            if assigned != split.assign(content_hash):
                raise ValueError("document content hash appears in the wrong split")
            counts[assigned] += 1
        # PRIMARY KEY(hash) ensures the deduplicated ledger cannot contain the
        # same content in both splits; the loop independently verifies assignment.
        counts["overlap"] = 0
    finally:
        connection.close()
    return counts
