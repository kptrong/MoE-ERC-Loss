"""Read-only validation and fixed-order batching of existing pilot artifacts."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess

import numpy as np
from torch.utils.data import default_collate

from .data_processing import validate_dataset
from .pilot_config import DATASET_FP, PILOT_SHA, TOKENIZER_SHA, PilotConfig, PilotPlan
from .tokenizer_utils import FrozenTokenizer


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def safe_file(root, name):
    if not isinstance(name, str) or not name:
        raise ValueError("empty artifact path")
    rel = PurePosixPath(name)
    if (rel.is_absolute() or ".." in rel.parts or "\\" in name or ":" in name
            or rel.as_posix() != name or any(ord(c) < 32 for c in name)):
        raise ValueError(f"unsafe artifact path: {name!r}")
    root = Path(root)
    for p in (root, *root.parents):
        if p.is_symlink():
            raise ValueError(f"symlink: {p}")
    path = root
    for part in rel.parts:
        path /= part
        if path.is_symlink():
            raise ValueError(f"symlink: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def microbatches(dataset, indices, batch_size):
    """No iterator RNG, shuffling, padding, repetition or dropped final records."""
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(indices), batch_size):
        yield default_collate([dataset[int(i)] for i in indices[start:start + batch_size]])


@dataclass(frozen=True)
class PilotArtifacts:
    root: Path
    tokenizer_path: Path
    plan: PilotPlan
    identity: dict
    checksums: dict

    def verify_unchanged(self):
        for name, expected in self.checksums.items():
            require(file_sha256(name) == expected, f"protected artifact changed: {name}")


def load_pilot_artifacts(root, tokenizer_path, config=PilotConfig(), *, full_training=False):
    root, tokenizer_path = Path(root), Path(tokenizer_path)
    tokenizer_path = safe_file(tokenizer_path.parent, tokenizer_path.name)
    require(file_sha256(tokenizer_path) == TOKENIZER_SHA, "tokenizer SHA mismatch")
    tokenizer = FrozenTokenizer(tokenizer_path)
    require(tokenizer.vocab_size == 16000 and tokenizer.eos_id == 2, "tokenizer config")
    manifest_path = safe_file(root, "dataset_manifest.json")
    marker = read_json(safe_file(root, "COMPLETE.json"))
    require(not (root / "INCOMPLETE.json").exists(), "incomplete dataset marker present")
    require(marker["status"] == "COMPLETE" and marker["tests_status"] == "PASS"
            and file_sha256(manifest_path) == marker["dataset_manifest_sha256"], "dataset completion")
    manifest = read_json(manifest_path)
    require(manifest["combined_source_fingerprint"] == DATASET_FP
            and manifest["tokenizer_sha256"] == TOKENIZER_SHA, "dataset identity")
    checksums, names = {}, set()
    for entry in manifest["files"]:
        require(entry["path"] not in names, "duplicate manifest entry")
        names.add(entry["path"])
        p = safe_file(root, entry["path"])
        require(p.stat().st_size == entry["bytes"] and file_sha256(p) == entry["sha256"],
                f"dataset checksum: {p}")
        checksums[str(p)] = entry["sha256"]
    design = read_json(safe_file(root, "packing_design.json"))
    encoded = json.dumps(design, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    require(hashlib.sha256(encoded).hexdigest() == manifest["design_sha256"], "design fingerprint")
    require(design["combined_source_fingerprint"] == DATASET_FP, "design source identity")
    recorded = design["pipeline_source_sha256"]
    require(set(recorded) == {"model.py", "dataset.py", "data_processing.py", "tokenizer_utils.py"},
            "protected source set")
    package = Path(__file__).resolve().parent
    creation = "2fd3478e478d3a20381e95b06890f8dcdf2ae013"
    for name, historical in recorded.items():
        original = subprocess.check_output(
            ["git", "-C", str(package.parent), "show", f"{creation}:moe_erc/{name}"])
        require(hashlib.sha256(original).hexdigest() == historical, "historical source identity")
        expected = historical
        if name == "model.py":
            fixed = subprocess.check_output([
                "git", "-C", str(package.parent), "show",
                "665cb7b1cfbca4c6ed3507345b21008321129a2a:moe_erc/model.py"])
            expected = hashlib.sha256(fixed).hexdigest()
        require(file_sha256(package / name) == expected, f"data compatibility: {name}")
    metadata = validate_dataset(root, tokenizer=tokenizer, checksums=True)
    require(metadata["sequence_length"] == config.sequence_length == 512
            and metadata["dtype"] == "<u2"
            and metadata["plan"]["provenance"]["git_commit"] == creation
            and metadata["splits"]["train"]["sequences"] == 107478
            and metadata["splits"]["validation"]["sequences"] == 1006, "dataset layout/provenance")
    require(metadata["plan"]["provenance"]["combined_source_fingerprint"] == DATASET_FP
            and metadata["plan"]["provenance"]["design_sha256"] == manifest["design_sha256"],
            "packing provenance")
    required = {"dataset.json", "packing_design.json", "tokenizer/tokenizer.json",
                "tokenizer/tokenizer.meta.json"}
    required.update(segment["file"] for split in metadata["splits"].values()
                    for segment in split["segments"])
    require(required <= names, "manifest omits protected dataset files")
    require(file_sha256(safe_file(root, "tokenizer/tokenizer.json")) == TOKENIZER_SHA
            and file_sha256(safe_file(root, "tokenizer/tokenizer.meta.json"))
                == file_sha256(tokenizer_path.with_suffix(".meta.json")), "bundled tokenizer mismatch")
    if full_training:
        from .data_processing import audit_document_splits
        require(audit_document_splits(root)["overlap"] == 0, "train/validation document overlap")
        import hashlib as _hashlib
        indices = np.random.Generator(np.random.PCG64(config.seed)).permutation(107478).astype("<i8")
        index_sha = _hashlib.sha256(indices.tobytes()).hexdigest()
        full = {"schema": "full55m-v1", "seed": config.seed,
                "ordering": "PCG64(seed).permutation(107478)", "sequences": 107478,
                "input_positions": 55028736, "effective_targets": 54921258,
                "index_bytes_sha256": index_sha, "dataset_manifest_sha256": file_sha256(manifest_path),
                "dataset_fp": DATASET_FP, "tokenizer_sha256": TOKENIZER_SHA}
        fingerprint = _hashlib.sha256(json.dumps(full, sort_keys=True).encode()).hexdigest()
        for p in (tokenizer_path, tokenizer_path.with_suffix(".meta.json"), manifest_path, root / "COMPLETE.json"):
            checksums[str(p)] = file_sha256(p)
        return PilotArtifacts(root, tokenizer_path, PilotPlan(tuple(map(int, indices)), config),
                {"dataset_fp": DATASET_FP, "tokenizer_sha256": TOKENIZER_SHA,
                 "pilot_manifest_sha256": fingerprint, "pilot_index_sha256": index_sha,
                 "training_manifest": full, "creation_commit": creation}, checksums)
    pilot_path = safe_file(root, "pilot_50m_manifest.json")
    require(file_sha256(pilot_path) == PILOT_SHA
            == safe_file(root, "pilot_50m_manifest.sha256").read_text().strip(), "pilot SHA")
    pilot = read_json(pilot_path)
    require(pilot["combined_source_fingerprint"] == DATASET_FP
            and pilot["tokenizer_sha256"] == TOKENIZER_SHA
            and pilot["dataset_manifest_sha256"] == file_sha256(manifest_path)
            and pilot["split"] == "train" and pilot["repeat_sequences"] is False, "pilot identity")
    require(pilot["sequence_length"] == 512 and pilot["seed"] == 42
            and pilot["train_binary_fingerprint"] == manifest["split_fingerprints"]["train"]
            and pilot["validation_binary_fingerprint"] == manifest["split_fingerprints"]["validation"],
            "pilot binary identity")
    index_path = safe_file(root, pilot["index_file"])
    require(file_sha256(index_path) == pilot["index_file_sha256"], "index SHA")
    indices = np.load(index_path, allow_pickle=False)
    require(indices.ndim == 1 and indices.dtype.str == pilot["index_dtype"] == "<i8"
            and len(indices) == pilot["selected_sequences"] == 97657
            and len(np.unique(indices)) == len(indices)
            and int(indices.min()) >= 0 and int(indices.max()) < 107478, "index content")
    plan = PilotPlan(tuple(int(i) for i in indices), config)
    require(plan.total_input_tokens == pilot["selected_input_tokens"] == 50_000_384
            and plan.total_targets == 49_902_727, "pilot budget")
    for p in (tokenizer_path, tokenizer_path.with_suffix(".meta.json"), manifest_path,
              root / "COMPLETE.json", pilot_path, root / "pilot_50m_manifest.sha256", index_path):
        checksums[str(p)] = file_sha256(p)
    identity = {
        "dataset_fp": DATASET_FP, "tokenizer_sha256": TOKENIZER_SHA,
        "pilot_manifest_sha256": PILOT_SHA, "pilot_index_sha256": file_sha256(index_path),
        "dataset_manifest_sha256": file_sha256(manifest_path), "creation_commit": creation,
    }
    return PilotArtifacts(root, tokenizer_path, plan, identity, checksums)


def inspect_pilot_artifacts(root, tokenizer_path):
    try:
        assets = load_pilot_artifacts(root, tokenizer_path)
        return {"status": "PASS", "identity": assets.identity,
                "sequences": len(assets.plan.indices), "updates": assets.plan.total_updates}
    except FileNotFoundError as exc:
        return {"status": "UNKNOWN", "reason": str(exc)}
    except (ValueError, KeyError, subprocess.CalledProcessError) as exc:
        return {"status": "FAIL", "reason": str(exc)}
