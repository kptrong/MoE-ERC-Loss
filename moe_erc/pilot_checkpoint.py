"""Step 6.5C/7.5C checkpoint port with pilot-sized boundary validation.

Same step65c-v1 payload/manifest envelope, atomic staging, pre-save digest,
weights_only loading and exact immediate restore comparison as notebook cell 89.
Identity is now pilot-v1, so historical smoke checkpoints cannot initialize a
pilot. Drive is a configured filesystem path; this module never mounts services.
"""

import copy
import math
from pathlib import Path
import shutil
import uuid

import torch

from . import pilot_checkpoint_core as core
from .pilot_config import PilotConfig, erc_config_for
from .pilot_data import file_sha256, read_json
from .erc_training import validate_erc_checkpoint_config


def validate_identity(identity, expected):
    core.exact(expected, identity, "identity")
    core.rt_require(identity["pilot_protocol"] == "pilot-v1", "not a pilot checkpoint")
    PilotConfig.from_dict(identity["training"])
    validate_erc_checkpoint_config(identity["erc_config"], erc_config_for(identity["method"]))


def new_progress(index_sha):
    return ({"step": 0, "input_tokens": 0, "targets": 0, "evaluated": [], "train_seconds": 0.0},
            {"cursor": 0, "epoch": 0, "shuffle": False, "pilot_index_sha256": index_sha})


def validate_progress(progress, sampler, plan, index_sha, require_evaluated=True):
    core.rt_require(set(progress) == {"step", "input_tokens", "targets", "evaluated", "train_seconds"},
                    "progress fields")
    core.rt_require(set(sampler) == {"cursor", "epoch", "shuffle", "pilot_index_sha256"},
                    "sampler fields")
    step = progress["step"]
    core.rt_require(type(step) is int and 0 <= step <= plan.total_updates, "global step")
    cursor = min(step * plan.config.effective_sequences, len(plan.indices))
    for actual, expected, name in (
        (sampler["cursor"], cursor, "cursor"),
        (progress["input_tokens"], cursor * plan.config.sequence_length, "input tokens"),
        (progress["targets"], cursor * (plan.config.sequence_length - 1), "targets"),
    ):
        core.rt_require(type(actual) is int and actual == expected, name)
    core.rt_require(sampler["epoch"] == 0 and sampler["shuffle"] is False
                    and sampler["pilot_index_sha256"] == index_sha, "sampler identity")
    evaluated = progress["evaluated"]
    due = [m for m in plan.milestones if m <= progress["input_tokens"]]
    core.rt_require(isinstance(evaluated, list)
                    and all(type(m) is int for m in evaluated)
                    and evaluated == due[:len(evaluated)], "evaluation milestone ordering")
    if require_evaluated:
        core.exact(due, evaluated, "completed evaluation milestones")
    seconds = progress["train_seconds"]
    core.rt_require(isinstance(seconds, (float, int)) and math.isfinite(seconds) and seconds >= 0,
                    "training time")


def copy_verified(source, target):
    source, target = Path(source), Path(target)
    core.rt_require(not target.exists() and not target.is_symlink(), "refusing file overwrite")
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".partial-" + uuid.uuid4().hex)
    with source.open("rb") as src, part.open("xb") as dst:
        shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)
    core.rt_require(part.stat().st_size == source.stat().st_size
                    and file_sha256(part) == file_sha256(source), "copy checksum mismatch")
    core.rt_require(not target.exists(), "refusing file overwrite")
    part.rename(target)
    core.rt_require(file_sha256(target) == file_sha256(source), "read-back checksum mismatch")


class CheckpointManager:
    def __init__(self, local_root, drive_root, identity, plan, guard=lambda: None, *, scaler=None):
        self.local, self.drive = Path(local_root), Path(drive_root) if drive_root is not None else None
        self.scaler = scaler
        self.identity, self.plan, self.guard = copy.deepcopy(identity), plan, guard
        core.rt_require(self.drive is None or self.local.resolve() != self.drive.resolve(), "persistence paths must differ")
        validate_identity(identity, identity)
        core.exact(plan.config.to_dict(), identity["training"], "plan config")
        core.rt_require(identity["total_sequences"] == len(plan.indices), "plan sequence count")
        for root in ((self.local, self.drive) if self.drive is not None else (self.local,)):
            for p in (root, *root.parents):
                core.rt_require(not p.is_symlink(), f"symlink namespace: {p}")
            root.mkdir(parents=True, exist_ok=True)
            owner = root / "run-identity.json"
            if owner.exists():
                validate_identity(read_json(owner), self.identity)
            else:
                core.write_json(owner, self.identity)
        self.last_equality = None

    def _check(self, progress, sampler):
        validate_progress(progress, sampler, self.plan, self.identity["pilot_index_sha256"])

    def _tying(self, model):
        a, b = model.token_embedding.weight, model.lm_head.weight
        core.rt_require(a is b and a.data_ptr() == b.data_ptr(), "weight tying lost")
        core.rt_require(sum(p.numel() for p in model.parameters()) == self.identity["parameters"],
                        "parameter count mismatch")

    def _publish(self, stage, root, manifest_sha):
        core.verify(stage, self.identity, manifest_sha, unpublished=True)
        core.write_json(stage / "COMPLETE.json", {"status": "COMPLETE", "manifest_sha256": manifest_sha})
        (stage / "INCOMPLETE.json").unlink()  # Only our newly created staging marker.
        core.rt_require(not root.exists(), "refusing checkpoint overwrite")
        stage.rename(root)
        core.verify(root, self.identity, manifest_sha)

    def save(self, model, optimizer, scheduler, progress, sampler, reason="milestone"):
        self.guard()
        self._check(progress, sampler)
        self._tying(model)
        core.rt_require(scheduler.last_epoch == progress["step"], "save scheduler boundary")
        core.rt_require(all(p.grad is None for p in model.parameters()), "unresolved gradients")
        state = core.snapshot(model, optimizer, scheduler, progress, sampler, self.scaler)
        digest = core.digest_tree(state)
        payload = {"schema": "step65c-v1", "identity": self.identity, "state": state}
        core.validate_payload(payload, self.identity)
        name = f"step{progress['step']:06d}-tokens{progress['input_tokens']:09d}-{uuid.uuid4().hex[:12]}"
        root = self.local / "checkpoints" / name
        root.parent.mkdir(parents=True, exist_ok=True)
        stage = root.with_name(root.name + ".partial-" + uuid.uuid4().hex)
        stage.mkdir(exist_ok=False)
        core.write_json(stage / "INCOMPLETE.json", {"status": "INCOMPLETE"})
        with (stage / "checkpoint.pt.part").open("xb") as f:
            torch.save(payload, f)
            f.flush()
            import os
            os.fsync(f.fileno())
        (stage / "checkpoint.pt.part").rename(stage / "checkpoint.pt")
        manifest = {
            "schema": "step65c-v1", "identity": self.identity, "state_digest": digest,
            "payload": {"file": "checkpoint.pt", "bytes": (stage / "checkpoint.pt").stat().st_size,
                        "sha256": file_sha256(stage / "checkpoint.pt")},
        }
        core.write_json(stage / "manifest.json", manifest)
        ms = file_sha256(stage / "manifest.json")
        with (stage / "manifest.sha256").open("x") as f:
            f.write(ms + "\n")
        core.verify(stage, self.identity, ms, unpublished=True)
        loaded = torch.load(stage / "checkpoint.pt", map_location="cpu", weights_only=True)
        core.validate_payload(loaded, self.identity)
        core.exact(state, loaded["state"], "serialization")
        core.rt_require(core.digest_tree(loaded["state"]) == digest, "serialization digest")
        del loaded
        self._publish(stage, root, ms)
        core.exact(state["rng"], core.rng_get(), "save RNG side effect")
        del state, payload

        if self.drive is None:
            receipt = {"path": str(root.resolve()), "manifest_sha256": ms,
                       "payload_sha256": manifest["payload"]["sha256"], "reason": reason,
                       "step": progress["step"], "input_tokens": progress["input_tokens"]}
            directory = self.local / "receipts"
            directory.mkdir(exist_ok=True)
            core.write_json(directory / f"{name}.json", receipt)
            return receipt

        # Same staged/read-back Drive copy protocol as notebook cell 90.
        destination = self.drive / "checkpoints" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        size = sum(p.stat().st_size for p in root.iterdir())
        core.rt_require(shutil.disk_usage(self.drive).free >= size + 32 * 1024**2,
                        "insufficient persistence free-space estimate")
        stage = destination.with_name(destination.name + ".partial-" + uuid.uuid4().hex)
        stage.mkdir(exist_ok=False)
        core.write_json(stage / "INCOMPLETE.json", {"status": "INCOMPLETE"})
        for filename in ("checkpoint.pt", "manifest.json", "manifest.sha256"):
            copy_verified(root / filename, stage / filename)
        self._publish(stage, destination, ms)
        receipt = {"path": str(destination.resolve()), "manifest_sha256": ms,
                   "payload_sha256": manifest["payload"]["sha256"], "reason": reason,
                   "step": progress["step"], "input_tokens": progress["input_tokens"]}
        for directory in (self.local / "receipts", self.drive / "receipts"):
            directory.mkdir(exist_ok=True)
            core.write_json(directory / f"{name}.json", receipt)
        return receipt

    def restore(self, receipt, model, optimizer, scheduler):
        """Exact immediate restore before the caller can run another update."""
        self.guard()
        manifest = core.verify(receipt["path"], self.identity, receipt["manifest_sha256"])
        payload = torch.load(Path(receipt["path"]) / "checkpoint.pt", map_location="cpu", weights_only=True)
        core.validate_payload(payload, self.identity)
        validate_identity(payload["identity"], self.identity)
        core.rt_require(core.digest_tree(payload["state"]) == manifest["state_digest"], "state digest")
        reference = core.frozen(payload["state"])
        core.no_alias(reference, payload["state"])
        self._check(reference["progress"], reference["sampler"])
        core.exact(reference["parameter_names"], core.names_by_group(model, optimizer), "parameter mapping")
        core.rt_require(all(p.grad is None for p in model.parameters()), "restore requires fresh gradients")
        model.load_state_dict(payload["state"]["model"], strict=True, assign=False)
        buffers = dict(model.named_buffers())
        core.rt_require(buffers.keys() == reference["buffers"].keys(), "buffer keys")
        with torch.no_grad():
            for name, value in reference["buffers"].items():
                buffers[name].copy_(value)
        optimizer.load_state_dict(payload["state"]["optimizer"])
        for group in optimizer.param_groups:
            core.rt_require(not group.get("capturable", False) and not group.get("fused", False),
                            "unsupported AdamW mode")
            for p in group["params"]:
                for key, value in list(optimizer.state.get(p, {}).items()):
                    if isinstance(value, torch.Tensor):
                        optimizer.state[p][key] = value.to("cpu" if key == "step" else p.device)
        scheduler.load_state_dict(payload["state"]["scheduler"])
        if self.scaler is not None:
            self.scaler.load_state_dict(payload["state"]["scaler"])
        modules = dict(model.named_modules())
        core.rt_require(modules.keys() == reference["modes"].keys(), "module modes")
        for name, flag in reference["modes"].items():
            modules[name].training = flag
        progress, sampler = copy.deepcopy(reference["progress"]), copy.deepcopy(reference["sampler"])
        self._tying(model)
        core.rng_set(reference["rng"])
        actual = core.snapshot(model, optimizer, scheduler, progress, sampler, self.scaler)
        core.no_alias(reference, actual)
        self.last_equality = {}
        for key in sorted(core.STATE_FIELDS):
            core.exact(reference[key], actual[key], f"restore.{key}")
            self.last_equality[key] = "PASS"
        core.rt_require(core.digest_tree(actual) == manifest["state_digest"], "restored content digest")
        core.rt_require(scheduler.last_epoch == progress["step"], "scheduler restore step")
        core.exact(optimizer.param_groups[0]["lr"], scheduler.get_last_lr()[0], "restored LR")
        return progress, sampler

    def publish_log(self, path):
        """Immutable log snapshots; resume starts a new local session segment."""
        path = Path(path)
        if self.drive is None:
            return path
        target = self.drive / "logs" / f"{path.stem}-{file_sha256(path)[:16]}{path.suffix}"
        if target.exists():
            core.rt_require(file_sha256(target) == file_sha256(path), "log snapshot collision")
        else:
            copy_verified(path, target)
        return target
