"""One pilot loop for Vanilla, AoE and ERC; CLI defaults to read-only preflight.

Fresh-process resume is the same CLI invocation with --resume RECEIPT.json.
No notebook objects, downloads, service mounts or dataset creation are needed.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import random
import re
import shutil
import subprocess
import time
import uuid

import numpy as np
import torch

from .config import AoEConfig, ModelConfig
from .dataset import PackedTokenDataset
from .erc_training import finish_optimizer_step
from .model_factory import create_model
from .pilot_checkpoint import CheckpointManager, new_progress, validate_progress
from .pilot_checkpoint_core import digest_tree, write_json
from .pilot_config import METHODS, PilotConfig, erc_config_for, learning_rate
from .pilot_data import (file_sha256, inspect_pilot_artifacts, load_pilot_artifacts,
                         microbatches, read_json, require)
from .pilot_eval import diagnostics, to_device, validate
from .pilot_losses import compute_data_loss
from .pilot_metrics import MetricLog
from .tokenizer_utils import FrozenTokenizer


def initialize(method, architecture, config, plan, device):
    erc_config_for(method)  # Validate before consuming RNG.
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    model = create_model("aoe" if method == "aoe" else "vanilla_moe", architecture)
    initial_hash = digest_tree(dict(model.named_parameters()))
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.peak_lr,
                                 betas=(config.beta1, config.beta2), weight_decay=config.weight_decay,
                                 foreach=False, fused=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda index: learning_rate(min(index + 1, plan.total_updates),
                                               plan.total_updates, config) / config.peak_lr)
    return model, optimizer, scheduler, initial_hash


def make_identity(method, model, plan, artifacts, source, run_id, initial_hash):
    require(bool(re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id)), "unsafe run ID")
    require(model.config.max_seq_len == plan.config.sequence_length, "architecture/context mismatch")
    return {"pilot_protocol": "pilot-v1", "method": method, "run_id": run_id,
            "training": plan.config.to_dict(), "architecture": asdict(model.config),
            "parameters": sum(p.numel() for p in model.parameters()),
            "erc_config": erc_config_for(method).to_dict(), "total_sequences": len(plan.indices),
            "initial_parameter_sha256": initial_hash, "source": source,
            "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "cuda_build": torch.version.cuda, "cuda_devices": torch.cuda.device_count()},
            **artifacts}


def source_identity(root, strict=True):
    root = Path(root)
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args]).decode().strip()
    commit = git("rev-parse", "HEAD")
    if strict:
        require(not git("status", "--porcelain", "--untracked-files=no"), "tracked source is dirty")
        require(not git("ls-files", "--others", "--exclude-standard", "moe_erc"),
                "untracked package source; commit pilot modules before launch")
    names = git("ls-files", "moe_erc", "pyproject.toml").splitlines()
    require("moe_erc/pilot_runner.py" in names, "commit Step 8.2 sources before official launch")
    hashes = {name: file_sha256(root / name) for name in names if (root / name).is_file()}
    return {"git_commit": commit, "files_sha256": hashes}


class PilotRunner:
    """Explicit datasets/plan permit tiny synthetic tests without production artifacts."""

    def __init__(self, model, optimizer, scheduler, train, validation, plan, manager, receipt=None):
        self.model, self.optimizer, self.scheduler = model, optimizer, scheduler
        self.train, self.validation, self.plan, self.manager = train, validation, plan, manager
        self.config, self.method = plan.config, manager.identity["method"]
        self.device = model.token_embedding.weight.device
        require(max(plan.indices) < len(train), "pilot index outside training split")
        self.progress, self.sampler = new_progress(manager.identity["pilot_index_sha256"])
        if receipt is not None:
            self.progress, self.sampler = manager.restore(receipt, model, optimizer, scheduler)
        self.log = MetricLog(manager.local, manager.identity, receipt)
        self.last_receipt = receipt
        if receipt is not None:
            self.log.write("restore", {"immediate_state_equality": manager.last_equality,
                                        "step": self.progress["step"],
                                        "sampler_cursor": self.sampler["cursor"],
                                        "exact_deterministic_replay": "NOT VERIFIED"})

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def update(self):
        """All CE/LB backwards, then one existing ERC/clipping/optimizer boundary."""
        config, model = self.config, self.model
        ids = self.plan.update_indices(self.progress["step"])
        require(bool(ids), "pilot already complete")
        expected_targets = len(ids) * (config.sequence_length - 1)
        sums = {"ce": 0.0, "lb_raw": 0.0, "lb_weighted": 0.0}
        actual_targets, actual_inputs, sizes = 0, 0, []
        model.train()
        self.optimizer.zero_grad(set_to_none=True)
        self._sync()
        started = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        for batch in microbatches(self.train, ids, config.microbatch_size):
            batch = to_device(batch, self.device)
            require(batch["input_ids"].shape[1] == config.sequence_length
                    and torch.equal(batch["input_ids"], batch["labels"]),
                    "pilot requires unshifted, unmasked packed labels")
            q = int((batch["labels"][:, 1:] != -100).sum())
            result = compute_data_loss(model, self.method, batch, config)
            weight = q / expected_targets
            (result["total"] * weight).backward()
            for key in sums:
                sums[key] += float(result[key].detach()) * weight
            actual_targets += q
            actual_inputs += batch["input_ids"].numel()
            sizes.append(batch["input_ids"].shape[0])
            del result, batch
        require(actual_targets == expected_targets, "effective target accounting")
        metrics = finish_optimizer_step(
            model, erc_config_for(self.method), self.optimizer, self.scheduler,
            **sums, global_step=self.progress["step"], effective_targets=actual_targets,
            clip_norm=config.clip_norm)
        self.optimizer.zero_grad(set_to_none=True)
        self._sync()
        seconds = time.perf_counter() - started
        self.progress["step"] += 1
        self.progress["input_tokens"] += actual_inputs
        self.progress["targets"] += actual_targets
        self.progress["train_seconds"] += seconds
        self.sampler["cursor"] += len(ids)
        validate_progress(self.progress, self.sampler, self.plan,
                          self.manager.identity["pilot_index_sha256"], require_evaluated=False)
        row = {**metrics, **self.progress, "sampler_cursor": self.sampler["cursor"],
               "sequence_indices": list(ids), "microbatch_sizes": sizes,
               "update_seconds": seconds, "input_tokens_per_second": actual_inputs / seconds,
               "cuda_peak_bytes": torch.cuda.max_memory_allocated(self.device)
                   if self.device.type == "cuda" else None}
        self.log.write("update", row)
        if self.progress["step"] % config.console_interval_updates == 0:
            print(json.dumps({k: row[k] for k in ("global_optimizer_step", "ce", "input_tokens",
                                                "input_tokens_per_second")}))
        return row

    def evaluate_due(self):
        due = [m for m in self.plan.milestones if m <= self.progress["input_tokens"]
               and m not in self.progress["evaluated"]]
        if not due:
            return
        result = validate(self.model, self.validation, self.config)
        result.update(diagnostics(self.model, self.method, self.train, self.plan))
        for milestone in due:
            self.log.write("evaluation", {**result, "milestone": milestone,
                                           "step": self.progress["step"],
                                           "actual_input_tokens": self.progress["input_tokens"]})
            self.progress["evaluated"].append(milestone)
        print(json.dumps({"milestones": due, "step": self.progress["step"],
                          "validation_ce": result["validation_ce"]}))
        self.save("evaluation")

    def save(self, reason="recovery"):
        started = time.perf_counter()
        self.last_receipt = self.manager.save(self.model, self.optimizer, self.scheduler,
                                             self.progress, self.sampler, reason)
        self.log.write("checkpoint", {**self.last_receipt,
                                       "checkpoint_seconds": time.perf_counter() - started})
        self.manager.publish_log(self.log.path)
        return self.last_receipt

    def run(self, max_updates=None, recovery_interval=None):
        """None means full pilot. A bounded launch is available for Step 8.3 smoke."""
        for name, value in (("max_updates", max_updates), ("recovery_interval", recovery_interval)):
            require(value is None or type(value) is int and value > 0, f"invalid {name}")
        self.evaluate_due()
        stop = self.plan.total_updates if max_updates is None else min(
            self.plan.total_updates, self.progress["step"] + max_updates)
        while self.progress["step"] < stop:
            self.update()
            self.evaluate_due()
            if recovery_interval and self.progress["step"] % recovery_interval == 0:
                if self.last_receipt["step"] != self.progress["step"]:
                    self.save()
        if self.last_receipt is None or self.last_receipt["step"] != self.progress["step"]:
            self.save()
        self.manager.guard()
        self.log.write("stop", {"progress": self.progress, "complete": stop == self.plan.total_updates,
                                "exact_deterministic_replay": "NOT VERIFIED"})
        self.manager.publish_log(self.log.path)
        return self.last_receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default="vanilla_moe")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--drive-root", type=Path)
    parser.add_argument("--resume", type=Path, help="Verified Drive receipt JSON")
    parser.add_argument("--run", action="store_true", help="Explicitly allow optimizer updates")
    parser.add_argument("--max-updates", type=int, help="Bound this launch for operational smoke")
    parser.add_argument("--recovery-interval", type=int, help="Optional extra boundary checkpoints")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    print(json.dumps({"artifacts": inspect_pilot_artifacts(args.dataset, args.tokenizer),
                      "cuda": torch.cuda.is_available(), "config": PilotConfig().to_dict(),
                      "disk_free_bytes": shutil.disk_usage(root).free,
                      "storage_policy": "Keep every milestone/recovery; no automatic deletion",
                      "checkpoint_estimate_bytes_each": 52_288_000 * 4 * 3 + 16000 * 512 * 4,
                      "storage_note": "Budget ~7.6 GiB/method for 11 snapshots; extra recoveries add copies. "
                                      "Mounted Drive free space is not an authoritative account quota."}, indent=2))
    if not args.run:
        return
    require(args.output_root is not None and args.drive_root is not None, "specify both persistence roots")
    require(args.drive_root.is_dir(), "mount/prepare persistence root before launch")
    require(torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            and "A100" in torch.cuda.get_device_name(0), "official pilot requires A100 with BF16")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    source = source_identity(root)
    config = PilotConfig()
    assets = load_pilot_artifacts(args.dataset, args.tokenizer, config)
    architecture = AoEConfig() if args.method == "aoe" else ModelConfig()
    model, optimizer, scheduler, initial = initialize(args.method, architecture, config, assets.plan, "cuda")
    receipt = read_json(args.resume) if args.resume else None
    if receipt:
        manifest_path = Path(receipt["path"]) / "manifest.json"
        require(file_sha256(manifest_path) == receipt["manifest_sha256"], "resume manifest checksum")
        run_id = read_json(manifest_path)["identity"]["run_id"]
    else:
        run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:12]
    identity = make_identity(args.method, model, assets.plan, assets.identity, source, run_id, initial)
    require(identity["parameters"] == (52_263_424 if args.method == "aoe" else 52_288_000),
            "production parameter count")
    relative = Path("pilot") / source["git_commit"][:12] / args.method / f"seed-{config.seed}" / run_id
    local, drive = args.output_root / relative, args.drive_root / relative
    if not receipt:
        require(not local.exists() and not drive.exists(), "run namespace exists; resume explicitly")
    def guard():
        require(source_identity(root) == source, "source changed since launch")
        assets.verify_unchanged()
    manager = CheckpointManager(local, drive, identity, assets.plan, guard)
    tokenizer = FrozenTokenizer(args.tokenizer)
    runner = PilotRunner(model, optimizer, scheduler,
                         PackedTokenDataset(args.dataset, "train", tokenizer=tokenizer),
                         PackedTokenDataset(args.dataset, "validation", tokenizer=tokenizer),
                         assets.plan, manager, receipt)
    write_json(local / f"launch-{uuid.uuid4().hex}.json", {"identity": identity,
               "resume": receipt, "checkpoint_provenance": "Step 6.5C cell 89 / Step 7.5C cell 96",
               "exact_deterministic_replay": "NOT VERIFIED"})
    runner.run(args.max_updates, args.recovery_interval)


if __name__ == "__main__":
    main()
