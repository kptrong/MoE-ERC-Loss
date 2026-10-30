"""Full55M FP16 training using the existing pilot loop/atomic checkpoint APIs."""
import copy, json, os, signal, time, traceback, uuid
from pathlib import Path
import torch
from . import pilot_checkpoint_core as core
from .config import ModelConfig, AoEConfig
from .dataset import PackedTokenDataset
from .erc_training import compute_model_erc
from .pilot_config import PilotConfig, erc_config_for
from .pilot_data import load_pilot_artifacts, read_json, require, file_sha256, microbatches
from .pilot_eval import to_device, validate, diagnostics
from .pilot_losses import compute_data_loss
from .pilot_runner import PilotRunner, initialize, make_identity, source_identity
from .pilot_checkpoint import CheckpointManager
from .tokenizer_utils import FrozenTokenizer

class StateBundle:
    def __init__(self, amp):
        self.amp, self.history = amp, []
    def state_dict(self):
        return {"amp": self.amp.state_dict(), "history": copy.deepcopy(self.history)}
    def load_state_dict(self, state):
        self.amp.load_state_dict(state["amp"])
        self.history = copy.deepcopy(state["history"])

class FullRunner(PilotRunner):
    def __init__(self, *args, amp, bundle, tokenizer, output, **kwargs):
        self.amp, self.bundle, self.tokenizer, self.output = amp, bundle, tokenizer, Path(output)
        super().__init__(*args, **kwargs)
        self.old_history = copy.deepcopy(bundle.history)
    def events(self):
        return self.old_history + [json.loads(l) for l in self.log.path.read_text().splitlines()]
    def update(self):
        ids = self.plan.update_indices(self.progress["step"])
        targets = len(ids) * (self.config.sequence_length - 1)
        rng = core.rng_get()
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        for attempt in range(33):
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            sums = {"ce": 0., "lb_raw": 0., "lb_weighted": 0.}
            before = self.amp.get_scale()
            for b in microbatches(self.train, ids, self.config.microbatch_size):
                b = to_device(b, self.device)
                require(torch.equal(b["input_ids"], b["labels"]), "unshifted packed labels required")
                q = int((b["labels"][:, 1:] != -100).sum()) / targets
                result = compute_data_loss(self.model, self.method, b, self.config, dtype=torch.float16)
                self.amp.scale(result["total"] * q).backward()
                for key in sums:
                    sums[key] += float(result[key].detach()) * q
                del result, b
            erc = compute_model_erc(self.model, erc_config_for(self.method))
            if erc.metrics["erc_applied"]:
                self.amp.scale(erc.weighted_loss).backward()
            self.amp.unscale_(self.optimizer)
            finite = bool(torch.stack([torch.isfinite(p.grad).all() for p in self.model.parameters()
                                      if p.grad is not None]).all())
            if finite:
                norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1., error_if_nonfinite=True)
            lr = self.optimizer.param_groups[0]["lr"]
            executed = []
            hook = self.optimizer.register_step_post_hook(lambda *a: executed.append(True))
            try:
                self.amp.step(self.optimizer)
                self.amp.update()
            finally:
                hook.remove()
            if executed:
                require(finite, "nonfinite optimizer update")
                self.scheduler.step()
                break
            require(not finite and self.amp.get_scale() < before, "Unexpected skipped optimizer update")
            self.log.write("overflow", {"step": self.progress["step"], "attempt": attempt,
                                       "scale_before": before, "scale_after": self.amp.get_scale()})
            core.rng_set(rng)
        else:
            raise RuntimeError("FP16 overflow retry limit; counters unchanged")
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        self.progress["step"] += 1
        self.progress["input_tokens"] += len(ids) * self.config.sequence_length
        self.progress["targets"] += targets
        self.progress["train_seconds"] += seconds
        self.sampler["cursor"] += len(ids)
        row = {**sums, **erc.metrics, **self.progress, "lr_used": lr,
               "gradient_norm": float(norm), "tokens_per_second": len(ids)*self.config.sequence_length/seconds,
               "peak_gpu_bytes": torch.cuda.max_memory_allocated(), "scale": self.amp.get_scale(),
               "overflow_retries": attempt, "total_loss": sums["ce"]+sums["lb_weighted"]+erc.metrics["erc_weighted"]}
        self.log.write("update", row)
        if self.progress["step"] % 50 == 0:
            print(json.dumps(row), flush=True)
        return row
    def save(self, reason="recovery"):
        self.bundle.history = self.events()
        receipt = super().save(reason)
        logs = self.output / "logs" / self.method
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "events.json").write_text(json.dumps(self.events()), encoding="utf-8")
        # Current run only: retain two verified recovery heads and final.
        root = self.manager.local / "checkpoints"
        paths = sorted([p for p in root.iterdir() if p.is_dir() and (p / "COMPLETE.json").exists()],
                       key=lambda p: p.name, reverse=True)
        for p in paths[:2]:
            core.verify(p, self.manager.identity, (p / "manifest.sha256").read_text().strip())
        for p in paths[2:]:
            if "tokens055028736" in p.name:
                continue
            require(p.parent == root and not p.is_symlink(), "unsafe retention path")
            m = core.verify(p, self.manager.identity, (p / "manifest.sha256").read_text().strip())
            archive = self.output / "logs" / self.method / "retired" / p.name
            archive.mkdir(parents=True, exist_ok=True)
            for name in ("manifest.json", "manifest.sha256", "COMPLETE.json"):
                (archive / name).write_bytes((p / name).read_bytes())
            for name in ("checkpoint.pt", "manifest.json", "manifest.sha256", "COMPLETE.json"):
                (p / name).unlink()
            p.rmdir()
        return receipt
    def evaluate_due(self):
        due = [m for m in self.plan.milestones if m <= self.progress["input_tokens"]
               and m not in self.progress["evaluated"]]
        if due:
            result = validate(self.model, self.validation, self.config, dtype=torch.float16)
            result.update(diagnostics(self.model, self.method, self.train, self.plan))
            for m in due:
                self.log.write("evaluation", {**result, "milestone": m,
                               "input_tokens": self.progress["input_tokens"]})
                self.progress["evaluated"].append(m)
            self.save("validation")
        done = {e["milestone"] for e in self.events() if e.get("event") == "benchmark_complete"}
        for m in (0, 11000000, 22000000, 33000000, 44000000, 55028736):
            if m > self.progress["input_tokens"] or m in done:
                continue
            self.save("before-benchmark")  # Preserve training even if download/evaluation fails.
            from .full_benchmarks import evaluate
            rows = evaluate(self.model, self.tokenizer, self.output / "benchmarks" / self.method,
                            self.method, self.progress["input_tokens"])
            for row in rows:
                old = [v for e in self.events() if e.get("event") == "benchmark_complete"
                       for v in e["rows"] if v["task"] == row["task"]]
                require(all(v["samples_sha256"] == row["samples_sha256"] for v in old),
                        "benchmark documents/subset changed since checkpoint")
            full = evaluate(self.model, self.tokenizer, self.output / "benchmarks" / self.method,
                            self.method, self.progress["input_tokens"], full=True) if m == 55028736 else []
            self.log.write("benchmark_complete", {"milestone": m, "rows": rows, "full_rows": full})
            self.save("benchmark")

def worker(settings, method, deadline):
    out = Path(settings["output"])
    report = {"method": method, "status": "FAILED"}
    try:
        require(torch.cuda.device_count() == 1, "Worker must see one GPU")
        torch.set_num_threads(2)
        cfg = PilotConfig(bf16=False)
        assets = load_pilot_artifacts(settings["dataset"], settings["tokenizer"], cfg, full_training=True)
        require(assets.plan.total_updates == 3359, "full horizon")
        source = source_identity(settings["source"])
        model, opt, sched, initial = initialize(method, AoEConfig() if method == "aoe" else ModelConfig(),
                                               cfg, assets.plan, "cuda")
        identity = make_identity(method, model, assets.plan, assets.identity, source,
                                 "full55m-fp16-v1-" + method, initial)
        identity["precision"] = "fp16-full55m-v1"
        try:
            amp = torch.amp.GradScaler("cuda")
        except (AttributeError, TypeError):
            amp = torch.cuda.amp.GradScaler()
        bundle = StateBundle(amp)
        def guard():
            core.exact(source_identity(settings["source"]), source)
            assets.verify_unchanged()
        manager = CheckpointManager(out / "checkpoints" / method, None, identity, assets.plan,
                                    guard, scaler=bundle)
        valid, matched = [], 0
        for root in [out / "checkpoints", *map(Path, settings["resume_roots"])]:
            for p in root.rglob("manifest.json") if root.exists() else []:
                if not (p.parent / "checkpoint.pt").exists():
                    continue
                m = read_json(p)
                if m.get("identity", {}).get("run_id") != identity["run_id"]:
                    continue
                core.exact(m["identity"], identity, "resume identity")
                matched += 1
                try:
                    ms = (p.parent / "manifest.sha256").read_text().strip()
                    core.verify(p.parent, identity, ms)
                    valid.append({"path": str(p.parent), "manifest_sha256": ms,
                                  "step": int(p.parent.name[4:10])})
                except Exception as exc:
                    print("Rejected checkpoint:", p, str(exc), flush=True)
        require(not matched or valid, "No valid matching checkpoint; refusing fresh restart")
        receipt = max(valid, key=lambda r:r["step"]) if valid else None
        tokenizer = FrozenTokenizer(settings["tokenizer"])
        r = FullRunner(model,opt,sched,PackedTokenDataset(settings["dataset"],"train",tokenizer=tokenizer),
                       PackedTokenDataset(settings["dataset"],"validation",tokenizer=tokenizer),assets.plan,
                       manager,receipt,amp=amp,bundle=bundle,tokenizer=tokenizer,output=out)
        stopped = []
        signal.signal(signal.SIGTERM, lambda *_: stopped.append(True))
        r.evaluate_due()
        while r.progress["step"] < 3359 and time.time() < deadline and not stopped:
            r.update(); r.evaluate_due()
            if r.progress["step"] % 100 == 0:
                r.save()
        r.save("session-end")
        report.update(status="COMPLETED" if r.progress["step"] == 3359 else "INCOMPLETE",
                      progress=r.progress, initial_hash=initial, receipt=r.last_receipt)
        (out / "logs" / method).mkdir(parents=True,exist_ok=True)
        (out / "logs" / method / "events.json").write_text(json.dumps(r.events()))
    except Exception as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
        print(report["traceback"],flush=True)
    (out / "results").mkdir(parents=True,exist_ok=True)
    (out / "results" / f"{method}.json").write_text(json.dumps(report,indent=2))
    return report

if __name__ == "__main__":
    import sys
    result = worker(read_json(sys.argv[1]),sys.argv[2],float(sys.argv[3]))
    raise SystemExit(result["status"] == "FAILED")
