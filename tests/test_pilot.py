"""Offline pilot fixtures: no Dolma, tokenizer downloads or official training."""

import copy
from dataclasses import replace
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from moe_erc.config import AoEConfig, ModelConfig
from moe_erc import erc_training
from moe_erc.pilot_checkpoint import CheckpointManager, validate_progress
from moe_erc import pilot_checkpoint_core as core
from moe_erc.pilot_config import PilotConfig, PilotPlan, learning_rate, erc_config_for
from moe_erc.pilot_data import file_sha256, inspect_pilot_artifacts, microbatches, safe_file
from moe_erc.pilot_eval import diagnostics, validate
from moe_erc.pilot_losses import compute_data_loss
from moe_erc.pilot_runner import PilotRunner, initialize, make_identity


def fixture(root, method="vanilla_moe", receipt=None):
    config = PilotConfig(sequence_length=8, microbatch_size=2, accumulation_steps=2,
                         validation_microbatch_size=2, warmup_updates=1, bf16=False,
                         evaluation_interval_tokens=32, diagnostic_sequences=3)
    plan = PilotPlan(tuple(range(11)), config)
    settings = dict(vocab_size=29, d_model=8, n_heads=2, n_layers=1, n_experts=3,
                    expert_hidden_dim=4, max_seq_len=8)
    arch = AoEConfig(**settings, aoe_rank=3) if method == "aoe" else ModelConfig(**settings)
    model, opt, scheduler, initial = initialize(method, arch, config, plan, "cpu")
    train = []
    for i in range(11):
        ids = (torch.arange(8) + i) % 29
        train.append({"input_ids": ids, "labels": ids.clone()})
    identity = make_identity(method, model, plan, {"pilot_index_sha256": "fixture-indices"},
                             {"git_commit": "fixture"}, "fixture", initial)
    manager = CheckpointManager(Path(root) / method / "local", Path(root) / method / "drive",
                                identity, plan)
    return PilotRunner(model, opt, scheduler, train, train[:5], plan, manager, receipt)


def fresh_process(root, method, receipt_file, report_file):
    """Executed only by the subprocess regression test in a new interpreter."""
    torch.set_num_threads(1)
    receipt = json.loads(Path(receipt_file).read_text())
    runner = fixture(root, method, receipt)
    before = core.frozen(dict(runner.model.named_parameters()))
    equality = runner.manager.last_equality.copy()
    start = runner.sampler["cursor"]
    runner.run()
    updated = any(not torch.equal(p, before[n]) for n, p in runner.model.named_parameters())
    Path(report_file).write_text(json.dumps({"equality": equality, "start_cursor": start,
           "progress": runner.progress, "sampler": runner.sampler, "updated": updated,
           "log": str(runner.log.path)}))


class PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_config_roundtrip_and_validation(self):
        c = PilotConfig()
        self.assertEqual(c, PilotConfig.from_dict(c.to_dict()))
        for fields in ({"sequence_length": 1}, {"microbatch_size": True}, {"peak_lr": float("nan")},
                       {"beta2": 1}, {"bf16": 1}, {"lb_coefficient": 0.02}, {"seed": -1}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                replace(c, **fields)
        with self.assertRaises(ValueError):
            PilotConfig.from_dict({})
        with self.assertRaises(ValueError):
            erc_config_for("unknown")

    def test_exact_full_budget_and_final_partial(self):
        p = PilotPlan(tuple(range(97657)), PilotConfig())
        self.assertEqual((p.total_updates, p.total_input_tokens, p.total_targets),
                         (3052, 50000384, 49902727))
        self.assertEqual(len(p.update_indices(3051)), 25)
        sizes = [len(p.update_indices(3051)[i:i+2]) for i in range(0, 25, 2)]
        self.assertEqual(sizes, [2] * 12 + [1])
        ids = [i for u in range(p.total_updates) for i in p.update_indices(u)]
        self.assertEqual(ids, list(p.indices))
        self.assertEqual(p.update_indices(3052), ())

    def test_invalid_indices_rejected(self):
        for ids in ((), (1, 1), (-1,), (True,), [1, 2]):
            with self.assertRaises(ValueError):
                PilotPlan(ids, PilotConfig())

    def test_lr_used_schedule_endpoints(self):
        c = PilotConfig()
        self.assertEqual(learning_rate(1, 3052, c), 4e-4 / 64)
        self.assertEqual(learning_rate(64, 3052, c), 4e-4)
        self.assertEqual(learning_rate(3052, 3052, c), 4e-5)
        self.assertGreater(learning_rate(65, 3052, c), learning_rate(100, 3052, c))

    def test_milestones_first_boundary_and_final(self):
        p = PilotPlan(tuple(range(97657)), PilotConfig())
        self.assertEqual(p.milestones, (0, *range(5000000, 50000000, 5000000), 50000384))
        for threshold in p.milestones[1:-1]:
            step = (threshold + 16383) // 16384
            self.assertLess((step - 1) * 16384, threshold)
            self.assertGreaterEqual(step * 16384, threshold)

    def test_identical_vanilla_erc_initialization_and_parameter_count(self):
        a, b = fixture(self.root), fixture(self.root, "vanilla_moe_erc")
        self.assertEqual(a.manager.identity["initial_parameter_sha256"],
                         b.manager.identity["initial_parameter_sha256"])
        self.assertEqual(a.manager.identity["parameters"], b.manager.identity["parameters"])
        for runner in (a, b):
            self.assertIs(runner.model.token_embedding.weight, runner.model.lm_head.weight)

    def test_target_weighted_accumulation_matches_reference(self):
        r = fixture(self.root)
        r.progress.update(step=2, input_tokens=64, targets=56)
        r.sampler["cursor"] = 8
        # Advance only scheduler for this isolated final-update test; no checkpoint is saved.
        r.scheduler.last_epoch = 2
        reference = copy.deepcopy(r.model)
        opt = torch.optim.AdamW(reference.parameters(), lr=r.optimizer.param_groups[0]["lr"],
                                betas=(0.9, 0.95), weight_decay=0.1, foreach=False, fused=False)
        ce = 0.0
        for batch in microbatches(r.train, (8, 9, 10), 2):
            result = compute_data_loss(reference, r.method, batch, r.config)
            q = batch["input_ids"].shape[0] / 3
            ce += float(result["ce"].detach()) * q
            (q * result["total"]).backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0)
        opt.step()
        row = r.update()
        self.assertEqual(row["microbatch_sizes"], [2, 1])
        self.assertAlmostEqual(row["ce"], ce, places=7)
        self.assertEqual((r.progress["input_tokens"], r.progress["targets"]), (88, 77))
        for a, b in zip(reference.parameters(), r.model.parameters()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_boundary_single_calls_and_erc_noise(self):
        for method in ("vanilla_moe", "aoe", "vanilla_moe_erc"):
            with self.subTest(method=method):
                r = fixture(self.root, method)
                with patch.object(r.optimizer, "step", wraps=r.optimizer.step) as opt_step, \
                     patch.object(r.scheduler, "step", wraps=r.scheduler.step) as sched_step, \
                     patch("moe_erc.erc_training.compute_erc_loss",
                           wraps=erc_training.compute_erc_loss) as erc:
                    row = r.update()
                self.assertEqual((opt_step.call_count, sched_step.call_count), (1, 1))
                self.assertEqual(erc.call_count, 1 if method == "vanilla_moe_erc" else 0)
                self.assertEqual(row["erc_applied"], method == "vanilla_moe_erc")
                self.assertEqual(r.scheduler.last_epoch, 1)
                self.assertEqual(row["lr_used"], [r.config.peak_lr])
                self.assertTrue(all(p.grad is None for p in r.model.parameters()))

    def test_disabled_baseline_does_not_draw_erc_noise(self):
        r = fixture(self.root)
        state = core.rng_get()
        with patch("moe_erc.erc_loss.generate_router_proxies", side_effect=AssertionError("noise")):
            r.update()
        core.exact(state, core.rng_get())

    def test_full_validation_target_weighting(self):
        r = fixture(self.root)
        data = copy.deepcopy(r.validation)
        data[-1]["labels"][2:] = -100
        expected, q_total = 0.0, 0
        with torch.no_grad():
            for item in data:
                q = int((item["labels"][1:] != -100).sum())
                output = r.model(**{k: v.unsqueeze(0) for k, v in item.items()})
                expected += float(output.loss) * q
                q_total += q
        result = validate(r.model, data, r.config)
        self.assertEqual(result["validation_sequences"], 5)
        self.assertEqual(result["validation_targets"], q_total)
        self.assertAlmostEqual(result["validation_ce"], expected / q_total, places=6)

    def test_eval_diagnostics_isolate_rng_modes_cursor_and_optimizer(self):
        for method in ("vanilla_moe", "aoe", "vanilla_moe_erc"):
            r = fixture(self.root, method)
            r.model.layers[0].eval()  # Mixed module flags must also be restored.
            before = core.snapshot(r.model, r.optimizer, r.scheduler, r.progress, r.sampler)
            validate(r.model, r.validation, r.config)
            diag = diagnostics(r.model, method, r.train, r.plan)
            after = core.snapshot(r.model, r.optimizer, r.scheduler, r.progress, r.sampler)
            core.exact(before, after)
            self.assertEqual(diag["probe_indices"], [0, 1, 2])
            self.assertAlmostEqual(sum(diag["routing"][0]["assignment_frequencies"]), 1, places=6)
            self.assertEqual(len(diag["erc_diagnostics"]), 0 if method == "aoe" else 1)

    def test_save_restore_exact_all_states_and_no_alias(self):
        for method in ("vanilla_moe", "aoe", "vanilla_moe_erc"):
            with self.subTest(method=method):
                r = fixture(self.root, method)
                receipt = r.run(max_updates=1)
                restored = fixture(self.root, method, receipt)
                self.assertEqual(set(restored.manager.last_equality), core.STATE_FIELDS)
                self.assertEqual(set(restored.manager.last_equality.values()), {"PASS"})
                a = core.snapshot(r.model, r.optimizer, r.scheduler, r.progress, r.sampler)
                b = core.snapshot(restored.model, restored.optimizer, restored.scheduler,
                                  restored.progress, restored.sampler)
                core.exact(a, b)
                core.no_alias(a, b)
                self.assertIs(restored.model.token_embedding.weight, restored.model.lm_head.weight)

    def test_fresh_process_resume_all_methods(self):
        for method in ("vanilla_moe", "aoe", "vanilla_moe_erc"):
            with self.subTest(method=method):
                r = fixture(self.root, method)
                receipt = r.run(max_updates=1)
                receipt_file, report_file = self.root / "receipt.json", self.root / "report.json"
                receipt_file.write_text(json.dumps(receipt))
                program = ("from tests.test_pilot import fresh_process; import sys; "
                           "fresh_process(*sys.argv[1:])")
                result = subprocess.run([sys.executable, "-c", program, str(self.root), method,
                                         str(receipt_file), str(report_file)],
                                        capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                report = json.loads(report_file.read_text())
                self.assertEqual(set(report["equality"].values()), {"PASS"})
                self.assertEqual(report["start_cursor"], 4)
                self.assertEqual(report["progress"]["step"], 3)
                self.assertEqual(report["progress"]["input_tokens"], 88)
                self.assertEqual(report["progress"]["targets"], 77)
                self.assertEqual(report["progress"]["evaluated"], [0, 32, 64, 88])
                self.assertTrue(report["updated"])
                rows = [json.loads(line) for line in Path(report["log"]).read_text().splitlines()]
                trace = [i for row in rows if row["event"] == "update" for i in row["sequence_indices"]]
                self.assertEqual(trace, list(range(4, 11)))
                self.assertEqual([row["milestone"] for row in rows if row["event"] == "evaluation"], [64, 88])

    def test_checkpoint_corruption_and_identity_rejected(self):
        r = fixture(self.root)
        receipt = r.run(max_updates=1)
        altered = copy.deepcopy(r.manager.identity)
        altered["training"]["seed"] = 7
        with self.assertRaises(RuntimeError):
            core.verify(receipt["path"], altered, receipt["manifest_sha256"])
        copy_path = self.root / "corrupt"
        shutil.copytree(receipt["path"], copy_path)
        with (copy_path / "checkpoint.pt").open("r+b") as f:
            f.write(b"BROKEN")
        with self.assertRaisesRegex(RuntimeError, "CHECKSUM_MISMATCH"):
            core.verify(copy_path, r.manager.identity, receipt["manifest_sha256"])

    def test_full_identity_fields_rejected_before_loading(self):
        r = fixture(self.root)
        receipt = r.run(max_updates=1)
        for key, value in (("method", "aoe"), ("source", {"git_commit": "wrong"}),
                           ("pilot_index_sha256", "wrong"), ("initial_parameter_sha256", "wrong"),
                           ("erc_config", erc_config_for("vanilla_moe_erc").to_dict()),
                           ("architecture", {})):
            identity = copy.deepcopy(r.manager.identity)
            identity[key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                core.verify(receipt["path"], identity, receipt["manifest_sha256"])

    def test_cpu_bf16_shared_loop_all_methods(self):
        # A CPU arithmetic check, not evidence about Colab CUDA kernels/VRAM.
        for method in ("vanilla_moe", "aoe", "vanilla_moe_erc"):
            r = fixture(self.root, method)
            r.config = replace(r.config, bf16=True)
            row = r.update()
            self.assertTrue(np.isfinite(row["ce"]))
            self.assertTrue(np.isfinite(row["gradient_norm_before_clip"]))
            self.assertTrue(all(p.dtype == torch.float32 for p in r.model.parameters()))

    def test_namespace_owner_cannot_be_overwritten(self):
        r = fixture(self.root)
        old = (r.manager.local / "run-identity.json").read_bytes()
        identity = copy.deepcopy(r.manager.identity)
        identity["method"] = "aoe"
        with self.assertRaises(RuntimeError):
            CheckpointManager(r.manager.local, r.manager.drive, identity, r.plan)
        self.assertEqual((r.manager.local / "run-identity.json").read_bytes(), old)

    def test_checkpoint_paths_unique_immutable(self):
        r = fixture(self.root)
        first = r.run(max_updates=1)
        digest = file_sha256(Path(first["path"]) / "checkpoint.pt")
        second = r.save()
        self.assertNotEqual(first["path"], second["path"])
        self.assertEqual(file_sha256(Path(first["path"]) / "checkpoint.pt"), digest)

    def test_invalid_cursor_and_milestone_rejected(self):
        r = fixture(self.root)
        r.progress["evaluated"] = [0]
        r.sampler["cursor"] = 1
        with self.assertRaises(RuntimeError):
            validate_progress(r.progress, r.sampler, r.plan, "fixture-indices")
        r.sampler["cursor"] = 0
        r.progress["evaluated"] = [0, 32]
        with self.assertRaises(RuntimeError):
            validate_progress(r.progress, r.sampler, r.plan, "fixture-indices")

    def test_artifacts_missing_unknown_and_path_rejections(self):
        self.assertEqual(inspect_pilot_artifacts(self.root / "missing", self.root / "tokenizer.json")["status"],
                         "UNKNOWN")
        for name in ("../x", "/x", "C:/x", "a\\x", "a//x"):
            with self.assertRaises(ValueError):
                safe_file(self.root, name)

    def test_logging_cannot_hold_graph(self):
        r = fixture(self.root)
        row = r.update()
        json.dumps(row, allow_nan=False)
        with self.assertRaises(TypeError):
            r.log.write("invalid", {"loss": torch.tensor(1.0, requires_grad=True)})


if __name__ == "__main__":
    unittest.main()
