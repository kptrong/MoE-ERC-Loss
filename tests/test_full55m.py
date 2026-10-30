import json, tempfile, unittest
from pathlib import Path
import torch
from unittest.mock import patch
from moe_erc.pilot_config import PilotConfig, PilotPlan, learning_rate
from moe_erc.full_benchmarks import sample_ids, score_tokens
from moe_erc.full_plots import render
from tests.test_erc_training import small_model

class FullTests(unittest.TestCase):
    def test_full_budget_order_and_partial(self):
        import numpy as np
        ids=tuple(map(int,np.random.Generator(np.random.PCG64(42)).permutation(107478)))
        plan=PilotPlan(ids,PilotConfig(bf16=False))
        self.assertEqual((plan.total_updates,plan.total_input_tokens,plan.total_targets),(3359,55028736,54921258))
        self.assertEqual(len(plan.update_indices(3358)),22)
        self.assertEqual([i for s in range(3359) for i in plan.update_indices(s)],list(ids))
        self.assertEqual(learning_rate(3359,3359,plan.config),4e-5)
    def test_fixed_samples(self):
        self.assertEqual(sample_ids(1000),sample_ids(1000))
        self.assertEqual(len(set(sample_ids(1000))),200)
        self.assertEqual(sample_ids(3),[0,1,2])
    def test_likelihood_independent_reference(self):
        model=small_model().eval()
        context,cont=[1,2],[3,4]
        actual,_=score_tokens(model,context,cont,2)
        with torch.no_grad():
            logits=model(torch.tensor([[1,2,3]]))[0].float().log_softmax(-1)
        self.assertAlmostEqual(actual,float(logits[1,3]+logits[2,4]),places=5)
    def test_plot_csv_fixture_only(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"logs"/"fixture"; p.mkdir(parents=True)
            (p/"events.json").write_text(json.dumps([{"event":"update","input_tokens":32,
                "ce":3.,"lb_raw":1.,"erc_raw":0.,"lr_used":.001,"tokens_per_second":10.}]))
            figures=render(d)
            self.assertEqual(len(figures),8)
            self.assertTrue((Path(d)/"benchmark_final_full.csv").is_file())
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA FP16 unavailable")
    def test_fp16_update_and_scaler_checkpoint_restore(self):
        from tests.test_pilot import fixture
        from moe_erc.full55m import FullRunner, StateBundle
        from moe_erc.pilot_checkpoint import CheckpointManager
        from moe_erc.pilot_eval import validate
        with tempfile.TemporaryDirectory() as d:
            b=fixture(Path(d)/"base", "vanilla_moe_erc")
            b.model.cuda()
            identity=dict(b.manager.identity,precision="fp16-full55m-v1")
            amp=torch.amp.GradScaler("cuda",init_scale=8.)
            bundle=StateBundle(amp)
            manager=CheckpointManager(Path(d)/"fp16",None,identity,b.plan,scaler=bundle)
            r=FullRunner(b.model,b.optimizer,b.scheduler,b.train,b.validation,b.plan,manager,
                         amp=amp,bundle=bundle,tokenizer=None,output=d)
            r.update()
            validate(r.model,r.validation,r.config,dtype=torch.float16)
            r.progress["evaluated"]=[0,32]
            receipt=r.save()
            manager.restore(receipt,r.model,r.optimizer,r.scheduler)
            self.assertEqual(manager.last_equality["scaler"],"PASS")
            self.assertEqual(r.progress["input_tokens"],32)


class RetentionTests(unittest.TestCase):
    """Checkpoint-only fixtures: no training or real dataset access."""
    def setUp(self):
        from tests.test_pilot import fixture
        from moe_erc.full55m import FullRunner, StateBundle
        from moe_erc.pilot_checkpoint import CheckpointManager
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        b = fixture(self.root / "fixture")
        manager = CheckpointManager(self.root / "owned", None, b.manager.identity, b.plan)
        self.r = FullRunner(b.model,b.optimizer,b.scheduler,b.train,b.validation,b.plan,manager,
                            amp=None,bundle=StateBundle(None),tokenizer=None,output=self.root)
        self.r.progress["evaluated"] = [0]

    def paths(self):
        return set((self.r.manager.local / "checkpoints").glob("step*"))

    def test_keep_sole_and_only_two_verified_heads_and_preserve_logs(self):
        from moe_erc import pilot_checkpoint_core as core
        self.r.log.write("benchmark_complete", {"rows": [{"acc": .5}]})
        benchmarks = self.root / "benchmarks" / "result.json"
        benchmarks.parent.mkdir(); benchmarks.write_text('{"acc":0.5}')
        first = Path(self.r.save()["path"])
        self.assertEqual(self.paths(), {first})
        second = Path(self.r.save()["path"])
        third = Path(self.r.save("session-end")["path"])
        self.assertEqual(self.paths(), {second, third})
        self.assertFalse(first.exists())
        for p in self.paths():
            core.verify(p,self.r.manager.identity,(p / "manifest.sha256").read_text().strip())
        archive = self.root / "logs" / self.r.method / "retired" / first.name
        self.assertEqual({p.name for p in archive.iterdir()},
                         {"manifest.json","manifest.sha256","COMPLETE.json"})
        self.assertTrue(self.r.log.path.exists())
        self.assertEqual(benchmarks.read_text(), '{"acc":0.5}')
        events = json.loads((self.root / "logs" / self.r.method / "events.json").read_text())
        self.assertTrue(any(e["event"] == "benchmark_complete" for e in events))

    def test_failed_new_save_never_deletes_resume_points(self):
        self.r.save(); self.r.save()
        before = {p:(p/"checkpoint.pt").read_bytes() for p in self.paths()}
        with patch.object(self.r.manager,"save",side_effect=RuntimeError("disk failure")):
            with self.assertRaisesRegex(RuntimeError,"disk failure"):
                self.r.save()
        self.assertEqual(self.paths(),set(before))
        for p, data in before.items():
            self.assertEqual((p/"checkpoint.pt").read_bytes(),data)

    def test_corrupt_replacement_cannot_trigger_retention(self):
        from moe_erc.full_storage import retain_two
        first = self.r.save(); second = self.r.save()
        p = Path(second["path"])
        with (p/"checkpoint.pt").open("ab") as f:
            f.write(b"corrupt")
        with self.assertRaises(RuntimeError):
            retain_two(self.r.manager,second,self.root/"archive")
        self.assertTrue(Path(first["path"]).exists())
        self.assertEqual(len(self.paths()),2)

    def test_corrupt_or_partial_old_directories_preserved_not_counted_as_valid(self):
        first = Path(self.r.save()["path"])
        with (first/"checkpoint.pt").open("ab") as f:
            f.write(b"corrupt")
        partial = first.parent / "unfinished.partial"
        partial.mkdir(); (partial/"INCOMPLETE.json").write_text("{}")
        second = Path(self.r.save()["path"])
        third = Path(self.r.save()["path"])
        self.assertTrue(first.exists()); self.assertTrue(partial.exists())
        self.assertEqual(self.paths(),{first,second,third})

    def test_budget_accounts_other_working_files_and_blocks_before_save(self):
        from moe_erc.full_storage import check_budget, tree_bytes
        self.r.save()
        unrelated = self.root/"unrelated.bin"; unrelated.write_bytes(b"x"*128)
        used = tree_bytes(self.root)
        self.assertEqual(check_budget(self.root,limit=used+100,reserve=100)["used_bytes"],used)
        with self.assertRaisesRegex(RuntimeError,"Output budget"):
            check_budget(self.root,limit=used+99,reserve=100)
        self.r.working_root = self.root
        self.r.output_limit_bytes = used+100
        before = self.paths()
        with patch.object(self.r.manager,"save") as save:
            with self.assertRaisesRegex(RuntimeError,"Output budget"):
                self.r.save()
            save.assert_not_called()
        self.assertEqual(self.paths(),before)

    def test_shared_lock_serializes_concurrent_writers(self):
        import threading
        from moe_erc.full_storage import checkpoint_lock
        acquired, started = threading.Event(), threading.Event()
        def other_writer():
            started.set()
            with checkpoint_lock(self.root):
                acquired.set()
        with checkpoint_lock(self.root):
            t = threading.Thread(target=other_writer)
            t.start()
            self.assertTrue(started.wait(5))
            self.assertFalse(acquired.wait(.1))
        t.join(5)
        self.assertFalse(t.is_alive())
        self.assertTrue(acquired.is_set())
