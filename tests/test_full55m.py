import json, tempfile, unittest
from pathlib import Path
import torch
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
