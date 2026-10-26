"""Update-boundary integration with synthetic data; no external training assets."""

import copy
from dataclasses import FrozenInstanceError, replace
from io import BytesIO
import json
import unittest
from unittest.mock import patch

import torch

from moe_erc import AoEConfig, ModelConfig
from moe_erc.config import ERCConfig
from moe_erc.erc_loss import compute_erc_loss, mean_erc_loss
from moe_erc.erc_training import (
    compute_model_erc, finish_optimizer_step, validate_erc_checkpoint_config,
)
from moe_erc.model_factory import create_model


def small_model(kind="vanilla_moe"):
    settings = dict(vocab_size=29, d_model=8, n_heads=2, n_layers=2,
                    n_experts=3, expert_hidden_dim=4)
    config = AoEConfig(**settings, aoe_rank=3) if kind == "aoe" else ModelConfig(**settings)
    return create_model(kind, config)


def batch():
    ids = torch.tensor([[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]])
    return {"input_ids": ids, "labels": ids.clone()}


def data_losses(model, item):
    """Test-only Step 6 semantics: attached router probabilities, detached counts.

    Not a second production LB implementation. The notebook remains the source
    of the actual CE/LB loop; this reference exercises the adapter's boundary.
    """
    probabilities = {}
    handles = []
    for index, layer in enumerate(model.layers):
        def record(module, args, output, index=index):
            probabilities[index] = output.probabilities
        handles.append(layer.moe.router.register_forward_hook(record))
    try:
        output = model(**item, return_routing_info=True)
    finally:
        for handle in handles:
            handle.remove()
    terms = []
    for index, info in enumerate(output.routing_info):
        fractions = info.diagnostics.assignment_fractions
        mean_probability = probabilities[index].reshape(-1, model.config.n_experts).mean(0)
        terms.append(model.config.n_experts * (fractions * mean_probability).sum())
    return output.loss, torch.stack(terms).mean()


def backward_data(model, batches, scaler=None):
    counts = [int((item["labels"][:, 1:] != -100).sum()) for item in batches]
    totals = {"ce": 0., "lb_raw": 0., "lb_weighted": 0.}
    for item, count in zip(batches, counts):
        ce, lb = data_losses(model, item)
        q = count / sum(counts)
        loss = q * (ce + 0.01 * lb)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
        totals["ce"] += q * float(ce.detach())
        totals["lb_raw"] += q * float(lb.detach())
        totals["lb_weighted"] += q * float((0.01 * lb).detach())
    return totals, sum(counts)


def optimizer_pair(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.1,
                                  foreach=False, fused=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1))
    return optimizer, scheduler


class ERCConfigTests(unittest.TestCase):
    def test_defaults_and_separation_from_architecture(self):
        config = ERCConfig()
        self.assertFalse(config.enabled)
        self.assertEqual(config.alpha, 1.)
        self.assertEqual(config.coefficient, 1.)
        self.assertEqual(config.layer_reduction, "mean")
        self.assertEqual(config.noise_timing, "once_per_optimizer_step")
        self.assertNotIn("enabled", ModelConfig().to_dict())
        with self.assertRaises(FrozenInstanceError):
            config.enabled = True

    def test_validation_and_strict_json_identity(self):
        config = ERCConfig(enabled=True, alpha=0.4, coefficient=0.7)
        stored = json.loads(json.dumps(config.to_dict()))
        self.assertEqual(ERCConfig.from_dict(stored), config)
        validate_erc_checkpoint_config(stored, config)
        for values in (
            {"enabled": 1}, {"alpha": -0.1}, {"alpha": 1.1}, {"alpha": True},
            {"alpha": float("nan")}, {"coefficient": -1}, {"coefficient": float("inf")},
            {"layer_reduction": "sum"}, {"noise_timing": "microbatch"},
            {"algorithm_version": "unknown"},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ERCConfig(**values)
        for key in stored:
            missing = dict(stored)
            missing.pop(key)
            with self.assertRaises(ValueError):
                validate_erc_checkpoint_config(missing, config)
        for wrong in (None, {**stored, "extra": 1}, {**stored, "algorithm_version": "old"}):
            with self.assertRaises(ValueError):
                validate_erc_checkpoint_config(wrong, config)
        for different in (replace(config, enabled=False), replace(config, alpha=0.5),
                          replace(config, coefficient=0.8)):
            with self.assertRaisesRegex(ValueError, "mismatch"):
                validate_erc_checkpoint_config(stored, different)


class ModelERCIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_both_factories_budgets_and_no_added_parameters(self):
        for kind, expected in (("vanilla_moe", 52_288_000), ("aoe", 52_263_424)):
            model = create_model(kind)
            before = {name: id(p) for name, p in model.named_parameters()}
            config = ERCConfig(enabled=(kind == "vanilla_moe"), coefficient=0.)
            compute_model_erc(model, config)
            self.assertEqual(sum(p.numel() for p in model.parameters()), expected)
            self.assertEqual(before, {name: id(p) for name, p in model.named_parameters()})
            self.assertIs(model.token_embedding.weight, model.lm_head.weight)
            self.assertLess(abs(float(model.token_embedding.weight.detach().std()) - 0.02), 1e-4)
            del model

    def test_disabled_and_zero_coefficient_preserve_logits_ce_lb_rng(self):
        model = small_model()
        item = batch()
        baseline_logits = model(**item).logits.detach().clone()
        ce, lb = data_losses(model, item)
        for config in (ERCConfig(), ERCConfig(enabled=True, coefficient=0.)):
            rng = torch.get_rng_state().clone()
            with patch("moe_erc.erc_training.compute_erc_loss") as core:
                result = compute_model_erc(model, config)
                core.assert_not_called()
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertFalse(result.metrics["erc_applied"])
            self.assertFalse(result.weighted_loss.requires_grad)
            self.assertEqual(result.metrics["erc_weighted"], 0.)
            torch.testing.assert_close(model(**item).logits, baseline_logits, rtol=0, atol=0)
            after_ce, after_lb = data_losses(model, item)
            torch.testing.assert_close(after_ce, ce, rtol=0, atol=0)
            torch.testing.assert_close(after_lb, lb, rtol=0, atol=0)

    def test_eval_skips_noise_and_regular_forward_never_calls_erc(self):
        model = small_model().eval()
        rng = torch.get_rng_state().clone()
        with patch("moe_erc.erc_training.compute_erc_loss") as core, torch.no_grad():
            model(**batch())
            result = compute_model_erc(model, ERCConfig(enabled=True))
            core.assert_not_called()
        self.assertFalse(result.metrics["erc_applied"])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        model.train()
        with torch.no_grad(), self.assertRaisesRegex(RuntimeError, "autograd"):
            compute_model_erc(model, ERCConfig(enabled=True))

    def test_layer_mean_coefficient_and_logging_have_no_graph(self):
        model = small_model()
        layers = [torch.tensor(2., requires_grad=True), torch.tensor(4., requires_grad=True)]
        with patch("moe_erc.erc_training.compute_erc_loss", side_effect=layers) as core:
            result = compute_model_erc(model, ERCConfig(enabled=True, coefficient=2.))
        self.assertEqual(core.call_count, 2)
        self.assertEqual(result.raw_loss.item(), 3.)
        self.assertEqual(result.weighted_loss.item(), 6.)
        self.assertEqual(result.metrics["erc_per_layer"], [2., 4.])
        json.dumps(result.metrics, allow_nan=False)
        result.weighted_loss.backward()
        for layer in layers:
            self.assertEqual(layer.grad.item(), 1.)

    def test_real_erc_gradients_are_gate_only_and_parameters_unchanged(self):
        model = small_model()
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        result = compute_model_erc(model, ERCConfig(enabled=True, alpha=0.))
        self.assertTrue(torch.isfinite(result.raw_loss))
        self.assertGreater(result.raw_loss.item(), 0.)
        result.weighted_loss.backward()
        for layer in model.layers:
            self.assertIsNone(layer.moe.router.weight.grad)
            for expert in layer.moe.experts:
                self.assertIsNotNone(expert.gate_proj.weight.grad)
                self.assertTrue(torch.isfinite(expert.gate_proj.weight.grad).all())
                self.assertGreater(expert.gate_proj.weight.grad.abs().sum().item(), 0.)
                self.assertIsNone(expert.up_proj.weight.grad)
                self.assertIsNone(expert.down_proj.weight.grad)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)

    def test_ce_and_lb_keep_their_own_router_gradients(self):
        model = small_model()
        ce, lb = data_losses(model, batch())
        router = model.layers[0].moe.router.weight
        for loss in (ce, lb):
            gradient, = torch.autograd.grad(loss, router, retain_graph=True)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0.)

    def test_aoe_disabled_unchanged_and_enabled_rejected(self):
        model = small_model("aoe")
        item = batch()
        original = model(**item)
        rng = torch.get_rng_state().clone()
        result = compute_model_erc(model, ERCConfig())
        self.assertFalse(result.metrics["erc_applied"])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        after = model(**item)
        torch.testing.assert_close(after.logits, original.logits, rtol=0, atol=0)
        torch.testing.assert_close(after.loss, original.loss, rtol=0, atol=0)
        for coefficient in (0., 1.):
            with self.assertRaisesRegex(ValueError, "AoE"):
                compute_model_erc(model, ERCConfig(enabled=True, coefficient=coefficient))


class OptimizerBoundaryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_accumulated_data_plus_one_erc_matches_reference_clip_and_update(self):
        model = small_model()
        reference = copy.deepcopy(model)
        optimizer, scheduler = optimizer_pair(model)
        ref_optimizer, ref_scheduler = optimizer_pair(reference)
        config = ERCConfig(enabled=True, alpha=0.3, coefficient=0.7)
        first, second = batch(), batch()
        second["labels"][0, 2:] = -100  # Unequal effective-target counts.
        totals, targets = backward_data(model, [first, second])
        ref_totals, _ = backward_data(reference, [first, second])
        self.assertIsNotNone(model.layers[0].moe.router.weight.grad)
        data_router_grad = model.layers[0].moe.router.weight.grad.clone()
        rng = torch.get_rng_state().clone()
        # Explicit reference: data gradients already exist, add ERC exactly once.
        reference_erc = mean_erc_loss([
            compute_erc_loss(layer.moe.router.weight, layer.moe.get_gate_weights(), config.alpha)
            for layer in reference.layers
        ])
        (config.coefficient * reference_erc).backward()
        combined = {name: p.grad.clone() if p.grad is not None else None
                    for name, p in reference.named_parameters()}
        expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.1,
                                                      error_if_nonfinite=True)
        ref_optimizer.step()
        ref_scheduler.step()
        torch.set_rng_state(rng)
        real_clip = torch.nn.utils.clip_grad_norm_
        calls = []

        def inspect_clip(parameters, *args, **kwargs):
            calls.append("clip")
            for name, parameter in model.named_parameters():
                if combined[name] is not None:
                    torch.testing.assert_close(parameter.grad, combined[name], rtol=0, atol=0)
            return real_clip(parameters, *args, **kwargs)

        before_gate = model.layers[0].moe.experts[0].gate_proj.weight.detach().clone()
        with patch("moe_erc.erc_training.compute_erc_loss", wraps=compute_erc_loss) as core, \
                patch("torch.nn.utils.clip_grad_norm_", side_effect=inspect_clip):
            metrics = finish_optimizer_step(
                model, config, optimizer, scheduler, **totals,
                global_step=0, effective_targets=targets, clip_norm=0.1,
            )
        self.assertEqual(core.call_count, len(model.layers))
        self.assertEqual(calls, ["clip"])
        self.assertFalse(torch.equal(before_gate, model.layers[0].moe.experts[0].gate_proj.weight))
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(metrics["gradient_norm_before_clip"], float(expected_norm))
        self.assertEqual(metrics["global_optimizer_step"], 1)
        self.assertEqual(metrics["effective_targets"], targets)
        self.assertEqual(scheduler.last_epoch, 1)
        self.assertEqual(metrics["lr_used"], [0.003])
        self.assertEqual(metrics["lr_next"], [0.0015])
        self.assertEqual(totals, ref_totals)
        self.assertEqual(metrics["total_loss"],
                         metrics["ce"] + metrics["lb_weighted"] + metrics["erc_weighted"])
        self.assertAlmostEqual(metrics["erc_weighted"],
                               config.coefficient * metrics["erc_raw"], places=6)
        json.dumps(metrics, allow_nan=False)
        # Router has CE/LB gradients; ERC adds no separate router contribution.
        torch.testing.assert_close(combined["layers.0.moe.router.projection.weight"],
                                   data_router_grad, rtol=0, atol=0)

    def test_one_and_two_microbatches_do_not_multiply_or_divide_erc(self):
        first = small_model()
        second = copy.deepcopy(first)
        config = ERCConfig(enabled=True, alpha=0.)
        totals_one, count_one = backward_data(first, [batch()])
        totals_two, count_two = backward_data(second, [batch(), batch()])
        rng = torch.get_rng_state().clone()
        one = finish_optimizer_step(
            first, config, torch.optim.SGD(first.parameters(), lr=0.01),
            **totals_one, global_step=0, effective_targets=count_one,
        )
        torch.set_rng_state(rng)
        two = finish_optimizer_step(
            second, config, torch.optim.SGD(second.parameters(), lr=0.01),
            **totals_two, global_step=0, effective_targets=count_two,
        )
        self.assertEqual(one["erc_raw"], two["erc_raw"])
        self.assertEqual(one["erc_weighted"], two["erc_weighted"])
        for a, b in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(a, b, rtol=2e-6, atol=1e-8)

    def test_disabled_and_zero_coefficient_update_match_original_boundary(self):
        baseline = small_model()
        base_optimizer, base_scheduler = optimizer_pair(baseline)
        models = [copy.deepcopy(baseline) for _ in range(2)]
        totals, targets = backward_data(baseline, [batch()])
        torch.nn.utils.clip_grad_norm_(baseline.parameters(), 1., error_if_nonfinite=True)
        base_optimizer.step()
        base_scheduler.step()
        for model, config in zip(models, (ERCConfig(), ERCConfig(enabled=True, coefficient=0.))):
            optimizer, scheduler = optimizer_pair(model)
            local_totals, local_targets = backward_data(model, [batch()])
            rng = torch.get_rng_state().clone()
            metrics = finish_optimizer_step(
                model, config, optimizer, scheduler, **local_totals,
                global_step=0, effective_targets=local_targets,
            )
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(totals, local_totals)
            self.assertEqual(targets, local_targets)
            self.assertEqual(metrics["erc_weighted"], 0.)
            for a, b in zip(model.parameters(), baseline.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_cpu_grad_scaler_includes_erc_before_unscale_clip(self):
        first = small_model()
        second = copy.deepcopy(first)
        config = ERCConfig(enabled=True, alpha=0.4)
        scaler = torch.amp.GradScaler("cpu", init_scale=128.)
        totals_one, targets = backward_data(first, [batch()])
        totals_two, _ = backward_data(second, [batch()], scaler)
        rng = torch.get_rng_state().clone()
        metrics_one = finish_optimizer_step(
            first, config, torch.optim.SGD(first.parameters(), lr=0.01),
            **totals_one, global_step=0, effective_targets=targets,
        )
        torch.set_rng_state(rng)
        metrics_two = finish_optimizer_step(
            second, config, torch.optim.SGD(second.parameters(), lr=0.01),
            **totals_two, global_step=0, effective_targets=targets, scaler=scaler,
        )
        self.assertEqual(metrics_one["total_loss"], metrics_two["total_loss"])
        for a, b in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_missing_data_gradients_and_invalid_update_fail_before_noise(self):
        model = small_model()
        optimizer, scheduler = optimizer_pair(model)
        kwargs = dict(ce=1., lb_raw=1., lb_weighted=0.01,
                      global_step=0, effective_targets=8)
        with patch("moe_erc.erc_training.compute_erc_loss") as core:
            with self.assertRaisesRegex(RuntimeError, "gradients are missing"):
                finish_optimizer_step(model, ERCConfig(enabled=True), optimizer, scheduler, **kwargs)
            for changes in ({"global_step": -1}, {"effective_targets": 0},
                            {"clip_norm": 0.}, {"ce": torch.tensor(1.)}, {"global_step": 1}):
                with self.assertRaises(ValueError):
                    finish_optimizer_step(model, ERCConfig(enabled=True), optimizer, scheduler,
                                          **{**kwargs, **changes})
            core.assert_not_called()

    def test_nonfinite_combined_gradient_does_not_step_optimizer_or_scheduler(self):
        model = small_model()
        optimizer, scheduler = optimizer_pair(model)
        totals, targets = backward_data(model, [batch()])
        model.final_norm.weight.grad.fill_(float("inf"))
        with patch.object(optimizer, "step") as step:
            with self.assertRaises(RuntimeError):
                finish_optimizer_step(model, ERCConfig(enabled=True), optimizer, scheduler,
                                      **totals, global_step=0, effective_targets=targets)
            step.assert_not_called()
        self.assertEqual(scheduler.last_epoch, 0)


class ERCCheckpointAdapterTests(unittest.TestCase):
    def test_in_memory_config_and_rng_restore_replays_next_proxy_loss(self):
        torch.manual_seed(44)
        model = small_model()
        config = ERCConfig(enabled=True, alpha=0.4)
        # Only adapter/RNG semantics, not a replacement for the Step 6 checkpoint.
        buffer = BytesIO()
        torch.save({"erc_config": config.to_dict(), "rng": torch.get_rng_state()}, buffer)
        expected = compute_model_erc(model, config)
        buffer.seek(0)
        loaded = torch.load(buffer, weights_only=True)
        validate_erc_checkpoint_config(loaded["erc_config"], config)
        torch.set_rng_state(loaded["rng"])
        actual = compute_model_erc(model, config)
        torch.testing.assert_close(actual.raw_loss, expected.raw_loss, rtol=0, atol=0)
        self.assertEqual(actual.metrics, expected.metrics)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class ERCIntegrationCUDATests(unittest.TestCase):
    def test_bf16_data_backward_fp32_erc_and_disabled_rng(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("CUDA BF16 unavailable")
        torch.manual_seed(32)
        model = small_model().cuda()
        item = {k: v.cuda() for k, v in batch().items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ce, lb = data_losses(model, item)
        (ce + 0.01 * lb).backward()
        rng = torch.cuda.get_rng_state().clone()
        compute_model_erc(model, ERCConfig())
        self.assertTrue(torch.equal(rng, torch.cuda.get_rng_state()))
        metrics = finish_optimizer_step(
            model, ERCConfig(enabled=True), torch.optim.AdamW(model.parameters(), lr=0.001),
            ce=float(ce.detach()), lb_raw=float(lb.detach()), lb_weighted=float((0.01*lb).detach()),
            global_step=0, effective_targets=8,
        )
        self.assertTrue(metrics["erc_applied"])
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
        json.dumps(metrics, allow_nan=False)
        torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
