"""Regression coverage for the shared embedding/head initialization scale."""

from io import BytesIO
import math
import unittest
from unittest.mock import patch

import torch

from moe_erc import AoEConfig, ModelConfig
from moe_erc.model_factory import create_model


MODEL_TYPES = ("vanilla_moe", "aoe")
SEEDS = (0, 42, 123)
PARAMETER_COUNTS = {"vanilla_moe": 52_288_000, "aoe": 52_263_424}


def small_config(kind):
    settings = dict(vocab_size=257, d_model=32, n_heads=4, n_layers=2,
                    n_experts=4, top_k=2, expert_hidden_dim=16)
    return (AoEConfig(**settings, aoe_rank=11) if kind == "aoe"
            else ModelConfig(**settings))


def inputs(vocab_size, seed, length=16):
    return torch.randint(vocab_size, (2, length),
                         generator=torch.Generator().manual_seed(seed + 1000))


class InitializationAssertions:
    def assert_tied(self, model):
        embedding = model.token_embedding.weight
        head = model.lm_head.weight
        self.assertIs(embedding, head)
        self.assertEqual(embedding.data_ptr(), head.data_ptr())
        self.assertEqual(embedding.untyped_storage().data_ptr(),
                         head.untyped_storage().data_ptr())

    def assert_shifted_ce(self, output, labels):
        targets = labels[:, 1:]
        valid = targets != -100
        log_probs = output.logits[:, :-1].float().log_softmax(-1)
        expected = -log_probs[valid].gather(1, targets[valid, None]).mean()
        torch.testing.assert_close(output.loss, expected, rtol=1e-5, atol=1e-5)

    def assert_reasonable_initial_ce(self, output, vocab_size):
        self.assertTrue(torch.isfinite(output.logits).all().item())
        self.assertTrue(torch.isfinite(output.loss).item())
        self.assertGreater(output.loss.item(), 0)
        # Broad catastrophic-scale guard, not an assertion that CE = log(V)
        # or that this initialization is optimal for language modeling.
        self.assertLess(output.loss.item(), math.log(vocab_size) + 10)

    def assert_gradients(self, model, output, kind):
        def check(parameter):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())

        check(model.token_embedding.weight)
        check(model.final_norm.weight)
        for layer, info in zip(model.layers, output.routing_info):
            check(layer.norm_1.weight)
            check(layer.norm_2.weight)
            for parameter in layer.attention.parameters():
                check(parameter)
            if kind == "vanilla_moe":
                check(layer.moe.router.weight)
                self.assertGreater(layer.moe.router.weight.grad.abs().sum().item(), 0)
            else:
                # Every gate-down projection participates in AoE routing even
                # when that expert's remaining FFN projections are unselected.
                for expert in layer.moe.experts:
                    check(expert.gate_down_proj.weight)
            active = info.diagnostics.token_counts > 0
            self.assertTrue(active.any().item())
            for expert_id, expert in enumerate(layer.moe.experts):
                if active[expert_id].item():
                    for parameter in expert.parameters():
                        check(parameter)
        for parameter in model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all().item())


class ModelInitializationTests(InitializationAssertions, unittest.TestCase):
    def test_full_size_scale_budget_and_initial_ce_across_seeds(self):
        for kind in MODEL_TYPES:
            for seed in SEEDS:
                with self.subTest(kind=kind, seed=seed):
                    torch.manual_seed(seed)
                    model = create_model(kind).eval()
                    self.assert_tied(model)
                    weight = model.token_embedding.weight.detach()
                    self.assertAlmostEqual(weight.std(unbiased=False).item(), 0.02, delta=1e-4)
                    self.assertAlmostEqual(weight.mean().item(), 0, delta=1e-4)
                    self.assertEqual(sum(p.numel() for p in model.parameters()),
                                     PARAMETER_COUNTS[kind])
                    ids = inputs(16000, seed)
                    with torch.no_grad():
                        output = model(ids, labels=ids)
                    self.assertEqual(output.logits.shape, (2, 16, 16000))
                    self.assert_reasonable_initial_ce(output, 16000)
                    self.assert_shifted_ce(output, ids)
                    del output, model

    def test_explicit_shared_initialization_is_called_once(self):
        original_normal = torch.nn.init.normal_
        for kind in MODEL_TYPES:
            with self.subTest(kind=kind):
                with patch("torch.nn.init.normal_", wraps=original_normal) as initialize:
                    model = create_model(kind, small_config(kind)).eval()
                    with torch.no_grad():
                        model(inputs(model.config.vocab_size, 42))
                    model.load_state_dict(model.state_dict())
                explicit_calls = [call for call in initialize.call_args_list
                                  if call.kwargs.get("std") == 0.02]
                # nn.Embedding also performs its own default initialization;
                # only the intentional final scale override is counted here.
                self.assertEqual(len(explicit_calls), 1)
                self.assertIs(explicit_calls[0].args[0], model.token_embedding.weight)
                self.assertEqual(explicit_calls[0].kwargs["mean"], 0.0)
                self.assert_tied(model)

    def test_other_parameters_keep_their_previous_initialization(self):
        original_normal = torch.nn.init.normal_

        def legacy_embedding(tensor, *args, **kwargs):
            # Emulate the old constructor by skipping only the new final call.
            if kwargs.get("std") == 0.02:
                return tensor
            return original_normal(tensor, *args, **kwargs)

        for kind in MODEL_TYPES:
            for seed in SEEDS:
                with self.subTest(kind=kind, seed=seed):
                    config = small_config(kind)
                    torch.manual_seed(seed)
                    with patch("torch.nn.init.normal_", side_effect=legacy_embedding):
                        legacy = create_model(kind, config)
                    torch.manual_seed(seed)
                    fixed = create_model(kind, config)
                    old_parameters = dict(legacy.named_parameters())
                    self.assertEqual(set(old_parameters), set(dict(fixed.named_parameters())))
                    for name, parameter in fixed.named_parameters():
                        if parameter is not fixed.token_embedding.weight:
                            torch.testing.assert_close(parameter, old_parameters[name], rtol=0, atol=0)
                    self.assertFalse(torch.equal(legacy.token_embedding.weight,
                                                 fixed.token_embedding.weight))

    def test_cpu_shift_gradients_and_state_dict_round_trip(self):
        for kind in MODEL_TYPES:
            for seed in SEEDS:
                with self.subTest(kind=kind, seed=seed):
                    config = small_config(kind)
                    torch.manual_seed(seed)
                    model = create_model(kind, config).eval()
                    ids = inputs(config.vocab_size, seed)
                    labels = ids.clone()
                    labels[0, 3] = -100
                    output = model(ids, labels=labels, return_routing_info=True)
                    self.assert_shifted_ce(output, labels)
                    self.assert_reasonable_initial_ce(output, config.vocab_size)
                    changed_first_label = labels.clone()
                    changed_first_label[:, 0] = (labels[:, 0] + 1) % config.vocab_size
                    with torch.no_grad():
                        torch.testing.assert_close(model(ids, labels=changed_first_label).loss,
                                                   output.loss)
                    output.loss.backward()
                    self.assert_gradients(model, output, kind)

                    buffer = BytesIO()
                    torch.save(model.state_dict(), buffer)
                    buffer.seek(0)
                    restored = create_model(kind, config).eval()
                    restored.load_state_dict(torch.load(buffer, weights_only=True))
                    self.assert_tied(restored)
                    torch.testing.assert_close(restored.token_embedding.weight,
                                               model.token_embedding.weight, rtol=0, atol=0)
                    with torch.no_grad():
                        restored_output = restored(ids, labels=labels)
                    torch.testing.assert_close(restored_output.logits, output.logits, rtol=0, atol=0)
                    torch.testing.assert_close(restored_output.loss, output.loss, rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class ModelInitializationCUDATests(InitializationAssertions, unittest.TestCase):
    def test_full_models_fp32_and_bf16_gradients_across_seeds(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("CUDA device does not support BF16")
        for kind in MODEL_TYPES:
            for seed in (42, 123):
                torch.manual_seed(seed)
                model = create_model(kind).cuda().eval()
                ids = inputs(16000, seed, length=32).cuda()
                try:
                    for autocast in (False, True):
                        with self.subTest(kind=kind, seed=seed, bf16=autocast):
                            model.zero_grad(set_to_none=True)
                            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                                output = model(ids, labels=ids, return_routing_info=True)
                            expected_dtype = torch.bfloat16 if autocast else torch.float32
                            self.assertEqual(output.logits.dtype, expected_dtype)
                            self.assertEqual(output.loss.dtype, torch.float32)
                            self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))
                            self.assert_tied(model)
                            self.assert_reasonable_initial_ce(output, 16000)
                            self.assert_shifted_ce(output, ids)
                            output.loss.backward()
                            self.assert_gradients(model, output, kind)
                            del output
                finally:
                    del model, ids
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
