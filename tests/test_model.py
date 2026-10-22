"""Full-size budget and small-model language modeling behavior."""

from dataclasses import asdict, fields
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from moe_erc import ModelConfig, count_parameters
from moe_erc.model import LanguageModelOutput, MoELanguageModel


def small_config():
    return ModelConfig(vocab_size=41, d_model=16, n_heads=4, n_layers=2,
                       n_experts=4, top_k=2, expert_hidden_dim=8)


class FullSizeModelTests(unittest.TestCase):
    def test_exact_budget_tying_shape_and_cpu_backward(self):
        torch.manual_seed(123)
        config = ModelConfig()
        model = MoELanguageModel(config)
        self.assertEqual(len(model.layers), 6)
        self.assertIs(model.lm_head.weight, model.token_embedding.weight)
        actual = {
            "token_embeddings": model.token_embedding.weight.numel(),
            "attention_projections": sum(p.numel() for layer in model.layers for p in layer.attention.parameters()),
            "swiglu_experts": sum(p.numel() for layer in model.layers for p in layer.moe.experts.parameters()),
            "router_weights": sum(layer.moe.router.weight.numel() for layer in model.layers),
            "normalization": model.final_norm.weight.numel() + sum(
                layer.norm_1.weight.numel() + layer.norm_2.weight.numel() for layer in model.layers),
        }
        self.assertEqual(actual, asdict(count_parameters(config)))
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 52_288_000)
        self.assertEqual(sum(actual.values()), 52_288_000)
        # Every block parameter is independent, including all routers/experts.
        block_parameter_ids = [id(p) for layer in model.layers for p in layer.parameters()]
        self.assertEqual(len(block_parameter_ids), len(set(block_parameter_ids)))
        ids = torch.randint(config.vocab_size, (2, 16))
        output = model(ids, labels=ids, return_routing_info=True)
        self.assertEqual(output.logits.shape, (2, 16, 16000))
        self.assertTrue(torch.isfinite(output.loss).item())
        output.loss.backward()
        for p in (model.token_embedding.weight, *model.final_norm.parameters(),
                  *(layer.attention.q_proj.weight for layer in model.layers),
                  *(layer.moe.router.weight for layer in model.layers)):
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all().item())
            self.assertGreater(p.grad.abs().sum().item(), 0)
        for layer, info in zip(model.layers, output.routing_info):
            for i, expert in enumerate(layer.moe.experts):
                if info.diagnostics.token_counts[i].item() > 0:
                    for parameter in expert.parameters():
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all().item())


class LanguageModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.config = small_config()
        self.model = MoELanguageModel(self.config)

    def test_output_interface_and_no_default_diagnostics(self):
        ids = torch.randint(41, (2, 7))
        with patch.object(self.model.layers[0].moe, "forward", wraps=self.model.layers[0].moe.forward) as forward:
            logits = self.model(ids)
            self.assertIsInstance(logits, torch.Tensor)
            self.assertFalse(forward.call_args.kwargs.get("return_diagnostics", False))
        self.assertEqual(logits.shape, (2, 7, 41))
        result = self.model(ids, labels=ids)
        self.assertIsInstance(result, LanguageModelOutput)
        self.assertIsNone(result.routing_info)
        # The modules never cache activations or routing output attributes.
        for module in self.model.modules():
            self.assertFalse(any(isinstance(v, torch.Tensor) for v in vars(module).values()))

    def test_next_token_shift_ignore_and_loss_backward(self):
        ids = torch.tensor([[2, 4, 8, 16, 3], [7, 9, 1, 6, 5]])
        labels = ids.clone()
        labels[0, 2] = -100
        labels[:, 0] = -100
        result = self.model(ids, labels=labels)
        expected = F.cross_entropy(result.logits[:, :-1].float().reshape(-1, 41),
                                   labels[:, 1:].reshape(-1), ignore_index=-100)
        torch.testing.assert_close(result.loss, expected)
        # Independent selected-token reduction verifies ignore semantics.
        valid = labels[:, 1:] != -100
        log_probs = result.logits[:, :-1].float().log_softmax(-1)
        manual = -log_probs[valid].gather(1, labels[:, 1:][valid, None]).mean()
        torch.testing.assert_close(result.loss, manual)
        changed_first_label = labels.clone()
        changed_first_label[:, 0] = 40
        torch.testing.assert_close(self.model(ids, labels=changed_first_label).loss, result.loss)
        unshifted = F.cross_entropy(result.logits.float().reshape(-1, 41), labels.reshape(-1), ignore_index=-100)
        self.assertGreater(abs(result.loss.item() - unshifted.item()), 1e-3)
        result.loss.backward()
        for p in (self.model.token_embedding.weight, self.model.layers[0].attention.q_proj.weight,
                  self.model.layers[0].moe.router.weight):
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all().item())
            self.assertGreater(p.grad.abs().sum().item(), 0)

    def test_no_valid_targets_produce_differentiable_zero(self):
        for length, ignore_all in ((1, False), (5, True)):
            with self.subTest(length=length):
                ids = torch.randint(41, (2, length))
                labels = torch.full_like(ids, -100) if ignore_all else ids.clone()
                output = self.model(ids, labels=labels)
                self.assertEqual(output.loss.item(), 0)
                self.assertTrue(output.loss.requires_grad)
                self.model.zero_grad(set_to_none=True)
                output.loss.backward()
                self.assertIsNotNone(self.model.token_embedding.weight.grad)
                self.assertTrue(torch.isfinite(self.model.token_embedding.weight.grad).all().item())

    def test_causality_under_future_token_changes(self):
        self.model.eval()
        ids = torch.randint(41, (2, 12))
        changed = ids.clone()
        changed[:, 5:] = (changed[:, 5:] + 11) % 41
        with torch.no_grad():
            original_logits, changed_logits = self.model(ids), self.model(changed)
            prefix_logits = self.model(ids[:, :5])
        torch.testing.assert_close(original_logits[:, :5], changed_logits[:, :5], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(original_logits[:, :5], prefix_logits, rtol=1e-5, atol=1e-6)
        self.assertGreater((original_logits[:, 5:] - changed_logits[:, 5:]).abs().max().item(), 1e-3)

    def test_sequence_boundaries_and_integer_types(self):
        with torch.no_grad():
            for length in (1, 2, 16, 511, 512):
                for dtype in (torch.int32, torch.int64):
                    ids = torch.randint(41, (1, length), dtype=dtype)
                    self.assertEqual(self.model(ids).shape, (1, length, 41))

    def test_invalid_inputs_and_labels(self):
        invalid_ids = [torch.ones(2, 513, dtype=torch.long), torch.ones(2, 0, dtype=torch.long),
                       torch.ones(0, 2, dtype=torch.long), torch.ones(2, dtype=torch.long),
                       torch.ones(2, 3, 1, dtype=torch.long), torch.tensor([[-1]]), torch.tensor([[41]])]
        for ids in invalid_ids:
            with self.subTest(shape=ids.shape):
                with self.assertRaises(ValueError):
                    self.model(ids)
        for dtype in (torch.float32, torch.bool, torch.int16):
            with self.assertRaises(TypeError):
                self.model(torch.ones(1, 3, dtype=dtype))
        ids = torch.ones(1, 3, dtype=torch.long)
        for labels in (torch.ones(1, 2, dtype=torch.long), torch.tensor([[1, -1, 2]]), torch.tensor([[1, 41, 2]])):
            with self.assertRaises(ValueError):
                self.model(ids, labels=labels)
        with self.assertRaises(TypeError):
            self.model(ids, labels=ids.float())

    def test_diagnostics_layer_mapping_and_erc_access(self):
        ids = torch.randint(41, (2, 7))
        captured = []
        def record(module, args, output):
            captured.append(output[1])
        handles = [layer.moe.register_forward_hook(record) for layer in self.model.layers]
        try:
            result = self.model(ids, return_routing_info=True)
        finally:
            for handle in handles:
                handle.remove()
        self.assertIsNone(result.loss)
        self.assertEqual(len(result.routing_info), 2)
        parameter_ids = []
        for index, (layer, info) in enumerate(zip(self.model.layers, result.routing_info)):
            self.assertEqual(info.layer_index, index)
            self.assertIs(info.diagnostics, captured[index])
            self.assertEqual(info.diagnostics.topk_indices.shape, (2, 7, 2))
            self.assertEqual(info.diagnostics.token_counts.sum().item(), 28)
            torch.testing.assert_close(info.diagnostics.assignment_fractions.sum(), torch.tensor(1.))
            torch.testing.assert_close(info.diagnostics.mean_probabilities.sum(), torch.tensor(1.))
            for field in fields(info.diagnostics):
                self.assertFalse(getattr(info.diagnostics, field.name).requires_grad)
            self.assertTrue(layer.moe.router.weight.requires_grad)
            gates = layer.moe.get_gate_weights()
            self.assertTrue(gates.requires_grad)
            self.assertEqual(gates.shape, (4, 8, 16))
            parameter_ids.extend(id(p) for p in layer.parameters())
        self.assertEqual(len(parameter_ids), len(set(parameter_ids)))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class LanguageModelCUDATests(unittest.TestCase):
    def test_bf16_forward_backward_and_tying(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("CUDA device does not support BF16")
        torch.manual_seed(123)
        for explicit in (False, True):
            with self.subTest(explicit_bf16=explicit):
                dtype = torch.bfloat16 if explicit else torch.float32
                model = MoELanguageModel(small_config()).to(device="cuda", dtype=dtype)
                self.assertIs(model.lm_head.weight, model.token_embedding.weight)
                ids = torch.randint(41, (2, 16), device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not explicit):
                    result = model(ids, labels=ids, return_routing_info=True)
                self.assertEqual(result.logits.dtype, torch.bfloat16)
                self.assertEqual(result.loss.dtype, torch.float32)
                self.assertTrue(torch.isfinite(result.loss).item())
                expected = F.cross_entropy(result.logits[:, :-1].float().reshape(-1, 41), ids[:, 1:].reshape(-1))
                torch.testing.assert_close(result.loss, expected)
                result.loss.backward()
                for p in (model.token_embedding.weight, model.layers[0].attention.q_proj.weight,
                          model.layers[0].moe.router.weight):
                    self.assertIsNotNone(p.grad)
                    self.assertTrue(torch.isfinite(p.grad).all().item())
                    self.assertGreater(p.grad.abs().sum().item(), 0)
                # Diagnostics remain detached; all selected gates retain gradients.
                for layer, info in zip(model.layers, result.routing_info):
                    for i, expert in enumerate(layer.moe.experts):
                        if info.diagnostics.token_counts[i].item():
                            self.assertIsNotNone(expert.gate_proj.weight.grad)
                            self.assertTrue(torch.isfinite(expert.gate_proj.weight.grad).all().item())


if __name__ == "__main__":
    unittest.main()
