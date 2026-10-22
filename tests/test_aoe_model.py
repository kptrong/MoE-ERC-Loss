"""Shared decoder/LM semantics, independent layers and the full AoE budget."""

from dataclasses import asdict, fields
import unittest

import torch
from torch.nn import functional as F

from moe_erc import AoEConfig, ModelConfig, count_parameters
from moe_erc.aoe import AoEDecoderBlock, SparseAoE
from moe_erc.decoder import MoEDecoderBlock
from moe_erc.model import AoELanguageModel, MoELanguageModel
from moe_erc.router import MoERouter


def small_config():
    return AoEConfig(vocab_size=41, d_model=16, n_heads=4, n_layers=2,
                     n_experts=4, top_k=2, expert_hidden_dim=8, aoe_rank=5)


class AoEModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_full_budget_and_shape_backward(self):
        config = AoEConfig()
        model = AoELanguageModel(config)
        self.assertEqual(len(model.layers), 6)
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 52_263_424)
        expected = count_parameters(config, "aoe")
        self.assertEqual(expected.total, 52_263_424)
        self.assertEqual(expected.swiglu_experts, 37_773_312)
        self.assertEqual(expected.router_weights, 0)
        actual = {
            "token_embeddings": model.token_embedding.weight.numel(),
            "attention_projections": sum(p.numel() for layer in model.layers for p in layer.attention.parameters()),
            "swiglu_experts": sum(p.numel() for layer in model.layers for p in layer.moe.parameters()),
            "router_weights": 0,
            "normalization": model.final_norm.weight.numel() + sum(
                layer.norm_1.weight.numel() + layer.norm_2.weight.numel() for layer in model.layers),
        }
        self.assertEqual(actual, asdict(expected))
        self.assertEqual(count_parameters(ModelConfig()).total - expected.total, 24_576)
        self.assertIs(model.lm_head.weight, model.token_embedding.weight)
        self.assertFalse(any(isinstance(m, MoERouter) for m in model.modules()))
        ids = torch.randint(16000, (2, 16))
        result = model(ids, labels=ids, return_routing_info=True)
        self.assertEqual(result.logits.shape, (2, 16, 16000))
        self.assertTrue(torch.isfinite(result.loss).item())
        result.loss.backward()
        for p in (model.token_embedding.weight, *(layer.attention.q_proj.weight for layer in model.layers),
                  *(layer.moe.experts[0].gate_down_proj.weight for layer in model.layers)):
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all().item())
            self.assertGreater(p.grad.abs().sum().item(), 0)

    def test_shared_forward_implementation_and_independent_parameters(self):
        self.assertIs(AoELanguageModel.forward, MoELanguageModel.forward)
        self.assertIs(AoEDecoderBlock.forward, MoEDecoderBlock.forward)
        model = AoELanguageModel(small_config())
        self.assertTrue(all(isinstance(layer.moe, SparseAoE) for layer in model.layers))
        ids = [id(p) for layer in model.layers for p in layer.parameters()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_causal_lm_loss_masking_and_layer_diagnostics(self):
        model = AoELanguageModel(small_config()).eval()
        ids = torch.randint(41, (2, 12))
        changed = ids.clone()
        changed[:, 5:] = (changed[:, 5:] + 7) % 41
        labels = ids.clone()
        labels[0, 3] = -100
        result = model(ids, labels=labels, return_routing_info=True)
        expected = F.cross_entropy(result.logits[:, :-1].float().reshape(-1, 41),
                                   labels[:, 1:].reshape(-1), ignore_index=-100)
        torch.testing.assert_close(result.loss, expected)
        with torch.no_grad():
            torch.testing.assert_close(result.logits[:, :5], model(changed)[:, :5], rtol=1e-5, atol=1e-6)
        for index, info in enumerate(result.routing_info):
            self.assertEqual(info.layer_index, index)
            self.assertEqual(info.diagnostics.scores.shape, (2, 12, 4))
            self.assertEqual(info.diagnostics.token_counts.sum().item(), 48)
            for field in fields(info.diagnostics):
                self.assertFalse(getattr(info.diagnostics, field.name).requires_grad)
        default = model(ids, labels=labels)
        self.assertIsNone(default.routing_info)
        self.assertTrue(all(not any(isinstance(value, torch.Tensor) for value in vars(m).values())
                            for m in model.modules()))
        for length in (1, 5):
            inputs = ids[:, :length]
            output = model(inputs, labels=torch.full_like(inputs, -100))
            self.assertEqual(output.loss.item(), 0)
            output.loss.backward()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class AoEModelCUDATests(unittest.TestCase):
    def test_bf16_model_forward_backward(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("CUDA device does not support BF16")
        torch.manual_seed(123)
        model = AoELanguageModel(small_config()).cuda()
        ids = torch.randint(41, (2, 16), device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model(ids, labels=ids)
        self.assertEqual(result.logits.dtype, torch.bfloat16)
        self.assertEqual(result.loss.dtype, torch.float32)
        self.assertIs(model.lm_head.weight, model.token_embedding.weight)
        result.loss.backward()
        self.assertTrue(torch.isfinite(result.loss).item())
        self.assertTrue(all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None))


if __name__ == "__main__":
    unittest.main()
