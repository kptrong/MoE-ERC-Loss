"""Factorized SwiGLU formula, reusable z, exact budget and autograd."""

import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from moe_erc import AoEConfig
from moe_erc.aoe_expert import AoEExpert


class AoEExpertTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_default_shape_parameter_count_and_backward(self):
        expert = AoEExpert(AoEConfig())
        self.assertEqual(sum(p.numel() for p in expert.parameters()), 393_472)
        self.assertEqual(expert.gate_down_proj.weight.shape, (171, 512))
        self.assertEqual(expert.gate_up_proj.weight.shape, (256, 171))
        self.assertTrue(all(module.bias is None for module in expert.modules() if isinstance(module, torch.nn.Linear)))
        x = torch.randn(2, 16, 512, requires_grad=True)
        output = expert(x)
        self.assertEqual(output.shape, x.shape)
        output.square().mean().backward()
        for value in (x, *expert.parameters()):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all().item())
            self.assertGreater(value.grad.abs().sum().item(), 0)

    def test_independent_factorized_reference_and_cached_activation(self):
        config = AoEConfig(d_model=8, n_heads=2, expert_hidden_dim=4, aoe_rank=3)
        expert = AoEExpert(config).double()
        x = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
        z = x @ expert.gate_down_proj.weight.T
        gate = z @ expert.gate_up_proj.weight.T
        expected = (F.silu(gate) * (x @ expert.up_proj.weight.T)) @ expert.down_proj.weight.T
        torch.testing.assert_close(expert(x), expected, rtol=1e-12, atol=1e-12)
        with patch.object(expert.gate_down_proj, "forward", side_effect=AssertionError("must reuse z")):
            cached = expert(x, gate_down_activation=z)
        torch.testing.assert_close(cached, expected, rtol=1e-12, atol=1e-12)
        cached.sum().backward()
        self.assertGreater(expert.gate_down_proj.weight.grad.abs().sum().item(), 0)

    def test_rank_validation(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "aoe_rank"):
                AoEConfig(aoe_rank=value)
        self.assertEqual(AoEConfig().to_dict()["aoe_rank"], 171)


if __name__ == "__main__":
    unittest.main()
