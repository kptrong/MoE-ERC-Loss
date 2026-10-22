"""Shape, formula, budget and gradient checks for a SwiGLU expert."""

import unittest

import torch
from torch.nn import functional as F

from moe_erc import ModelConfig
from moe_erc.experts import SwiGLUExpert


class ExpertTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_shape_budget_and_backward(self):
        expert = SwiGLUExpert(ModelConfig())
        self.assertEqual(sum(p.numel() for p in expert.parameters()), 393_216)
        self.assertEqual(expert.gate_proj.weight.shape, (256, 512))
        self.assertEqual(set(dict(expert.named_parameters())),
                         {"gate_proj.weight", "up_proj.weight", "down_proj.weight"})
        x = torch.randn(2, 128, 512, requires_grad=True)
        y = expert(x)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all().item())
        y.square().mean().backward()
        for value in (x, *expert.parameters()):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all().item())
            self.assertGreater(value.grad.abs().sum().item(), 0)

    def test_formula(self):
        expert = SwiGLUExpert(ModelConfig(d_model=8, n_heads=2, expert_hidden_dim=4)).double()
        x = torch.randn(2, 3, 8, dtype=torch.float64)
        gate = F.silu(x @ expert.gate_proj.weight.T)
        up = x @ expert.up_proj.weight.T
        expected = (gate * up) @ expert.down_proj.weight.T
        torch.testing.assert_close(expert(x), expected, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
