"""Pre-Norm topology, residual arithmetic and selected-module gradients."""

import unittest
from unittest.mock import patch

import torch

from moe_erc import ModelConfig
from moe_erc.attention import CausalSelfAttention
from moe_erc.decoder import MoEDecoderBlock
from moe_erc.moe import SparseMoE
from moe_erc.rmsnorm import RMSNorm


class DecoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_default_shape_structure_and_backward(self):
        block = MoEDecoderBlock(ModelConfig())
        self.assertEqual(sum(isinstance(m, RMSNorm) for m in block.modules()), 2)
        self.assertEqual(sum(isinstance(m, CausalSelfAttention) for m in block.modules()), 1)
        self.assertEqual(sum(isinstance(m, SparseMoE) for m in block.modules()), 1)
        x = torch.randn(2, 16, 512, requires_grad=True)
        y, diagnostic = block(x, return_routing_info=True)
        self.assertEqual(y.shape, x.shape)
        y.square().mean().backward()
        selected = [p for i, expert in enumerate(block.moe.experts)
                    if diagnostic.token_counts[i].item() > 0 for p in expert.parameters()]
        for p in (x, block.norm_1.weight, block.norm_2.weight,
                  *block.attention.parameters(), block.moe.router.weight, *selected):
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all().item())
            self.assertGreater(p.grad.abs().sum().item(), 0)

    def test_pre_norm_residual_equations(self):
        config = ModelConfig(d_model=8, n_heads=2, n_experts=4, expert_hidden_dim=4)
        block = MoEDecoderBlock(config).eval()
        x = torch.randn(2, 5, 8)
        with torch.no_grad():
            h = x + block.attention(block.norm_1(x))
            expected = h + block.moe(block.norm_2(h))
            torch.testing.assert_close(block(x), expected)
            with patch.object(block.norm_1, "forward", wraps=block.norm_1.forward) as first:
                with patch.object(block.norm_2, "forward", wraps=block.norm_2.forward) as second:
                    block(x)
                    self.assertEqual(first.call_count, 1)
                    self.assertEqual(second.call_count, 1)
            # Zero branches must leave x intact through both residual paths.
            with patch.object(block.attention, "forward", side_effect=torch.zeros_like):
                with patch.object(block.moe, "forward", side_effect=torch.zeros_like):
                    torch.testing.assert_close(block(x), x, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
