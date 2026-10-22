"""Full-softmax Top-K routing, with no post-selection renormalization."""

import unittest

import torch

from moe_erc import ModelConfig
from moe_erc.router import MoERouter


class RouterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.router = MoERouter(ModelConfig())

    def test_shapes_formula_and_budget(self):
        x = torch.randn(2, 128, 512)
        result = self.router(x)
        self.assertEqual(result.logits.shape, (2, 128, 16))
        self.assertEqual(result.probabilities.shape, (2, 128, 16))
        self.assertEqual(result.topk_indices.shape, (2, 128, 2))
        self.assertEqual(result.topk_weights.shape, (2, 128, 2))
        torch.testing.assert_close(result.logits, x @ self.router.weight.T)
        expected = torch.softmax(x @ self.router.weight.T, dim=-1)
        torch.testing.assert_close(result.probabilities, expected)
        torch.testing.assert_close(result.probabilities.sum(-1), torch.ones(2, 128))
        expected_weights, expected_indices = expected.topk(2, dim=-1)
        torch.testing.assert_close(result.topk_indices, expected_indices)
        torch.testing.assert_close(result.topk_weights, expected_weights)
        torch.testing.assert_close(result.topk_weights, result.probabilities.gather(-1, result.topk_indices))
        self.assertTrue((result.topk_weights.sum(-1) < 1).all().item())
        self.assertEqual(sum(p.numel() for p in self.router.parameters()), 8_192)
        self.assertIs(self.router.weight, self.router.projection.weight)

    def test_uniform_scores_are_not_renormalized(self):
        with torch.no_grad():
            self.router.weight.zero_()
        result = self.router(torch.ones(1, 1, 512))
        torch.testing.assert_close(result.topk_weights, torch.full((1, 1, 2), 1/16))
        torch.testing.assert_close(result.topk_weights.sum(-1), torch.tensor([[1/8]]))
        self.assertNotEqual(result.topk_indices[0, 0, 0], result.topk_indices[0, 0, 1])

    def test_selected_weights_backpropagate(self):
        x = torch.randn(3, 512, requires_grad=True)
        self.router(x).topk_weights.sum().backward()
        for value in (x, self.router.weight):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all().item())
            self.assertGreater(value.grad.abs().sum().item(), 0)

    def test_bf16_probabilities_use_fp32(self):
        router = self.router.to(dtype=torch.bfloat16)
        result = router(torch.randn(1, 3, 512, dtype=torch.bfloat16))
        self.assertEqual(result.logits.dtype, torch.bfloat16)
        self.assertEqual(result.probabilities.dtype, torch.float32)
        self.assertEqual(result.topk_weights.dtype, torch.float32)
        torch.testing.assert_close(result.probabilities, result.logits.float().softmax(-1))


if __name__ == "__main__":
    unittest.main()
