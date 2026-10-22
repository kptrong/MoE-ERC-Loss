"""Norm routing, sparse dispatch, dense reference and mixed-precision tests."""

from dataclasses import fields, replace
import unittest

import torch

from moe_erc import AoEConfig
from moe_erc.aoe import SparseAoE
from moe_erc.router import MoERouter


def config():
    return AoEConfig(d_model=8, n_heads=2, n_experts=4, top_k=2, expert_hidden_dim=4, aoe_rank=3)


def dense_reference(layer, x):
    scores = torch.stack([(x @ e.gate_down_proj.weight.T).norm(p=2, dim=-1) for e in layer.experts], dim=-1)
    weights, indices = scores.softmax(-1).topk(layer.config.top_k, dim=-1)
    all_outputs = torch.stack([e(x) for e in layer.experts], dim=-2)
    selected = all_outputs.gather(-2, indices.unsqueeze(-1).expand(*indices.shape, x.shape[-1]))
    return (selected * weights.unsqueeze(-1)).sum(-2)


class AoERoutingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_norms_before_silu_full_softmax_and_no_router(self):
        layer = SparseAoE(config()).double()
        x = torch.zeros(1, 8, dtype=torch.float64)
        x[0, 0] = 1
        with torch.no_grad():
            for i, expert in enumerate(layer.experts):
                expert.gate_down_proj.weight.zero_()
                expert.gate_down_proj.weight[:, 0].copy_(torch.tensor([-3., -4., 0.]) * (i+1))
        routing = layer.compute_routing(x)
        self.assertEqual(routing.gate_down_activations.shape, (1, 4, 3))
        torch.testing.assert_close(routing.scores, torch.tensor([[5., 10., 15., 20.]], dtype=torch.float64))
        torch.testing.assert_close(routing.probabilities, routing.scores.softmax(-1))
        torch.testing.assert_close(routing.probabilities.sum(-1), torch.ones(1, dtype=torch.float64))
        torch.testing.assert_close(routing.topk_indices, torch.tensor([[3, 2]]))
        torch.testing.assert_close(routing.topk_weights, routing.probabilities.gather(-1, routing.topk_indices))
        self.assertLess(routing.topk_weights.sum().item(), 1)
        self.assertFalse(any(isinstance(m, MoERouter) for m in layer.modules()))
        self.assertFalse(hasattr(layer, "router"))
        self.assertFalse(torch.allclose(routing.scores, torch.nn.functional.silu(routing.gate_down_activations).norm(dim=-1)))

    def test_routing_gradients_reach_all_gate_down_projections(self):
        layer = SparseAoE(config()).double()
        x = torch.randn(7, 8, dtype=torch.float64, requires_grad=True)
        routing = layer.compute_routing(x)
        routing.topk_weights.sum().backward()
        for value in (x, *(e.gate_down_proj.weight for e in layer.experts)):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all().item())
            self.assertGreater(value.grad.abs().sum().item(), 0)

    def test_ties_zero_norm_and_no_top_k_renormalization(self):
        layer = SparseAoE(config())
        x = torch.zeros(3, 8, requires_grad=True)
        routing = layer.compute_routing(x)
        torch.testing.assert_close(routing.topk_weights, torch.full((3, 2), .25))
        routing.topk_weights.sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all().item())


class SparseAoETests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_default_shape_budget_backward_and_diagnostics(self):
        layer = SparseAoE(AoEConfig())
        self.assertEqual(sum(p.numel() for p in layer.parameters()), 6_295_552)
        x = torch.randn(2, 16, 512, requires_grad=True)
        y, diagnostic = layer(x, return_diagnostics=True)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(diagnostic.scores.shape, (2, 16, 16))
        self.assertEqual(diagnostic.probabilities.shape, (2, 16, 16))
        self.assertEqual(diagnostic.topk_indices.shape, (2, 16, 2))
        self.assertTrue((diagnostic.topk_indices[..., 0] != diagnostic.topk_indices[..., 1]).all().item())
        torch.testing.assert_close(diagnostic.mean_activation_norms, diagnostic.scores.mean((0, 1)))
        torch.testing.assert_close(diagnostic.mean_probabilities, diagnostic.probabilities.mean((0, 1)))
        self.assertEqual(diagnostic.token_counts.sum().item(), 64)
        torch.testing.assert_close(diagnostic.assignment_fractions.sum(), torch.tensor(1.))
        for field in fields(diagnostic):
            self.assertFalse(getattr(diagnostic, field.name).requires_grad)
        y.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all().item())
        for i, expert in enumerate(layer.experts):
            self.assertIsNotNone(expert.gate_down_proj.weight.grad)
            if diagnostic.token_counts[i].item() > 0:
                for p in expert.parameters():
                    self.assertIsNotNone(p.grad)
                    self.assertTrue(torch.isfinite(p.grad).all().item())
                    self.assertGreater(p.grad.abs().sum().item(), 0)

    def test_sparse_dense_outputs_and_gradients(self):
        for top_k in (1, 2, 4):
            with self.subTest(top_k=top_k):
                layer = SparseAoE(replace(config(), top_k=top_k)).double()
                x = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
                sparse, dense = layer(x), dense_reference(layer, x)
                torch.testing.assert_close(sparse, dense, rtol=1e-12, atol=1e-12)
                upstream = torch.randn_like(sparse)
                variables = (x, *layer.parameters())
                a = torch.autograd.grad((sparse * upstream).sum(), variables, allow_unused=True)
                b = torch.autograd.grad((dense * upstream).sum(), variables, allow_unused=True)
                for value, expected, p in zip(a, b, variables):
                    value = torch.zeros_like(p) if value is None else value
                    expected = torch.zeros_like(p) if expected is None else expected
                    torch.testing.assert_close(value, expected, rtol=1e-10, atol=1e-12)

    def test_gate_down_once_and_only_selected_experts_continue(self):
        layer = SparseAoE(config()).double()
        x = torch.ones(1, 1, 8, dtype=torch.float64, requires_grad=True)
        with torch.no_grad():
            for i, expert in enumerate(layer.experts):
                expert.gate_down_proj.weight.fill_(float(i+1) / 8)
        down_calls, continuation_tokens = [0]*4, [0]*4
        def count_down(i):
            def hook(module, args):
                down_calls[i] += 1
            return hook
        def count_continuation(i):
            def hook(module, args):
                continuation_tokens[i] += args[0].shape[0]
            return hook
        handles = []
        for i, expert in enumerate(layer.experts):
            handles += [expert.gate_down_proj.register_forward_pre_hook(count_down(i)),
                        expert.gate_up_proj.register_forward_pre_hook(count_continuation(i))]
        try:
            y = layer(x)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(down_calls, [1, 1, 1, 1])
        self.assertEqual(continuation_tokens, [0, 0, 1, 1])
        y.sum().backward()
        for expert in layer.experts[:2]:
            self.assertIsNotNone(expert.gate_down_proj.weight.grad)
            self.assertIsNone(expert.gate_up_proj.weight.grad)
            self.assertIsNone(expert.up_proj.weight.grad)
            self.assertIsNone(expert.down_proj.weight.grad)

    def test_zero_input_and_noncontiguous_input(self):
        layer = SparseAoE(config()).double()
        zeros = torch.zeros(1, 1, 8, dtype=torch.float64, requires_grad=True)
        y = layer(zeros)
        y.sum().backward()
        self.assertTrue(torch.isfinite(zeros.grad).all().item())
        x = torch.randn(2, 8, 5, dtype=torch.float64).transpose(1, 2)
        torch.testing.assert_close(layer(x), dense_reference(layer, x))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class AoECUDATests(unittest.TestCase):
    def test_bf16_forward_backward(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("CUDA device does not support BF16")
        torch.manual_seed(123)
        for explicit in (False, True):
            with self.subTest(explicit_bf16=explicit):
                dtype = torch.bfloat16 if explicit else torch.float32
                layer = SparseAoE(AoEConfig()).to(device="cuda", dtype=dtype)
                x = torch.randn(2, 16, 512, device="cuda", dtype=dtype, requires_grad=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not explicit):
                    y, diagnostic = layer(x, return_diagnostics=True)
                self.assertEqual(y.dtype, x.dtype)
                self.assertEqual(diagnostic.scores.dtype, torch.float32)
                self.assertEqual(diagnostic.probabilities.dtype, torch.float32)
                y.float().square().mean().backward()
                self.assertTrue(torch.isfinite(x.grad).all().item())
                for p in layer.parameters():
                    if p.grad is not None:
                        self.assertTrue(torch.isfinite(p.grad).all().item())


if __name__ == "__main__":
    unittest.main()
