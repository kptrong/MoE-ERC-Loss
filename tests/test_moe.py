"""Sparse dispatch versus a test-only dense reference, autograd and diagnostics."""

from dataclasses import fields, replace
import unittest

import torch

from moe_erc import ModelConfig, count_parameters
from moe_erc.moe import SparseMoE


def small_config():
    return ModelConfig(d_model=8, n_heads=2, n_experts=4, top_k=2, expert_hidden_dim=4)


def dense_reference(moe, x):
    """Test only: execute all experts, then gather selected full-softmax terms."""
    probabilities = (x @ moe.router.weight.T).softmax(-1)
    weights, indices = probabilities.topk(moe.config.top_k, dim=-1)
    all_outputs = torch.stack([expert(x) for expert in moe.experts], dim=-2)
    selected = all_outputs.gather(-2, indices.unsqueeze(-1).expand(*indices.shape, x.shape[-1]))
    return (selected * weights.unsqueeze(-1)).sum(-2)


def route_only_to_first_two(moe):
    with torch.no_grad():
        moe.router.weight.zero_()
        moe.router.weight[:, 0].copy_(torch.tensor([2., 1., -2., -3.]))


class SparseMoETests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_default_shape_budget_and_cpu_backward(self):
        config = ModelConfig()
        moe = SparseMoE(config)
        self.assertEqual(sum(p.numel() for p in moe.parameters()), 6_299_648)
        self.assertEqual(sum(p.numel() for p in moe.parameters()) * 6, 37_797_888)
        counts = count_parameters(config)
        self.assertEqual(sum(p.numel() for p in moe.parameters()) * config.n_layers,
                         counts.swiglu_experts + counts.router_weights)
        x = torch.randn(2, 128, 512, requires_grad=True)
        y, diagnostic = moe(x, return_diagnostics=True)
        self.assertEqual(y.shape, (2, 128, 512))
        self.assertEqual(y.device.type, "cpu")
        self.assertTrue(torch.isfinite(y).all().item())
        y.square().mean().backward()
        for value in (x, moe.router.weight):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all().item())
            self.assertGreater(value.grad.abs().sum().item(), 0)
        for index, expert in enumerate(moe.experts):
            if diagnostic.token_counts[index].item() > 0:
                for parameter in expert.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all().item())
                    self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_dense_reference_output_and_gradients(self):
        moe = SparseMoE(small_config()).double()
        x = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
        sparse = moe(x)
        dense = dense_reference(moe, x)
        torch.testing.assert_close(sparse, dense, rtol=1e-12, atol=1e-12)
        variables = (x, *moe.parameters())
        # Identical upstream signal verifies dispatch gradients, not just values.
        signal = torch.randn_like(sparse)
        sparse_grads = torch.autograd.grad((sparse * signal).sum(), variables, allow_unused=True)
        dense_grads = torch.autograd.grad((dense * signal).sum(), variables, allow_unused=True)
        for sparse_grad, dense_grad, variable in zip(sparse_grads, dense_grads, variables):
            if sparse_grad is None:
                sparse_grad = torch.zeros_like(variable)
            if dense_grad is None:
                dense_grad = torch.zeros_like(variable)
            torch.testing.assert_close(sparse_grad, dense_grad, rtol=1e-10, atol=1e-12)

    def test_sparse_calls_empty_experts_and_single_token(self):
        moe = SparseMoE(small_config())
        route_only_to_first_two(moe)
        calls = [0] * 4
        def record(index):
            def hook(module, args):
                calls[index] += args[0].shape[0]
            return hook
        handles = [expert.register_forward_pre_hook(record(i)) for i, expert in enumerate(moe.experts)]
        try:
            x = torch.ones(1, 1, 8, requires_grad=True)
            y, diagnostic = moe(x, return_diagnostics=True)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(y.shape, (1, 1, 8))
        self.assertEqual(calls, [1, 1, 0, 0])
        self.assertEqual(diagnostic.token_counts.tolist(), calls)
        y.square().sum().backward()
        for expert in moe.experts[:2]:
            for parameter in expert.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)
        for expert in moe.experts[2:]:
            self.assertTrue(all(p.grad is None for p in expert.parameters()))
        self.assertGreater(moe.router.weight.grad.abs().sum().item(), 0)

    def test_dispatch_counts_and_diagnostics(self):
        moe = SparseMoE(small_config())
        x = torch.randn(2, 7, 8)
        observed_counts = [0] * 4
        def record(index):
            def hook(module, args):
                observed_counts[index] += args[0].shape[0]
            return hook
        handles = [expert.register_forward_pre_hook(record(i)) for i, expert in enumerate(moe.experts)]
        try:
            output, diagnostic = moe(x, return_diagnostics=True)
        finally:
            for handle in handles:
                handle.remove()
        indices = diagnostic.topk_indices
        self.assertEqual(indices.shape, (2, 7, 2))
        self.assertTrue((indices[..., 0] != indices[..., 1]).all().item())
        self.assertEqual(sum(observed_counts), 2 * 7 * 2)
        self.assertEqual(diagnostic.token_counts.tolist(), observed_counts)
        torch.testing.assert_close(diagnostic.assignment_fractions,
                                   diagnostic.token_counts.float() / 28)
        torch.testing.assert_close(diagnostic.assignment_fractions.sum(), torch.tensor(1.))
        torch.testing.assert_close(diagnostic.mean_probabilities,
                                   moe.router(x).probabilities.mean((0, 1)))
        torch.testing.assert_close(diagnostic.mean_probabilities.sum(), torch.tensor(1.))
        for field in fields(diagnostic):
            self.assertFalse(getattr(diagnostic, field.name).requires_grad)
        torch.testing.assert_close(moe(x), output)
        self.assertIsInstance(moe(x), torch.Tensor)

    def test_erc_parameter_access_preserves_autograd(self):
        moe = SparseMoE(small_config())
        gates = moe.get_gate_weights()
        self.assertEqual(gates.shape, (4, 4, 8))
        self.assertEqual(moe.router.weight.shape, (4, 8))
        self.assertTrue(gates.requires_grad)
        for index, expert in enumerate(moe.experts):
            torch.testing.assert_close(gates[index], expert.gate_proj.weight)
        # Check graph connectivity only; this is not an ERC implementation.
        (gates.sum() + moe.router.weight.sum()).backward()
        torch.testing.assert_close(moe.router.weight.grad, torch.ones_like(moe.router.weight))
        for expert in moe.experts:
            torch.testing.assert_close(expert.gate_proj.weight.grad,
                                       torch.ones_like(expert.gate_proj.weight))

    def test_noncontiguous_inputs(self):
        moe = SparseMoE(small_config()).double()
        x = torch.randn(2, 8, 5, dtype=torch.float64).transpose(1, 2)
        self.assertFalse(x.is_contiguous())
        torch.testing.assert_close(moe(x), dense_reference(moe, x))

    def test_top_k_one_and_all_experts(self):
        for top_k in (1, 4):
            with self.subTest(top_k=top_k):
                moe = SparseMoE(replace(small_config(), top_k=top_k)).double()
                x = torch.randn(1, 3, 8, dtype=torch.float64)
                torch.testing.assert_close(moe(x), dense_reference(moe, x))

    def test_invalid_inputs(self):
        moe = SparseMoE(small_config())
        for shape in ((0, 2, 8), (1, 0, 8), (1, 2, 4), (2, 8)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    moe(torch.randn(*shape))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class MoECUDATests(unittest.TestCase):
    def test_bf16_forward_backward(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("CUDA device does not support BF16")
        torch.manual_seed(123)
        for explicit_bf16 in (False, True):
            with self.subTest(explicit_bf16=explicit_bf16):
                dtype = torch.bfloat16 if explicit_bf16 else torch.float32
                moe = SparseMoE(ModelConfig()).to(device="cuda", dtype=dtype)
                x = torch.randn(2, 32, 512, device="cuda", dtype=dtype, requires_grad=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not explicit_bf16):
                    y, diagnostic = moe(x, return_diagnostics=True)
                self.assertEqual(y.shape, x.shape)
                self.assertEqual(y.dtype, x.dtype)
                self.assertTrue(torch.isfinite(y).all().item())
                y.float().square().mean().backward()
                for value in (x, moe.router.weight):
                    self.assertIsNotNone(value.grad)
                    self.assertTrue(torch.isfinite(value.grad).all().item())
                    self.assertGreater(value.grad.abs().sum().item(), 0)
                for index, expert in enumerate(moe.experts):
                    if diagnostic.token_counts[index].item() > 0:
                        for parameter in expert.parameters():
                            self.assertIsNotNone(parameter.grad)
                            self.assertTrue(torch.isfinite(parameter.grad).all().item())


if __name__ == "__main__":
    unittest.main()
