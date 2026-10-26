"""Independent mathematical references and stop-gradient regression for ERC."""

import unittest

import torch
from torch.nn import functional as F

from moe_erc import ModelConfig
from moe_erc.erc_loss import (
    compute_activation_matrix, compute_erc_loss, erc_loss_from_matrix,
    generate_router_proxies, mean_erc_loss,
)
from moe_erc.moe import SparseMoE


def reference_matrix(proxy, gates):
    return torch.stack([
        torch.stack([torch.linalg.vector_norm(gate @ token) for gate in gates])
        for token in proxy
    ])


def reference_loss(matrix, alpha):
    # Literal Equation (3), including its M[j,i] column term.
    terms = []
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if i != j:
                terms.extend((
                    (matrix[i, j] - alpha * matrix[i, i]).clamp_min(0),
                    (matrix[j, i] - alpha * matrix[i, i]).clamp_min(0),
                ))
    return torch.stack(terms).sum() / len(matrix) ** 2


class ProxyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_shape_distances_epsilon_and_no_graph(self):
        router = torch.tensor([[3., 0.], [0., 4.], [3., 4.]], requires_grad=True)
        proxy, epsilon = generate_router_proxies(router)
        self.assertEqual(proxy.shape, (3, 2))
        self.assertEqual(epsilon.shape, (3,))
        self.assertEqual(proxy.dtype, torch.float32)
        self.assertEqual(proxy.device, router.device)
        torch.testing.assert_close(epsilon, torch.tensor([4/6, 3/8, 3/10]))
        self.assertFalse(proxy.requires_grad)
        self.assertFalse(epsilon.requires_grad)
        self.assertIsNone(proxy.grad_fn)
        self.assertIsNone(epsilon.grad_fn)
        self.assertIsNone(router.grad)

    def test_uniform_formula_bounds_and_controlled_rng(self):
        router = torch.tensor([[1., -2., 3.], [-3., 4., 2.], [2., 1., -4.]])
        seed = 100
        uniform = torch.rand(router.shape, generator=torch.Generator().manual_seed(seed))
        # FP64 loop reference for distances/norms; diagonal is never a candidate.
        r = router.double()
        eps64 = torch.stack([
            min(torch.linalg.vector_norm(r[i] - r[j]) for j in range(3) if j != i)
            / (2 * torch.linalg.vector_norm(r[i])) for i in range(3)
        ])
        torch.manual_seed(seed)
        proxy, epsilon = generate_router_proxies(router)
        expected = r * (1 - eps64[:, None] + 2 * eps64[:, None] * uniform.double())
        torch.testing.assert_close(epsilon.double(), eps64, rtol=2e-6, atol=1e-7)
        torch.testing.assert_close(proxy.double(), expected, rtol=2e-6, atol=5e-7)
        delta = proxy / router
        self.assertTrue((delta >= 1 - epsilon[:, None] - 1e-6).all())
        self.assertTrue((delta <= 1 + epsilon[:, None] + 1e-6).all())
        # The coordinate-wise bound implies this Euclidean perturbation bound.
        self.assertTrue((torch.linalg.vector_norm(proxy - router, dim=1)
                         <= epsilon * torch.linalg.vector_norm(router, dim=1) + 1e-6).all())
        torch.manual_seed(seed)
        again, again_eps = generate_router_proxies(router)
        torch.testing.assert_close(proxy, again, rtol=0, atol=0)
        torch.testing.assert_close(epsilon, again_eps, rtol=0, atol=0)
        torch.manual_seed(seed + 1)
        other, _ = generate_router_proxies(router)
        self.assertFalse(torch.equal(proxy, other))

    def test_duplicate_nonzero_rows_and_no_epsilon_clipping(self):
        router = torch.tensor([[1., 2.], [1., 2.], [3., 4.]])
        proxy, epsilon = generate_router_proxies(router)
        torch.testing.assert_close(epsilon[:2], torch.zeros(2), rtol=0, atol=0)
        torch.testing.assert_close(proxy[:2], router[:2], rtol=0, atol=0)
        _, large = generate_router_proxies(torch.tensor([[0.1, 0.], [10., 0.]]))
        self.assertGreater(large[0].item(), 1)
        self.assertAlmostEqual(large[0].item(), 49.5, places=4)

    def test_dtype_inputs_compute_fp32(self):
        for dtype in (torch.float64, torch.float32, torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                proxy, epsilon = generate_router_proxies(torch.tensor([[1., 2.], [3., 4.]], dtype=dtype))
                self.assertEqual(proxy.dtype, torch.float32)
                self.assertEqual(epsilon.dtype, torch.float32)
                self.assertTrue(torch.isfinite(proxy).all())

    def test_invalid_router_inputs(self):
        for router in (torch.ones(1, 2), torch.ones(0, 2), torch.ones(2, 0),
                       torch.ones(2), torch.ones(2, 2, 1), torch.tensor([[0., 0.], [1., 2.]])):
            with self.subTest(shape=router.shape):
                with self.assertRaises(ValueError):
                    generate_router_proxies(router)
        for bad in (float("inf"), float("nan"), -float("inf")):
            with self.assertRaisesRegex(ValueError, "finite"):
                generate_router_proxies(torch.tensor([[1., bad], [2., 3.]]))
        with self.assertRaisesRegex(ValueError, "zero router-row"):
            generate_router_proxies(torch.zeros(2, 3))
        for bad in (torch.ones(2, 2, dtype=torch.long), torch.ones(2, 2, dtype=torch.complex64), None):
            with self.assertRaises(TypeError):
                generate_router_proxies(bad)
        with self.assertRaisesRegex(ValueError, "FP32"):
            generate_router_proxies(torch.full((2, 2), 1e100, dtype=torch.float64))


class MatrixAndHingeTests(unittest.TestCase):
    def test_orientation_and_pre_silu_projection(self):
        proxy = torch.tensor([[1., 0.], [0., 2.]])
        gates = torch.tensor([[[-3., 4.]], [[2., -1.]]], requires_grad=True)
        matrix = compute_activation_matrix(proxy, gates)
        torch.testing.assert_close(matrix, torch.tensor([[3., 2.], [8., 2.]]), rtol=0, atol=0)
        self.assertEqual(matrix.shape, (2, 2))
        self.assertTrue(matrix.requires_grad)
        after_silu = F.silu(torch.einsum("jDd,id->ijD", gates, proxy)).norm(dim=-1)
        self.assertFalse(torch.allclose(matrix, after_silu))

    def test_loop_values_and_gradients_with_fixed_proxy(self):
        torch.manual_seed(72)
        proxy = torch.randn(4, 5)
        gates = torch.randn(4, 3, 5, requires_grad=True)
        matrix = compute_activation_matrix(proxy, gates)
        gates64 = gates.detach().double().requires_grad_()
        expected = reference_matrix(proxy.double(), gates64)
        torch.testing.assert_close(matrix.double(), expected, rtol=2e-6, atol=1e-6)
        for alpha in (0., 0.4, 1.):
            with self.subTest(alpha=alpha):
                actual_loss = erc_loss_from_matrix(matrix, alpha)
                expected_loss = reference_loss(expected, alpha)
                torch.testing.assert_close(actual_loss.double(), expected_loss, rtol=3e-6, atol=1e-6)
                actual_grad, = torch.autograd.grad(actual_loss, gates, retain_graph=True)
                expected_grad, = torch.autograd.grad(expected_loss, gates64, retain_graph=True)
                torch.testing.assert_close(actual_grad.double(), expected_grad, rtol=3e-5, atol=2e-6)

    def test_row_column_terms_n_squared_and_diagonal_mask(self):
        # Asymmetric diagonals distinguish the two broadcasting axes.
        matrix = torch.tensor([[2., 5.], [1., 3.]], requires_grad=True)
        # alpha=1: row sum=3, column sum=2, divided by 4 -> 1.25.
        self.assertEqual(erc_loss_from_matrix(matrix).item(), 1.25)
        # alpha=.5: off-diagonal row sum=4, column sum=3.5 -> 1.875.
        # Including diagonal hinges would give a different answer.
        self.assertEqual(erc_loss_from_matrix(matrix, 0.5).item(), 1.875)
        for alpha in (0., 0.5, 1.):
            reference = matrix.detach().clone().requires_grad_()
            actual_loss = erc_loss_from_matrix(matrix, alpha)
            expected_loss = reference_loss(reference, alpha)
            torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
            actual_grad, = torch.autograd.grad(actual_loss, matrix)
            expected_grad, = torch.autograd.grad(expected_loss, reference)
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)

    def test_satisfied_and_boundary_hinges_allow_zero_gradients(self):
        for matrix in (torch.tensor([[4., 1.], [1., 5.]]), torch.tensor([[5., 0.], [0., 4.]])):
            matrix.requires_grad_()
            loss = erc_loss_from_matrix(matrix, 0.5)
            self.assertEqual(loss.item(), 0)
            loss.backward()
            torch.testing.assert_close(matrix.grad, torch.zeros_like(matrix), rtol=0, atol=0)

    def test_boundary_subgradient_matches_figure8_clamp(self):
        matrix = torch.zeros(2, 2, requires_grad=True)
        loss = erc_loss_from_matrix(matrix, 0.5)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        # A zero hinge value does not imply zero clamp subgradient at its kink.
        reference = matrix.detach().clone().requires_grad_()
        reference_loss(reference, 0.5).backward()
        torch.testing.assert_close(matrix.grad, reference.grad, rtol=0, atol=0)

    def test_zero_gate_norm_has_finite_zero_gradient(self):
        proxy = torch.tensor([[1., 2.], [3., 4.]])
        gates = torch.zeros(2, 3, 2, requires_grad=True)
        loss = erc_loss_from_matrix(compute_activation_matrix(proxy, gates))
        loss.backward()
        self.assertEqual(loss.item(), 0)
        torch.testing.assert_close(gates.grad, torch.zeros_like(gates), rtol=0, atol=0)

    def test_alpha_validation_and_default(self):
        matrix = torch.tensor([[1., 3.], [2., 1.]])
        torch.testing.assert_close(erc_loss_from_matrix(matrix), erc_loss_from_matrix(matrix, 1.))
        for bad in (-0.1, 1.1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                erc_loss_from_matrix(matrix, bad)
        for bad in (True, "1", None, torch.tensor(1.)):
            with self.assertRaises(TypeError):
                erc_loss_from_matrix(matrix, bad)

    def test_invalid_activation_inputs_and_matrix(self):
        proxy, gates = torch.ones(2, 3), torch.ones(2, 4, 3)
        for bad in (torch.ones(1, 4, 3), torch.ones(2, 0, 3), torch.ones(2, 4, 2), torch.ones(2, 3)):
            with self.assertRaises(ValueError):
                compute_activation_matrix(proxy, bad)
        with self.assertRaises(ValueError):
            compute_activation_matrix(torch.ones(2, 3, 1), gates)
        for target in ("proxy", "gates"):
            p, g = proxy.clone(), gates.clone()
            (p if target == "proxy" else g).flatten()[0] = float("nan")
            with self.assertRaisesRegex(ValueError, "finite"):
                compute_activation_matrix(p, g)
        for bad in (torch.ones(1, 1), torch.ones(2, 3), torch.ones(2), -torch.ones(2, 2),
                    torch.full((2, 2), float("inf"))):
            with self.assertRaises(ValueError):
                erc_loss_from_matrix(bad)

    def test_fp32_under_cpu_autocast_and_float64_inputs(self):
        proxy = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64)
        gates = torch.ones(2, 3, 2, dtype=torch.float64, requires_grad=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            matrix = compute_activation_matrix(proxy, gates)
            loss = erc_loss_from_matrix(matrix)
        self.assertEqual(matrix.dtype, torch.float32)
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertTrue(torch.isfinite(gates.grad).all())


class LayerAndAggregationTests(unittest.TestCase):
    def test_one_layer_end_to_end_gradient_and_no_parameter_mutation(self):
        config = ModelConfig(d_model=4, n_heads=1, n_experts=2, expert_hidden_dim=2)
        moe = SparseMoE(config)
        with torch.no_grad():
            # Equal nonzero rows remove stochasticity without a noise override.
            moe.router.weight.copy_(torch.tensor([[1., 0., 0., 0.]]).expand(2, -1))
            moe.experts[0].gate_proj.weight.fill_(1.)
            moe.experts[1].gate_proj.weight.fill_(3.)
        before = {name: p.detach().clone() for name, p in moe.named_parameters()}
        parameter_ids = [id(p) for p in moe.parameters()]
        loss = compute_erc_loss(moe.router.weight, moe.get_gate_weights())
        self.assertGreater(loss.item(), 0)
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertIsNone(moe.router.weight.grad)
        for expert in moe.experts:
            self.assertIsNotNone(expert.gate_proj.weight.grad)
            self.assertTrue(torch.isfinite(expert.gate_proj.weight.grad).all())
            self.assertGreater(expert.gate_proj.weight.grad.abs().sum().item(), 0)
            self.assertIsNone(expert.up_proj.weight.grad)
            self.assertIsNone(expert.down_proj.weight.grad)
        self.assertEqual(parameter_ids, [id(p) for p in moe.parameters()])
        for name, parameter in moe.named_parameters():
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)

    def test_layer_composes_helpers_with_same_rng(self):
        torch.manual_seed(35)
        router = torch.randn(4, 8, requires_grad=True)
        gates = torch.randn(4, 5, 8, requires_grad=True)
        rng = torch.get_rng_state()
        actual = compute_erc_loss(router, gates, 0.7)
        actual_grad, = torch.autograd.grad(actual, gates)
        torch.set_rng_state(rng)
        proxy, _ = generate_router_proxies(router)
        ref_gates = gates.detach().double().requires_grad_()
        expected = reference_loss(reference_matrix(proxy.double(), ref_gates), 0.7)
        expected_grad, = torch.autograd.grad(expected, ref_gates)
        torch.testing.assert_close(actual.double(), expected, rtol=3e-6, atol=1e-6)
        torch.testing.assert_close(actual_grad.double(), expected_grad, rtol=3e-5, atol=2e-6)
        self.assertIsNone(router.grad)

    def test_invalid_layer_input_does_not_draw_noise(self):
        router, gates = torch.ones(2, 3), torch.ones(2, 4, 3)
        before = torch.get_rng_state().clone()
        for bad_gates, alpha in ((torch.ones(3, 4, 3), 1.), (gates, -1.)):
            with self.assertRaises(ValueError):
                compute_erc_loss(router, bad_gates, alpha)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_mean_aggregation_six_layers_preserves_gradient(self):
        layers = [torch.tensor(float(i), requires_grad=True) for i in range(1, 7)]
        loss = mean_erc_loss(iter(layers))
        self.assertEqual(loss.item(), 3.5)
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        for layer in layers:
            torch.testing.assert_close(layer.grad, torch.tensor(1/6))
        self.assertEqual(mean_erc_loss([torch.tensor(4.)]).item(), 4.)

    def test_invalid_aggregation(self):
        for losses in ([], [torch.ones(1)], [torch.tensor(float("nan"))]):
            with self.assertRaises(ValueError):
                mean_erc_loss(losses)
        with self.assertRaises(TypeError):
            mean_erc_loss([1.])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class ERCCUDATests(unittest.TestCase):
    def test_bf16_autocast_fp32_agreement_backward_and_rng_replay(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("CUDA BF16 unavailable")
        torch.manual_seed(42)
        router = torch.randn(16, 512, device="cuda", requires_grad=True)
        gates = torch.randn(16, 256, 512, device="cuda", requires_grad=True)
        rng = torch.cuda.get_rng_state().clone()
        baseline = compute_erc_loss(router, gates, 0.6)
        baseline_grad, = torch.autograd.grad(baseline, gates)
        torch.cuda.set_rng_state(rng)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = compute_erc_loss(router, gates, 0.6)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertEqual(loss.device, router.device)
        self.assertTrue(torch.isfinite(loss))
        grad, router_grad = torch.autograd.grad(loss, (gates, router), allow_unused=True)
        self.assertIsNone(router_grad)
        self.assertTrue(torch.isfinite(grad).all())
        torch.testing.assert_close(loss, baseline, rtol=0, atol=0)
        torch.testing.assert_close(grad, baseline_grad, rtol=0, atol=0)
        torch.cuda.synchronize()

    def test_explicit_bf16_inputs_and_device_mismatch_rejected(self):
        router = torch.tensor([[1., 2.], [3., 4.]], device="cuda", dtype=torch.bfloat16,
                              requires_grad=True)
        gates = torch.ones(2, 3, 2, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        loss = compute_erc_loss(router, gates, 0.5)
        loss.backward()
        self.assertEqual(loss.dtype, torch.float32)
        self.assertIsNone(router.grad)
        self.assertTrue(torch.isfinite(gates.grad).all())
        proxy, epsilon = generate_router_proxies(router)
        self.assertEqual(proxy.device, router.device)
        self.assertEqual(epsilon.device, router.device)
        with self.assertRaisesRegex(ValueError, "same device"):
            compute_activation_matrix(proxy.cpu(), gates)
        with self.assertRaisesRegex(ValueError, "same device"):
            compute_erc_loss(router.cpu(), gates)
        with self.assertRaisesRegex(ValueError, "same device"):
            mean_erc_loss([loss, loss.cpu()])


if __name__ == "__main__":
    unittest.main()
