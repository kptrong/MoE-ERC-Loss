"""Numerical, causal, autograd and budget checks for STEP 2.

PyTorch is required: missing dependencies are reported as import errors, not
silently skipped. Only hardware-dependent CUDA tests may be skipped.
"""

import math
from dataclasses import replace
import unittest

import torch

from moe_erc import ModelConfig, count_parameters
from moe_erc.attention import CausalSelfAttention
from moe_erc.rmsnorm import RMSNorm
from moe_erc.rope import RotaryPositionalEmbedding


def assert_finite_backward(test, module, x, output):
    output.float().square().mean().backward()
    test.assertIsNotNone(x.grad)
    test.assertTrue(torch.isfinite(x.grad).all().item())
    for name, parameter in module.named_parameters():
        with test.subTest(parameter=name):
            test.assertIsNotNone(parameter.grad)
            test.assertTrue(torch.isfinite(parameter.grad).all().item())
            test.assertGreater(parameter.grad.abs().sum().item(), 0)


class RMSNormTests(unittest.TestCase):
    def test_shape_dtype_and_parameter_count(self):
        norm = RMSNorm(512)
        for dtype in (torch.float32, torch.float64, torch.bfloat16, torch.float16):
            with self.subTest(dtype=dtype):
                x = torch.randn(2, 7, 512).to(dtype)
                y = norm(x)
                self.assertEqual(y.shape, x.shape)
                self.assertEqual(y.dtype, x.dtype)
                self.assertTrue(torch.isfinite(y).all().item())
        self.assertEqual(sum(p.numel() for p in norm.parameters()), 512)
        self.assertEqual(set(dict(norm.named_parameters())), {"weight"})

    def test_independent_scalar_reference(self):
        norm = RMSNorm(4, eps=0.25).double()
        values = [[1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 0.0, 0.0]]
        scales = [0.5, 1.0, 1.5, 2.0]
        with torch.no_grad():
            norm.weight.copy_(torch.tensor(scales))
        expected = [[v * s / math.sqrt(sum(z*z for z in row) / 4 + 0.25)
                     for v, s in zip(row, scales)] for row in values]
        torch.testing.assert_close(norm(torch.tensor(values, dtype=torch.float64)),
                                   torch.tensor(expected, dtype=torch.float64), rtol=1e-12, atol=1e-12)

    def test_bf16_large_values_use_fp32_and_backward(self):
        norm = RMSNorm(4)
        x = torch.tensor([[300.0, -400.0, 500.0, -600.0]], dtype=torch.bfloat16, requires_grad=True)
        reference = (x.float() / torch.sqrt(x.float().square().mean(-1, keepdim=True) + norm.eps)).bfloat16()
        y = norm(x)
        torch.testing.assert_close(y, reference, rtol=0, atol=0)
        assert_finite_backward(self, norm, x, y)

    def test_validation(self):
        for kwargs in ({"d_model": 0}, {"d_model": 4, "eps": 0}):
            with self.assertRaises(ValueError):
                RMSNorm(**kwargs)
        with self.assertRaises(ValueError):
            RMSNorm(4)(torch.ones(2, 3))
        with self.assertRaises(TypeError):
            RMSNorm(4)(torch.ones(2, 4, dtype=torch.int64))


class RoPETests(unittest.TestCase):
    def test_shape_position_zero_and_no_parameters(self):
        rope = RotaryPositionalEmbedding(64)
        q, k = torch.randn(2, 8, 512, 64), torch.randn(2, 8, 512, 64)
        rq, rk = rope(q, k)
        self.assertEqual(rq.shape, q.shape)
        self.assertEqual(rk.shape, k.shape)
        torch.testing.assert_close(rq[:, :, 0], q[:, :, 0], rtol=0, atol=0)
        torch.testing.assert_close(rk[:, :, 0], k[:, :, 0], rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in rope.parameters()), 0)
        self.assertEqual(len(rope.state_dict()), 0)

    def test_independent_rotation_at_nonzero_positions(self):
        rope = RotaryPositionalEmbedding(4, base=10000.0)
        q = torch.tensor([[[[1., 2., 3., 4.]] * 3]], dtype=torch.float64)
        k = -2 * q
        expected = []
        for position in range(3):
            row = []
            for pair, (even, odd) in enumerate(((1., 2.), (3., 4.))):
                angle = position * 10000.0 ** (-2 * pair / 4)
                row.extend([even * math.cos(angle) - odd * math.sin(angle),
                            even * math.sin(angle) + odd * math.cos(angle)])
            expected.append(row)
        rq, rk = rope(q, k)
        reference = torch.tensor([[expected]], dtype=torch.float64)
        torch.testing.assert_close(rq, reference, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(rk, -2 * reference, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(rq.norm(dim=-1), q.norm(dim=-1))

    def test_dtype_and_gradients(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                q = torch.randn(1, 2, 16, 64).to(dtype).requires_grad_()
                k = torch.randn_like(q, requires_grad=True)
                rq, rk = RotaryPositionalEmbedding(64).to(dtype=dtype)(q, k)
                self.assertEqual(rq.dtype, dtype)
                self.assertEqual(rk.dtype, dtype)
                (rq.float().square().mean() + rk.float().square().mean()).backward()
                for tensor in (q, k):
                    self.assertIsNotNone(tensor.grad)
                    self.assertTrue(torch.isfinite(tensor.grad).all().item())

    def test_validation(self):
        for kwargs in ({"head_dim": 3}, {"head_dim": 64, "max_seq_len": 0},
                       {"head_dim": 64, "base": 0}):
            with self.assertRaises(ValueError):
                RotaryPositionalEmbedding(**kwargs)
        rope = RotaryPositionalEmbedding(64)
        for length in (0, 513):
            q = torch.randn(1, 8, length, 64)
            with self.assertRaises(ValueError):
                rope(q, q)
        with self.assertRaises(ValueError):
            rope(torch.randn(1, 8, 2, 64), torch.randn(1, 8, 3, 64))


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = ModelConfig()
        self.attention = CausalSelfAttention(self.config)

    def test_required_shape_and_finite_backward_on_cpu(self):
        x = torch.randn(2, 128, 512, requires_grad=True)
        y = self.attention(x)
        self.assertEqual(y.shape, (2, 128, 512))
        self.assertEqual(y.device.type, "cpu")
        self.assertEqual(y.dtype, x.dtype)
        self.assertTrue(torch.isfinite(y).all().item())
        assert_finite_backward(self, self.attention, x, y)

    def test_parameter_budget_and_no_bias(self):
        self.assertEqual(sum(p.numel() for p in self.attention.parameters()), 1_048_576)
        self.assertEqual(sum(p.numel() for p in self.attention.parameters()) * self.config.n_layers,
                         count_parameters(self.config).attention_projections)
        self.assertEqual(set(dict(self.attention.named_parameters())),
                         {"q_proj.weight", "k_proj.weight", "v_proj.weight", "out_proj.weight"})

    def test_future_token_changes_cannot_change_prefix(self):
        self.attention.eval()
        x = torch.randn(2, 32, 512)
        changed = x.clone()
        changed[:, 11:] = torch.randn_like(changed[:, 11:]) * 5
        with torch.no_grad():
            original_output = self.attention(x)
            changed_output = self.attention(changed)
            prefix_only_output = self.attention(x[:, :11])
        torch.testing.assert_close(original_output[:, :11], changed_output[:, :11], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(original_output[:, :11], prefix_only_output, rtol=1e-5, atol=1e-6)
        self.assertGreater((original_output[:, 11:] - changed_output[:, 11:]).abs().max().item(), 1e-3)

    def test_independent_attention_reference_including_rope(self):
        # Tiny double-precision reference catches axis/scaling errors, omitted
        # RoPE, or accidental rotation of V. Production still uses only SDPA.
        config = replace(self.config, d_model=8, n_heads=2)
        attention = CausalSelfAttention(config).double().eval()
        x = torch.randn(1, 3, 8, dtype=torch.float64)
        q = (x @ attention.q_proj.weight.T).reshape(1, 3, 2, 4).transpose(1, 2)
        k = (x @ attention.k_proj.weight.T).reshape(1, 3, 2, 4).transpose(1, 2)
        v = (x @ attention.v_proj.weight.T).reshape(1, 3, 2, 4).transpose(1, 2)
        def rotate_reference(value):
            result = torch.empty_like(value)
            for position in range(3):
                for pair in range(2):
                    angle = position * config.rope_theta ** (-2 * pair / 4)
                    even, odd = value[:, :, position, 2*pair], value[:, :, position, 2*pair+1]
                    result[:, :, position, 2*pair] = even * math.cos(angle) - odd * math.sin(angle)
                    result[:, :, position, 2*pair+1] = even * math.sin(angle) + odd * math.cos(angle)
            return result
        logits = rotate_reference(q) @ rotate_reference(k).transpose(-2, -1) / 2
        logits = logits.masked_fill(torch.ones(3, 3, dtype=torch.bool).triu(1), -float("inf"))
        expected = (logits.softmax(-1) @ v).transpose(1, 2).reshape(1, 3, 8) @ attention.out_proj.weight.T
        torch.testing.assert_close(attention(x), expected, rtol=1e-12, atol=1e-12)

    def test_sequence_boundaries_and_invalid_input(self):
        with torch.no_grad():
            for length in (1, 512):
                self.assertEqual(self.attention(torch.randn(1, length, 512)).shape, (1, length, 512))
        for shape in ((1, 0, 512), (1, 513, 512), (1, 3, 256), (3, 512)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    self.attention(torch.randn(*shape))

    def test_eval_disables_attention_dropout(self):
        attention = CausalSelfAttention(replace(self.config, dropout=0.5)).eval()
        x = torch.randn(1, 8, 512)
        with torch.no_grad():
            torch.testing.assert_close(attention(x), attention(x), rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class CUDABFloat16Tests(unittest.TestCase):
    def setUp(self):
        if not torch.cuda.is_bf16_supported():
            self.skipTest("CUDA device does not support BF16")
        torch.manual_seed(42)

    def test_autocast_forward_backward(self):
        config = ModelConfig()
        norm = RMSNorm(config.d_model, config.rms_norm_eps).cuda()
        attention = CausalSelfAttention(config).cuda()
        x = torch.randn(2, 128, 512, device="cuda", requires_grad=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            normalized = norm(x)
            self.assertEqual(normalized.dtype, x.dtype)
            y = attention(normalized)
            self.assertEqual(y.dtype, torch.bfloat16)
        assert_finite_backward(self, attention, x, y)
        self.assertIsNotNone(norm.weight.grad)
        self.assertTrue(torch.isfinite(norm.weight.grad).all().item())

    def test_explicit_bf16_modules_and_causality(self):
        norm = RMSNorm(512).to(device="cuda", dtype=torch.bfloat16)
        attention = CausalSelfAttention(ModelConfig()).to(device="cuda", dtype=torch.bfloat16)
        x = torch.randn(2, 128, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        normalized = norm(x)
        self.assertEqual(normalized.dtype, x.dtype)
        y = attention(normalized)
        self.assertEqual(y.dtype, x.dtype)
        assert_finite_backward(self, attention, x, y)
        self.assertIsNotNone(norm.weight.grad)
        self.assertTrue(torch.isfinite(norm.weight.grad).all().item())
        attention.eval()
        changed = x.detach().clone()
        changed[:, 16:] = torch.randn_like(changed[:, 16:]) * 5
        with torch.no_grad():
            original, modified = attention(x.detach()), attention(changed)
        torch.testing.assert_close(original[:, :16], modified[:, :16], rtol=1e-2, atol=1e-2)
        self.assertGreater((original[:, 16:] - modified[:, 16:]).abs().max().item(), 0.01)


if __name__ == "__main__":
    unittest.main()
