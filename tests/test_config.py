"""Configuration contract and invalid-input tests (standard-library unittest)."""

from dataclasses import FrozenInstanceError, replace
import unittest

from moe_erc import ModelConfig


class ConfigTests(unittest.TestCase):
    def test_architecture_defaults(self):
        config = ModelConfig()
        expected = {
            "vocab_size": 16000, "d_model": 512, "n_layers": 6,
            "n_heads": 8, "n_experts": 16, "top_k": 2,
            "expert_hidden_dim": 256, "max_seq_len": 512,
            "dropout": 0.0, "linear_bias": False, "tie_embeddings": True,
            "attention_type": "multi_head_causal", "positional_embedding": "rope",
            "normalization": "rmsnorm", "expert_activation": "swiglu",
            "shared_experts": 0, "rope_theta": 10000.0, "rms_norm_eps": 1e-6,
        }
        self.assertEqual(config.to_dict(), expected)
        self.assertEqual(config.head_dim, 64)

    def test_positive_integer_fields(self):
        for name in ("vocab_size", "d_model", "n_layers", "n_heads",
                     "n_experts", "top_k", "expert_hidden_dim", "max_seq_len"):
            for value in (0, -1, 1.5, True, "512"):
                with self.subTest(field=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        replace(ModelConfig(), **{name: value})

    def test_vocabulary_minimum(self):
        with self.assertRaisesRegex(ValueError, "vocab_size"):
            ModelConfig(vocab_size=1)

    def test_attention_divisibility(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            ModelConfig(d_model=510)

    def test_rope_head_dimension(self):
        with self.assertRaisesRegex(ValueError, "even head_dim"):
            ModelConfig(d_model=24, n_heads=8)

    def test_top_k_limit_and_boundary(self):
        with self.assertRaisesRegex(ValueError, "top_k"):
            ModelConfig(top_k=17)
        self.assertEqual(ModelConfig(top_k=16).top_k, 16)

    def test_dropout_range(self):
        for value in (-0.1, 1.0, float("nan"), float("inf"), True, "0"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "dropout"):
                    ModelConfig(dropout=value)
        self.assertEqual(ModelConfig(dropout=0.5).dropout, 0.5)

    def test_numerical_settings(self):
        for name in ("rope_theta", "rms_norm_eps"):
            for value in (0, -1, float("nan"), float("inf"), True, "1"):
                with self.subTest(field=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        replace(ModelConfig(), **{name: value})

    def test_unsupported_architectures_are_rejected(self):
        for name, value in {
            "linear_bias": True, "tie_embeddings": False, "shared_experts": 1,
            "attention_type": "gqa", "positional_embedding": "learned",
            "normalization": "layernorm", "expert_activation": "gelu",
        }.items():
            with self.subTest(field=name):
                with self.assertRaisesRegex(ValueError, name):
                    replace(ModelConfig(), **{name: value})

    def test_config_is_immutable(self):
        with self.assertRaises(FrozenInstanceError):
            ModelConfig().d_model = 1024


if __name__ == "__main__":
    unittest.main()
