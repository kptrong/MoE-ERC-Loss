"""Independent component budgets and architecture-sensitivity checks."""

from dataclasses import replace
import unittest

from moe_erc import ModelConfig, ParameterCounts, count_parameters


class ParameterCountTests(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig()
        self.counts = count_parameters(self.config)

    def test_default_components(self):
        self.assertEqual(self.counts, ParameterCounts(
            token_embeddings=8_192_000,
            attention_projections=6_291_456,
            swiglu_experts=37_748_736,
            router_weights=49_152,
            normalization=6_656,
        ))

    def test_exact_total(self):
        self.assertEqual(self.counts.total, 52_288_000)

    def test_tied_embedding_is_counted_once(self):
        changed = count_parameters(replace(self.config, vocab_size=16001))
        self.assertEqual(changed.total - self.counts.total, 512)

    def test_top_k_does_not_change_total_parameters(self):
        for k in (1, 4, 16):
            self.assertEqual(count_parameters(replace(self.config, top_k=k)), self.counts)

    def test_context_length_and_heads_do_not_add_parameters(self):
        changed = replace(self.config, max_seq_len=1024, n_heads=16)
        self.assertEqual(count_parameters(changed), self.counts)

    def test_independent_small_architecture(self):
        config = ModelConfig(vocab_size=10, d_model=8, n_layers=1,
                             n_heads=2, n_experts=3, top_k=1, expert_hidden_dim=4)
        # Embedding=80; Q/K/V/O=4*64; experts=3*(32+32+32);
        # router=24; three scale-only norms=24. No hard-coded target in utility.
        self.assertEqual(count_parameters(config), ParameterCounts(80, 256, 288, 24, 24))
        self.assertEqual(count_parameters(config).total, 672)

    def test_additional_layer_includes_two_norms_and_own_router(self):
        changed = count_parameters(replace(self.config, n_layers=7))
        self.assertEqual(changed.total - self.counts.total, 7_349_248)


if __name__ == "__main__":
    unittest.main()
