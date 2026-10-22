"""Architecture selection, analytical counts and state_dict round trips."""

from io import BytesIO
from pathlib import Path
import tempfile
import unittest

import torch

from moe_erc import AoEConfig, ModelConfig, count_parameters
from moe_erc.model import AoELanguageModel, MoELanguageModel
from moe_erc.model_factory import create_model


class ModelFactoryTests(unittest.TestCase):
    def test_existing_data_pipeline_batch_into_aoe(self):
        from moe_erc.data_processing import PreprocessingConfig, prepare_dataset
        from moe_erc.dataset import PackedTokenDataset, make_dataloader
        from tests.data_fixtures import fixture
        torch.manual_seed(123)
        with tempfile.TemporaryDirectory() as directory:
            split, sources, tokenizer = fixture(directory)
            root = Path(directory) / "packed"
            prepare_dataset(sources, tokenizer, root, allow_test_tokenizer=True,
                            config=PreprocessingConfig(1024, 512, split=split))
            dataset = PackedTokenDataset(root, "train", tokenizer=tokenizer)
            try:
                batch = next(iter(make_dataloader(dataset, batch_size=1)))
                config = AoEConfig(vocab_size=tokenizer.vocab_size, d_model=16, n_heads=4,
                                   n_layers=2, n_experts=4, expert_hidden_dim=8, aoe_rank=5)
                tokenizer.validate_model(config, allow_test=True)
                model = create_model("aoe", config)
                result = model(**batch)
                self.assertEqual(result.logits.shape, (1, 512, tokenizer.vocab_size))
                self.assertTrue(torch.isfinite(result.loss).item())
                result.loss.backward()
                self.assertIsNotNone(model.token_embedding.weight.grad)
                self.assertTrue(torch.isfinite(model.token_embedding.weight.grad).all().item())
            finally:
                dataset.close()

    def test_model_selection_and_state_dict_round_trip(self):
        torch.manual_seed(123)
        base = dict(vocab_size=41, d_model=16, n_layers=2, n_heads=4, n_experts=4, expert_hidden_dim=8)
        ids = torch.randint(41, (2, 7))
        for kind, cls, config in (("vanilla_moe", MoELanguageModel, ModelConfig(**base)),
                                  ("aoe", AoELanguageModel, AoEConfig(**base, aoe_rank=5))):
            with self.subTest(model_type=kind):
                model = create_model(kind, config).eval()
                self.assertIs(type(model), cls)
                self.assertEqual(sum(p.numel() for p in model.parameters()), count_parameters(config, kind).total)
                result = model(ids, labels=ids)
                buffer = BytesIO()
                torch.save(model.state_dict(), buffer)
                buffer.seek(0)
                restored = create_model(kind, config).eval()
                restored.load_state_dict(torch.load(buffer, weights_only=True))
                self.assertIs(restored.lm_head.weight, restored.token_embedding.weight)
                torch.testing.assert_close(restored(ids), result.logits, rtol=0, atol=0)
                result.loss.backward()
                self.assertIsNotNone(model.token_embedding.weight.grad)

    def test_conversion_and_unknown_architecture(self):
        base = ModelConfig(vocab_size=41, d_model=16, n_layers=1, n_heads=4, n_experts=4, expert_hidden_dim=8)
        model = create_model("aoe", base)
        self.assertIsInstance(model.config, AoEConfig)
        self.assertEqual(model.config.aoe_rank, 171)
        for field, value in base.to_dict().items():
            self.assertEqual(getattr(model.config, field), value)
        for invalid in ("erc", "unknown"):
            with self.assertRaises(ValueError):
                create_model(invalid, base)
            with self.assertRaises(ValueError):
                count_parameters(base, invalid)


if __name__ == "__main__":
    unittest.main()
