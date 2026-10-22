"""Memmap, unshifted labels, reproducible workers and offline model integration."""

from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import torch

from moe_erc.data_processing import PreprocessingConfig, prepare_dataset
from moe_erc.data_smoke import run_smoke
from moe_erc.dataset import PackedTokenDataset, make_dataloader
from moe_erc.tokenizer_utils import train_tokenizer
from tests.data_fixtures import fixture


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.split, self.sources, self.tokenizer = fixture(self.root)
        prepare_dataset(self.sources, self.tokenizer, self.root / "packed", allow_test_tokenizer=True,
                        config=PreprocessingConfig(8192, 4096, split=self.split))
        self.datasets = []

    def tearDown(self):
        for dataset in self.datasets:
            dataset.close()
        self.temporary.cleanup()

    def dataset(self, split="train"):
        dataset = PackedTokenDataset(self.root / "packed", split, tokenizer=self.tokenizer,
                                     verify_checksums=True)
        self.datasets.append(dataset)
        return dataset

    def test_sample_memmap_dtype_unshifted_independent_labels(self):
        dataset = self.dataset()
        sample = dataset[0]
        self.assertEqual(set(sample), {"input_ids", "labels"})
        self.assertEqual(sample["input_ids"].shape, (512,))
        self.assertEqual(sample["input_ids"].dtype, torch.long)
        self.assertEqual(sample["labels"].dtype, torch.long)
        self.assertTrue(torch.equal(sample["input_ids"], sample["labels"]))
        self.assertTrue(dataset._maps)
        saved = sample["input_ids"].clone()
        sample["labels"][0] = -100
        sample["input_ids"][1] = 0
        self.assertTrue(torch.equal(dataset[0]["input_ids"], saved))
        self.assertTrue(torch.equal(dataset[-1]["input_ids"], dataset[len(dataset)-1]["input_ids"]))
        with self.assertRaises(IndexError):
            dataset[len(dataset)]

    def test_seeded_training_and_ordered_validation_loader(self):
        train = self.dataset()
        first = list(make_dataloader(train, batch_size=2, seed=12))
        second = list(make_dataloader(train, batch_size=2, seed=12))
        self.assertEqual(first[0]["input_ids"].shape, (2, 512))
        for a, b in zip(first, second):
            self.assertTrue(torch.equal(a["input_ids"], b["input_ids"]))
        validation = self.dataset("validation")
        batches = list(make_dataloader(validation, batch_size=2))
        collected = torch.cat([b["input_ids"] for b in batches])
        ordered = torch.stack([validation[i]["input_ids"] for i in range(len(validation))])
        self.assertTrue(torch.equal(collected, ordered))

    def test_worker_pickle_and_two_worker_loading(self):
        dataset = self.dataset()
        dataset[0]
        restored = pickle.loads(pickle.dumps(dataset))
        self.datasets.append(restored)
        self.assertFalse(restored._maps)
        self.assertTrue(torch.equal(restored[0]["input_ids"], dataset[0]["input_ids"]))
        batches = list(make_dataloader(dataset, batch_size=2, seed=12, num_workers=2))
        reference = list(make_dataloader(dataset, batch_size=2, seed=12, num_workers=0))
        self.assertEqual(len(batches), len(reference))
        for a, b in zip(batches, reference):
            self.assertTrue(torch.equal(a["input_ids"], b["input_ids"]))

    def test_tokenizer_metadata_mismatch_rejected(self):
        other = train_tokenizer(self.sources, self.root / "other.json", split=self.split,
                                vocab_size=280, test_only=True)
        with self.assertRaisesRegex(ValueError, "tokenizer mismatch"):
            PackedTokenDataset(self.root / "packed", "train", tokenizer=other)

    def test_offline_end_to_end_smoke(self):
        with patch("moe_erc.dolma_reader.urlopen", side_effect=AssertionError("network forbidden")):
            report = run_smoke(self.root / "smoke")
        self.assertEqual(report["batch_shapes"], {"input_ids": [2, 512], "labels": [2, 512]})
        self.assertEqual(report["logits_shape"], [2, 512, report["vocab_size"]])
        self.assertTrue(report["labels_equal_inputs"])
        self.assertEqual(report["backward"], "PASS")
        self.assertLess(report["token_range"][1], report["vocab_size"])


if __name__ == "__main__":
    unittest.main()
