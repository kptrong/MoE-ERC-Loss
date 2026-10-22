"""Training-only, frozen byte-level BPE and reproducible saved artifacts."""

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from moe_erc import ModelConfig
from moe_erc.dolma_reader import iter_documents
from moe_erc.tokenizer_utils import FrozenTokenizer, SPECIAL_TOKENS, SplitConfig, chain_hash, train_tokenizer
from tests.data_fixtures import fixture, write_jsonl


class TokenizerTests(unittest.TestCase):
    def test_encode_decode_specials_save_load_and_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, tokenizer = fixture(directory)
            text = "Hello 🌏! tiếng Việt\n日本語"
            ids = tokenizer.encode(text)
            self.assertTrue(all(type(i) is int and 0 <= i < tokenizer.vocab_size for i in ids))
            self.assertEqual(tokenizer.decode(ids), text)
            self.assertEqual(tokenizer.special_ids, dict(zip(SPECIAL_TOKENS, range(4))))
            loaded = FrozenTokenizer(tokenizer.path)
            self.assertEqual(ids, loaded.encode(text))
            self.assertEqual(tokenizer.fingerprint, loaded.fingerprint)
            with self.assertRaises(ValueError):
                tokenizer.validate_model(ModelConfig())
            config = replace(ModelConfig(), vocab_size=tokenizer.vocab_size)
            with self.assertRaisesRegex(ValueError, "test-only"):
                tokenizer.validate_model(config)
            tokenizer.validate_model(config, allow_test=True)

    def test_training_sample_contains_only_unique_train_documents(self):
        with tempfile.TemporaryDirectory() as directory:
            split, sources, tokenizer = fixture(directory)
            expected_hash = "00" * 32
            seen = set()
            expected_count = 0
            for source in sources:
                for document in iter_documents(source):
                    if split.assign(document.content_hash) == "train" and document.content_hash not in seen:
                        seen.add(document.content_hash)
                        expected_hash = chain_hash(expected_hash, document.content_hash)
                        expected_count += 1
            self.assertEqual(tokenizer.metadata["training_documents"], expected_count)
            self.assertEqual(tokenizer.metadata["training_content_fingerprint"], expected_hash)
            with patch("moe_erc.dolma_reader.urlopen", side_effect=AssertionError("network forbidden")):
                same = train_tokenizer(sources, Path(directory) / "same.json", split=split,
                                       max_documents=8, vocab_size=320, test_only=True)
            self.assertEqual(tokenizer.fingerprint, same.fingerprint)
            with self.assertRaises(FileExistsError):
                train_tokenizer(sources, tokenizer.path, test_only=True)

    def test_small_corpus_does_not_claim_production_16000(self):
        with tempfile.TemporaryDirectory() as directory:
            source = write_jsonl(Path(directory) / "tiny.jsonl", [{"text": "tiny text"}])
            with self.assertRaisesRegex(ValueError, "only .*16000"):
                train_tokenizer([str(source)], Path(directory) / "production.json", max_documents=1)
            self.assertFalse((Path(directory) / "production.json").exists())

    def test_tokenizer_hash_detects_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, tokenizer = fixture(directory)
            with tokenizer.path.open("a", encoding="utf-8") as stream:
                stream.write(" ")
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                FrozenTokenizer(tokenizer.path)


if __name__ == "__main__":
    unittest.main()
