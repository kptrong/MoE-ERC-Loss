"""Packing references, stable splits, integrity, budget and crash/resume tests."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from moe_erc.data_processing import (
    PreprocessingConfig, audit_document_splits, prepare_dataset, validate_dataset,
)
from moe_erc.dolma_reader import atomic_json, iter_documents
from tests.data_fixtures import fixture, write_jsonl


def read_tokens(root, info):
    return np.concatenate([np.fromfile(Path(root) / s["file"], dtype="<u2")
                           for s in info["segments"]]).tolist()


class ProcessingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.split, self.sources, self.tokenizer = fixture(self.root)
        self.config = PreprocessingConfig(1_000_000, 100_000, split=self.split)

    def prepare(self, name, **kwargs):
        return prepare_dataset(self.sources, self.tokenizer, self.root / name,
                               config=self.config, allow_test_tokenizer=True, **kwargs)

    def test_eos_packing_split_separation_and_discarded_tokens(self):
        state = self.prepare("packed")
        expected = {"train": [], "validation": []}
        hashes = {s: set() for s in expected}
        for source in self.sources:
            for document in iter_documents(source):
                split = self.split.assign(document.content_hash)
                hashes[split].add(document.content_hash)
                expected[split].extend(self.tokenizer.encode(document.text) + [self.tokenizer.eos_id])
        self.assertFalse(hashes["train"] & hashes["validation"])
        for split in expected:
            info = state["splits"][split]
            size = len(expected[split]) // 512 * 512
            self.assertEqual(read_tokens(self.root / "packed", info), expected[split][:size])
            self.assertEqual(info["tokens"], size)
            self.assertEqual(info["discarded_tokens"], len(expected[split]) - size)
            self.assertEqual(info["sequences"], size // 512)
            self.assertTrue(all(0 <= i < self.tokenizer.vocab_size for i in expected[split]))
        self.assertEqual(audit_document_splits(self.root / "packed")["overlap"], 0)
        validate_dataset(self.root / "packed", tokenizer=self.tokenizer)

    def test_same_content_deduplicated_before_tokenization(self):
        duplicate = next(iter_documents(self.sources[0]))
        extra = write_jsonl(self.root / "duplicate.jsonl", [{"text": "\n" + duplicate.text + "  "}, {}, {"text": ""}])
        sources = self.sources + [str(extra)]
        state = prepare_dataset(sources, self.tokenizer, self.root / "dedup", config=self.config,
                                allow_test_tokenizer=True)
        self.assertEqual(state["duplicates"], 1)
        self.assertEqual(state["reader_stats"]["invalid_documents"], 2)
        self.assertEqual(sum(s["documents"] for s in state["splits"].values()), 12)
        audit = audit_document_splits(self.root / "dedup")
        self.assertEqual(audit, {"train": 8, "validation": 4, "overlap": 0})

    def test_resume_and_reproducibility(self):
        partial = self.prepare("resumed", max_shards_per_run=1)
        self.assertFalse(partial["complete"])
        self.assertFalse((self.root / "resumed" / "dataset.json").exists())
        resumed = self.prepare("resumed", resume=True)
        reference = self.prepare("reference")
        self.assertEqual(resumed, reference)
        self.assertEqual(self.prepare("resumed", resume=True), resumed)
        with self.assertRaises(FileExistsError):
            self.prepare("resumed")

    def test_crash_after_sqlite_commit_does_not_duplicate_sequences(self):
        def interrupt(path, state):
            if Path(path).name == "checkpoint.json" and state["next_shard"] == 1:
                raise RuntimeError("simulated process interruption")
            atomic_json(path, state)
        with patch("moe_erc.data_processing.atomic_json", side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                self.prepare("crashed")
        recovered = self.prepare("crashed", resume=True)
        reference = self.prepare("reference")
        self.assertEqual(recovered, reference)
        self.assertEqual(audit_document_splits(self.root / "crashed")["train"], 8)

    def test_budgets_and_frozen_validation_when_train_expands(self):
        states = []
        for train_limit in (1100, 7000):
            config = replace(self.config, train_max_tokens=train_limit, validation_max_tokens=800)
            root = self.root / str(train_limit)
            state = prepare_dataset(self.sources, self.tokenizer, root, config=config,
                                    allow_test_tokenizer=True)
            states.append((root, state))
            for split, limit in (("train", train_limit), ("validation", 800)):
                info = state["splits"][split]
                self.assertEqual(info["accepted_tokens"], limit)
                self.assertEqual(info["tokens"], limit // 512 * 512)
                self.assertEqual(info["tail_discarded_tokens"], limit % 512)
                self.assertEqual(info["encoded_tokens"], info["tokens"] + info["discarded_tokens"])
        self.assertEqual(read_tokens(states[0][0], states[0][1]["splits"]["validation"]),
                         read_tokens(states[1][0], states[1][1]["splits"]["validation"]))
        self.assertEqual(states[0][1]["splits"]["validation"]["content_fingerprint"],
                         states[1][1]["splits"]["validation"]["content_fingerprint"])

    def test_empty_validation_is_reported_without_reassignment(self):
        documents = [dict(text=d.text) for source in self.sources for d in iter_documents(source)
                     if self.split.assign(d.content_hash) == "train"]
        source = write_jsonl(self.root / "train-only.jsonl", documents)
        with self.assertLogs("moe_erc.data_processing", level="WARNING") as logs:
            state = prepare_dataset([str(source)], self.tokenizer, self.root / "no-val",
                                    config=self.config, allow_test_tokenizer=True)
        self.assertEqual(state["splits"]["validation"]["documents"], 0)
        self.assertEqual(state["splits"]["validation"]["sequences"], 0)
        self.assertTrue(any("No validation" in message for message in logs.output))
        self.assertFalse(state["splits"]["train"]["budget_reached"])
        from moe_erc.dataset import PackedTokenDataset, make_dataloader
        dataset = PackedTokenDataset(self.root / "no-val", "validation", tokenizer=self.tokenizer)
        try:
            self.assertEqual(len(dataset), 0)
            with self.assertRaisesRegex(ValueError, "no packed sequences"):
                make_dataloader(dataset)
        finally:
            dataset.close()

    def test_resume_rejects_configuration_or_source_changes(self):
        self.prepare("partial", max_shards_per_run=1)
        with self.assertRaisesRegex(ValueError, "resume configuration"):
            prepare_dataset(self.sources, self.tokenizer, self.root / "partial", resume=True,
                            config=replace(self.config, train_max_tokens=1234), allow_test_tokenizer=True)
        with Path(self.sources[0]).open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.prepare("partial", resume=True)

    def test_binary_corruption_and_metadata_accounting_are_detected(self):
        state = self.prepare("packed")
        path = self.root / "packed" / state["splits"]["train"]["segments"][0]["file"]
        with path.open("r+b") as stream:
            stream.write(b"\xff\xff")
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_dataset(self.root / "packed", tokenizer=self.tokenizer)
        # Independent metadata accounting failure without corrupting binaries.
        self.prepare("metadata")
        path = self.root / "metadata" / "dataset.json"
        metadata = json.loads(path.read_text())
        metadata["splits"]["train"]["tokens"] += 1
        atomic_json(path, metadata)
        with self.assertRaisesRegex(ValueError, "token count"):
            validate_dataset(self.root / "metadata", tokenizer=self.tokenizer)


if __name__ == "__main__":
    unittest.main()
