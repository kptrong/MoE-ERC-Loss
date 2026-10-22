"""No-network reader tests, including mocked timeout/retry/download limits."""

import gzip
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from moe_erc.dolma_reader import (
    DownloadOptions, compressed_size, iter_documents, parse_manifest, read_manifest,
    resolve_source, select_shards,
)


class Response(BytesIO):
    def __init__(self, value, length=None):
        super().__init__(value)
        self.headers = {} if length is None else {"Content-Length": str(length)}


class DolmaReaderTests(unittest.TestCase):
    def test_manifest_and_deterministic_selection(self):
        urls = [f"https://example.invalid/{s}/{i}.json.gz" for s in ("web", "books") for i in range(8)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.txt"
            path.write_text("# official-format fixture\n" + "\n".join(urls) + "\n" + urls[0], encoding="utf-8")
            with patch("moe_erc.dolma_reader.urlopen", side_effect=AssertionError("network forbidden")):
                manifest = read_manifest(str(path))
            self.assertEqual(manifest["urls"], urls)
            self.assertEqual(len(manifest["manifest_sha256"]), 64)
            a = select_shards(urls, max_shards=3, seed=42)
            self.assertEqual(a, select_shards(list(reversed(urls)), max_shards=3, seed=42))
            self.assertNotEqual(a, select_shards(urls, max_shards=3, seed=43))
            self.assertTrue(all("books" in x for x in select_shards(urls, max_shards=2, source_contains="books")))
        for invalid in ("", "not a URL", "https://example.invalid/a.txt"):
            with self.assertRaises(ValueError):
                parse_manifest(invalid)

    def test_jsonl_and_gzip_invalid_lines_and_limit(self):
        raw = '\n'.join([json.dumps({"text": " hello\r\nworld "}), '{broken', '{}',
                         '{"text":7}', '{"text":"  "}', json.dumps({"text": "second"})])
        with tempfile.TemporaryDirectory() as directory:
            plain, zipped = Path(directory) / "x.jsonl", Path(directory) / "x.json.gz"
            plain.write_text(raw, encoding="utf-8")
            with gzip.open(zipped, "wt", encoding="utf-8") as stream:
                stream.write(raw)
            for path in (plain, zipped):
                stats = {}
                documents = list(iter_documents(path, stats=stats))
                self.assertEqual([d.text for d in documents], ["hello\nworld", "second"])
                self.assertEqual(stats["invalid_documents"], 4)
                self.assertEqual(len(list(iter_documents(path, max_documents=1))), 1)

    def test_remote_requires_explicit_opt_in(self):
        with patch("moe_erc.dolma_reader.urlopen", side_effect=AssertionError("network forbidden")):
            with self.assertRaises(PermissionError):
                resolve_source("https://example.invalid/a.json.gz", DownloadOptions())

    def test_mock_download_retry_integrity_and_budget(self):
        source = "https://example.invalid/a.json.gz"
        payload = gzip.compress(b'{"text":"hello"}\n')
        with tempfile.TemporaryDirectory() as directory:
            options = DownloadOptions(True, directory, 1000, retries=1)
            with patch("moe_erc.dolma_reader.time.sleep"), patch(
                "moe_erc.dolma_reader.urlopen", side_effect=[TimeoutError("timeout"), Response(payload, len(payload))],
            ) as request:
                path = resolve_source(source, options)
                self.assertEqual(request.call_count, 2)
                self.assertEqual(path.read_bytes(), payload)
            with patch("moe_erc.dolma_reader.urlopen", side_effect=AssertionError("must reuse cache")):
                self.assertEqual(resolve_source(source, options), path)
            path.write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "checksum"):
                resolve_source(source, options)
        for known_length in (None, len(payload)):
            with tempfile.TemporaryDirectory() as directory:
                options = DownloadOptions(True, directory, 2, retries=0)
                with patch("moe_erc.dolma_reader.urlopen", return_value=Response(payload, known_length)):
                    with self.assertRaisesRegex(ValueError, "budget"):
                        resolve_source(source, options)
                self.assertFalse(list(Path(directory).glob("*.part")))

    def test_unknown_size_and_manifest_retry_failure(self):
        with patch("moe_erc.dolma_reader.urlopen", return_value=Response(b"")):
            self.assertIsNone(compressed_size("https://example.invalid/a.json.gz"))
        with patch("moe_erc.dolma_reader.urlopen", return_value=Response(b"", 123)):
            self.assertEqual(compressed_size("https://example.invalid/a.json.gz"), 123)
        with patch("moe_erc.dolma_reader.time.sleep"), patch("moe_erc.dolma_reader.urlopen", side_effect=TimeoutError("timeout")):
            with self.assertRaisesRegex(RuntimeError, "manifest"):
                read_manifest("https://example.invalid/manifest", retries=1)


if __name__ == "__main__":
    unittest.main()
