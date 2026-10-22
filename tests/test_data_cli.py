"""CLI help and dry-run inspection use fixtures only, with networking blocked."""

import contextlib
import importlib
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class DataCLITests(unittest.TestCase):
    def test_all_commands_have_help(self):
        for name in ("inspect_dolma", "train_tokenizer", "prepare_dataset", "inspect_batch"):
            with self.subTest(command=name), patch("sys.argv", [name, "--help"]):
                with contextlib.redirect_stdout(StringIO()), self.assertRaises(SystemExit) as raised:
                    importlib.import_module(f"scripts.{name}").main()
                self.assertEqual(raised.exception.code, 0)

    def test_inspect_defaults_to_dry_run(self):
        from scripts.inspect_dolma import main
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.txt"
            manifest.write_text("https://example.invalid/web/0.json.gz\nhttps://example.invalid/books/1.json.gz\n")
            output = Path(directory) / "selection.json"
            argv = ["inspect", "--manifest", str(manifest), "--max-shards", "1", "--save-selection", str(output)]
            with patch("sys.argv", argv), patch("moe_erc.dolma_reader.urlopen", side_effect=AssertionError("network forbidden")):
                with contextlib.redirect_stdout(StringIO()) as capture:
                    main()
            self.assertIn("dry-run", capture.getvalue())
            self.assertIn("unknown", capture.getvalue())
            self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
