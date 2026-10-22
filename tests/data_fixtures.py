"""Small local fixture helpers shared by STEP 5 tests."""

import json
from pathlib import Path

from moe_erc.data_smoke import synthetic_documents
from moe_erc.tokenizer_utils import SplitConfig, train_tokenizer


def write_jsonl(path, documents):
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for document in documents:
            stream.write(json.dumps(document, ensure_ascii=False) + "\n")
    return path


def fixture(root):
    root = Path(root)
    split = SplitConfig(seed=42, validation_fraction=0.25)
    documents = synthetic_documents(split, train_count=8, validation_count=4)
    sources = [str(write_jsonl(root / "first.jsonl", documents[:6])),
               str(write_jsonl(root / "second.jsonl", documents[6:]))]
    tokenizer = train_tokenizer(sources, root / "tokenizer.json", split=split,
                                max_documents=8, vocab_size=320, test_only=True)
    return split, sources, tokenizer
