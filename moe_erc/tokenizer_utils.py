"""Train-once byte-level BPE from a fixed training-only document iterator."""

from dataclasses import asdict, dataclass
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
import tokenizers

from .config import ModelConfig
from .dolma_reader import DownloadOptions, atomic_json, file_sha256, iter_documents, resolve_source


SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")
SPLIT_ALGORITHM = "sha256(seed:content_sha256)/2^256; canonical-lf-strip-v1"


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 42
    validation_fraction: float = 0.01

    def __post_init__(self):
        if type(self.seed) is not int or not 0 < self.validation_fraction < 1:
            raise ValueError("split seed must be an integer; validation_fraction must be in (0,1)")

    def assign(self, content_hash: str) -> str:
        value = int.from_bytes(hashlib.sha256(f"{self.seed}:{content_hash}".encode()).digest(), "big")
        return "validation" if value < int(self.validation_fraction * (1 << 256)) else "train"


def chain_hash(previous: str, content_hash: str) -> str:
    return hashlib.sha256(bytes.fromhex(previous) + bytes.fromhex(content_hash)).hexdigest()


class FrozenTokenizer:
    """Read-only encoding interface; no train/add-token method is exposed."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.metadata = json.loads(self.path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        self.fingerprint = file_sha256(self.path)
        if self.fingerprint != self.metadata["sha256"]:
            raise ValueError("tokenizer fingerprint mismatch")
        self._tokenizer = Tokenizer.from_file(str(self.path))
        self.vocab_size = self._tokenizer.get_vocab_size()
        self.special_ids = {s: self._tokenizer.token_to_id(s) for s in SPECIAL_TOKENS}
        if self.vocab_size != self.metadata["vocab_size"] or self.special_ids != self.metadata["special_ids"]:
            raise ValueError("tokenizer vocabulary or special-token metadata mismatch")
        if set(self._tokenizer.get_vocab().values()) != set(range(self.vocab_size)):
            raise ValueError("tokenizer IDs are not a contiguous vocabulary")

    @property
    def eos_id(self) -> int:
        return self.special_ids["<eos>"]

    def encode(self, text: str) -> list[int]:
        ids = self._tokenizer.encode(text, add_special_tokens=False).ids
        if any(i < 0 or i >= self.vocab_size for i in ids):
            raise ValueError("encoded IDs are out of vocabulary")
        return ids

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        if any(i < 0 or i >= self.vocab_size for i in ids):
            raise ValueError("decode IDs are out of vocabulary")
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def validate_model(self, config: ModelConfig, *, allow_test: bool = False) -> None:
        if self.vocab_size != config.vocab_size:
            raise ValueError("tokenizer vocabulary must exactly match model vocab_size")
        if self.metadata["test_only"] and not allow_test:
            raise ValueError("test-only tokenizer requires explicit allow_test=True")


def train_tokenizer(
    sources: list[str], output: str | Path, *, split: SplitConfig = SplitConfig(),
    vocab_size: int = 16000, max_documents: int = 100_000, test_only: bool = False,
    downloads: DownloadOptions | None = None, provenance: dict | None = None,
) -> FrozenTokenizer:
    """Use first max_documents unique training documents in the frozen source order.

    Source order, byte fingerprints and consumed training content hashes are
    recorded. Validation documents are filtered before the trainer sees text.
    SQLite bounds deduplication RAM; BPE training itself still uses trainer RAM.
    """
    output = Path(output)
    if output.suffix != ".json" or not sources or max_documents <= 0:
        raise ValueError("require sources, positive max_documents and a .json output")
    if not 260 <= vocab_size <= 65536:
        raise ValueError("byte-level vocab_size must be in [260,65536]")
    if not test_only and vocab_size != 16000:
        raise ValueError("production tokenizer must request exactly 16000 tokens")
    if output.exists() or output.with_suffix(".meta.json").exists():
        raise FileExistsError("refusing to overwrite tokenizer; reuse the saved JSON")
    options = downloads or DownloadOptions()
    metadata = {
        "algorithm": "byte-level-bpe", "tokenizers_version": tokenizers.__version__,
        "requested_vocab_size": vocab_size, "test_only": test_only,
        "trainer": {"min_frequency": 2, "initial_alphabet": "all-256-bytes",
                    "add_prefix_space": False, "automatic_bos_eos": False},
        "special_ids": dict(zip(SPECIAL_TOKENS, range(4))),
        "split": asdict(split), "split_algorithm": SPLIT_ALGORITHM,
        "source_plan": list(sources), "sources_consumed": [],
        "max_training_documents": max_documents, "training_documents": 0,
        "training_content_fingerprint": "00" * 32, "provenance": provenance or {},
    }
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=2, show_progress=False,
        special_tokens=list(SPECIAL_TOKENS), initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    with tempfile.TemporaryDirectory(prefix="moe-tokenizer-") as directory:
        # tokenizers may consume the Python iterator on a producer thread.
        # Only that iterator accesses SQLite during training; the main thread
        # closes it after train_from_iterator returns, never concurrently.
        with closing(sqlite3.connect(str(Path(directory) / "seen.sqlite"), check_same_thread=False)) as seen:
            seen.execute("CREATE TABLE seen (hash TEXT PRIMARY KEY)")

            def texts():
                for source in sources:
                    path = resolve_source(source, options)
                    metadata["sources_consumed"].append({"source": source, "sha256": file_sha256(path)})
                    for document in iter_documents(path):
                        if split.assign(document.content_hash) != "train":
                            continue
                        if seen.execute("INSERT OR IGNORE INTO seen VALUES (?)", (document.content_hash,)).rowcount == 0:
                            continue
                        metadata["training_documents"] += 1
                        metadata["training_content_fingerprint"] = chain_hash(
                            metadata["training_content_fingerprint"], document.content_hash,
                        )
                        yield document.text
                        if metadata["training_documents"] >= max_documents:
                            return

            tokenizer.train_from_iterator(texts(), trainer=trainer)
    actual = tokenizer.get_vocab_size()
    if metadata["training_documents"] == 0:
        raise ValueError("no training documents; no validation documents were reassigned")
    if not test_only and actual != 16000:
        raise ValueError(f"production vocabulary reached only {actual}/16000; supply more training text")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    tokenizer.save(str(temporary))
    temporary.replace(output)
    metadata.update(vocab_size=actual, sha256=file_sha256(output))
    atomic_json(output.with_suffix(".meta.json"), metadata)
    return FrozenTokenizer(output)
