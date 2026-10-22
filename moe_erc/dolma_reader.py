"""Bounded, opt-in shard acquisition and line-at-a-time Dolma/local readers."""

from dataclasses import dataclass
import gzip
import hashlib
import json
import logging
from pathlib import Path
import random
import time
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


REPOSITORY = "allenai/dolma"
VERSION = "v1_5-sample"
MANIFEST_URL = "https://huggingface.co/datasets/allenai/dolma/resolve/main/urls/v1_5-sample.txt"
LOGGER = logging.getLogger(__name__)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        import os
        os.fsync(stream.fileno())
    temporary.replace(path)


def is_remote(source: str) -> bool:
    return urlparse(str(source)).scheme in ("https", "http")


def parse_manifest(text: str) -> list[str]:
    urls = []
    for line in text.splitlines():
        url = line.strip()
        if not url or url.startswith("#"):
            continue
        if not is_remote(url) or not urlparse(url).path.endswith(".gz"):
            raise ValueError(f"invalid shard URL in manifest: {url}")
        if url not in urls:
            urls.append(url)
    if not urls:
        raise ValueError("manifest contains no shard URLs")
    return urls


def read_manifest(source: str = MANIFEST_URL, *, timeout: float = 30, retries: int = 2) -> dict:
    """Fetch only the small URL manifest, never a data shard."""
    if timeout <= 0 or retries < 0:
        raise ValueError("timeout must be positive and retries nonnegative")
    if is_remote(source):
        for attempt in range(retries + 1):
            try:
                with urlopen(Request(source, headers={"User-Agent": "moe-erc-research"}), timeout=timeout) as response:
                    raw = response.read(16 * 1024 * 1024 + 1)
                if len(raw) > 16 * 1024 * 1024:
                    raise ValueError("manifest exceeds the 16 MiB safety limit")
                break
            except (OSError, URLError) as exc:
                if attempt == retries:
                    raise RuntimeError(f"cannot read manifest {source}: {exc}") from exc
                time.sleep(min(attempt + 1, 3))
    else:
        raw = Path(source).read_bytes()
    return {
        "repository": REPOSITORY, "version": VERSION,
        "manifest_source": str(source), "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "urls": parse_manifest(raw.decode("utf-8")),
    }


def select_shards(urls: list[str], *, max_shards: int, seed: int = 42, source_contains: str | None = None) -> list[str]:
    if type(max_shards) is not int or max_shards <= 0:
        raise ValueError("max_shards must be positive; unlimited selection is not supported")
    candidates = sorted(set(u for u in urls if not source_contains or source_contains in urlparse(u).path))
    random.Random(seed).shuffle(candidates)
    if not candidates:
        raise ValueError("source filter selected no shards")
    return candidates[:max_shards]


def compressed_size(url: str, *, timeout: float = 30) -> int | None:
    """HEAD only. Unknown size remains None, never an estimate."""
    try:
        with urlopen(Request(url, method="HEAD"), timeout=timeout) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length is not None else None
    except (OSError, URLError, ValueError):
        return None


@dataclass
class DownloadOptions:
    allow_download: bool = False
    cache_dir: str = "data/raw"
    max_bytes: int = 1_000_000_000
    timeout: float = 30
    retries: int = 2
    bytes_received: int = 0

    def __post_init__(self):
        if self.max_bytes <= 0 or self.timeout <= 0 or self.retries < 0:
            raise ValueError("invalid download byte limit, timeout or retry count")


def resolve_source(source: str, options: DownloadOptions) -> Path:
    if not is_remote(source):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    # Cached remote sources also require explicit consent to use the remote plan.
    if not options.allow_download:
        raise PermissionError("remote shards require --allow-download (or inspect --download)")
    cache = Path(options.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(source.encode()).hexdigest()
    target = cache / f"{key}-{Path(urlparse(source).path).name}"
    receipt = target.with_name(target.name + ".receipt.json")
    if target.exists():
        if not receipt.exists() or json.loads(receipt.read_text())["sha256"] != file_sha256(target):
            raise ValueError(f"cached shard checksum missing or invalid: {target}")
        return target
    temporary = target.with_name(target.name + ".part")
    for attempt in range(options.retries + 1):
        try:
            with urlopen(Request(source, headers={"User-Agent": "moe-erc-research"}), timeout=options.timeout) as response:
                length = response.headers.get("Content-Length")
                if length is not None and int(length) > options.max_bytes - options.bytes_received:
                    raise ValueError(f"download byte budget would be exceeded by {source}")
                actual = 0
                with temporary.open("wb") as stream:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        options.bytes_received += len(chunk)
                        if options.bytes_received > options.max_bytes:
                            raise ValueError(f"download byte budget exceeded while reading {source}")
                        stream.write(chunk)
                        actual += len(chunk)
                if length is not None and actual != int(length):
                    raise OSError(f"incomplete transfer: expected {length} bytes, received {actual}")
            temporary.replace(target)
            atomic_json(receipt, {"url": source, "sha256": file_sha256(target), "bytes": actual})
            return target
        except (OSError, URLError) as exc:
            temporary.unlink(missing_ok=True)
            if attempt == options.retries:
                raise RuntimeError(f"download failed for {source}: {exc}") from exc
            time.sleep(min(attempt + 1, 3))
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    raise AssertionError("unreachable")


@dataclass(frozen=True)
class Document:
    text: str
    content_hash: str


def canonical_text(text: str) -> str:
    """Explicit identity policy: normalize line endings and strip outer whitespace."""
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def iter_documents(path: str | Path, *, stats: dict | None = None, max_documents: int | None = None):
    """Stream JSONL or gzip JSONL, with one document's text resident at a time."""
    if max_documents is not None and max_documents <= 0:
        raise ValueError("max_documents must be positive")
    path = Path(path)
    counters = stats if stats is not None else {}
    opener = gzip.open if path.suffix == ".gz" else open
    yielded = 0
    try:
        with opener(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                counters["lines"] = counters.get("lines", 0) + 1
                try:
                    value = json.loads(line)
                    text = canonical_text(value["text"]) if isinstance(value, dict) and isinstance(value.get("text"), str) else ""
                except (ValueError, TypeError):
                    text = ""
                if not text:
                    counters["invalid_documents"] = counters.get("invalid_documents", 0) + 1
                    continue
                counters["valid_documents"] = counters.get("valid_documents", 0) + 1
                yield Document(text, hashlib.sha256(text.encode("utf-8")).hexdigest())
                yielded += 1
                if max_documents is not None and yielded >= max_documents:
                    break
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(f"cannot read JSONL shard {path}: {exc}") from exc
