"""Map-style, worker-safe memory-mapped packed token dataset."""

from bisect import bisect_right
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .data_processing import validate_dataset
from .tokenizer_utils import FrozenTokenizer


class PackedTokenDataset(Dataset):
    def __init__(self, root: str | Path, split: str, *, tokenizer: FrozenTokenizer | None = None,
                 verify_checksums: bool = False):
        if split not in ("train", "validation"):
            raise ValueError("split must be train or validation")
        self.root = Path(root)
        self.split = split
        self.metadata = validate_dataset(self.root, tokenizer=tokenizer, checksums=verify_checksums)
        self.sequence_length = self.metadata["sequence_length"]
        self.segments = [s for s in self.metadata["splits"][split]["segments"] if s["sequences"]]
        self.ends = []
        total = 0
        for segment in self.segments:
            total += segment["sequences"]
            self.ends.append(total)
        self._maps = {}

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        segment_index = bisect_right(self.ends, index)
        offset = index - (self.ends[segment_index - 1] if segment_index else 0)
        if segment_index not in self._maps:
            segment = self.segments[segment_index]
            self._maps[segment_index] = np.memmap(
                self.root / segment["file"], dtype="<u2", mode="r",
                shape=(segment["sequences"], self.sequence_length),
            )
        values = np.array(self._maps[segment_index][offset], dtype=np.int64, copy=True)
        if values.min() < 0 or values.max() >= self.metadata["tokenizer"]["vocab_size"]:
            raise ValueError("sample contains out-of-vocabulary token IDs")
        input_ids = torch.from_numpy(values)
        return {"input_ids": input_ids, "labels": input_ids.clone()}

    def __getstate__(self):
        # Windows spawn / Linux workers reopen files instead of serializing maps.
        return {**self.__dict__, "_maps": {}}

    def close(self):
        for view in self._maps.values():
            view._mmap.close()
        self._maps.clear()


def _seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def make_dataloader(dataset: PackedTokenDataset, *, batch_size: int = 4, seed: int = 42,
                    num_workers: int = 0, pin_memory: bool = False, drop_last: bool = False):
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    if not len(dataset):
        raise ValueError(f"{dataset.split} contains no packed sequences; inspect corpus and budgets")
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=dataset.split == "train",
        num_workers=num_workers, generator=generator, worker_init_fn=_seed_worker,
        pin_memory=pin_memory, drop_last=drop_last,
    )
