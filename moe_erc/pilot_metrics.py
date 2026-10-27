"""Append-only JSONL session segments; values must already be detached scalars."""

import json
from pathlib import Path
import uuid


class MetricLog:
    def __init__(self, root, identity, parent_receipt=None):
        directory = Path(root) / "logs"
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"session-{uuid.uuid4().hex}.jsonl"
        self.identity = identity
        with self.path.open("x", encoding="utf-8"):
            pass
        self.write("session", {"identity": identity, "parent_receipt": parent_receipt})

    def write(self, event, values):
        record = {"event": event, "method": self.identity["method"],
                  "seed": self.identity["training"]["seed"], **values}
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
