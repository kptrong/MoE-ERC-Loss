"""Full55M output budget and verified, per-model checkpoint retention.

Never prunes Input, logs, benchmark results, partial saves or unknown files.
The shared advisory lock serializes checkpoint writes across both GPU workers.
"""
from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import time

from . import pilot_checkpoint_core as core

OUTPUT_BUDGET_BYTES = 18_000_000_000  # Below Kaggle's documented 20 GB output limit.
CHECKPOINT_RESERVE_BYTES = 1024**3
CHECKPOINT_FILES = ("checkpoint.pt", "manifest.json", "manifest.sha256", "COMPLETE.json")


def tree_bytes(root):
    """Measure all regular files without following symlinks (including .git)."""
    total = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if not Path(directory, d).is_symlink()]
        for name in files:
            p = Path(directory, name)
            if not p.is_symlink():
                try:
                    total += p.stat().st_size
                except FileNotFoundError:  # Concurrent atomic rename/retention.
                    pass
    return total


def check_budget(root, *, reserve=0, limit=OUTPUT_BUDGET_BYTES):
    root = Path(root).resolve(strict=True)
    core.rt_require(0 < limit <= OUTPUT_BUDGET_BYTES and reserve >= 0, "invalid output budget")
    used, free = tree_bytes(root), shutil.disk_usage(root).free
    core.rt_require(used + reserve <= limit,
                    f"Output budget: {used:,} used + {reserve:,} reserved > {limit:,} bytes. "
                    "Stop; preserve existing checkpoints. Download/save outputs before resuming.")
    core.rt_require(free >= reserve + 256 * 1024**2, "Insufficient filesystem free space")
    return {"used_bytes": used, "reserved_bytes": reserve, "limit_bytes": limit, "free_bytes": free}


@contextmanager
def checkpoint_lock(output):
    """OS lock releases on crash; lock file stays to avoid inode replacement races."""
    path = Path(output) / ".checkpoint-storage.lock"
    core.rt_require(not path.is_symlink(), "unsafe storage lock")
    with path.open("a+b") as f:
        if os.name == "nt":
            import msvcrt
            if f.tell() == 0:
                f.write(b"\0"); f.flush()
            while True:
                f.seek(0)
                try:
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(.1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def retain_two(manager, receipt, archive_root):
    """Run only AFTER the manager publishes and validates the replacement.

    Invalid/incomplete directories are reported, preserved and charged to budget.
    A final checkpoint is one of the two heads, never a third exception.
    """
    root = (manager.local / "checkpoints").resolve(strict=True)
    current = Path(receipt["path"]).resolve(strict=True)
    core.rt_require(current.parent == root, "replacement outside owned checkpoint root")
    core.verify(current, manager.identity, receipt["manifest_sha256"])
    valid, rejected = [], []
    for p in root.iterdir():
        if not p.is_dir() or p.is_symlink() or not (p / "COMPLETE.json").exists():
            continue
        try:
            m = core.verify(p, manager.identity, (p / "manifest.sha256").read_text().strip())
            valid.append((int(p.name.split("-")[0][4:]), p.stat().st_mtime_ns, p, m))
        except Exception as exc:
            rejected.append({"path": str(p), "error": str(exc)})
    valid.sort(key=lambda e: (e[0], e[2] == current, e[1]), reverse=True)
    # The just-published state must be among the latest two; do not prune after
    # accidentally saving an older state over an existing newer recovery point.
    keep = valid[:2]
    core.rt_require(any(e[2] == current for e in keep), "replacement is older than retained heads")
    retired = []
    for _, _, p, _ in valid[2:]:
        # Revalidate protected replacements immediately before each deletion.
        for _, _, head, _ in keep:
            core.verify(head, manager.identity, (head / "manifest.sha256").read_text().strip())
        core.rt_require(p.parent == root and not p.is_symlink(), "unsafe retention path")
        core.verify(p, manager.identity, (p / "manifest.sha256").read_text().strip())
        archive = Path(archive_root) / p.name
        archive.mkdir(parents=True, exist_ok=True)
        for name in CHECKPOINT_FILES[1:]:
            source, target = p / name, archive / name
            if target.exists():
                core.rt_require(not target.is_symlink() and target.read_bytes() == source.read_bytes(),
                                "retention metadata archive conflict")
            else:
                with target.open("xb") as f:
                    f.write(source.read_bytes())
            core.rt_require(target.read_bytes() == source.read_bytes(), "retention metadata copy mismatch")
        # Delete only the known four files, never recursive deletion.
        for name in CHECKPOINT_FILES:
            (p / name).unlink()
        p.rmdir()
        retired.append(str(p))
    return {"retained": [str(e[2]) for e in keep], "retired": retired, "rejected_preserved": rejected}
