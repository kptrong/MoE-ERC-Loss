"""Portable primitives extracted from the PASS Step 6.5C notebook cell 89.

Serialization envelope and exact-content validation are preserved. Only snapshot
synchronization is guarded for CPU tests. No notebook execution/dependency at runtime.
Pilot boundary/identity checks live in pilot_checkpoint, not the old smoke runner.
"""
from pathlib import Path
import copy
import hashlib
import json
import random
import numpy as np
import torch
from .pilot_data import file_sha256 as rt_sha, read_json as rt_json

def rt_require(ok, message):
    if not ok:
        raise RuntimeError(message)

NOTEBOOK_SHA256 = 'f93bb1c18b845bec10c93aee5278229decb3e0ca4267ca4fd9e8f1300c82cb45'
UTILITY_SOURCE_SHA256 = '323f9230c19b682a4713e4fdca3727d496415d8cac39e08a740ea135eee96eac'

STATE_FIELDS = {
    "model", "buffers", "optimizer", "parameter_names", "scheduler",
    "rng", "progress", "sampler", "modes", "scaler",
}

def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")

def frozen(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().clone()
    if isinstance(x, dict):
        return {k: frozen(v) for k, v in x.items()}
    if isinstance(x, tuple):
        return tuple(frozen(v) for v in x)
    if isinstance(x, list):
        return [frozen(v) for v in x]
    return copy.deepcopy(x)

def exact(a, b, path="state"):
    if isinstance(a, torch.Tensor):
        rt_require(isinstance(b, torch.Tensor), f"{path}: tensor type")
        a, b = a.detach().cpu(), b.detach().cpu()
        rt_require(a.dtype == b.dtype and a.shape == b.shape, f"{path}: dtype/shape")
        aa = a.contiguous().reshape(-1).view(torch.uint8)
        bb = b.contiguous().reshape(-1).view(torch.uint8)
        rt_require(torch.equal(aa, bb), f"{path}: tensor content differs")
    elif isinstance(a, dict):
        rt_require(type(a) is type(b) and a.keys() == b.keys(), f"{path}: keys/type")
        for k in a:
            exact(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (tuple, list)):
        rt_require(type(a) is type(b) and len(a) == len(b), f"{path}: length/type")
        for i, (x, y) in enumerate(zip(a, b)):
            exact(x, y, f"{path}[{i}]")
    else:
        rt_require(type(a) is type(b) and a == b, f"{path}: {a!r} != {b!r}")

def tensor_addresses(x):
    if isinstance(x, torch.Tensor):
        return {(str(x.device), x.untyped_storage().data_ptr())} if x.numel() else set()
    if isinstance(x, dict):
        values = x.values()
    elif isinstance(x, (tuple, list)):
        values = x
    else:
        return set()
    result = set()
    for v in values:
        result.update(tensor_addresses(v))
    return result

def no_alias(a, b):
    rt_require(not tensor_addresses(a).intersection(tensor_addresses(b)),
               "Comparison reference aliasing.")

def digest_tree(x):
    h = hashlib.sha256()
    def visit(v):
        h.update(type(v).__name__.encode() + b"\0")
        if isinstance(v, torch.Tensor):
            t = v.detach().cpu().contiguous()
            h.update(str(t.dtype).encode())
            h.update(str(tuple(t.shape)).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=repr):
                visit(k)
                visit(v[k])
        elif isinstance(v, (list, tuple)):
            h.update(str(len(v)).encode() + b"\0")
            for item in v:
                visit(item)
        else:
            h.update(repr(v).encode() + b"\0")
    visit(x)
    return h.hexdigest()

def rng_get():
    n = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (n[0], n[1].tolist(), n[2], n[3], n[4]),
        "cpu": torch.get_rng_state().clone(),
        "cuda": [t.clone() for t in torch.cuda.get_rng_state_all()],
    }

def rng_set(r):
    random.setstate(r["python"])
    n = r["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(r["cpu"])
    rt_require(len(r["cuda"]) == torch.cuda.device_count(), "CUDA RNG count.")
    torch.cuda.set_rng_state_all(r["cuda"])

def names_by_group(model, optimizer):
    names = {id(p): n for n, p in model.named_parameters()}
    groups = [[names[id(p)] for p in g["params"]] for g in optimizer.param_groups]
    flat = [n for group in groups for n in group]
    rt_require(len(flat) == len(set(flat)), "Repeated optimizer parameter.")
    return groups

def snapshot(model, optimizer, scheduler, progress, sampler, scaler=None):
    if next(model.parameters()).device.type == "cuda":
        torch.cuda.synchronize(next(model.parameters()).device)
    state = {
        "model": frozen(model.state_dict()),
        "buffers": frozen(dict(model.named_buffers())),
        "optimizer": frozen(optimizer.state_dict()),
        "parameter_names": names_by_group(model, optimizer),
        "scheduler": frozen(scheduler.state_dict()),
        "rng": frozen(rng_get()),
        "progress": copy.deepcopy(progress),
        "sampler": copy.deepcopy(sampler),
        "modes": {n: m.training for n, m in model.named_modules()},
        "scaler": frozen(scaler.state_dict()) if scaler is not None else None,
    }
    no_alias(model.state_dict(), state["model"])
    no_alias(optimizer.state_dict(), state["optimizer"])
    return state

def validate_payload(payload, identity):
    rt_require(isinstance(payload, dict)
               and set(payload) == {"schema", "identity", "state"}, "PAYLOAD_FIELDS")
    rt_require(payload["schema"] == "step65c-v1", "PAYLOAD_SCHEMA")
    rt_require(payload["identity"] == identity, "IDENTITY_MISMATCH")
    rt_require(isinstance(payload["state"], dict)
               and set(payload["state"]) == STATE_FIELDS, "STATE_FIELDS")
    if identity.get("precision") == "fp16-full55m-v1":
        rt_require(isinstance(payload["state"]["scaler"], dict), "Missing scaler.")
    else:
        rt_require(payload["state"]["scaler"] is None, "Unexpected scaler.")

def file_check(path, expected_sha):
    p = Path(path)
    rt_require(p.is_file() and not p.is_symlink(), f"Missing/unsafe file: {p}")
    rt_require(rt_sha(p) == expected_sha, "CHECKSUM_MISMATCH")

def verify(root, identity, manifest_sha, unpublished=False):
    root = Path(root)
    rt_require(root.is_dir() and not root.is_symlink(), "Unsafe checkpoint directory.")
    marker = "INCOMPLETE.json" if unpublished else "COMPLETE.json"
    required = {"checkpoint.pt", "manifest.json", "manifest.sha256", marker}
    rt_require({p.name for p in root.iterdir()} == required, "CHECKPOINT_INVENTORY")
    rt_require(all(p.is_file() and not p.is_symlink() for p in root.iterdir()),
               "Unsafe checkpoint entry.")
    file_check(root / "manifest.json", manifest_sha)
    rt_require((root / "manifest.sha256").read_text().strip() == manifest_sha,
               "MANIFEST_SIDECAR")
    m = rt_json(root / "manifest.json")
    rt_require(set(m) == {"schema", "identity", "payload", "state_digest"}
               and m["schema"] == "step65c-v1", "MANIFEST_FIELDS")
    rt_require(m["identity"] == identity, "IDENTITY_MISMATCH")
    e = m["payload"]
    rt_require(set(e) == {"file", "bytes", "sha256"}
               and e["file"] == "checkpoint.pt", "PAYLOAD_ENTRY")
    rt_require((root / "checkpoint.pt").stat().st_size == e["bytes"], "PAYLOAD_SIZE")
    file_check(root / "checkpoint.pt", e["sha256"])
    if not unpublished:
        rt_require(rt_json(root / marker) == {
            "status": "COMPLETE", "manifest_sha256": manifest_sha,
        }, "COMPLETE_MARKER")
    return m
