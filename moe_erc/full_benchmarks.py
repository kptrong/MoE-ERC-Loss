"""Pinned harness MC adapter; no external tokenizer/model weights."""
import hashlib
import json
from pathlib import Path
import random
import torch
from .pilot_eval import isolated_evaluation

TASKS = ("arc_challenge", "hellaswag", "boolq", "winogrande", "sciq")

def sample_ids(count, seed=42, maximum=200):
    return sorted(random.Random(seed).sample(range(count), min(count, maximum)))

def score_tokens(model, context, continuation, eos, dtype=torch.float16):
    """Exact conditional token LL, sliding left context when longer than 512.

    Score each continuation token once. Never drop continuation tokens silently.
    """
    history = list(context) or [eos]
    if 0 < len(continuation) <= model.config.max_seq_len:
        tokens = (history + list(continuation))[-(model.config.max_seq_len + 1):]
        ids = torch.tensor([tokens[:-1]], device=next(model.parameters()).device)
        with torch.no_grad(), torch.autocast(ids.device.type, dtype=dtype, enabled=ids.device.type == "cuda"):
            logits = model(ids)[0, -len(continuation):].float()
        logp = logits.log_softmax(-1)
        target = torch.tensor(continuation, device=ids.device)
        if not torch.isfinite(logp).all():
            raise RuntimeError("nonfinite benchmark logits")
        return float(logp.gather(1,target[:,None]).sum()), bool((logits.argmax(-1)==target).all())
    score, greedy = 0.0, True
    for target in continuation:
        ids = torch.tensor([history[-model.config.max_seq_len:]], device=next(model.parameters()).device)
        with torch.no_grad(), torch.autocast(ids.device.type, dtype=dtype, enabled=ids.device.type == "cuda"):
            logits = model(ids)[0, -1].float()
        logp = logits.log_softmax(-1)
        if not torch.isfinite(logp).all():
            raise RuntimeError("nonfinite benchmark logits")
        score += float(logp[target])
        greedy = greedy and int(logits.argmax()) == target
        history.append(target)
    return score, greedy

def adapter(model, tokenizer):
    from lm_eval.api.model import TemplateLM
    class ResearchLM(TemplateLM):
        def __init__(self):
            super().__init__()
        @property
        def eot_token_id(self):
            return tokenizer.eos_id
        @property
        def tokenizer_name(self):
            return "moe-erc-bpe16k-" + tokenizer.fingerprint
        def tok_encode(self, text, **kwargs):
            return tokenizer.encode(text)
        def _loglikelihood_tokens(self, requests, **kwargs):
            return [score_tokens(model, context, continuation, self.eot_token_id)
                    for _, context, continuation in requests]
        def loglikelihood_rolling(self, requests, **kwargs):
            raise NotImplementedError("Only the five configured MC tasks are supported")
        def generate_until(self, requests, **kwargs):
            raise NotImplementedError("MC likelihood evaluation only")
    return ResearchLM()

def evaluate(model, tokenizer, directory, method, tokens, full=False):
    from lm_eval import evaluator
    from lm_eval.tasks import TaskManager, get_task_dict
    from importlib.metadata import version
    if version("lm_eval") != "0.4.9":
        raise RuntimeError("lm_eval==0.4.9 required for pinned task templates")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    with isolated_evaluation(model):
        for name in TASKS:
            task = get_task_dict([name], task_manager=TaskManager())[name]
            docs = list(task.eval_docs)
            selected = sample_ids(len(docs))
            doc_hashes = [hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()
                          for d in docs]
            plan = {"task": name, "harness": "0.4.9", "num_fewshot": 0, "seed": 42,
                    "split": "test" if task.has_test_docs() else "validation",
                    "doc_count": len(docs), "sample_ids": selected,
                    "document_hashes": doc_hashes, "task_config": "lm_eval==0.4.9 default YAML; no chat template"}
            plan_path = directory / f"{name}-samples.json"
            if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
                raise RuntimeError(f"Benchmark documents/template changed: {name}")
            plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
            result = evaluator.simple_evaluate(model=adapter(model, tokenizer), tasks=[task],
                        num_fewshot=0, samples=None if full else {name: selected},
                        bootstrap_iters=0, log_samples=False, random_seed=42,
                        numpy_random_seed=42, torch_random_seed=42, fewshot_random_seed=42)
            metrics = result["results"][name]
            row = {"method": method, "input_tokens": tokens, "task": name,
                   "scope": "full" if full else "fixed_subset", "acc": metrics["acc,none"],
                   "acc_norm": metrics.get("acc_norm,none"), "num_fewshot": 0,
                   "samples_sha256": hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()}
            rows.append(row)
            path = directory / f"{method}-{tokens}-{name}-{'full' if full else 'subset'}.json"
            path.write_text(json.dumps({"row": row, "harness_result": result}, default=str), encoding="utf-8")
    return rows
