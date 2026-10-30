"""Full CE validation and fixed-probe diagnostics, isolated from training RNG."""

from contextlib import contextmanager
import math
import time

import torch

from .pilot_checkpoint_core import rng_get, rng_set
from .pilot_data import microbatches, require
from .pilot_losses import RoutingLossTap
from .erc_loss import generate_router_proxies, compute_activation_matrix, erc_loss_from_matrix


@contextmanager
def isolated_evaluation(model):
    state = rng_get()
    flags = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        for module, flag in flags:
            module.training = flag
        rng_set(state)


def to_device(batch, device):
    return {name: value.to(device) for name, value in batch.items()}


def validate(model, dataset, config, *, dtype=None):
    """All records, CE only; model performs the sole next-token shift."""
    started = time.perf_counter()
    device = model.token_embedding.weight.device
    numerator, targets, sequences = 0.0, 0, 0
    with isolated_evaluation(model):
        for batch in microbatches(dataset, range(len(dataset)), config.validation_microbatch_size):
            batch = to_device(batch, device)
            q = int((batch["labels"][:, 1:] != -100).sum())
            with torch.autocast(device_type=device.type, dtype=dtype or torch.bfloat16,
                                enabled=config.bf16 or dtype is not None):
                output = model(**batch)
            require(bool(torch.isfinite(output.loss)), "nonfinite validation CE")
            numerator += float(output.loss) * q
            targets += q
            sequences += batch["input_ids"].shape[0]
    require(targets > 0 and sequences == len(dataset), "incomplete/empty validation")
    ce = numerator / targets
    require(math.isfinite(ce), "nonfinite validation CE")
    return {"validation_ce": ce, "validation_ppl": math.exp(ce) if ce < 700 else None,
            "ppl_overflow": ce >= 700, "validation_targets": targets,
            "validation_sequences": sequences, "validation_seconds": time.perf_counter() - started}


def diagnostics(model, method, dataset, plan):
    """Same first 32 manifest IDs for every method, never the training cursor.

    ERC diagnostics use a separate fixed seed, including for Vanilla so the
    coupling trajectory can be compared. AoE has no independent router proxy.
    Entropy is averaged per token from full probabilities, not mean probabilities.
    """
    started = time.perf_counter()
    config = plan.config
    device = model.token_embedding.weight.device
    kind = "aoe" if method == "aoe" else "vanilla_moe"
    probe = plan.indices[:config.diagnostic_sequences]
    counts = [torch.zeros(model.config.n_experts, device=device) for _ in model.layers]
    entropy = [0.0 for _ in model.layers]
    tokens = 0
    with isolated_evaluation(model):
        for batch in microbatches(dataset, probe, config.microbatch_size):
            batch = to_device(batch, device)
            with RoutingLossTap(model, kind) as tap:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=config.bf16):
                    output = model(**batch, return_routing_info=True)
                for li, info in enumerate(output.routing_info):
                    if kind == "aoe":
                        p = torch.stack([tap.scores[li][e] for e in range(model.config.n_experts)], -1)
                        p = p.float().softmax(-1).reshape(-1, model.config.n_experts)
                    else:
                        p = tap.probabilities[li].float().reshape(-1, model.config.n_experts)
                    entropy[li] += float(-(p * p.clamp_min(torch.finfo(p.dtype).tiny).log()).sum())
                    ids = info.diagnostics.topk_indices.reshape(-1)
                    counts[li] += torch.bincount(ids, minlength=model.config.n_experts)
            tokens += batch["input_ids"].numel()
        routing = []
        for li, count in enumerate(counts):
            frequency = count / count.sum()
            routing.append({"layer": li, "assignment_counts": count.int().tolist(),
                            "assignment_frequencies": frequency.tolist(),
                            "expert_utilization": float((count > 0).float().mean()),
                            "load_cv": float(count.std(unbiased=False) / count.mean()),
                            "router_entropy": entropy[li] / tokens})
        erc = []
        if kind == "vanilla_moe":
            torch.manual_seed(config.seed + 10000)
            for li, layer in enumerate(model.layers):
                proxy, epsilon = generate_router_proxies(layer.moe.router.weight)
                matrix = compute_activation_matrix(proxy, layer.moe.get_gate_weights())
                mask = ~torch.eye(matrix.shape[0], dtype=torch.bool, device=device)
                diag = matrix.diagonal()
                erc.append({"layer": li, "erc_raw": float(erc_loss_from_matrix(matrix)),
                            "row_violation_rate": float((matrix > diag[:, None])[mask].float().mean()),
                            "column_violation_rate": float((matrix > diag[None, :])[mask].float().mean()),
                            "epsilon_min": float(epsilon.min()), "epsilon_mean": float(epsilon.mean()),
                            "epsilon_max": float(epsilon.max())})
    return {"probe_indices": list(probe), "diagnostic_noise_seed": config.seed + 10000,
            "routing": routing, "erc_diagnostics": erc,
            "diagnostic_seconds": time.perf_counter() - started}
