"""Small update-boundary adapter for the existing notebook training loop.

No data loading, CE/LB implementation, optimizer construction or checkpoint I/O.
Call finish_optimizer_step once after all target-weighted CE/LB backwards.
It adds ERC once, unscales if needed, clips the combined gradients, then steps.
Model forward remains CE-only. Layer mean/noise timing are project policies.
"""

from dataclasses import dataclass
import math

import torch

from .config import ERCConfig
from .erc_loss import compute_erc_loss, mean_erc_loss
from .model import AoELanguageModel, MoELanguageModel
from .moe import SparseMoE


@dataclass(frozen=True)
class ModelERCResult:
    raw_loss: torch.Tensor
    weighted_loss: torch.Tensor
    metrics: dict


def _validate_model(model: MoELanguageModel, config: ERCConfig) -> None:
    if not isinstance(config, ERCConfig):
        raise TypeError("config must be ERCConfig")
    if not isinstance(model, MoELanguageModel):
        raise TypeError("model must be a MoELanguageModel or AoELanguageModel")
    if config.enabled and isinstance(model, AoELanguageModel):
        raise ValueError("original ERC requires a Vanilla MoE router; ERC on AoE is unsupported")


def compute_model_erc(model: MoELanguageModel, config: ERCConfig) -> ModelERCResult:
    """Compute raw/weighted ERC and graph-free Python logging metrics.

    This explicit helper is never called by regular model forward. Disabled,
    zero-coefficient and eval cases do not stack gates, call ERC or draw noise.
    An explicitly enabled AoE config is rejected, even at coefficient zero.
    Active ERC requires autograd, so accidentally calling it inside no_grad is
    an error rather than a silently detached training loss.
    """
    _validate_model(model, config)
    if not config.enabled or config.coefficient == 0 or not model.training:
        zero = model.token_embedding.weight.new_zeros((), dtype=torch.float32)
        return ModelERCResult(zero, zero, {
            "erc_applied": False, "erc_raw": 0.0, "erc_weighted": 0.0,
            "erc_per_layer": [],
        })
    if not torch.is_grad_enabled():
        raise RuntimeError("training ERC requires autograd; use model.eval() for inference")
    if not model.layers or any(not isinstance(layer.moe, SparseMoE) for layer in model.layers):
        raise ValueError("ERC requires nonempty Vanilla SparseMoE layers")
    losses = tuple(
        compute_erc_loss(layer.moe.router.weight, layer.moe.get_gate_weights(), config.alpha)
        for layer in model.layers
    )
    raw = mean_erc_loss(losses)
    weighted = raw * config.coefficient
    if not torch.isfinite(weighted).item():
        raise ValueError("weighted ERC is nonfinite")
    return ModelERCResult(raw, weighted, {
        "erc_applied": True, "erc_raw": float(raw.detach()),
        "erc_weighted": float(weighted.detach()),
        "erc_per_layer": [float(loss.detach()) for loss in losses],
    })


def validate_erc_checkpoint_config(stored: dict, expected: ERCConfig) -> None:
    """Validate payload['erc_config'] before restoring any model/optimizer/RNG.

    Does not read/write checkpoints. Legacy/missing configs must not silently
    become a new default. Existing checkpoint code must also save/restore its
    normal CPU/CUDA RNG states; ERC uses those generators, with no new RNG state.
    """
    if not isinstance(expected, ERCConfig):
        raise TypeError("expected must be ERCConfig")
    restored = ERCConfig.from_dict(stored)
    if restored != expected:
        raise ValueError("ERC checkpoint config mismatch")


def finish_optimizer_step(
    model: MoELanguageModel, config: ERCConfig, optimizer: torch.optim.Optimizer,
    scheduler=None, *, ce: float, lb_raw: float, lb_weighted: float,
    global_step: int, effective_targets: int, clip_norm: float = 1.0, scaler=None,
) -> dict:
    """Finish ONE update after the caller's accumulated CE/LB backwards.

    ce/lb_* are detached sums weighted by q_m=targets_m/sum(targets).
    global_step is the number of completed updates before this call. The caller
    owns data cursor/input-token counters and commits them only after success.
    Do not zero/unscale/clip/step again between the data backwards and this call.
    With an enabled GradScaler, ALL data backwards must use that same scaler.
    Nonfinite combined gradients fail before stepping (no silent scaler retry).
    No zero_grad is performed here, preserving the Step 6 boundary convention.
    """
    _validate_model(model, config)
    if not model.training or not torch.is_grad_enabled():
        raise RuntimeError("finish_optimizer_step requires training mode and autograd")
    if type(global_step) is not int or global_step < 0:
        raise ValueError("global_step must be a nonnegative integer")
    if type(effective_targets) is not int or effective_targets <= 0:
        raise ValueError("effective_targets must be a positive integer")
    for name, value in (("ce", ce), ("lb_raw", lb_raw), ("lb_weighted", lb_weighted),
                        ("clip_norm", clip_norm)):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            raise ValueError(f"{name} must be a detached finite Python number")
    if clip_norm <= 0:
        raise ValueError("clip_norm must be positive")
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer_parameters = [p for group in optimizer.param_groups for p in group["params"]]
    if (len(optimizer_parameters) != len(parameters)
            or {id(p) for p in optimizer_parameters} != {id(p) for p in parameters}):
        raise ValueError("optimizer must contain each trainable model parameter exactly once")
    if scheduler is not None and scheduler.optimizer is not optimizer:
        raise ValueError("scheduler belongs to a different optimizer")
    if scheduler is not None and scheduler.last_epoch != global_step:
        raise ValueError("scheduler/global_step mismatch under the Step 6 update convention")
    if not any(p.grad is not None for p in parameters):
        raise RuntimeError("accumulated CE/LB gradients are missing; backward data losses first")

    result = compute_model_erc(model, config)
    scaled = scaler is not None and scaler.is_enabled()
    if result.metrics["erc_applied"]:
        if scaled:
            scaler.scale(result.weighted_loss).backward()
        else:
            result.weighted_loss.backward()
    if scaled:
        scaler.unscale_(optimizer)
    # Includes CE, LB and ERC. Nonfinite gradients raise before optimizer.step.
    norm = torch.nn.utils.clip_grad_norm_(parameters, clip_norm, error_if_nonfinite=True)
    lr_used = [float(group["lr"]) for group in optimizer.param_groups]
    if scaled:
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < previous_scale:
            raise RuntimeError("GradScaler skipped optimizer step; do not advance counters")
    else:
        optimizer.step()
    if scheduler is not None:
        scheduler.step()
        if scheduler.last_epoch != global_step + 1:
            raise RuntimeError("scheduler did not advance exactly once")
    return {
        "ce": float(ce), "lb_raw": float(lb_raw), "lb_weighted": float(lb_weighted),
        **result.metrics,
        "total_loss": float(ce + lb_weighted + result.metrics["erc_weighted"]),
        "global_optimizer_step": global_step + 1,
        "effective_targets": effective_targets,
        "gradient_norm_before_clip": float(norm.detach()),
        "lr_used": lr_used,
        "lr_next": [float(group["lr"]) for group in optimizer.param_groups],
    }
