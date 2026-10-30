"""Load-balancing tap ported from notebook Step 6.5B-RETEST cell 84.

Only autocast device selection is generalized to CPU/CUDA. Routing is untouched.
"""
import torch
from .pilot_checkpoint_core import rt_require

class RoutingLossTap:
    def __init__(self, model, kind):
        self.model, self.kind = model, kind
        self.handles = []
        self.probabilities = {}
        self.scores = {i: {} for i in range(len(model.layers))}

    def __enter__(self):
        try:
            for li, layer in enumerate(self.model.layers):
                if self.kind == "vanilla_moe":
                    def capture(module, args, output, index=li):
                        self.probabilities[index] = output.probabilities
                    self.handles.append(layer.moe.router.register_forward_hook(capture))
                else:
                    for ei, expert in enumerate(layer.moe.experts):
                        def capture(module, args, output, layer_index=li, expert_index=ei):
                            with torch.autocast(device_type=output.device.type, enabled=False):
                                self.scores[layer_index][expert_index] = (
                                    torch.linalg.vector_norm(output.float(), ord=2, dim=-1)
                                )
                            # Không return output mới: không thay đổi routing.
                        self.handles.append(expert.gate_down_proj.register_forward_hook(capture))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def raw_loss(self, routing_info, verify=False):
        losses = []
        experts = self.model.config.n_experts
        for li, layer_info in enumerate(routing_info):
            diagnostic = layer_info.diagnostics
            if self.kind == "vanilla_moe":
                probabilities = self.probabilities[li].reshape(-1, experts)
            else:
                rt_require(len(self.scores[li]) == experts, "AoE capture thiếu experts.")
                scores = torch.stack([self.scores[li][e] for e in range(experts)], dim=-1)
                with torch.autocast(device_type=scores.device.type, enabled=False):
                    probabilities = scores.float().softmax(-1)
                if verify:
                    torch.testing.assert_close(
                        probabilities.detach(),
                        diagnostic.probabilities.reshape_as(probabilities),
                        rtol=3e-6, atol=3e-7,
                    )
            indices = diagnostic.topk_indices.reshape(-1, self.model.config.top_k)
            rt_require(indices.shape[0] == probabilities.shape[0], "Routing shape sai.")
            fractions = (
                torch.bincount(indices.reshape(-1), minlength=experts).float()
                / indices.numel()
            )
            mean_probability = probabilities.mean(0)
            if verify:
                torch.testing.assert_close(
                    mean_probability.detach(), diagnostic.mean_probabilities,
                    rtol=3e-6, atol=3e-7,
                )
            losses.append(experts * (fractions * mean_probability).sum())
        return torch.stack(losses).mean()

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.probabilities.clear()
        self.scores.clear()


def compute_data_loss(model, method, batch, config, *, dtype=None):
    """STEP 6 CE/LB semantics, shared by all methods; no ERC in this function."""
    kind = "aoe" if method == "aoe" else "vanilla_moe"
    device = model.token_embedding.weight.device
    with RoutingLossTap(model, kind) as tap:
        with torch.autocast(device_type=device.type, dtype=dtype or torch.bfloat16,
                            enabled=config.bf16 or dtype is not None):
            output = model(**batch, return_routing_info=True)
        raw = tap.raw_loss(output.routing_info)
    weighted = config.lb_coefficient * raw
    total = output.loss + weighted
    for name, value in (("ce", output.loss), ("lb_raw", raw), ("total", total)):
        rt_require(value.ndim == 0 and bool(torch.isfinite(value)), f"nonfinite {name}")
    if torch.is_grad_enabled():
        rt_require(raw.requires_grad, "load balancing was detached")
    return {"output": output, "ce": output.loss, "lb_raw": raw,
            "lb_weighted": weighted, "total": total}
