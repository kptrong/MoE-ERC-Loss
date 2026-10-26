"""Parameter-free ERC core: paper Sections 3.1--3.2 and Figure 8.

All arithmetic is FP32, including inside BF16 autocast. Proxy generation is
entirely stop-gradient, as in the authors' pseudocode; gate weights stay attached.
The layer loss is raw: no CE, load balancing, coefficient or training integration.
Mean aggregation across layers is a project policy, not a verified paper rule.
"""

from collections.abc import Iterable
import math
from numbers import Real

import torch


def _floating_finite(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a floating-point tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must contain only finite values")


def _router_shape(router_weights: torch.Tensor) -> None:
    _floating_finite(router_weights, "router_weights")
    if router_weights.ndim != 2 or router_weights.shape[0] < 2 or router_weights.shape[1] < 1:
        raise ValueError("router_weights must have shape [n,d] with n >= 2 and d > 0")


def _alpha(alpha: float) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise TypeError("alpha must be a real scalar in [0,1]")
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and in [0,1]")
    return float(alpha)


@torch.no_grad()
def generate_router_proxies(router_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return detached FP32 (proxy [n,d], epsilon [n]) on the input device.

    epsilon_i = min_{j != i} ||R_i-R_j||_2 / (2 ||R_i||_2).
    proxy_ik = R_ik * (1-epsilon_i + 2*epsilon_i*U_ik), U_ik ~ U(0,1).
    Uses the device's default PyTorch RNG (covered by normal RNG checkpoints).
    No epsilon clipping or denominator clamp is applied. Zero row norms fail;
    duplicate nonzero rows are valid and give epsilon=0. Epsilon is diagnostic
    output, not a learned parameter or a retained computation graph.
    """
    _router_shape(router_weights)
    with torch.autocast(device_type=router_weights.device.type, enabled=False):
        router = router_weights.float()
        _floating_finite(router, "FP32 router_weights")
        norms = torch.linalg.vector_norm(router, dim=1)
        if (norms == 0).any().item():
            raise ValueError("zero router-row norm: bounded ERC noise is undefined")
        _floating_finite(norms, "router-row norms")
        distances = torch.cdist(router, router, p=2)
        _floating_finite(distances, "router distances")
        distances.fill_diagonal_(float("inf"))
        epsilon = distances.min(dim=1).values / 2 / norms
        _floating_finite(epsilon, "epsilon")
        noise = torch.rand_like(router)
        epsilon_column = epsilon.unsqueeze(1)
        proxy = router * (1 - epsilon_column + 2 * epsilon_column * noise)
        _floating_finite(proxy, "proxy")
        return proxy, epsilon


def _activation_inputs(proxy: torch.Tensor, gate_weights: torch.Tensor) -> None:
    _floating_finite(proxy, "proxy")
    _floating_finite(gate_weights, "gate_weights")
    if proxy.ndim != 2 or proxy.shape[0] < 2 or proxy.shape[1] < 1:
        raise ValueError("proxy must have shape [n,d] with n >= 2 and d > 0")
    if (gate_weights.ndim != 3 or gate_weights.shape[0] != proxy.shape[0]
            or gate_weights.shape[1] < 1 or gate_weights.shape[2] != proxy.shape[1]):
        raise ValueError("gate_weights must have shape [n,D,d], matching proxy, with D > 0")
    if proxy.device != gate_weights.device:
        raise ValueError("proxy and gate_weights must be on the same device")


def compute_activation_matrix(proxy: torch.Tensor, gate_weights: torch.Tensor) -> torch.Tensor:
    """Return FP32 M[i,j] = ||gate_weights[j] @ proxy[i]||_2, shape [n,n].

    gate_weights uses PyTorch layout [n,D,d], e.g. moe.get_gate_weights().
    The projection is before SiLU; no up/output projection or bias participates.
    This helper preserves autograd of supplied inputs. Proxies from
    generate_router_proxies are already detached; never detach gate_weights.
    """
    _activation_inputs(proxy, gate_weights)
    with torch.autocast(device_type=proxy.device.type, enabled=False):
        activations = torch.einsum("jDd,id->ijD", gate_weights.float(), proxy.float())
        matrix = torch.linalg.vector_norm(activations, dim=-1)
        _floating_finite(matrix, "activation matrix")
        return matrix


def erc_loss_from_matrix(matrix: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Equation (3): sum both off-diagonal hinge terms, divided by n**2.

    Column broadcasting uses M[i,j]-alpha*M[j,j], equivalent to the paper's
    M[j,i]-alpha*M[i,i] after exchanging the summation indices.
    """
    alpha = _alpha(alpha)
    _floating_finite(matrix, "matrix")
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must have square shape [n,n] with n >= 2")
    if (matrix < 0).any().item():
        raise ValueError("matrix contains activation norms and must be nonnegative")
    with torch.autocast(device_type=matrix.device.type, enabled=False):
        value = matrix.float()
        diagonal = value.diagonal()
        # Figure 8 uses clamp, which selects a different subgradient at exactly
        # zero than torch.relu. Preserve that boundary behavior as well as values.
        row = torch.clamp(value - alpha * diagonal.unsqueeze(1), min=0.0)
        column = torch.clamp(value - alpha * diagonal.unsqueeze(0), min=0.0)
        mask = ~torch.eye(value.shape[0], dtype=torch.bool, device=value.device)
        loss = ((row + column) * mask).mean()
        _floating_finite(loss, "ERC loss")
        return loss


def compute_erc_loss(
    router_weights: torch.Tensor, gate_weights: torch.Tensor, alpha: float = 1.0,
) -> torch.Tensor:
    """Raw ERC scalar for one layer; gradient flows to gates, not the router.

    Validate shapes/alpha before drawing noise. No model state is mutated.
    The caller decides when ERC is enabled and when noise is sampled.
    """
    alpha = _alpha(alpha)
    _router_shape(router_weights)
    _activation_inputs(router_weights, gate_weights)
    proxy, _ = generate_router_proxies(router_weights)
    return erc_loss_from_matrix(compute_activation_matrix(proxy, gate_weights), alpha)


def mean_erc_loss(layer_losses: Iterable[torch.Tensor]) -> torch.Tensor:
    """Differentiable FP32 mean of a nonempty iterable of layer scalars.

    Project policy for future integration: mean across layers, once per optimizer
    step. This helper neither schedules computations nor applies a coefficient.
    """
    losses = tuple(layer_losses)
    if not losses:
        raise ValueError("layer_losses must not be empty")
    for loss in losses:
        _floating_finite(loss, "layer loss")
        if loss.ndim != 0:
            raise ValueError("each layer loss must be a scalar tensor")
    if any(loss.device != losses[0].device for loss in losses):
        raise ValueError("all layer losses must be on the same device")
    with torch.autocast(device_type=losses[0].device.type, enabled=False):
        result = torch.stack([loss.float() for loss in losses]).mean()
        _floating_finite(result, "mean ERC loss")
        return result
