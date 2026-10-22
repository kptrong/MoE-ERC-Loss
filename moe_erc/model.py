"""Decoder-only MoE language model with tied embeddings and next-token loss."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig
from .decoder import MoEDecoderBlock
from .moe import RoutingDiagnostics
from .rmsnorm import RMSNorm


@dataclass(frozen=True)
class LayerRoutingInfo:
    layer_index: int
    diagnostics: RoutingDiagnostics


@dataclass(frozen=True)
class LanguageModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    routing_info: tuple[LayerRoutingInfo, ...] | None = None


class MoELanguageModel(nn.Module):
    """Embedding -> independent Pre-Norm blocks -> final norm -> tied LM head.

    Default forward returns logits [B,T,V]. Providing labels or requesting
    routing info returns LanguageModelOutput. Labels are unshifted [B,T]; the
    model performs the next-token shift internally, ignoring targets equal to
    -100. No-target batches return differentiable zero rather than NaN.

    Diagnostics are detached, per-layer observations including *mean* full
    routing probabilities, assignment counts/fractions and Top-K indices. Full
    per-token router logits/probabilities are not retained. No auxiliary loss,
    padding mask, KV cache or extra positional parameters are introduced.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(MoEDecoderBlock(config) for _ in range(config.n_layers))
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Share the actual Parameter, not a one-time copy of its values.
        self.lm_head.weight = self.token_embedding.weight

    def _validate_inputs(self, input_ids: torch.Tensor, labels: torch.Tensor | None) -> None:
        if input_ids.ndim != 2 or input_ids.shape[0] == 0:
            raise ValueError("input_ids must have shape [B,T] with a positive batch size")
        if not 1 <= input_ids.shape[1] <= self.config.max_seq_len:
            raise ValueError(f"sequence length must be in [1, {self.config.max_seq_len}]")
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must use int32 or int64 token IDs")
        if input_ids.device != self.token_embedding.weight.device:
            raise ValueError("input_ids and model must be on the same device")
        if ((input_ids < 0) | (input_ids >= self.config.vocab_size)).any().item():
            raise ValueError("input_ids contain out-of-vocabulary token IDs")
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same [B,T] shape as input_ids")
            if labels.dtype not in (torch.int32, torch.int64):
                raise TypeError("labels must use int32 or int64 token IDs")
            if labels.device != input_ids.device:
                raise ValueError("labels and input_ids must be on the same device")
            invalid = (labels != -100) & ((labels < 0) | (labels >= self.config.vocab_size))
            if invalid.any().item():
                raise ValueError("labels must be vocabulary IDs or ignore_index=-100")

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None,
        *, return_routing_info: bool = False,
    ) -> torch.Tensor | LanguageModelOutput:
        self._validate_inputs(input_ids, labels)
        hidden = self.token_embedding(input_ids)
        routing_info = [] if return_routing_info else None
        for index, layer in enumerate(self.layers):
            if return_routing_info:
                hidden, diagnostics = layer(hidden, return_routing_info=True)
                routing_info.append(LayerRoutingInfo(index, diagnostics))
            else:
                hidden = layer(hidden)
        logits = self.lm_head(self.final_norm(hidden))
        loss = None
        if labels is not None:
            # CE uses FP32 even under BF16 autocast; logits remain in their
            # native dtype for callers. Position t predicts label t+1.
            with torch.autocast(device_type=logits.device.type, enabled=False):
                shifted_logits = logits[:, :-1, :].float()
                shifted_labels = labels[:, 1:].to(torch.int64)
                if (shifted_labels != -100).any().item():
                    loss = F.cross_entropy(
                        shifted_logits.reshape(-1, self.config.vocab_size),
                        shifted_labels.reshape(-1), ignore_index=-100,
                    )
                else:
                    loss = shifted_logits.sum() * 0.0
        if labels is None and not return_routing_info:
            return logits
        return LanguageModelOutput(
            logits=logits, loss=loss,
            routing_info=tuple(routing_info) if routing_info is not None else None,
        )
