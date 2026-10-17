"""Categorical policies with explicit action validity."""

import torch
from torch.distributions import Categorical


def masked_categorical(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    """Construct categorical policies from floating (B, P) logits and bool masks.

    Every state must have at least one valid action and finite valid logits.
    Invalid entries are replaced by negative infinity, giving exactly zero
    probability and gradient. PyTorch's Categorical entropy clamps log-probs
    internally before multiplication, avoiding zero times negative infinity.
    Never pass terminal states here. Invalid-action log-probs are -inf.
    """
    if logits.ndim != 2 or logits.shape[0] == 0 or not logits.is_floating_point():
        raise ValueError("logits must be a floating matrix with shape (B, P)")
    if mask.dtype != torch.bool or mask.shape != logits.shape or mask.device != logits.device:
        raise ValueError("mask must be Boolean and match logits shape and device")
    if not mask.any(dim=1).all():
        raise ValueError("each state must have at least one valid action")
    if not torch.isfinite(logits[mask]).all():
        raise ValueError("valid action logits must be finite")
    return Categorical(logits=logits.masked_fill(~mask, -torch.inf))
