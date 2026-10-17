"""Small shared helpers for pair policies and rowwise feedforward networks."""

from numbers import Integral

import torch
from torch import nn

from geogb.observations import PairBatch


def positive_integer(value: int, name: str) -> None:
    """Reject nonintegral or nonpositive architecture dimensions."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def feedforward(input_dim: int, output_dim: int, *, hidden_dim: int,
                depth: int, activation: type[nn.Module] = nn.ReLU,
                activate_final: bool = False) -> nn.Sequential:
    """Build a rowwise network; depth counts Linear layers, including output."""
    for name, value in (("input_dim", input_dim), ("output_dim", output_dim),
                        ("hidden_dim", hidden_dim), ("depth", depth)):
        positive_integer(value, name)
    widths = [input_dim] + [hidden_dim] * (depth - 1) + [output_dim]
    layers = []
    for i, (left, right) in enumerate(zip(widths, widths[1:])):
        layers.append(nn.Linear(left, right))
        if i < depth - 1 or activate_final:
            layers.append(activation())
    return nn.Sequential(*layers)


def pair_features(batch: PairBatch, feature_dim: int) -> torch.Tensor:
    """Validate policy inputs and sanitize padding before any projection."""
    if batch.features.shape[-1] != feature_dim:
        raise ValueError("batch feature dimension does not match model")
    if not batch.mask.any(dim=1).all():
        raise ValueError("policy cannot score a state with no valid actions")
    return batch.features.masked_fill(~batch.mask.unsqueeze(-1), 0)
