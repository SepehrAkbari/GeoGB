"""An independent shared MLP scorer for pair rows."""

from numbers import Integral
from typing import Sequence

import torch
from torch import nn

from geogb.observations import PairBatch


class PairMLP(nn.Module):
    """Score each row independently with shared weights and ReLU activations.

    Parameters
    ----------
    feature_dim : int
        Number of features per pair.
    hidden_dims : sequence of int
        Hidden widths, default (128,). An empty sequence gives a linear scorer.
    activation : type of nn.Module
        Activation constructor, default ReLU.

    Notes
    -----
    ``model(batch)`` returns raw (B, P) logits. Use ``masked_categorical`` for
    action selection. Invalid logits are zero; the explicit mask is still
    required. This is a new initialization, with no TensorFlow weight mapping.
    Future graph policies may accept a different observation type.
    """

    def __init__(self, feature_dim: int, hidden_dims: Sequence[int] = (128,), *,
                 activation: type[nn.Module] = nn.ReLU) -> None:
        super().__init__()
        widths = (feature_dim, *hidden_dims, 1)
        if any(isinstance(w, bool) or not isinstance(w, Integral) or w < 1 for w in widths):
            raise ValueError("feature and hidden dimensions must be positive integers")
        self.feature_dim = feature_dim
        layers = []
        for i, (left, right) in enumerate(zip(widths, widths[1:])):
            layers.append(nn.Linear(left, right))
            if i < len(widths) - 2:
                layers.append(activation())
        self.scorer = nn.Sequential(*layers)

    def forward(self, batch: PairBatch) -> torch.Tensor:
        """Return one raw logit per action; reject states without actions."""
        if batch.features.shape[-1] != self.feature_dim:
            raise ValueError("batch feature dimension does not match model")
        if not batch.mask.any(dim=1).all():
            raise ValueError("policy cannot score a state with no valid actions")
        # Sanitize padding before Linear so even NaN padding has no effect.
        features = batch.features.masked_fill(~batch.mask.unsqueeze(-1), 0)
        return self.scorer(features).squeeze(-1).masked_fill(~batch.mask, 0)
