"""Contextual Deep Sets policies with explicitly masked aggregation."""

import torch
from torch import nn

from geogb.observations import PairBatch
from .common import feedforward, pair_features


class DeepSetsPolicy(nn.Module):
    """Score each pair using its embedding, pooled context, and log cardinality.

    Parameters
    ----------
    feature_dim : int
        Input features per pair.
    embed_dim : int
        Embedding width, default 64.
    encoder_depth, scorer_depth : int
        Number of Linear layers in phi and rho, respectively; default 2 each.
    hidden_dim : int
        Width of intermediate layers in both networks, default 64.
    activation : type of nn.Module
        Constructor for hidden activations and phi's final activation (ReLU).
    aggregation : {'mean', 'sum'}
        Masked pooling mode. Mean is the default.

    Notes
    -----
    h_i = phi(x_i), z = MaskedMean(h) (or MaskedSum), and
    logits_i = rho(concat(h_i, z, log(1 + p))). Returns raw (B, P) scores
    with zero padding. Terminal states are rejected; use masked_categorical
    for action selection. No parameters depend on action order or padded size.
    """

    def __init__(self, feature_dim: int, embed_dim: int = 64, *,
                 encoder_depth: int = 2, scorer_depth: int = 2,
                 hidden_dim: int = 64, activation: type[nn.Module] = nn.ReLU,
                 aggregation: str = "mean") -> None:
        super().__init__()
        if aggregation not in {"mean", "sum"}:
            raise ValueError("aggregation must be 'mean' or 'sum'")
        self.feature_dim = feature_dim
        self.aggregation = aggregation
        self.encoder = feedforward(feature_dim, embed_dim, hidden_dim=hidden_dim,
                                   depth=encoder_depth, activation=activation,
                                   activate_final=True)
        self.scorer = feedforward(2 * embed_dim + 1, 1, hidden_dim=hidden_dim,
                                  depth=scorer_depth, activation=activation)

    def forward(self, batch: PairBatch) -> torch.Tensor:
        """Return contextual raw logits in the original action order."""
        features = pair_features(batch, self.feature_dim)
        embeddings = self.encoder(features).masked_fill(~batch.mask.unsqueeze(-1), 0)
        count = batch.mask.sum(dim=1, keepdim=True).to(embeddings.dtype)
        context = embeddings.sum(dim=1, keepdim=True)
        if self.aggregation == "mean":
            context = context / count.unsqueeze(-1)
        context = context.expand(-1, embeddings.shape[1], -1)
        cardinality = count.log1p().unsqueeze(-1).expand(-1, embeddings.shape[1], -1)
        scores = self.scorer(torch.cat((embeddings, context, cardinality), dim=-1))
        return scores.squeeze(-1).masked_fill(~batch.mask, 0)
