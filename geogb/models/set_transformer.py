"""A full-attention Set Transformer encoder with a per-action scorer."""

import torch
from torch import nn

from geogb.observations import PairBatch
from .attention import SetAttentionBlock
from .common import pair_features, positive_integer


class SetTransformerPolicy(nn.Module):
    """Score pairs using stacked full self-attention blocks without positions.

    Parameters
    ----------
    feature_dim : int
        Input features per pair.
    embed_dim : int
        Linear input projection width, default 64.
    num_heads : int
        Attention heads per SAB; must divide embed_dim, default 4.
    depth : int
        Number of SABs, default 2.
    feedforward_dim : int
        Hidden width of each SAB's feedforward residual, default 128.
    dropout : float
        Attention-weight and residual-branch dropout probability, default 0.1.
    activation : type of nn.Module
        Feedforward activation constructor, default ReLU.

    Notes
    -----
    Returns raw (B, P) logits, with zero padded scores. Attention uses O(P**2)
    interactions per block. There are no inducing points or pooling decoder.
    Equivariance and batch/single consistency hold in evaluation mode up to
    floating-point rounding; independent dropout masks affect training samples.
    """

    def __init__(self, feature_dim: int, embed_dim: int = 64, *,
                 num_heads: int = 4, depth: int = 2, feedforward_dim: int = 128,
                 dropout: float = 0.1, activation: type[nn.Module] = nn.ReLU) -> None:
        super().__init__()
        positive_integer(feature_dim, "feature_dim")
        positive_integer(depth, "depth")
        # Construct blocks first so dimensions are validated before projections.
        self.blocks = nn.ModuleList([
            SetAttentionBlock(embed_dim, num_heads, feedforward_dim=feedforward_dim,
                              dropout=dropout, activation=activation)
            for _ in range(depth)
        ])
        self.feature_dim = feature_dim
        self.projection = nn.Linear(feature_dim, embed_dim)
        self.scorer = nn.Linear(embed_dim, 1)

    def forward(self, batch: PairBatch) -> torch.Tensor:
        """Return one score per original action, rejecting terminal states."""
        features = pair_features(batch, self.feature_dim)
        hidden = self.projection(features).masked_fill(~batch.mask.unsqueeze(-1), 0)
        for block in self.blocks:
            hidden = block(hidden, batch.mask)
        return self.scorer(hidden).squeeze(-1).masked_fill(~batch.mask, 0)
