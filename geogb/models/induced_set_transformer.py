"""An inducing-point Set Transformer with one score per candidate pair."""

import torch
from torch import nn

from geogb.observations import PairBatch
from .attention import InducedSetAttentionBlock
from .common import pair_features, positive_integer


class InducedSetTransformerPolicy(nn.Module):
    """Score candidates using stacked ISABs and a shared linear scorer.

    Parameters
    ----------
    feature_dim : int
        Input features per pair.
    embed_dim : int
        Input projection and attention width, default 64.
    num_heads : int
        Attention heads in both stages; must divide embed_dim, default 4.
    num_inducing : int
        Learned inducing vectors per block, default 16. Any positive count is
        valid, including counts larger than the number of candidates.
    depth : int
        Number of ISABs, default 2. Each has its own inducing vectors and MABs.
    feedforward_dim : int
        Hidden width of each MAB's feedforward residual, default 128.
    dropout : float
        Attention-weight and residual-branch dropout probability, default 0.1.
    activation : type of nn.Module
        Feedforward activation constructor, default ReLU.

    Notes
    -----
    Accepts PairBatch and returns raw (B, P) logits in original action order,
    with zero padding. Empty states are rejected; use masked_categorical for
    action selection. There are no positions or action-collapsing pooling.
    Both cross-attention stages cost O(P*m*d) for fixed depth and width; projection
    and rowwise feedforward costs also remain. Dropout must be disabled with
    eval() for deterministic permutation and batch/single comparisons.
    """

    def __init__(self, feature_dim: int, embed_dim: int = 64, *,
                 num_heads: int = 4, num_inducing: int = 16, depth: int = 2,
                 feedforward_dim: int = 128, dropout: float = 0.1,
                 activation: type[nn.Module] = nn.ReLU) -> None:
        super().__init__()
        positive_integer(feature_dim, "feature_dim")
        positive_integer(depth, "depth")
        self.blocks = nn.ModuleList([
            InducedSetAttentionBlock(embed_dim, num_heads, num_inducing=num_inducing,
                                     feedforward_dim=feedforward_dim, dropout=dropout,
                                     activation=activation)
            for _ in range(depth)
        ])
        self.feature_dim = feature_dim
        self.projection = nn.Linear(feature_dim, embed_dim)
        self.scorer = nn.Linear(embed_dim, 1)

    def forward(self, batch: PairBatch) -> torch.Tensor:
        """Return one raw score per original action, rejecting terminal states."""
        features = pair_features(batch, self.feature_dim)
        hidden = self.projection(features).masked_fill(~batch.mask.unsqueeze(-1), 0)
        for block in self.blocks:
            hidden = block(hidden, batch.mask)
        return self.scorer(hidden).squeeze(-1).masked_fill(~batch.mask, 0)
