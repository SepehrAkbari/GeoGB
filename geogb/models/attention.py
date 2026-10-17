"""Masked attention blocks reusable for self-attention and future inducing sets."""

import math

import torch
from torch import nn

from .common import feedforward, positive_integer


class MultiheadAttentionBlock(nn.Module):
    """Post-normalized multihead attention with a rowwise feedforward residual.

    Queries and key/value rows have the same embed_dim but may have different
    lengths. Masks use True for valid rows, the opposite of PyTorch's
    key_padding_mask. Every context must have at least one valid key. Invalid
    query outputs are zero. This block has no positional information.

    Attention weight dropout and both residual-branch dropouts use ``dropout``.
    The feedforward network is Linear(embed_dim, feedforward_dim), activation,
    Linear(feedforward_dim, embed_dim). LayerNorm acts only on features.
    """

    def __init__(self, embed_dim: int = 64, num_heads: int = 4, *,
                 feedforward_dim: int = 128, dropout: float = 0.1,
                 activation: type[nn.Module] = nn.ReLU) -> None:
        super().__init__()
        positive_integer(embed_dim, "embed_dim")
        positive_integer(num_heads, "num_heads")
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if not math.isfinite(dropout) or not 0 <= dropout <= 1:
            raise ValueError("dropout must be finite and in [0, 1]")
        self.embed_dim = embed_dim
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout,
                                               batch_first=True)
        self.feedforward = feedforward(embed_dim, embed_dim, hidden_dim=feedforward_dim,
                                       depth=2, activation=activation)
        self.attention_dropout = nn.Dropout(dropout)
        self.feedforward_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(embed_dim)
        self.feedforward_norm = nn.LayerNorm(embed_dim)

    def forward(self, queries: torch.Tensor, context: torch.Tensor,
                query_mask: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
        """Attend from (B, Q, d) queries to (B, K, d) masked keys and values."""
        for rows, mask in ((queries, query_mask), (context, context_mask)):
            if rows.ndim != 3 or rows.shape[-1] != self.embed_dim:
                raise ValueError("attention rows must have shape (B, length, embed_dim)")
            if mask.dtype != torch.bool or mask.shape != rows.shape[:2] or mask.device != rows.device:
                raise ValueError("attention mask must be Boolean and match rows shape and device")
        if queries.shape[0] == 0 or queries.shape[0] != context.shape[0]:
            raise ValueError("queries and context must share a positive batch size")
        if not context_mask.any(dim=1).all():
            raise ValueError("attention context must have at least one valid key per state")
        # A key_padding_mask alone cannot remove NaNs before Q/K/V projection.
        queries = queries.masked_fill(~query_mask.unsqueeze(-1), 0)
        context = context.masked_fill(~context_mask.unsqueeze(-1), 0)
        attended, _ = self.attention(queries, context, context,
                                     key_padding_mask=~context_mask, need_weights=False)
        hidden = self.attention_norm(queries + self.attention_dropout(attended))
        output = self.feedforward_norm(hidden + self.feedforward_dropout(self.feedforward(hidden)))
        return output.masked_fill(~query_mask.unsqueeze(-1), 0)


class SetAttentionBlock(nn.Module):
    """Self-attention specialization: SAB(X) = MAB(X, X)."""

    def __init__(self, embed_dim: int = 64, num_heads: int = 4, *,
                 feedforward_dim: int = 128, dropout: float = 0.1,
                 activation: type[nn.Module] = nn.ReLU) -> None:
        super().__init__()
        self.block = MultiheadAttentionBlock(embed_dim, num_heads,
                                             feedforward_dim=feedforward_dim,
                                             dropout=dropout, activation=activation)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return masked contextual features in the input row order."""
        return self.block(features, features, mask, mask)
