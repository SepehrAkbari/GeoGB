"""A small constructor helper without a registry or configuration framework."""

from typing import Any

from torch import nn

from .deep_sets import DeepSetsPolicy
from .mlp import PairMLP
from .set_transformer import SetTransformerPolicy


def make_policy(name: str, feature_dim: int, **kwargs: Any) -> nn.Module:
    """Build mlp, deep_sets, or set_transformer with model-specific keywords."""
    if name == "mlp":
        return PairMLP(feature_dim, **kwargs)
    if name == "deep_sets":
        return DeepSetsPolicy(feature_dim, **kwargs)
    if name == "set_transformer":
        return SetTransformerPolicy(feature_dim, **kwargs)
    raise ValueError("policy name must be 'mlp', 'deep_sets', or 'set_transformer'")
