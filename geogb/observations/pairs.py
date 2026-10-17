"""Lossless NumPy observations and explicitly masked PyTorch batches."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class PairObservation:
    """Unpadded nonnegative integer rows in native action order.

    ``features`` is retained without copying; callers must not mutate it while
    in use. Zero rows denote a terminal state, and zero-valued rows are valid
    actions. Float32 conversion happens only during collation and may round
    large integers; the original NumPy data remains available here.
    """

    features: np.ndarray

    def __post_init__(self) -> None:
        x = self.features
        if not isinstance(x, np.ndarray) or x.ndim != 2 or x.shape[1] == 0:
            raise ValueError("pair features must be a NumPy matrix with positive feature dimension")
        if not np.issubdtype(x.dtype, np.integer) or np.any(x < 0):
            raise ValueError("pair features must contain nonnegative integers without padding")


@dataclass(frozen=True)
class PairBatch:
    """Float32 (B, P, d) features with Boolean (B, P) validity mask.

    Padding values carry no semantics. Masks are explicit and never inferred
    from features. Empty states can be represented, but policies reject them.
    """

    features: torch.Tensor
    mask: torch.Tensor

    def __post_init__(self) -> None:
        x, mask = self.features, self.mask
        if x.ndim != 3 or x.shape[0] == 0 or x.shape[2] == 0:
            raise ValueError("features must have shape (B, P, d), with B and d positive")
        if x.dtype != torch.float32:
            raise ValueError("features must use torch.float32")
        if mask.dtype != torch.bool or mask.shape != x.shape[:2]:
            raise ValueError("mask must be Boolean with shape (B, P)")
        if mask.device != x.device:
            raise ValueError("features and mask must share a device")
        if not torch.isfinite(x[mask]).all() or (x[mask] < 0).any():
            raise ValueError("valid features must be finite and nonnegative")

    def to(self, device: torch.device | str) -> "PairBatch":
        """Move both tensors together, retaining their dtypes."""
        return PairBatch(self.features.to(device), self.mask.to(device))


def collate_pairs(observations: Sequence[PairObservation | np.ndarray], *,
                  device: torch.device | str = "cpu") -> PairBatch:
    """Pad observations with zeros, preserving every row and its action index."""
    if len(observations) == 0:
        raise ValueError("cannot collate an empty batch")
    pairs = [x if isinstance(x, PairObservation) else PairObservation(x) for x in observations]
    d = pairs[0].features.shape[1]
    if any(x.features.shape[1] != d for x in pairs):
        raise ValueError("inconsistent feature dimensions within batch")
    p = max(len(x.features) for x in pairs)
    features = torch.zeros((len(pairs), p, d), dtype=torch.float32)
    mask = torch.zeros((len(pairs), p), dtype=torch.bool)
    for i, observation in enumerate(pairs):
        x = observation.features
        # Copy directly into the float32 batch, including read-only/strided arrays.
        features[i, :len(x)].numpy()[:] = x
        mask[i, :len(x)] = True
    return PairBatch(features, mask).to(device)
