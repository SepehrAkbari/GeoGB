"""Architecture-independent pair representations."""

from .pairs import PairBatch, PairObservation, collate_pairs

__all__ = ["PairBatch", "PairObservation", "collate_pairs"]
