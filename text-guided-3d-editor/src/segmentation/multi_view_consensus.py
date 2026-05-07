"""Multi-view voting over Gaussian indices (PRD 2.3)."""
from __future__ import annotations

import numpy as np


def consensus_indices(
    index_sets: list[np.ndarray],
    min_votes: int = 2,
) -> np.ndarray:
    """Indices that appear in at least min_votes of the views."""
    if not index_sets:
        return np.array([], dtype=np.int64)
    counts: dict[int, int] = {}
    for arr in index_sets:
        for i in np.unique(arr):
            counts[int(i)] = counts.get(int(i), 0) + 1
    out = [i for i, c in counts.items() if c >= min_votes]
    return np.array(sorted(out), dtype=np.int64)
