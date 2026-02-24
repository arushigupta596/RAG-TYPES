# backend/utils/rrf.py
# Reciprocal Rank Fusion
from __future__ import annotations

from typing import List, Tuple, Dict

RRF_K = 60  # standard constant


def reciprocal_rank_fusion(
    ranked_lists: List[List[str]],
    k: int = RRF_K,
) -> List[Tuple[str, float]]:
    """
    Merge multiple ranked lists via Reciprocal Rank Fusion.

    Args:
        ranked_lists: Each inner list is a ranked sequence of doc IDs
                      (index 0 = highest rank).
        k:            RRF smoothing constant (default 60).

    Returns:
        List of (doc_id, rrf_score) sorted by score descending.
        Higher score = more relevant.
    """
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
