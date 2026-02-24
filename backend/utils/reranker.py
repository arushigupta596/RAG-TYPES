# backend/utils/reranker.py
# Optional cross-encoder reranking using sentence-transformers
from __future__ import annotations

from typing import List, Tuple

from langchain_core.documents import Document

_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:
            _cross_encoder = None
    return _cross_encoder


def rerank(
    query: str,
    docs: List[Document],
    top_k: int = 5,
) -> List[Tuple[Document, float]]:
    """
    Rerank documents using a cross-encoder model.
    Falls back to original order with score=1.0 if model unavailable.

    Returns: [(doc, score), ...] sorted by score descending.
    """
    model = _get_cross_encoder()
    if model is None or not docs:
        return [(doc, 1.0) for doc in docs[:top_k]]

    pairs = [(query, doc.page_content[:512]) for doc in docs]
    scores = model.predict(pairs)

    scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return scored[:top_k]
