"""
Pipeline 5 — Hybrid RAG (Dense + Sparse)
BM25 keyword search + vector similarity retrieval → Reciprocal Rank Fusion →
answer from fused results.
"""
from __future__ import annotations

from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from rank_bm25 import BM25Okapi

from .base import BaseRAGPipeline, RAGResult
from utils.ingest import get_vector_store, get_raw_chunks
from utils.llm import get_llm

TOP_K_DENSE = 10
TOP_K_SPARSE = 10
TOP_K_FINAL = 6
RRF_K = 60  # standard RRF constant

SYSTEM_PROMPT = (
    "You are a financial analyst assistant. "
    "Answer the user's question using ONLY the provided context. "
    "If the context does not contain enough information, say so clearly."
)


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return text.lower().split()


def reciprocal_rank_fusion(
    ranked_lists: List[List[str]],
    k: int = RRF_K,
) -> List[Tuple[str, float]]:
    """
    Merge multiple ranked lists via Reciprocal Rank Fusion.
    Returns list of (doc_id, rrf_score) sorted by score descending.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRAGPipeline(BaseRAGPipeline):
    name = "Hybrid RAG"
    description = (
        "Combines BM25 keyword search (sparse) with vector similarity (dense). "
        "Results are merged via Reciprocal Rank Fusion to leverage both exact "
        "keyword matches and semantic similarity."
    )
    color = "#c0392b"

    def __init__(self):
        self._store = None
        self._llm = None
        self._chunks: List[Document] = []
        self._bm25: BM25Okapi | None = None

    def _init(self):
        if self._store is None:
            self._store = get_vector_store()
        if self._llm is None:
            self._llm = get_llm()
        if self._bm25 is None:
            self._chunks = get_raw_chunks()
            tokenized = [_tokenize(c.page_content) for c in self._chunks]
            self._bm25 = BM25Okapi(tokenized)

    def run(self, query: str) -> RAGResult:
        self._init()

        trace = ["**Step 1 — Dual retrieval: BM25 (sparse) + Vector (dense)**"]

        # ── Sparse: BM25 ─────────────────────────────────────────────────
        query_tokens = _tokenize(query)
        bm25_scores = self._bm25.get_scores(query_tokens)
        bm25_ranked_indices = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:TOP_K_SPARSE]

        bm25_results = {
            str(idx): self._chunks[idx] for idx in bm25_ranked_indices
        }
        bm25_ranked_ids = [str(i) for i in bm25_ranked_indices]

        trace.append(f"- BM25 top-{TOP_K_SPARSE} chunks retrieved")
        for rank, idx in enumerate(bm25_ranked_indices[:5], 1):
            doc = self._chunks[idx]
            src = f"{doc.metadata.get('source', '?')} p.{doc.metadata.get('page', '?')}"
            trace.append(f"  {rank}. {src} (BM25 score: {bm25_scores[idx]:.3f})")

        # ── Dense: vector similarity ──────────────────────────────────────
        dense_docs = self._store.similarity_search_with_score(query, k=TOP_K_DENSE)

        dense_id_map: dict[str, tuple[Document, float]] = {}
        dense_ranked_ids: list[str] = []
        for i, (doc, score) in enumerate(dense_docs):
            dense_id = f"dense_{i}"
            dense_id_map[dense_id] = (doc, score)
            dense_ranked_ids.append(dense_id)

        trace.append(f"- Vector similarity top-{TOP_K_DENSE} chunks retrieved")
        for rank, (doc, score) in enumerate(dense_docs[:5], 1):
            src = f"{doc.metadata.get('source', '?')} p.{doc.metadata.get('page', '?')}"
            trace.append(f"  {rank}. {src} (cosine score: {score:.4f})")

        # ── RRF fusion ────────────────────────────────────────────────────
        trace.append("\n**Step 2 — Reciprocal Rank Fusion (RRF)**")
        trace.append(f"- RRF constant k={RRF_K}")

        # We need a unified ID space: BM25 uses "str(idx)", dense uses "dense_i"
        fused = reciprocal_rank_fusion([bm25_ranked_ids, dense_ranked_ids])

        trace.append(f"- Top fused results (showing first {TOP_K_FINAL}):")
        sources = []
        chunks = []
        for doc_id, rrf_score in fused[:TOP_K_FINAL]:
            if doc_id.startswith("dense_"):
                doc, _ = dense_id_map[doc_id]
            else:
                doc = bm25_results.get(doc_id)
                if doc is None:
                    continue
            src = f"{doc.metadata.get('source', '?')} p.{doc.metadata.get('page', '?')}"
            sources.append(src)
            chunks.append(doc.page_content)
            trace.append(f"  • {src} — RRF score: {rrf_score:.5f}")

        # ── Answer generation ─────────────────────────────────────────────
        trace.append("\n**Step 3 — Answer generation from fused context**")
        context = "\n\n---\n\n".join(
            f"[Source: {src}]\n{chunk}" for src, chunk in zip(sources, chunks)
        )
        trace.append(f"- Combined context length: {len(context)} chars")

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
        ]
        response = self._llm.invoke(messages)

        return RAGResult(
            answer=response.content,
            sources=list(dict.fromkeys(sources)),
            reasoning_trace="\n".join(trace),
            retrieved_chunks=chunks,
            pipeline_name=self.name,
        )
