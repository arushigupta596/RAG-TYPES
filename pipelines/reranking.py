"""
Pipeline 7 — Reranking RAG
Retrieve a broad candidate set → LLM-based pairwise reranker scores
each chunk against the query → top reranked chunks go to answer generation.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from .base import BaseRAGPipeline, RAGResult
from utils.ingest import get_vector_store
from utils.llm import get_llm

CANDIDATE_K = 12
FINAL_K = 5

RERANK_PROMPT = (
    "You are a relevance scoring system for a financial Q&A pipeline. "
    "Given a user question and a document passage, rate how relevant the passage "
    "is for answering the question on a scale from 0 to 10. "
    "Respond with ONLY a single integer (0-10), nothing else.\n\n"
    "Question: {query}\n\nPassage:\n{passage}"
)

SYSTEM_PROMPT = (
    "You are a financial analyst assistant. "
    "Answer the user's question using ONLY the provided context. "
    "If the context does not contain enough information, say so clearly."
)


class RerankingRAGPipeline(BaseRAGPipeline):
    name = "Reranking RAG"
    description = (
        "Retrieves a broad candidate pool then uses an LLM to score each chunk's "
        "relevance to the query (0-10). Only the top reranked chunks are used "
        "for answer generation, dramatically improving precision."
    )
    color = "#d35400"

    def __init__(self):
        self._store = None
        self._llm = None

    def _init(self):
        if self._store is None:
            self._store = get_vector_store()
        if self._llm is None:
            self._llm = get_llm()

    def run(self, query: str) -> RAGResult:
        self._init()

        # ── Step 1: broad retrieval ───────────────────────────────────────
        trace = [f"**Step 1 — Broad retrieval (top-{CANDIDATE_K} candidates)**"]
        trace.append(f"- Query: `{query}`")

        docs_with_scores = self._store.similarity_search_with_score(query, k=CANDIDATE_K)
        candidates = []
        for doc, score in docs_with_scores:
            src = f"{doc.metadata.get('source', '?')} p.{doc.metadata.get('page', '?')}"
            candidates.append((src, doc.page_content, score))
            trace.append(f"  • {src} — initial cosine score: {score:.4f}")

        # ── Step 2: LLM reranking ─────────────────────────────────────────
        trace.append(f"\n**Step 2 — LLM relevance reranking**")
        trace.append(f"- Scoring each candidate chunk 0-10 for relevance to query")

        scored = []
        for src, content, initial_score in candidates:
            prompt = RERANK_PROMPT.format(query=query, passage=content[:600])
            resp = self._llm.invoke([HumanMessage(content=prompt)])
            raw = resp.content.strip()
            try:
                rerank_score = int("".join(c for c in raw if c.isdigit())[:2])
                rerank_score = min(max(rerank_score, 0), 10)
            except (ValueError, IndexError):
                rerank_score = 0
            scored.append((src, content, initial_score, rerank_score))

        # Sort by rerank score descending
        scored.sort(key=lambda x: x[3], reverse=True)

        trace.append("\n- Reranking results (all candidates):")
        for src, _, initial, rerank in scored:
            marker = "✓" if rerank >= 6 else "✗"
            trace.append(
                f"  {marker} {src} | initial: {initial:.3f} → rerank: {rerank}/10"
            )

        # ── Step 3: take top reranked chunks ──────────────────────────────
        top_chunks = scored[:FINAL_K]
        trace.append(
            f"\n- Selected top-{FINAL_K} chunks after reranking "
            f"(scores: {[s[3] for s in top_chunks]})"
        )

        sources = [s[0] for s in top_chunks]
        chunks = [s[1] for s in top_chunks]

        # ── Step 4: answer generation ─────────────────────────────────────
        trace.append("\n**Step 3 — Answer generation from reranked context**")
        context = "\n\n---\n\n".join(
            f"[Source: {src}]\n{chunk}" for src, chunk in zip(sources, chunks)
        )
        trace.append(f"- Reranked context length: {len(context)} chars")

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
            extra={"rerank_scores": [(s[0], s[3]) for s in scored]},
        )
