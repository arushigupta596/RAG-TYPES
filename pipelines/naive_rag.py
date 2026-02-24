"""
Pipeline 1 — Naive RAG
Embed query → cosine similarity top-k → stuff into prompt → answer.
The simplest possible RAG baseline.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from .base import BaseRAGPipeline, RAGResult
from utils.ingest import get_vector_store
from utils.llm import get_llm

TOP_K = 5

SYSTEM_PROMPT = (
    "You are a financial analyst assistant. "
    "Answer the user's question using ONLY the provided context. "
    "If the context does not contain enough information, say so clearly. "
    "Be concise and precise."
)


class NaiveRAGPipeline(BaseRAGPipeline):
    name = "Naive RAG"
    description = (
        "The simplest RAG baseline: embed the query, retrieve the top-k most "
        "similar chunks by cosine distance, and stuff them into a single prompt."
    )
    color = "#4a90e2"

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

        # ── Step 1: embed + retrieve ──────────────────────────────────────
        trace = ["**Step 1 — Vector similarity retrieval**"]
        trace.append(f"- Query: `{query}`")
        trace.append(f"- Embedding query with `all-MiniLM-L6-v2`")
        trace.append(f"- Retrieving top-{TOP_K} chunks by cosine similarity")

        docs = self._store.similarity_search_with_score(query, k=TOP_K)

        sources = []
        chunks = []
        for doc, score in docs:
            src = f"{doc.metadata.get('source', '?')} p.{doc.metadata.get('page', '?')}"
            sources.append(src)
            chunks.append(doc.page_content)
            trace.append(f"  • {src} — similarity score: {score:.4f}")

        # ── Step 2: build prompt ──────────────────────────────────────────
        trace.append("\n**Step 2 — Prompt construction**")
        context = "\n\n---\n\n".join(
            f"[Source: {src}]\n{chunk}" for src, chunk in zip(sources, chunks)
        )
        trace.append(f"- Context length: {len(context)} chars ({len(chunks)} chunks)")
        trace.append("- Strategy: *stuff all chunks into a single prompt*")

        # ── Step 3: LLM call ──────────────────────────────────────────────
        trace.append("\n**Step 3 — LLM generation**")
        trace.append(f"- Model: `{self._llm.model_name}`")
        trace.append("- Temperature: 0.0")

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=f"Context:\n{context}\n\nQuestion: {query}"
            ),
        ]
        response = self._llm.invoke(messages)
        answer = response.content

        trace.append("- Answer generated ✓")

        return RAGResult(
            answer=answer,
            sources=list(dict.fromkeys(sources)),
            reasoning_trace="\n".join(trace),
            retrieved_chunks=chunks,
            pipeline_name=self.name,
        )
