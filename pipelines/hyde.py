"""
Pipeline 4 — HyDE (Hypothetical Document Embeddings)
Generate a hypothetical answer → embed that answer → retrieve chunks
similar to the hypothetical → answer from retrieved real chunks.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from .base import BaseRAGPipeline, RAGResult
from utils.ingest import get_vector_store, get_embeddings
from utils.llm import get_llm

TOP_K = 5

HYDE_PROMPT = (
    "You are a financial analyst. Write a concise, factual passage (~150 words) "
    "that directly answers the following question as if you were writing it in a "
    "professional financial report. Do NOT say you don't know — write the most "
    "plausible answer based on general financial knowledge.\n\n"
    "Question: {query}"
)

SYSTEM_PROMPT = (
    "You are a financial analyst assistant. "
    "Answer the user's question using ONLY the provided context from real documents. "
    "If the context does not contain enough information, say so clearly."
)


class HyDEPipeline(BaseRAGPipeline):
    name = "HyDE"
    description = (
        "Hypothetical Document Embeddings: asks the LLM to write a plausible answer, "
        "embeds that hypothetical, then retrieves real chunks similar to it. "
        "Closes the semantic gap between short queries and long document chunks."
    )
    color = "#8e44ad"

    def __init__(self):
        self._store = None
        self._llm = None
        self._embeddings = None

    def _init(self):
        if self._store is None:
            self._store = get_vector_store()
        if self._llm is None:
            self._llm = get_llm()
        if self._embeddings is None:
            self._embeddings = get_embeddings()

    def run(self, query: str) -> RAGResult:
        self._init()

        # ── Step 1: generate hypothetical document ────────────────────────
        trace = ["**Step 1 — Hypothetical document generation**"]
        trace.append(f"- Query: `{query}`")
        trace.append("- Asking LLM to write a plausible financial passage that answers the query")

        hyde_resp = self._llm.invoke([
            HumanMessage(content=HYDE_PROMPT.format(query=query))
        ])
        hypothetical = hyde_resp.content.strip()
        preview = hypothetical[:200].replace("\n", " ")
        trace.append(f"- Hypothetical document generated ({len(hypothetical)} chars):")
        trace.append(f"  > `{preview}…`")

        # ── Step 2: embed hypothetical & retrieve ─────────────────────────
        trace.append("\n**Step 2 — Embed hypothetical → retrieve real chunks**")
        trace.append("- Embedding the hypothetical document (not the original query)")
        trace.append(f"- Retrieving top-{TOP_K} real chunks by similarity to hypothetical embedding")

        docs_with_scores = self._store.similarity_search_with_score(hypothetical, k=TOP_K)

        sources = []
        chunks = []
        for doc, score in docs_with_scores:
            src = f"{doc.metadata.get('source', '?')} p.{doc.metadata.get('page', '?')}"
            sources.append(src)
            chunks.append(doc.page_content)
            trace.append(f"  • {src} — similarity score: {score:.4f}")

        # ── Step 3: answer generation ─────────────────────────────────────
        trace.append("\n**Step 3 — Answer generation from real retrieved chunks**")
        context = "\n\n---\n\n".join(
            f"[Source: {src}]\n{chunk}" for src, chunk in zip(sources, chunks)
        )
        trace.append(f"- Context length: {len(context)} chars")
        trace.append("- Note: LLM answers from *real* documents, not the hypothetical")

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
            extra={"hypothetical_document": hypothetical},
        )
