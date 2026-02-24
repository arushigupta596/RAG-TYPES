"""
Pipeline 2 — Contextual Compression RAG
Retrieve a larger candidate set → LLM compressor filters each chunk to
only the sentences relevant to the query → pass compressed context to LLM.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from .base import BaseRAGPipeline, RAGResult
from utils.ingest import get_vector_store
from utils.llm import get_llm

CANDIDATE_K = 10   # retrieve more candidates before compression
SYSTEM_PROMPT = (
    "You are a financial analyst assistant. "
    "Answer the user's question using ONLY the provided context. "
    "If the context does not contain enough information, say so clearly."
)

COMPRESS_PROMPT = (
    "You are a document relevance filter. "
    "Given the passage below and a user question, extract ONLY the sentences "
    "directly relevant to the question. "
    "If nothing is relevant, respond with exactly: [NOT RELEVANT]\n\n"
    "Question: {query}\n\nPassage:\n{passage}"
)


class ContextualCompressionPipeline(BaseRAGPipeline):
    name = "Contextual Compression"
    description = (
        "Retrieves a larger candidate set then uses an LLM to compress each chunk "
        "down to only the sentences relevant to the query before generating the answer."
    )
    color = "#e67e22"

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

        trace = ["**Step 1 — Broad retrieval (larger candidate pool)**"]
        trace.append(f"- Retrieving top-{CANDIDATE_K} candidate chunks")

        docs_with_scores = self._store.similarity_search_with_score(query, k=CANDIDATE_K)

        sources_raw = []
        compressed_chunks = []
        kept = 0

        trace.append("\n**Step 2 — LLM-based contextual compression**")
        trace.append("- Each chunk is individually filtered by the LLM")

        for doc, score in docs_with_scores:
            src = f"{doc.metadata.get('source', '?')} p.{doc.metadata.get('page', '?')}"
            sources_raw.append(src)

            prompt = COMPRESS_PROMPT.format(query=query, passage=doc.page_content)
            resp = self._llm.invoke([HumanMessage(content=prompt)])
            compressed = resp.content.strip()

            if "[NOT RELEVANT]" in compressed:
                trace.append(f"  • {src} — ✗ filtered out (score={score:.4f})")
            else:
                kept += 1
                compressed_chunks.append((src, compressed))
                preview = compressed[:120].replace("\n", " ")
                trace.append(
                    f"  • {src} — ✓ kept (score={score:.4f}): `{preview}…`"
                )

        trace.append(f"\n- Chunks retained after compression: {kept} / {CANDIDATE_K}")

        if not compressed_chunks:
            return RAGResult(
                answer="The retrieved documents did not contain relevant information for this query.",
                sources=[],
                reasoning_trace="\n".join(trace),
                retrieved_chunks=[],
                pipeline_name=self.name,
            )

        # ── Step 3: answer generation ────────────────────────────────────
        trace.append("\n**Step 3 — Answer generation from compressed context**")
        context = "\n\n---\n\n".join(
            f"[Source: {src}]\n{chunk}" for src, chunk in compressed_chunks
        )
        trace.append(f"- Compressed context length: {len(context)} chars")

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
        ]
        response = self._llm.invoke(messages)

        return RAGResult(
            answer=response.content,
            sources=list(dict.fromkeys(src for src, _ in compressed_chunks)),
            reasoning_trace="\n".join(trace),
            retrieved_chunks=[chunk for _, chunk in compressed_chunks],
            pipeline_name=self.name,
        )
