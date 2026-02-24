# backend/pipelines/fusion.py
# RAG Fusion: generate 4 query variants → parallel Chroma searches → RRF → generate
from __future__ import annotations

import asyncio
from typing import List

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from backend.pipelines.base import BaseRAGPipeline, RAGResponse, TraceStep
from backend.ingest.embedder import get_vectorstore
from backend.llm import get_generation_llm
from backend.utils.rrf import reciprocal_rank_fusion

TOP_K_PER_QUERY = 5
FINAL_K = 5

VARIANT_PROMPT = (
    "You are a financial search query expert. Given the user question below, "
    "generate exactly 4 alternative phrasings that would help retrieve relevant "
    "financial documents. Each variant must focus on a different aspect of the question. "
    "Output ONLY the 4 questions, one per line, no numbering or extra text.\n\n"
    "Question: {query}"
)

SYSTEM_PROMPT = (
    "You are a senior financial analyst. Answer the question using ONLY the provided context. "
    "Be precise, cite key figures, and structure your answer clearly. "
    "If context is insufficient, say so explicitly."
)


class FusionRAGPipeline(BaseRAGPipeline):
    name = "fusion"
    label = "RAG Fusion"
    description = (
        "Generates 4 query variants via LLM, runs parallel Chroma searches for each, "
        "merges with Reciprocal Rank Fusion, and answers from the top-5 fused chunks."
    )
    color = "#f39c12"

    def __init__(self):
        self._vs = None
        self._llm = None

    def _init(self):
        if self._vs is None:
            self._vs = get_vectorstore()
        if self._llm is None:
            self._llm = get_generation_llm()

    def run(self, query: str) -> RAGResponse:
        self._init()
        trace: List[TraceStep] = []
        start_import = __import__("time").perf_counter

        # ── Step 1: generate query variants ──────────────────────────────
        variant_resp = self._llm.invoke([
            HumanMessage(content=VARIANT_PROMPT.format(query=query))
        ])
        raw_variants = [v.strip() for v in variant_resp.content.strip().split("\n") if v.strip()]
        variants = raw_variants[:4]
        all_queries = [query] + variants

        trace.append(TraceStep(
            step="EXPAND",
            description=f"Generated {len(variants)} query variants via {self._llm.model_name}.",
            data={"original": query, "variants": variants},
        ))

        # ── Step 2: parallel retrieval ────────────────────────────────────
        # Using a simple synchronous loop (Chroma client is not async)
        per_query_results: dict[str, list[tuple[Document, float]]] = {}
        for q in all_queries:
            docs_scores = self._vs.similarity_search_with_score(q, k=TOP_K_PER_QUERY)
            per_query_results[q] = docs_scores

        trace.append(TraceStep(
            step="RETRIEVE",
            description=(
                f"Retrieved top-{TOP_K_PER_QUERY} chunks for each of "
                f"{len(all_queries)} queries ({len(all_queries) * TOP_K_PER_QUERY} total candidate chunks)."
            ),
            data={
                q[:60]: [d.metadata.get("chunk_id", "?") for d, _ in docs]
                for q, docs in per_query_results.items()
            },
        ))

        # ── Step 3: RRF ────────────────────────────────────────────────────
        # Build ranked lists using chunk_id as doc ID
        id_to_doc: dict[str, tuple[Document, float]] = {}
        ranked_lists: list[list[str]] = []

        for q, docs_scores in per_query_results.items():
            ranked_for_q: list[str] = []
            for doc, score in docs_scores:
                cid = doc.metadata.get("chunk_id", doc.page_content[:40])
                id_to_doc[cid] = (doc, score)
                ranked_for_q.append(cid)
            ranked_lists.append(ranked_for_q)

        fused = reciprocal_rank_fusion(ranked_lists)
        top_fused = fused[:FINAL_K]

        trace.append(TraceStep(
            step="FUSE",
            description=f"RRF merged {len(id_to_doc)} unique chunks → selected top-{FINAL_K}.",
            data=[(cid[:50], round(score, 5)) for cid, score in top_fused],
        ))

        # ── Step 4: build context + generate ─────────────────────────────
        sources: list[dict] = []
        context_parts: list[str] = []

        for cid, rrf_score in top_fused:
            doc, init_score = id_to_doc[cid]
            meta = doc.metadata
            sources.append({
                "text": doc.page_content,
                "doc_name": meta.get("doc_name", "?"),
                "page": meta.get("page", "?"),
                "score": round(rrf_score, 5),
                "chunk_id": cid,
            })
            context_parts.append(
                f"[{meta.get('doc_name','?')} p.{meta.get('page','?')}]\n{doc.page_content}"
            )

        context = "\n\n---\n\n".join(context_parts)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
        ]
        answer_resp = self._llm.invoke(messages)

        trace.append(TraceStep(
            step="GENERATE",
            description=f"Generated answer from {FINAL_K} fused chunks using {self._llm.model_name}.",
            data={"context_chars": len(context)},
        ))

        return RAGResponse(
            answer=answer_resp.content,
            sources=sources,
            trace=trace,
            latency_ms=0,  # filled by _timed_run
            pipeline_name=self.name,
        )
