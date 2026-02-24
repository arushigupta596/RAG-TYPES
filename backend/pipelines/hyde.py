# backend/pipelines/hyde.py
# HyDE: generate hypothetical answer → embed → retrieve real docs → generate
from __future__ import annotations

from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from backend.pipelines.base import BaseRAGPipeline, RAGResponse, TraceStep
from backend.ingest.embedder import get_vectorstore
from backend.llm import get_generation_llm

TOP_K = 5

HYPOTHETICAL_PROMPT = (
    "You are a financial analyst writing a professional research note. "
    "Write a concise, factual passage of ~150 words that directly answers the "
    "following question as it might appear in a financial report or analysis. "
    "Write authoritatively — do NOT hedge with 'I don't know'. "
    "Use financial terminology naturally.\n\n"
    "Question: {query}"
)

SYSTEM_PROMPT = (
    "You are a senior financial analyst. Answer the question using ONLY the "
    "provided real document context. Be precise and cite specific data points "
    "where available. If context is insufficient, say so explicitly."
)


class HyDEPipeline(BaseRAGPipeline):
    name = "hyde"
    label = "HyDE"
    description = (
        "Generates a hypothetical answer document via LLM, embeds it, retrieves "
        "real chunks similar to that embedding, then answers from real evidence. "
        "Bridges the semantic gap between short queries and long document chunks."
    )
    color = "#8e44ad"

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

        # ── Step 1: generate hypothetical document ─────────────────────────
        hyp_resp = self._llm.invoke([
            HumanMessage(content=HYPOTHETICAL_PROMPT.format(query=query))
        ])
        hypothetical = hyp_resp.content.strip()

        trace.append(TraceStep(
            step="HYPOTHESIZE",
            description=(
                f"LLM wrote a {len(hypothetical)}-char hypothetical document "
                f"that will serve as the search vector."
            ),
            data={"hypothetical_document": hypothetical},
        ))

        # ── Step 2: embed hypothetical → retrieve real docs ─────────────────
        docs_scores = self._vs.similarity_search_with_score(hypothetical, k=TOP_K)

        sources: list[dict] = []
        context_parts: list[str] = []
        for doc, score in docs_scores:
            meta = doc.metadata
            sources.append({
                "text": doc.page_content,
                "doc_name": meta.get("doc_name", "?"),
                "page": meta.get("page", "?"),
                "score": round(float(score), 4),
                "chunk_id": meta.get("chunk_id", "?"),
            })
            context_parts.append(
                f"[{meta.get('doc_name','?')} p.{meta.get('page','?')}]\n{doc.page_content}"
            )

        trace.append(TraceStep(
            step="RETRIEVE",
            description=(
                f"Embedded the hypothetical document and retrieved top-{TOP_K} "
                f"real chunks by cosine similarity."
            ),
            data=[
                {"doc": s["doc_name"], "page": s["page"], "score": s["score"]}
                for s in sources
            ],
        ))

        # ── Step 3: generate answer from real chunks ───────────────────────
        context = "\n\n---\n\n".join(context_parts)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
        ]
        answer_resp = self._llm.invoke(messages)

        trace.append(TraceStep(
            step="GENERATE",
            description=(
                f"Generated final answer from {len(sources)} real document chunks "
                f"(not from the hypothetical). Model: {self._llm.model_name}."
            ),
        ))

        return RAGResponse(
            answer=answer_resp.content,
            sources=sources,
            trace=trace,
            latency_ms=0,
            pipeline_name=self.name,
        )
