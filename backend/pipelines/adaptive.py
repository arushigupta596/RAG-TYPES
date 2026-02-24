# backend/pipelines/adaptive.py
# Adaptive RAG: classify query tier → route to direct / standard / agentic pipeline
from __future__ import annotations

import json
import re
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from backend.pipelines.base import BaseRAGPipeline, RAGResponse, TraceStep
from backend.pipelines.self_rag_prompts import ADAPTIVE_ROUTE_PROMPT
from backend.ingest.embedder import get_vectorstore
from backend.llm import get_generation_llm, get_grading_llm
from backend.utils.reranker import rerank

SIMPLE_K = 3
MEDIUM_K = 5

SYSTEM_PROMPT = (
    "You are a senior financial analyst. Answer the question using ONLY the "
    "provided context. Be precise and structure your answer clearly."
)


def _parse_tier(raw: str) -> tuple[str, str]:
    """Parse JSON routing response → (tier, reason)."""
    try:
        data = json.loads(raw.strip())
        tier = str(data.get("tier", "MEDIUM")).upper()
        reason = str(data.get("reason", ""))
        if tier not in ("SIMPLE", "MEDIUM", "COMPLEX"):
            tier = "MEDIUM"
        return tier, reason
    except Exception:
        upper = raw.upper()
        for t in ("SIMPLE", "COMPLEX", "MEDIUM"):
            if t in upper:
                return t, f"extracted from: {raw[:60]}"
        return "MEDIUM", f"parse error: {raw[:60]}"


class AdaptiveRAGPipeline(BaseRAGPipeline):
    name = "adaptive"
    label = "Adaptive RAG"
    description = (
        "Classifies query complexity (SIMPLE / MEDIUM / COMPLEX) via the grading LLM. "
        "Routes to: direct top-3 lookup (SIMPLE), reranked top-5 (MEDIUM), "
        "or multi-step agentic retrieval (COMPLEX)."
    )
    color = "#3498db"

    def __init__(self):
        self._vs = None
        self._gen_llm = None
        self._grade_llm = None

    def _init(self):
        if self._vs is None:
            self._vs = get_vectorstore()
        if self._gen_llm is None:
            self._gen_llm = get_generation_llm()
        if self._grade_llm is None:
            self._grade_llm = get_grading_llm()

    # ── Tier implementations ─────────────────────────────────────────────

    def _run_simple(self, query: str, trace: List[TraceStep]) -> tuple[str, list[dict]]:
        """SIMPLE: direct Chroma search, top-3 chunks."""
        docs_scores = self._vs.similarity_search_with_score(query, k=SIMPLE_K)
        sources, context_parts = [], []
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
            description=f"SIMPLE tier: direct top-{SIMPLE_K} vector retrieval.",
            data=[{"doc": s["doc_name"], "page": s["page"], "score": s["score"]} for s in sources],
        ))
        context = "\n\n---\n\n".join(context_parts)
        return context, sources

    def _run_medium(self, query: str, trace: List[TraceStep]) -> tuple[str, list[dict]]:
        """MEDIUM: standard RAG with optional cross-encoder reranking, top-5."""
        docs_scores = self._vs.similarity_search_with_score(query, k=10)
        raw_docs = [d for d, _ in docs_scores]

        reranked = rerank(query, raw_docs, top_k=MEDIUM_K)
        sources, context_parts = [], []
        for doc, score in reranked:
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
            description=f"MEDIUM tier: retrieved 10 candidates, reranked to top-{MEDIUM_K}.",
            data=[{"doc": s["doc_name"], "page": s["page"], "score": s["score"]} for s in sources],
        ))
        context = "\n\n---\n\n".join(context_parts)
        return context, sources

    def _run_complex(self, query: str, trace: List[TraceStep]) -> tuple[str, list[dict]]:
        """
        COMPLEX: multi-step retrieval using 3 sub-question decompositions.
        Decomposes the query → retrieves for each sub-Q → deduplicates → generates.
        """
        decomp_prompt = (
            "Decompose the following complex financial question into exactly 3 specific "
            "sub-questions that together cover all aspects needed to answer it. "
            "Output ONLY the 3 questions, one per line.\n\nQuestion: {query}"
        )
        resp = self._gen_llm.invoke([
            HumanMessage(content=decomp_prompt.format(query=query))
        ])
        sub_qs = [q.strip() for q in resp.content.strip().split("\n") if q.strip()][:3]

        trace.append(TraceStep(
            step="ROUTE",
            description=f"COMPLEX tier: decomposed into {len(sub_qs)} sub-questions.",
            data={"sub_questions": sub_qs},
        ))

        seen: set[str] = set()
        sources, context_parts = [], []
        for sq in [query] + sub_qs:
            for doc, score in self._vs.similarity_search_with_score(sq, k=3):
                cid = doc.metadata.get("chunk_id", doc.page_content[:40])
                if cid not in seen:
                    seen.add(cid)
                    meta = doc.metadata
                    sources.append({
                        "text": doc.page_content,
                        "doc_name": meta.get("doc_name", "?"),
                        "page": meta.get("page", "?"),
                        "score": round(float(score), 4),
                        "chunk_id": cid,
                    })
                    context_parts.append(
                        f"[{meta.get('doc_name','?')} p.{meta.get('page','?')}]\n{doc.page_content}"
                    )

        trace.append(TraceStep(
            step="RETRIEVE",
            description=f"COMPLEX tier: retrieved {len(sources)} unique chunks across all sub-questions.",
            data=[{"sub_q": sq} for sq in sub_qs],
        ))
        context = "\n\n---\n\n".join(context_parts)
        return context, sources

    # ── Main run ─────────────────────────────────────────────────────────

    def run(self, query: str) -> RAGResponse:
        self._init()
        trace: List[TraceStep] = []

        # ── Step 1: classify tier ──────────────────────────────────────────
        route_resp = self._grade_llm.invoke([
            HumanMessage(content=ADAPTIVE_ROUTE_PROMPT.format(query=query))
        ])
        tier, tier_reason = _parse_tier(route_resp.content)

        trace.append(TraceStep(
            step="ROUTE",
            description=(
                f"Query classified as **{tier}** by {self._grade_llm.model_name}. "
                f"Reason: {tier_reason}"
            ),
            data={"tier": tier, "reason": tier_reason},
        ))

        # ── Step 2: route to appropriate retrieval tier ───────────────────
        if tier == "SIMPLE":
            context, sources = self._run_simple(query, trace)
        elif tier == "COMPLEX":
            context, sources = self._run_complex(query, trace)
        else:  # MEDIUM
            context, sources = self._run_medium(query, trace)

        # ── Step 3: generate answer ────────────────────────────────────────
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
        ]
        answer_resp = self._gen_llm.invoke(messages)

        trace.append(TraceStep(
            step="GENERATE",
            description=(
                f"Generated answer via {tier} path using {self._gen_llm.model_name}. "
                f"Context: {len(context)} chars from {len(sources)} chunks."
            ),
        ))

        return RAGResponse(
            answer=answer_resp.content,
            sources=sources,
            trace=trace,
            latency_ms=0,
            pipeline_name=self.name,
        )
