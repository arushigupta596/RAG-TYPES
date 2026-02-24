# backend/pipelines/self_rag.py
# Self-RAG: retrieve → grade each chunk (ISREL) → filter → generate → grade answer (ISSUP)
from __future__ import annotations

import json
from typing import List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

from backend.pipelines.base import BaseRAGPipeline, RAGResponse, TraceStep
from backend.pipelines.self_rag_prompts import ISREL_PROMPT, ISSUP_PROMPT
from backend.ingest.embedder import get_vectorstore
from backend.llm import get_generation_llm, get_grading_llm

RETRIEVE_K = 8
ISREL_THRESHOLD = 0.4

SYSTEM_PROMPT = (
    "You are a senior financial analyst. Answer the question using ONLY the "
    "provided context. Be precise, structured, and evidence-based. "
    "If context is insufficient, say so explicitly."
)


def _parse_grade(raw: str, field: str = "score") -> Tuple[float, str]:
    """Safely parse JSON grade response. Returns (score, reason)."""
    try:
        data = json.loads(raw.strip())
        score = float(data.get("score", 0.3))
        reason = str(data.get("reason", "no reason given"))
        return score, reason
    except Exception:
        # Fallback: try to extract a float
        import re
        nums = re.findall(r"0\.\d+|\d+\.\d+", raw)
        score = float(nums[0]) if nums else 0.3
        return score, f"parse error — raw: {raw[:80]}"


class SelfRAGPipeline(BaseRAGPipeline):
    name = "self_rag"
    label = "Self-RAG"
    description = (
        "Retrieves top-8 chunks, grades each for relevance (ISREL) with the grading LLM, "
        "filters below 0.4, generates an answer, then grades the answer's support (ISSUP). "
        "Every score is shown in the trace."
    )
    color = "#e74c3c"

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

    def run(self, query: str) -> RAGResponse:
        self._init()
        trace: List[TraceStep] = []

        # ── Step 1: retrieve ───────────────────────────────────────────────
        docs_scores = self._vs.similarity_search_with_score(query, k=RETRIEVE_K)
        trace.append(TraceStep(
            step="RETRIEVE",
            description=f"Retrieved top-{RETRIEVE_K} candidate chunks by cosine similarity.",
            data=[
                {
                    "doc_name": d.metadata.get("doc_name", "?"),
                    "page": d.metadata.get("page", "?"),
                    "cosine_score": round(float(s), 4),
                }
                for d, s in docs_scores
            ],
        ))

        # ── Step 2: grade each chunk (ISREL) ──────────────────────────────
        relevant_chunks: list[dict] = []
        isrel_results: list[dict] = []

        for doc, cosine_score in docs_scores:
            meta = doc.metadata
            prompt = ISREL_PROMPT.format(query=query, chunk=doc.page_content[:600])
            resp = self._grade_llm.invoke([HumanMessage(content=prompt)])
            isrel_score, isrel_reason = _parse_grade(resp.content)

            kept = isrel_score >= ISREL_THRESHOLD
            isrel_results.append({
                "doc_name": meta.get("doc_name", "?"),
                "page": meta.get("page", "?"),
                "isrel_score": round(isrel_score, 3),
                "reason": isrel_reason,
                "kept": kept,
            })

            if kept:
                relevant_chunks.append({
                    "doc": doc,
                    "meta": meta,
                    "isrel_score": isrel_score,
                    "cosine_score": float(cosine_score),
                })

        trace.append(TraceStep(
            step="GRADE",
            description=(
                f"ISREL grading complete. {len(relevant_chunks)}/{RETRIEVE_K} chunks "
                f"passed threshold ({ISREL_THRESHOLD}). Grading model: {self._grade_llm.model_name}."
            ),
            data=isrel_results,
        ))

        if not relevant_chunks:
            trace.append(TraceStep(
                step="ERROR",
                description="No chunks passed ISREL threshold. Returning fallback answer.",
            ))
            return RAGResponse(
                answer=(
                    "The retrieved documents did not contain sufficiently relevant "
                    "information to answer this question confidently."
                ),
                sources=[],
                trace=trace,
                latency_ms=0,
                pipeline_name=self.name,
            )

        # ── Step 3: generate answer ────────────────────────────────────────
        context_parts = []
        sources: list[dict] = []
        for item in relevant_chunks:
            doc = item["doc"]
            meta = item["meta"]
            context_parts.append(
                f"[{meta.get('doc_name','?')} p.{meta.get('page','?')}]\n{doc.page_content}"
            )
            sources.append({
                "text": doc.page_content,
                "doc_name": meta.get("doc_name", "?"),
                "page": meta.get("page", "?"),
                "score": item["isrel_score"],
                "chunk_id": meta.get("chunk_id", "?"),
            })

        context = "\n\n---\n\n".join(context_parts)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
        ]
        answer_resp = self._gen_llm.invoke(messages)
        answer = answer_resp.content

        trace.append(TraceStep(
            step="GENERATE",
            description=(
                f"Generated answer from {len(relevant_chunks)} relevant chunks "
                f"using {self._gen_llm.model_name}."
            ),
        ))

        # ── Step 4: grade answer support (ISSUP) ─────────────────────────
        issup_prompt = ISSUP_PROMPT.format(context=context[:2000], answer=answer)
        issup_resp = self._grade_llm.invoke([HumanMessage(content=issup_prompt)])
        issup_score, issup_reason = _parse_grade(issup_resp.content)

        trace.append(TraceStep(
            step="GRADE",
            description=(
                f"ISSUP (answer support) score: {issup_score:.2f} — {issup_reason}"
            ),
            data={"issup_score": round(issup_score, 3), "reason": issup_reason},
        ))

        return RAGResponse(
            answer=answer,
            sources=sources,
            trace=trace,
            latency_ms=0,
            pipeline_name=self.name,
        )
