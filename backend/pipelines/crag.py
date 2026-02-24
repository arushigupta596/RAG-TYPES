# backend/pipelines/crag.py
# CRAG: retrieve → grade set → if avg < threshold trigger Tavily → merge → generate
from __future__ import annotations

import json
import re
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from backend.pipelines.base import BaseRAGPipeline, RAGResponse, TraceStep
from backend.pipelines.self_rag_prompts import CRAG_GRADE_PROMPT
from backend.ingest.embedder import get_vectorstore
from backend.llm import get_generation_llm, get_grading_llm
from backend.config import CRAG_QUALITY_THRESHOLD, TAVILY_API_KEY

RETRIEVE_K = 5

SYSTEM_PROMPT = (
    "You are a senior financial analyst. Answer the question using ONLY the "
    "provided context. Context may include both document excerpts and live web results. "
    "Clearly distinguish if you are citing web sources. "
    "If context is insufficient, say so."
)


def _parse_quality(raw: str) -> tuple[float, str]:
    try:
        data = json.loads(raw.strip())
        return float(data.get("score", 0.3)), str(data.get("reason", ""))
    except Exception:
        nums = re.findall(r"0\.\d+|\d+\.\d+", raw)
        return (float(nums[0]) if nums else 0.3), f"parse error: {raw[:60]}"


def _tavily_search(query: str, max_results: int = 3) -> list[dict]:
    """Run Tavily web search. Returns list of {title, url, content}."""
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=TAVILY_API_KEY)
        results = client.search(query=query, max_results=max_results)
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", ""),
            }
            for r in results.get("results", [])
        ]
    except Exception as e:
        return [{"title": "Web search error", "url": "", "content": str(e)}]


class CRAGPipeline(BaseRAGPipeline):
    name = "crag"
    label = "CRAG"
    description = (
        "Retrieves corpus chunks, grades the set quality with the grading LLM. "
        "If quality < 0.5, triggers a live Tavily web search fallback. "
        "Merges corpus + web evidence and generates the final answer."
    )
    color = "#1abc9c"

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

        # ── Step 1: retrieve from corpus ───────────────────────────────────
        docs_scores = self._vs.similarity_search_with_score(query, k=RETRIEVE_K)

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
            description=f"Retrieved top-{RETRIEVE_K} corpus chunks by cosine similarity.",
            data=[{"doc": s["doc_name"], "page": s["page"], "score": s["score"]} for s in sources],
        ))

        # ── Step 2: grade overall document quality ─────────────────────────
        docs_text = "\n\n".join(context_parts[:3])  # grade top-3 to save tokens
        grade_prompt = CRAG_GRADE_PROMPT.format(query=query, docs_text=docs_text[:3000])
        grade_resp = self._grade_llm.invoke([HumanMessage(content=grade_prompt)])
        quality_score, quality_reason = _parse_quality(grade_resp.content)

        web_triggered = quality_score < CRAG_QUALITY_THRESHOLD
        trace.append(TraceStep(
            step="GRADE",
            description=(
                f"Document quality score: {quality_score:.2f} "
                f"(threshold: {CRAG_QUALITY_THRESHOLD}). "
                f"Reason: {quality_reason}. "
                f"Web fallback: {'YES — triggered' if web_triggered else 'NO — skipped'}."
            ),
            data={
                "quality_score": round(quality_score, 3),
                "reason": quality_reason,
                "threshold": CRAG_QUALITY_THRESHOLD,
                "web_triggered": web_triggered,
            },
        ))

        # ── Step 3: web fallback (conditional) ────────────────────────────
        if web_triggered:
            if not TAVILY_API_KEY:
                trace.append(TraceStep(
                    step="ERROR",
                    description=(
                        "Web fallback triggered but TAVILY_API_KEY is not set. "
                        "Set it in your .env file. Proceeding with corpus only."
                    ),
                ))
            else:
                web_results = _tavily_search(query)
                trace.append(TraceStep(
                    step="ROUTE",
                    description=(
                        f"Tavily web search returned {len(web_results)} results. "
                        f"Merging with corpus chunks."
                    ),
                    data=[{"title": r["title"], "url": r["url"]} for r in web_results],
                ))
                # Append web results to context and sources
                for r in web_results:
                    web_text = f"[WEB: {r['title']}]\n{r['content']}"
                    context_parts.append(web_text)
                    sources.append({
                        "text": r["content"],
                        "doc_name": f"WEB: {r['title']}",
                        "page": r["url"],
                        "score": 0.0,
                        "chunk_id": r["url"],
                    })
        else:
            trace.append(TraceStep(
                step="ROUTE",
                description="Quality gate passed — using corpus chunks only, no web search.",
            ))

        # ── Step 4: generate answer ────────────────────────────────────────
        context = "\n\n---\n\n".join(context_parts)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
        ]
        answer_resp = self._gen_llm.invoke(messages)

        trace.append(TraceStep(
            step="GENERATE",
            description=(
                f"Generated answer from {'corpus + web' if web_triggered else 'corpus'} "
                f"context ({len(context)} chars). Model: {self._gen_llm.model_name}."
            ),
        ))

        return RAGResponse(
            answer=answer_resp.content,
            sources=sources,
            trace=trace,
            latency_ms=0,
            pipeline_name=self.name,
        )
