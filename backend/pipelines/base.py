# backend/pipelines/base.py
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Any


@dataclass
class TraceStep:
    step: str          # Short label: RETRIEVE / GRADE / ROUTE / GENERATE / ERROR
    description: str   # One human-readable sentence
    data: Any = None   # Optional detail: scores, queries, paths, chunk texts


@dataclass
class RAGResponse:
    answer:        str
    sources:       List[dict]      # [{text, doc_name, page, score, chunk_id}]
    trace:         List[TraceStep]
    latency_ms:    int
    pipeline_name: str


class BaseRAGPipeline:
    name: str = "base"
    label: str = "Base"
    description: str = ""
    color: str = "#6c757d"

    def run(self, query: str) -> RAGResponse:
        raise NotImplementedError

    def _timed_run(self, query: str) -> RAGResponse:
        """Wrapper that measures latency and catches top-level errors."""
        start = time.perf_counter()
        try:
            response = self.run(query)
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            response.latency_ms = elapsed_ms
            return response
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            return RAGResponse(
                answer=f"Pipeline error: {exc}",
                sources=[],
                trace=[TraceStep("ERROR", str(exc))],
                latency_ms=elapsed_ms,
                pipeline_name=self.name,
            )
