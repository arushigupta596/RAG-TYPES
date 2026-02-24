"""
BaseRAGPipeline — abstract interface that every pipeline must implement.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Any


@dataclass
class RAGResult:
    """Standardised output from any RAG pipeline."""
    answer: str
    sources: List[str]           # document names / chunk references
    reasoning_trace: str         # step-by-step explanation shown in UI
    retrieved_chunks: List[str]  # raw retrieved text (for inspection)
    pipeline_name: str
    extra: dict = field(default_factory=dict)  # pipeline-specific extras


class BaseRAGPipeline(ABC):
    """
    Every pipeline must implement `run(query) -> RAGResult`.
    Pipelines are responsible for their own initialisation (lazy is fine).
    """

    name: str = "Base"
    description: str = ""
    color: str = "#6c757d"       # accent colour shown in the UI

    @abstractmethod
    def run(self, query: str) -> RAGResult:
        """Execute the full pipeline for a given query."""
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"
