"""
Pipeline 6 — Graph RAG
Build a knowledge graph from the corpus (spaCy NER + NetworkX).
Query → find relevant subgraph → combine graph context with vector chunks → answer.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from .base import BaseRAGPipeline, RAGResult
from utils.ingest import get_vector_store, load_all_pdfs
from utils.knowledge_graph import build_knowledge_graph, query_graph, graph_context_to_text
from utils.llm import get_llm

TOP_K_VECTOR = 3
HOP_DEPTH = 2

SYSTEM_PROMPT = (
    "You are a financial analyst assistant. "
    "You have been given two types of context: "
    "(1) a knowledge graph showing entities and their relationships, and "
    "(2) raw document chunks. "
    "Use both to answer the question accurately. "
    "If the context does not contain enough information, say so clearly."
)


class GraphRAGPipeline(BaseRAGPipeline):
    name = "Graph RAG"
    description = (
        "Builds a knowledge graph from the corpus using spaCy NER and NetworkX. "
        "Queries the graph for relevant entity subgraphs, combines them with "
        "vector-retrieved chunks, and generates a graph-aware answer."
    )
    color = "#16a085"

    def __init__(self):
        self._store = None
        self._llm = None
        self._graph = None

    def _init(self):
        if self._store is None:
            self._store = get_vector_store()
        if self._llm is None:
            self._llm = get_llm()
        if self._graph is None:
            docs = load_all_pdfs()
            self._graph = build_knowledge_graph(docs)

    def run(self, query: str) -> RAGResult:
        self._init()

        trace = ["**Step 1 — Knowledge graph construction**"]
        trace.append(
            f"- Graph loaded: {self._graph.number_of_nodes()} nodes, "
            f"{self._graph.number_of_edges()} edges"
        )

        # ── Graph query ───────────────────────────────────────────────────
        trace.append("\n**Step 2 — Subgraph retrieval**")
        graph_result = query_graph(
            self._graph, query, top_k=5, hop_depth=HOP_DEPTH
        )
        trace.append(graph_result["trace"])

        graph_context = graph_context_to_text(graph_result)
        trace.append(f"\n- Graph context length: {len(graph_context)} chars")

        # ── Vector retrieval ──────────────────────────────────────────────
        trace.append(f"\n**Step 3 — Vector retrieval (top-{TOP_K_VECTOR} chunks)**")
        docs_with_scores = self._store.similarity_search_with_score(query, k=TOP_K_VECTOR)

        sources = []
        vector_chunks = []
        for doc, score in docs_with_scores:
            src = f"{doc.metadata.get('source', '?')} p.{doc.metadata.get('page', '?')}"
            sources.append(src)
            vector_chunks.append(doc.page_content)
            trace.append(f"  • {src} — score: {score:.4f}")

        # ── Combined answer ───────────────────────────────────────────────
        trace.append("\n**Step 4 — Combined graph + vector answer generation**")
        vector_context = "\n\n---\n\n".join(
            f"[Source: {src}]\n{chunk}" for src, chunk in zip(sources, vector_chunks)
        )
        combined_context = (
            f"=== KNOWLEDGE GRAPH CONTEXT ===\n{graph_context}\n\n"
            f"=== DOCUMENT CHUNKS ===\n{vector_context}"
        )
        trace.append(f"- Total context length: {len(combined_context)} chars")

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Context:\n{combined_context}\n\nQuestion: {query}"),
        ]
        response = self._llm.invoke(messages)

        return RAGResult(
            answer=response.content,
            sources=list(dict.fromkeys(sources)),
            reasoning_trace="\n".join(trace),
            retrieved_chunks=vector_chunks + [graph_context],
            pipeline_name=self.name,
            extra={
                "graph_nodes": graph_result["nodes"],
                "graph_edges": graph_result["edges"],
                "seed_nodes": graph_result["seed_nodes"],
            },
        )
