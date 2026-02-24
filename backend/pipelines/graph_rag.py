# backend/pipelines/graph_rag.py
# Graph RAG: NER → NetworkX DiGraph → 2-hop BFS → Chroma chunk_id lookup → generate
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage

from backend.pipelines.base import BaseRAGPipeline, RAGResponse, TraceStep
from backend.ingest.embedder import get_vectorstore
from backend.llm import get_generation_llm
from backend.graph.traversal import traverse
from backend.config import GRAPH_PATH

TOP_K_FALLBACK = 5   # vector fallback if graph yields no chunk_ids
MAX_CHUNKS_FROM_GRAPH = 8

SYSTEM_PROMPT = (
    "You are a senior financial analyst with access to a knowledge graph and document corpus. "
    "Answer the question using the provided context which includes entity relationship paths "
    "and raw document chunks. Highlight any cross-document connections you find. "
    "If context is insufficient, say so explicitly."
)


@st.cache_resource(show_spinner="Loading knowledge graph…")
def _load_graph(graph_path: str):
    """Load persisted NetworkX graph. Cached as Streamlit resource."""
    p = Path(graph_path)
    if not p.exists():
        return None
    with open(graph_path, "rb") as f:
        return pickle.load(f)


class GraphRAGPipeline(BaseRAGPipeline):
    name = "graph_rag"
    label = "Graph RAG"
    description = (
        "Builds a knowledge graph from the corpus (spaCy NER → NetworkX DiGraph). "
        "At query time: extracts entities, runs 2-hop BFS, fetches the matched chunk IDs "
        "from Chroma, and generates a graph-aware answer with entity path in trace."
    )
    color = "#16a085"

    def __init__(self):
        self._vs = None
        self._gen_llm = None

    def _init(self):
        if self._vs is None:
            self._vs = get_vectorstore()
        if self._gen_llm is None:
            self._gen_llm = get_generation_llm()

    def run(self, query: str) -> RAGResponse:
        self._init()
        trace: List[TraceStep] = []

        # ── Step 1: load graph ─────────────────────────────────────────────
        G = _load_graph(GRAPH_PATH)

        if G is None:
            trace.append(TraceStep(
                step="ERROR",
                description=(
                    f"Graph file not found at {GRAPH_PATH}. "
                    "Run `python scripts/ingest.py` first to build the knowledge graph. "
                    "Falling back to pure vector retrieval."
                ),
            ))
            # Graceful fallback to vector search
            docs_scores = self._vs.similarity_search_with_score(query, k=TOP_K_FALLBACK)
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
            context = "\n\n---\n\n".join(context_parts)
            trace.append(TraceStep(
                step="RETRIEVE",
                description=f"Fallback: vector search returned {len(sources)} chunks.",
            ))
        else:
            trace.append(TraceStep(
                step="RETRIEVE",
                description=(
                    f"Knowledge graph loaded: {G.number_of_nodes()} nodes, "
                    f"{G.number_of_edges()} edges."
                ),
                data={
                    "nodes": G.number_of_nodes(),
                    "edges": G.number_of_edges(),
                },
            ))

            # ── Step 2: BFS traversal ──────────────────────────────────────
            traversal = traverse(G, query, max_hops=2, max_seed_nodes=5)

            trace.append(TraceStep(
                step="RETRIEVE",
                description=(
                    f"Graph traversal: entities found={traversal['query_entities']}, "
                    f"seeds={traversal['seed_nodes']}, "
                    f"nodes visited={len(traversal['visited_nodes'])}, "
                    f"chunk IDs collected={len(traversal['chunk_ids'])}."
                ),
                data={
                    "query_entities": traversal["query_entities"],
                    "seed_nodes": traversal["seed_nodes"],
                    "top_paths": traversal["paths"][:8],
                    "chunk_ids_count": len(traversal["chunk_ids"]),
                },
            ))

            # ── Step 3: fetch chunks from Chroma by chunk_id ──────────────
            sources, context_parts = [], []
            chunk_ids = traversal["chunk_ids"][:MAX_CHUNKS_FROM_GRAPH]

            if chunk_ids:
                # Chroma metadata filter
                try:
                    result = self._vs.get(
                        where={"chunk_id": {"$in": chunk_ids}},
                        include=["documents", "metadatas"],
                    )
                    fetched_docs = result.get("documents", [])
                    fetched_meta = result.get("metadatas", [])
                    for text, meta in zip(fetched_docs, fetched_meta):
                        if not meta:
                            meta = {}
                        sources.append({
                            "text": text,
                            "doc_name": meta.get("doc_name", "?"),
                            "page": meta.get("page", "?"),
                            "score": 1.0,   # graph-matched, no cosine score
                            "chunk_id": meta.get("chunk_id", "?"),
                        })
                        context_parts.append(
                            f"[{meta.get('doc_name','?')} p.{meta.get('page','?')}]\n{text}"
                        )
                except Exception as e:
                    trace.append(TraceStep(
                        step="ERROR",
                        description=f"Chroma chunk_id fetch failed: {e}. Falling back to vector.",
                    ))

            # Supplement with vector retrieval if graph yielded few results
            if len(sources) < 3:
                docs_scores = self._vs.similarity_search_with_score(query, k=TOP_K_FALLBACK)
                for doc, score in docs_scores:
                    meta = doc.metadata
                    cid = meta.get("chunk_id", "")
                    if cid not in {s["chunk_id"] for s in sources}:
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
                    description=f"Supplemented graph results with {TOP_K_FALLBACK} vector chunks.",
                ))

            context = "\n\n---\n\n".join(context_parts)

        # ── Step 4: generate answer ────────────────────────────────────────
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
        ]
        answer_resp = self._gen_llm.invoke(messages)

        trace.append(TraceStep(
            step="GENERATE",
            description=(
                f"Generated answer from {len(sources)} chunks "
                f"({len(context)} chars). Model: {self._gen_llm.model_name}."
            ),
        ))

        return RAGResponse(
            answer=answer_resp.content,
            sources=sources,
            trace=trace,
            latency_ms=0,
            pipeline_name=self.name,
        )
