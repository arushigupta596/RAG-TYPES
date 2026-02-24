"""
Pipeline 3 — Multi-Query RAG
Generate N alternative phrasings of the query → retrieve for each →
deduplicate → answer from the union of results.
"""
from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from .base import BaseRAGPipeline, RAGResult
from utils.ingest import get_vector_store
from utils.llm import get_llm

TOP_K_PER_QUERY = 3
NUM_VARIANTS = 4

REPHRASE_PROMPT = (
    "You are a query expansion expert for a financial document search system. "
    "Given the user question below, generate {n} alternative phrasings that "
    "would help retrieve relevant financial documents. "
    "Output ONLY the {n} questions, one per line, no numbering or bullets.\n\n"
    "Original question: {query}"
)

SYSTEM_PROMPT = (
    "You are a financial analyst assistant. "
    "Answer the user's question using ONLY the provided context. "
    "If the context does not contain enough information, say so clearly."
)


class MultiQueryPipeline(BaseRAGPipeline):
    name = "Multi-Query RAG"
    description = (
        "Generates multiple rephrased versions of the query, retrieves documents "
        "for each, deduplicates, and answers from the combined evidence pool."
    )
    color = "#27ae60"

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

        # ── Step 1: generate query variants ──────────────────────────────
        trace = ["**Step 1 — Query expansion**"]
        trace.append(f"- Original query: `{query}`")
        trace.append(f"- Generating {NUM_VARIANTS} alternative phrasings via LLM")

        rephrase_resp = self._llm.invoke([
            HumanMessage(content=REPHRASE_PROMPT.format(n=NUM_VARIANTS, query=query))
        ])
        raw_variants = rephrase_resp.content.strip().split("\n")
        variants = [v.strip() for v in raw_variants if v.strip()][:NUM_VARIANTS]

        trace.append("- Generated variants:")
        for i, v in enumerate(variants, 1):
            trace.append(f"  {i}. `{v}`")

        all_queries = [query] + variants

        # ── Step 2: retrieve for each variant ────────────────────────────
        trace.append("\n**Step 2 — Parallel retrieval across all queries**")
        seen_content = set()
        union_docs: list[tuple[str, str]] = []  # (source, content)

        for q in all_queries:
            docs = self._store.similarity_search(q, k=TOP_K_PER_QUERY)
            new_for_q = 0
            for doc in docs:
                content_key = doc.page_content[:200]
                if content_key not in seen_content:
                    seen_content.add(content_key)
                    src = f"{doc.metadata.get('source', '?')} p.{doc.metadata.get('page', '?')}"
                    union_docs.append((src, doc.page_content))
                    new_for_q += 1
            trace.append(f"  • `{q[:60]}…` → {new_for_q} new unique chunks")

        trace.append(f"\n- Total unique chunks after deduplication: {len(union_docs)}")

        # ── Step 3: answer generation ────────────────────────────────────
        trace.append("\n**Step 3 — Answer generation from union context**")
        context = "\n\n---\n\n".join(
            f"[Source: {src}]\n{chunk}" for src, chunk in union_docs
        )
        trace.append(f"- Combined context length: {len(context)} chars")

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}"),
        ]
        response = self._llm.invoke(messages)

        sources = list(dict.fromkeys(src for src, _ in union_docs))
        return RAGResult(
            answer=response.content,
            sources=sources,
            reasoning_trace="\n".join(trace),
            retrieved_chunks=[chunk for _, chunk in union_docs],
            pipeline_name=self.name,
        )
