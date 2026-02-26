"""
app.py — RAG Architecture Showcase · Homepage
Launch with: streamlit run app.py
"""
from __future__ import annotations

import sys, os
from pathlib import Path

# Ensure repo root is on sys.path so backend imports work
_REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, _REPO_ROOT)

import streamlit as st

# ── Bootstrap: copy st.secrets → os.environ before any cache functions run ───
# st.secrets IS available here (module level, after st import) but NOT inside
# st.cache_resource on cold boot. Copying to os.environ ensures _get() in
# config.py always finds the keys via os.getenv(), which works everywhere.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str) and _k not in os.environ:
            os.environ[_k] = _v
except Exception:
    pass  # Local dev: secrets come from .env via python-dotenv


st.set_page_config(
    page_title="RAG Architecture Showcase",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Building vector store on first run (approx. 2 min)...")
def _auto_ingest():
    repo_root   = Path(_REPO_ROOT)
    chroma_ready = (repo_root / "chroma_db" / "chroma.sqlite3").exists()
    graph_ready  = (repo_root / "data" / "graph.pkl").exists()
    if chroma_ready and graph_ready:
        return

    # Run ingest inline (avoids subprocess working-dir and st.secrets issues)
    from tqdm import tqdm
    from backend.config import DOCS_DIR, CHROMA_COLLECTION, GRAPH_PATH
    from backend.ingest.loader import load_pdf
    from backend.ingest.chunker import chunk_documents
    from backend.ingest.embedder import get_vectorstore_direct
    from backend.graph.builder import build_graph
    from langchain_chroma import Chroma

    docs_dir  = repo_root / "data" / "docs"
    pdf_paths = sorted(docs_dir.glob("*.pdf"))
    if not pdf_paths:
        st.error(f"No PDFs found in {docs_dir}. Please add PDFs to data/docs/ and redeploy.")
        st.stop()

    all_chunks = []
    for pdf_path in pdf_paths:
        pages  = load_pdf(pdf_path)
        chunks = chunk_documents(pages)
        all_chunks.extend(chunks)

    vs = get_vectorstore_direct()
    try:
        existing = vs.get(include=["metadatas"])
        existing_ids: set[str] = {
            m["chunk_id"] for m in existing.get("metadatas", []) if m and "chunk_id" in m
        }
    except Exception:
        existing_ids = set()

    new_chunks = [c for c in all_chunks if c.metadata.get("chunk_id", "") not in existing_ids]
    BATCH_SIZE = 100
    for i in range(0, len(new_chunks), BATCH_SIZE):
        vs.add_documents(new_chunks[i : i + BATCH_SIZE])

    build_graph(all_chunks, str(repo_root / "data" / "graph.pkl"))

_auto_ingest()

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #0d1117; }

  .hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #58a6ff 0%, #a371f7 50%, #f0883e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin-bottom: 6px;
  }
  .hero-sub {
    font-size: 1.1rem;
    color: #8b949e;
    margin-bottom: 32px;
    line-height: 1.6;
  }

  .pipeline-card {
    background: #161b22;
    border-radius: 12px;
    padding: 18px 20px;
    border-left: 4px solid;
    margin-bottom: 14px;
    transition: transform 0.15s ease;
  }
  .pipeline-card:hover { transform: translateX(3px); }
  .pipeline-card h3 {
    margin: 0 0 6px 0;
    font-size: 1rem;
    font-weight: 700;
  }
  .pipeline-card p {
    margin: 0;
    font-size: 0.85rem;
    color: #8b949e;
    line-height: 1.55;
  }

  .tag {
    display: inline-block;
    background: #21262d;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.75rem;
    color: #8b949e;
    margin: 3px 2px 0;
  }

  .doc-card {
    background: #161b22;
    border-radius: 8px;
    padding: 14px 16px;
    border-left: 3px solid #30363d;
    margin-bottom: 10px;
    font-size: 0.85rem;
  }
  .doc-card strong { color: #e6edf3; }
  .doc-card span { color: #8b949e; }

  .section-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: #e6edf3;
    margin: 36px 0 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid #21262d;
  }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">RAG Architecture Showcase</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">'
    'Seven retrieval-augmented generation strategies running live on financial documents —<br>'
    'with full reasoning traces, source inspection, and side-by-side comparison.'
    '</div>',
    unsafe_allow_html=True,
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Pipelines", "7")
m2.metric("Documents", "6 PDFs")
m3.metric("Generation", "Claude 3.5 Sonnet")
m4.metric("Grading / Routing", "Claude 3 Haiku")

st.divider()

# ── Pipeline overview ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">The Seven Architectures</div>', unsafe_allow_html=True)

PIPELINES = [
    {
        "name": "Agentic RAG",
        "color": "#9b59b6",
        "desc": (
            "A ReAct agent with a RetrieverTool wrapping ChromaDB. "
            "Autonomously decides how many searches to make and which sub-questions to ask. "
            "Every tool call is captured in the trace."
        ),
        "tags": ["tool calling", "multi-step", "sub-questions", "ReAct"],
    },
    {
        "name": "Graph RAG",
        "color": "#16a085",
        "desc": (
            "Builds a knowledge graph (spaCy NER → NetworkX DiGraph) from the corpus. "
            "At query time, extracts entities, runs 2-hop BFS, fetches matched chunks from Chroma "
            "by chunk_id, and generates a graph-aware answer."
        ),
        "tags": ["spaCy NER", "NetworkX", "BFS traversal", "entity linking"],
    },
    {
        "name": "Self-RAG",
        "color": "#e74c3c",
        "desc": (
            "Retrieves 8 chunks, grades each for relevance (ISREL score 0-1) with the grading LLM, "
            "filters those below 0.4, generates an answer, then grades the answer's support (ISSUP). "
            "All scores visible in trace."
        ),
        "tags": ["ISREL grading", "ISSUP grading", "self-reflection", "per-chunk scoring"],
    },
    {
        "name": "CRAG",
        "color": "#1abc9c",
        "desc": (
            "Corrective RAG: retrieves corpus chunks, grades the set quality. "
            "If quality < 0.5, automatically triggers a live Tavily web search fallback. "
            "Merges corpus and web evidence before generating."
        ),
        "tags": ["quality gate", "Tavily web search", "fallback", "adaptive"],
    },
    {
        "name": "Adaptive RAG",
        "color": "#3498db",
        "desc": (
            "Classifies each query as SIMPLE / MEDIUM / COMPLEX via the grading LLM, "
            "then routes to: direct top-3 lookup, reranked top-5, or multi-step "
            "question decomposition. Optimises latency and depth together."
        ),
        "tags": ["query classification", "tiered routing", "reranking", "decomposition"],
    },
    {
        "name": "HyDE",
        "color": "#8e44ad",
        "desc": (
            "Hypothetical Document Embeddings: asks the LLM to write a plausible 150-word "
            "financial passage answering the query, embeds it, retrieves real chunks similar "
            "to that embedding, then answers from real evidence only."
        ),
        "tags": ["hypothetical doc", "embedding bridge", "semantic gap", "no direct query embed"],
    },
    {
        "name": "RAG Fusion",
        "color": "#f39c12",
        "desc": (
            "Generates 4 alternative query phrasings via LLM, runs parallel Chroma searches "
            "for each, merges all results with Reciprocal Rank Fusion (RRF k=60), "
            "and answers from the top-5 fused chunks."
        ),
        "tags": ["query expansion", "parallel retrieval", "RRF fusion", "4 variants"],
    },
]

left_col, right_col = st.columns(2)
for i, p in enumerate(PIPELINES):
    target = left_col if i % 2 == 0 else right_col
    tags_html = "".join(f'<span class="tag">{t}</span>' for t in p["tags"])
    target.markdown(
        f'<div class="pipeline-card" style="border-left-color:{p["color"]};">'
        f'<h3 style="color:{p["color"]};">{p["name"]}</h3>'
        f'<p>{p["desc"]}</p>'
        f'<div style="margin-top:8px;">{tags_html}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.divider()

# ── Document corpus ───────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Document Corpus</div>', unsafe_allow_html=True)

DOCS = [
    ("05_tech_sector_analysis.pdf",       "Analyst report",     "Technology sector performance and company analysis"),
    ("06_interest_rate_impact.pdf",        "Research note",      "Impact of interest rate changes — Graph RAG multi-hop chain"),
    ("07_risk_factors_comparison.pdf",     "Comparative brief",  "Systematic vs idiosyncratic risk across companies"),
    ("08_balance_sheet_glossary.pdf",      "Glossary",           "Balance sheet terminology — ideal for HyDE and Adaptive"),
    ("09_mergers_acquisitions_2023.pdf",   "M&A summary",        "M&A process, deal structure, entity linking for Graph RAG"),
    ("10_equity_valuation_methods.pdf",    "Methods guide",      "DCF, multiples, and valuation frameworks — Fusion and HyDE"),
]

doc_cols = st.columns(3)
for i, (fname, doc_type, desc) in enumerate(DOCS):
    with doc_cols[i % 3]:
        st.markdown(
            f'<div class="doc-card">'
            f'<strong>{fname.split("_", 1)[1].replace("_", " ").replace(".pdf", "").title()}</strong><br>'
            f'<span style="font-size:0.75rem; color:#58a6ff;">{doc_type}</span><br>'
            f'<span style="font-size:0.82rem;">{desc}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.divider()

# ── Setup reminder ────────────────────────────────────────────────────────────
with st.expander("First-time setup", expanded=False):
    st.markdown("""
    **Before running queries, complete the ingest step:**

    ```bash
    # 1. Install dependencies
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm

    # 2. Copy and fill in your API keys
    cp .env.example .env
    # Edit .env — add OPENROUTER_API_KEY and TAVILY_API_KEY

    # 3. Place PDFs in data/docs/

    # 4. Run ingest (once — idempotent on re-run)
    python scripts/ingest.py

    # 5. Launch
    streamlit run app.py
    ```

    CRAG's web fallback requires a free Tavily key from [app.tavily.com](https://app.tavily.com).
    All other pipelines work with only the OpenRouter key.
    """)
