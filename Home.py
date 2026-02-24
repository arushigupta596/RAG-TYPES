"""
RAG Architecture Showcase — Homepage
"""
import streamlit as st

st.set_page_config(
    page_title="RAG Architecture Showcase",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Pipeline cards */
    .pipeline-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
        border-left: 4px solid;
        transition: transform 0.2s ease;
    }
    .pipeline-card:hover { transform: translateX(4px); }
    .pipeline-card h3 { margin: 0 0 8px 0; font-size: 1.1rem; }
    .pipeline-card p  { margin: 0; color: #9ca3af; font-size: 0.9rem; line-height: 1.5; }

    /* Hero */
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4a90e2, #8e44ad);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .hero-sub {
        font-size: 1.2rem;
        color: #9ca3af;
        margin-bottom: 32px;
    }

    /* Stat chips */
    .stat-chip {
        display: inline-block;
        background: #2d3748;
        border-radius: 20px;
        padding: 6px 16px;
        margin: 4px;
        font-size: 0.85rem;
        color: #e2e8f0;
    }

    /* Section header */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 32px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #2d3748;
    }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">RAG Architecture Showcase</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Seven retrieval-augmented generation strategies, '
    'running live on financial documents — with full reasoning traces.</div>',
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Pipelines", "7", help="Seven distinct RAG architectures")
with col2:
    st.metric("Documents", "6", help="Financial PDF documents in the corpus")
with col3:
    st.metric("LLM", "GPT-4o mini", help="Via OpenRouter")
with col4:
    st.metric("Embeddings", "MiniLM-L6", help="Local sentence-transformers model")

st.divider()

# ── Pipeline overview ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">The Seven Pipelines</div>', unsafe_allow_html=True)

PIPELINES = [
    {
        "name": "1. Naive RAG",
        "color": "#4a90e2",
        "desc": (
            "The simplest baseline: embed the query, retrieve the top-k most "
            "similar chunks by cosine distance, stuff into a single prompt. "
            "Fast and interpretable — but no relevance filtering."
        ),
        "tags": ["vector search", "cosine similarity", "prompt stuffing"],
    },
    {
        "name": "2. Contextual Compression",
        "color": "#e67e22",
        "desc": (
            "Retrieves a larger candidate pool, then uses a secondary LLM call to "
            "compress each chunk to only the sentences relevant to the query. "
            "Higher precision at the cost of extra LLM calls."
        ),
        "tags": ["broad retrieval", "LLM filtering", "noise reduction"],
    },
    {
        "name": "3. Multi-Query RAG",
        "color": "#27ae60",
        "desc": (
            "Generates N alternative phrasings of the query via LLM, retrieves "
            "documents for each, deduplicates, and answers from the union. "
            "Captures documents missed by single-phrasing retrieval."
        ),
        "tags": ["query expansion", "parallel retrieval", "deduplication"],
    },
    {
        "name": "4. HyDE",
        "color": "#8e44ad",
        "desc": (
            "Hypothetical Document Embeddings: asks the LLM to write a plausible "
            "answer, embeds that hypothetical, then retrieves real chunks similar "
            "to it. Closes the semantic gap between short queries and long passages."
        ),
        "tags": ["hypothetical doc", "embedding bridge", "semantic gap"],
    },
    {
        "name": "5. Hybrid RAG",
        "color": "#c0392b",
        "desc": (
            "Combines BM25 keyword search (sparse) with vector similarity (dense). "
            "Merges results via Reciprocal Rank Fusion to leverage both exact "
            "keyword matches and semantic understanding."
        ),
        "tags": ["BM25", "vector search", "RRF fusion"],
    },
    {
        "name": "6. Graph RAG",
        "color": "#16a085",
        "desc": (
            "Builds a knowledge graph from the corpus using spaCy NER and NetworkX. "
            "Queries the graph for relevant entity subgraphs, combines with "
            "vector-retrieved chunks, and generates a graph-aware answer."
        ),
        "tags": ["NER", "NetworkX", "entity relationships", "subgraph retrieval"],
    },
    {
        "name": "7. Reranking RAG",
        "color": "#d35400",
        "desc": (
            "Retrieves a broad candidate pool, then scores each chunk's relevance "
            "0-10 using an LLM reranker. Only the highest-scoring chunks proceed "
            "to answer generation — maximising answer precision."
        ),
        "tags": ["broad retrieval", "LLM reranking", "precision optimisation"],
    },
]

cols = st.columns(2)
for i, p in enumerate(PIPELINES):
    with cols[i % 2]:
        tags_html = "".join(
            f'<span class="stat-chip">{t}</span>' for t in p["tags"]
        )
        st.markdown(
            f"""
            <div class="pipeline-card" style="border-left-color: {p['color']};">
                <h3 style="color: {p['color']};">{p['name']}</h3>
                <p>{p['desc']}</p>
                <div style="margin-top: 10px;">{tags_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

# ── Corpus overview ───────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Document Corpus</div>', unsafe_allow_html=True)

docs = [
    ("Interest Rate Impact", "4 pages", "Impact of interest rate changes on financial markets"),
    ("Balance Sheet Analysis", "5 pages", "Corporate balance sheet structure and interpretation"),
    ("Equity Valuation Methods", "5 pages", "DCF, comparables, and other equity valuation approaches"),
    ("Mergers & Acquisitions", "5 pages", "M&A process, deal structure, and valuation"),
    ("Risk Factor Comparison", "4 pages", "Systematic vs idiosyncratic risk analysis"),
    ("Tech Sector Analysis", "4 pages", "Technology sector performance and outlook"),
]

doc_cols = st.columns(3)
for i, (title, pages, desc) in enumerate(docs):
    with doc_cols[i % 3]:
        st.markdown(
            f"""
            <div class="pipeline-card" style="border-left-color: #4a90e2;">
                <h3 style="color: #e2e8f0; font-size: 1rem;">📄 {title}</h3>
                <p style="color: #6b7280; font-size: 0.8rem; margin-bottom: 6px;">{pages}</p>
                <p>{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

# ── Navigation CTA ────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Get Started</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    st.markdown("""
    ### 🔎 Single Query
    Run any question through **one** RAG pipeline at a time.
    See the full reasoning trace, retrieved chunks, and answer side by side.

    👈 Select **Single Query** in the sidebar.
    """)
with c2:
    st.markdown("""
    ### ⚖️ Compare Pipelines
    Run the **same question** through multiple pipelines simultaneously.
    Compare answers, sources, and reasoning traces head-to-head.

    👈 Select **Compare Pipelines** in the sidebar.
    """)
