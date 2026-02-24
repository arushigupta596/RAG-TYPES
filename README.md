# RAG Architecture Showcase

A Streamlit app that runs seven distinct Retrieval-Augmented Generation (RAG) pipelines side-by-side on a financial document corpus, so you can compare how each architecture retrieves, reasons, and generates answers for the same query.

**Live demo:** [share.streamlit.io](https://share.streamlit.io) · **Stack:** LangChain · LangGraph · ChromaDB · OpenRouter · Streamlit

---

## The 7 RAG Pipelines

### 1. Agentic RAG
The LLM decides *when* and *how many times* to retrieve. Built with LangGraph's `create_react_agent`, the model is given a retriever as a tool and autonomously makes multiple tool calls for complex, multi-part questions — only generating a final answer when it decides it has enough context.

**Best for:** Open-ended questions that span multiple topics or require iterative reasoning.

---

### 2. Graph RAG
Builds a NetworkX knowledge graph from spaCy Named Entity Recognition over the corpus. At query time, entities in the question are matched to graph nodes and the graph is traversed to find connected concepts — giving structured, relationship-aware context that flat vector search misses.

**Best for:** Questions about relationships between entities (e.g. *"How are rising interest rates connected to tech sector layoffs?"*).

---

### 3. Self-RAG
Adds a self-grading loop: after retrieval, each chunk is scored for relevance (`ISREL`). Only chunks above the threshold are passed to the LLM. After generation, the answer is graded for support (`ISSUP`). Low-quality answers trigger a retry. The LLM reflects on its own output before returning it.

**Best for:** High-stakes queries where factual grounding and hallucination reduction matter most.

---

### 4. CRAG (Corrective RAG)
Retrieves documents and grades the entire set with an LLM judge. If the average quality score falls below a threshold, CRAG discards the corpus results and falls back to a live Tavily web search — then merges web and corpus results before generating. Corrects retrieval failures automatically.

**Best for:** Questions where the corpus may be incomplete or outdated (e.g. current events, recent prices).

---

### 5. Adaptive RAG
Classifies each query into one of three tiers before doing anything else:
- **Simple** — answered directly from top-3 chunks
- **Medium** — standard retrieval + cross-encoder reranking on top-5 chunks
- **Complex** — routes to the full Agentic pipeline

The routing decision is made by an LLM grader, so compute is matched to query difficulty.

**Best for:** Mixed workloads where query complexity varies widely.

---

### 6. HyDE (Hypothetical Document Embeddings)
Instead of embedding the raw question, HyDE first asks the LLM to write a *hypothetical answer* as if it appeared in a financial report. That synthetic passage is embedded and used as the search vector. Because the hypothetical answer lives in document-space rather than question-space, it retrieves semantically closer chunks.

**Best for:** Queries phrased as questions where relevant documents are phrased as statements/facts.

---

### 7. RAG Fusion
Generates four paraphrased variants of the original query in parallel, runs a separate vector search for each, then merges all result lists using **Reciprocal Rank Fusion (RRF)** — a rank-aggregation formula that promotes chunks that appear highly in multiple lists. Reduces sensitivity to any single query phrasing.

**Best for:** Queries where the exact wording is ambiguous or where coverage across multiple phrasings matters.

---

## Document Corpus

Six financial PDFs covering:

| File | Topic |
|------|-------|
| `05_tech_sector_analysis.pdf` | Tech sector fundamentals and trends |
| `06_interest_rate_impact.pdf` | Impact of rate changes on markets |
| `07_risk_factors_comparison.pdf` | Risk factor frameworks |
| `08_balance_sheet_glossary.pdf` | Balance sheet terminology |
| `09_mergers_acquisitions_2023.pdf` | M&A activity and deal structures |
| `10_equity_valuation_methods.pdf` | DCF, comparables, and valuation methods |

---

## Architecture

```
app.py                        # Streamlit homepage + auto-ingest on cold start
pages/
  2_Compare_Pipelines.py      # Side-by-side pipeline comparison UI
backend/
  config.py                   # All model names and paths (single source of truth)
  llm.py                      # LLM + embeddings clients (OpenRouter)
  ingest/
    loader.py                 # PyMuPDF PDF loader
    chunker.py                # RecursiveCharacterTextSplitter
    embedder.py               # ChromaDB PersistentClient
  graph/
    builder.py                # spaCy NER → NetworkX DiGraph
    traversal.py              # Entity-to-graph BFS traversal
  pipelines/
    base.py                   # RAGResponse + TraceStep dataclasses
    agentic.py                # LangGraph ReAct agent
    graph_rag.py              # Knowledge graph retrieval
    self_rag.py               # ISREL/ISSUP grading loops
    crag.py                   # Corrective retrieval + Tavily fallback
    adaptive.py               # Query-tier routing
    hyde.py                   # Hypothetical document embedding
    fusion.py                 # Multi-query + RRF
  utils/
    reranker.py               # Cross-encoder reranking (sentence-transformers)
    rrf.py                    # Reciprocal Rank Fusion
scripts/
  ingest.py                   # One-time ingest: PDF → chunks → ChromaDB + graph.pkl
data/
  docs/                       # Source PDFs (committed)
  graph.pkl                   # Knowledge graph (auto-generated, gitignored)
chroma_db/                    # Vector store (auto-generated, gitignored)
```

---

## Running Locally

**Prerequisites:** Python 3.10+, an [OpenRouter](https://openrouter.ai) API key, optionally a [Tavily](https://tavily.com) API key for CRAG web fallback.

```bash
# 1. Clone
git clone https://github.com/arushigupta596/RAG-TYPES.git
cd RAG-TYPES

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
cp .env.example .env
# Edit .env and fill in your keys

# 4. Run ingest (builds ChromaDB + knowledge graph — takes ~2 min)
python scripts/ingest.py

# 5. Launch
streamlit run app.py
```

---

## Deploying to Streamlit Cloud

1. Fork or push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Set **Main file path** to `app.py`
4. Under **Advanced settings → Secrets**, paste:
   ```toml
   OPENROUTER_API_KEY  = "your-key"
   OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
   GENERATION_MODEL    = "anthropic/claude-3.5-sonnet"
   GRADING_MODEL       = "anthropic/claude-3-haiku"
   EMBEDDING_MODEL     = "openai/text-embedding-3-small"
   TAVILY_API_KEY      = "your-key"
   ```
5. Click **Deploy** — the app auto-ingests on first cold start (~2 min), then stays warm.

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | OpenRouter API key | required |
| `OPENROUTER_BASE_URL` | OpenRouter base URL | `https://openrouter.ai/api/v1` |
| `GENERATION_MODEL` | Model for answer generation | `anthropic/claude-3.5-sonnet` |
| `GRADING_MODEL` | Model for ISREL/ISSUP/routing graders | `anthropic/claude-3-haiku` |
| `EMBEDDING_MODEL` | Embedding model (OpenRouter) | `openai/text-embedding-3-small` |
| `TAVILY_API_KEY` | Tavily web search (CRAG fallback) | optional |
