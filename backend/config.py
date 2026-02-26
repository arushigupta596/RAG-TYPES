# backend/config.py
import os
from dotenv import load_dotenv

load_dotenv()


def _get(key, default=""):
    # Always re-read at call time so st.secrets is available after session init
    val = os.getenv(key, "")
    if val:
        return val
    try:
        import streamlit as st
        val = st.secrets.get(key, "")
        if val:
            return val
    except Exception:
        pass
    return default


# These are called lazily (inside functions) in llm.py — do NOT cache at module level
def get_openrouter_api_key():  return _get("OPENROUTER_API_KEY")
def get_openrouter_base_url(): return _get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
def get_generation_model():    return _get("GENERATION_MODEL", "anthropic/claude-3.5-sonnet")
def get_grading_model():       return _get("GRADING_MODEL",    "anthropic/claude-3-haiku")
def get_embedding_model():     return _get("EMBEDDING_MODEL",  "openai/text-embedding-3-small")
def get_tavily_api_key():      return _get("TAVILY_API_KEY")

# Non-secret constants — safe to read at import time
CRAG_QUALITY_THRESHOLD = 0.5

# ChromaDB
_BASE_DIR         = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHROMA_PATH       = os.path.join(_BASE_DIR, "chroma_db")
CHROMA_COLLECTION = "rag_showcase"

# Chunking
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150

# Data
DOCS_DIR   = os.path.join(_BASE_DIR, "data", "docs")
GRAPH_PATH = os.path.join(_BASE_DIR, "data", "graph.pkl")
