# backend/config.py
import os
from dotenv import load_dotenv

load_dotenv()


def _get(key, default=""):
    val = os.getenv(key, "")
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default


OPENROUTER_API_KEY  = _get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = _get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Models — change here only, nowhere else in the codebase
GENERATION_MODEL = _get("GENERATION_MODEL", "anthropic/claude-3.5-sonnet")
GRADING_MODEL    = _get("GRADING_MODEL",    "anthropic/claude-3-haiku")
EMBEDDING_MODEL  = _get("EMBEDDING_MODEL",  "openai/text-embedding-3-small")

# CRAG
CRAG_QUALITY_THRESHOLD = 0.5
TAVILY_API_KEY = _get("TAVILY_API_KEY")

# ChromaDB
CHROMA_PATH       = "./chroma_db"
CHROMA_COLLECTION = "rag_showcase"

# Chunking
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150

# Data
DOCS_DIR   = "./data/docs"
GRAPH_PATH = "./data/graph.pkl"
