# backend/ingest/embedder.py
# Embed + store in ChromaDB; get_vectorstore() — shared across all pipelines
from __future__ import annotations

import streamlit as st
try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma  # type: ignore

from backend.config import CHROMA_PATH, CHROMA_COLLECTION
from backend.llm import get_embeddings


@st.cache_resource(show_spinner="Loading vector store…")
def get_vectorstore() -> Chroma:
    """
    Return the ChromaDB vector store backed by OpenRouter embeddings.
    Cached as a Streamlit resource so every pipeline reuses the same client.
    """
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PATH,
    )


def get_vectorstore_direct() -> Chroma:
    """
    Non-cached version for use outside Streamlit (e.g. ingest script).
    """
    return Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PATH,
    )
