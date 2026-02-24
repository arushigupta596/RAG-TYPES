# backend/ingest/chunker.py
# RecursiveCharacterTextSplitter → assigns chunk_id to every chunk
from __future__ import annotations

from typing import List

from langchain_core.documents import Document
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

from backend.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into chunks and assign a unique chunk_id to each.
    chunk_id = "{doc_name}::chunk::{i}" (globally unique within the corpus)

    Returns a new list of Documents with enriched metadata:
        chunk_id, doc_name, page, total_pages, source_path
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # Assign chunk_ids
    doc_counters: dict[str, int] = {}
    for chunk in chunks:
        doc_name = chunk.metadata.get("doc_name", "unknown")
        idx = doc_counters.get(doc_name, 0)
        chunk.metadata["chunk_id"] = f"{doc_name}::chunk::{idx}"
        doc_counters[doc_name] = idx + 1

    return chunks
