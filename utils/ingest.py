"""
PDF ingestion pipeline: PyMuPDF → RecursiveCharacterTextSplitter → ChromaDB
"""
import os
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_PATH = str(Path(__file__).parent.parent / "chroma_db")
DATA_PATH = str(Path(__file__).parent.parent / "data")
COLLECTION_NAME = "financial_docs"


def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """Extract text page-by-page from a PDF using PyMuPDF."""
    pages = []
    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            pages.append({
                "text": text,
                "page": page_num + 1,
                "source": Path(pdf_path).name,
                "source_path": pdf_path,
            })
    doc.close()
    return pages


def load_all_pdfs(data_dir: str = DATA_PATH) -> List[Document]:
    """Load all PDFs from the data directory into LangChain Documents."""
    documents = []
    pdf_files = list(Path(data_dir).glob("*.pdf"))

    for pdf_path in pdf_files:
        pages = extract_text_from_pdf(str(pdf_path))
        for page_data in pages:
            doc = Document(
                page_content=page_data["text"],
                metadata={
                    "source": page_data["source"],
                    "page": page_data["page"],
                    "source_path": page_data["source_path"],
                },
            )
            documents.append(doc)

    return documents


def get_embeddings():
    """Return a local HuggingFace embedding model (no API key needed)."""
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(force_rebuild: bool = False) -> Chroma:
    """
    Build or load the ChromaDB vector store.
    Chunks PDFs with RecursiveCharacterTextSplitter and embeds them.
    """
    embeddings = get_embeddings()

    # Check if store already exists and is populated
    if not force_rebuild and Path(CHROMA_PATH).exists():
        store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PATH,
        )
        if store._collection.count() > 0:
            return store

    # Load and split documents
    raw_docs = load_all_pdfs()
    if not raw_docs:
        raise ValueError(f"No PDF documents found in {DATA_PATH}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)

    # Build ChromaDB
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
    )
    return store


def get_vector_store() -> Chroma:
    """Get (or build) the shared vector store."""
    return build_vector_store(force_rebuild=False)


def get_raw_chunks() -> List[Document]:
    """Return all chunks as LangChain Documents (for BM25 etc.)."""
    raw_docs = load_all_pdfs()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(raw_docs)
