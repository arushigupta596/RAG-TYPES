# backend/ingest/loader.py
# PDF → text using PyMuPDF (fitz)
from __future__ import annotations

import fitz  # PyMuPDF
from pathlib import Path
from typing import List

from langchain_core.documents import Document


def load_pdf(path: str | Path) -> List[Document]:
    """
    Load a single PDF file page-by-page.
    Returns one Document per page with metadata:
        doc_name, page, total_pages, source_path
    """
    path = Path(path)
    doc_name = path.stem
    documents: List[Document] = []

    with fitz.open(str(path)) as pdf:
        total_pages = len(pdf)
        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "doc_name": doc_name,
                        "page": page_num,
                        "total_pages": total_pages,
                        "source_path": str(path),
                    },
                )
            )

    return documents


def load_all_pdfs(docs_dir: str | Path) -> List[Document]:
    """
    Discover and load all PDFs in docs_dir, sorted by filename.
    Returns a flat list of Documents across all files and pages.
    """
    docs_dir = Path(docs_dir)
    pdf_paths = sorted(docs_dir.glob("*.pdf"))

    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in {docs_dir}")

    all_docs: List[Document] = []
    for pdf_path in pdf_paths:
        pages = load_pdf(pdf_path)
        all_docs.extend(pages)
        print(f"  Loaded {pdf_path.name} → {len(pages)} pages")

    return all_docs
