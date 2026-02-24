#!/usr/bin/env python
"""
scripts/ingest.py — Run once before launching the app.

Steps:
  1. Discover all PDFs in data/docs/ (sorted by filename)
  2. Load pages via PyMuPDF
  3. Chunk with RecursiveCharacterTextSplitter; assign chunk_ids
  4. Embed and store in ChromaDB (idempotent — skips existing chunk_ids)
  5. Build knowledge graph (spaCy NER → NetworkX DiGraph) → data/graph.pkl

Usage:
    python scripts/ingest.py

Expected output:
    [1/6] 05_tech_sector_analysis.pdf → N pages → M chunks
    ...
    ChromaDB: X total chunks stored
    Graph: Y nodes, Z edges → saved to data/graph.pkl
"""

import sys
import os

# Allow running from repo root without installing as a package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
from tqdm import tqdm

from backend.config import DOCS_DIR, CHROMA_PATH, CHROMA_COLLECTION, GRAPH_PATH
from backend.ingest.loader import load_pdf
from backend.ingest.chunker import chunk_documents
from backend.ingest.embedder import get_vectorstore_direct
from backend.graph.builder import build_graph


def main():
    docs_dir = Path(DOCS_DIR)
    pdf_paths = sorted(docs_dir.glob("*.pdf"))

    if not pdf_paths:
        print(f"ERROR: No PDF files found in {docs_dir}")
        print("Place your PDF files in data/docs/ and re-run.")
        sys.exit(1)

    print(f"\nFound {len(pdf_paths)} PDF(s) in {docs_dir}\n")

    # ── Phase 1: Load + chunk all PDFs ────────────────────────────────────
    all_chunks = []
    for i, pdf_path in enumerate(pdf_paths, start=1):
        pages = load_pdf(pdf_path)
        chunks = chunk_documents(pages)
        all_chunks.extend(chunks)
        print(
            f"[{i}/{len(pdf_paths)}] {pdf_path.name} "
            f"→ {len(pages)} pages → {len(chunks)} chunks"
        )

    print(f"\nTotal chunks to embed: {len(all_chunks)}")

    # ── Phase 2: Embed into ChromaDB (idempotent) ─────────────────────────
    print("\nConnecting to ChromaDB…")
    vs = get_vectorstore_direct()

    # Get existing chunk_ids from ChromaDB
    try:
        existing = vs.get(include=["metadatas"])
        existing_ids: set[str] = set()
        for meta in existing.get("metadatas", []):
            if meta and "chunk_id" in meta:
                existing_ids.add(meta["chunk_id"])
        print(f"  Existing chunks in DB: {len(existing_ids)}")
    except Exception:
        existing_ids = set()
        print("  Could not read existing chunks — inserting all.")

    new_chunks = [
        c for c in all_chunks
        if c.metadata.get("chunk_id", "") not in existing_ids
    ]
    print(f"  New chunks to insert: {len(new_chunks)}")

    if new_chunks:
        BATCH_SIZE = 100
        for batch_start in tqdm(
            range(0, len(new_chunks), BATCH_SIZE),
            desc="  Embedding batches",
        ):
            batch = new_chunks[batch_start: batch_start + BATCH_SIZE]
            vs.add_documents(batch)

    total_in_db = len(existing_ids) + len(new_chunks)
    print(f"\nChromaDB: {total_in_db} total chunks stored in '{CHROMA_COLLECTION}'")

    # ── Phase 3: Build knowledge graph ────────────────────────────────────
    print("\nBuilding knowledge graph (spaCy NER)…")
    print("  This may take a few minutes on first run.")
    G = build_graph(all_chunks, GRAPH_PATH)

    print(f"\n{'='*60}")
    print("Ingest complete!")
    print(f"  PDFs processed : {len(pdf_paths)}")
    print(f"  Total chunks   : {total_in_db}")
    print(f"  Graph nodes    : {G.number_of_nodes()}")
    print(f"  Graph edges    : {G.number_of_edges()}")
    print(f"  Graph saved    : {GRAPH_PATH}")
    print(f"{'='*60}")
    print("\nYou can now launch the app:")
    print("  streamlit run app.py\n")


if __name__ == "__main__":
    main()
