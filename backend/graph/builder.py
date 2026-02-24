# backend/graph/builder.py
# NER → NetworkX DiGraph → data/graph.pkl
# Runs once during ingest; results are persisted and loaded at query time.
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import networkx as nx
from langchain_core.documents import Document

# Lazy-load spaCy to avoid import cost when not needed
_nlp = None

RELEVANT_LABELS = {
    "ORG", "PRODUCT", "GPE", "PERSON", "NORP",
    "MONEY", "PERCENT", "CARDINAL", "DATE",
    "FAC", "EVENT", "LAW",
}


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        model = "en_core_web_sm"
        try:
            _nlp = spacy.load(model)
        except OSError:
            from spacy.cli import download as spacy_download
            spacy_download(model)
            _nlp = spacy.load(model)
    return _nlp


def _extract_entities(text: str) -> List[tuple[str, str]]:
    """Return [(entity_text, label), ...] for the given text snippet."""
    nlp = _get_nlp()
    doc = nlp(text[:100_000])
    seen: set[tuple[str, str]] = set()
    entities: List[tuple[str, str]] = []
    for ent in doc.ents:
        if ent.label_ in RELEVANT_LABELS:
            key = (ent.text.strip(), ent.label_)
            if key not in seen and ent.text.strip():
                entities.append(key)
                seen.add(key)
    return entities


def build_graph(chunks: List[Document], graph_path: str) -> nx.DiGraph:
    """
    Build a directed knowledge graph from all corpus chunks.

    Node attributes: label (NER type), sources (set of doc_names)
    Edge attributes: relation, chunk_id, doc_name, weight (co-occurrence count)

    Persists the graph as a pickle at graph_path.
    """
    G = nx.DiGraph()

    for chunk in chunks:
        text = chunk.page_content
        chunk_id = chunk.metadata.get("chunk_id", "unknown")
        doc_name = chunk.metadata.get("doc_name", "unknown")

        entities = _extract_entities(text)

        # Add / update nodes
        for ent_text, ent_label in entities:
            if not G.has_node(ent_text):
                G.add_node(ent_text, label=ent_label, sources=set())
            G.nodes[ent_text]["sources"].add(doc_name)

        # Add edges: co-occurrence within same chunk
        for i, (head_text, _) in enumerate(entities):
            for tail_text, _ in entities[i + 1:]:
                if head_text == tail_text:
                    continue
                if G.has_edge(head_text, tail_text):
                    G[head_text][tail_text]["weight"] += 1
                    G[head_text][tail_text]["chunk_ids"].add(chunk_id)
                else:
                    G.add_edge(
                        head_text,
                        tail_text,
                        relation="co-occurs-with",
                        chunk_ids={chunk_id},
                        doc_name=doc_name,
                        weight=1,
                    )
                # Mirror edge for undirected traversal
                if G.has_edge(tail_text, head_text):
                    G[tail_text][head_text]["weight"] += 1
                    G[tail_text][head_text]["chunk_ids"].add(chunk_id)
                else:
                    G.add_edge(
                        tail_text,
                        head_text,
                        relation="co-occurs-with",
                        chunk_ids={chunk_id},
                        doc_name=doc_name,
                        weight=1,
                    )

    # Persist
    Path(graph_path).parent.mkdir(parents=True, exist_ok=True)
    with open(graph_path, "wb") as f:
        pickle.dump(G, f)

    print(
        f"Graph: {G.number_of_nodes()} nodes, "
        f"{G.number_of_edges()} edges → saved to {graph_path}"
    )
    return G


def load_graph(graph_path: str) -> nx.DiGraph:
    """Load the persisted graph from disk."""
    with open(graph_path, "rb") as f:
        return pickle.load(f)
