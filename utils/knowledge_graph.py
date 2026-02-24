"""
Knowledge graph construction using spaCy NER + NetworkX.
Used by the Graph RAG pipeline.
"""
from __future__ import annotations

import re
from typing import List, Tuple, Dict, Any

import networkx as nx
from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# NER-based entity extraction (spaCy, lazy-loaded)
# ---------------------------------------------------------------------------

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


RELEVANT_ENTITY_TYPES = {
    "ORG", "PRODUCT", "GPE", "MONEY", "PERCENT",
    "CARDINAL", "DATE", "NORP", "FAC", "EVENT",
}


def extract_entities(text: str) -> List[Tuple[str, str]]:
    """Return list of (entity_text, entity_label) for relevant NER types."""
    nlp = _get_nlp()
    # Truncate to avoid spaCy max-length issues
    doc = nlp(text[:100_000])
    entities = []
    seen = set()
    for ent in doc.ents:
        if ent.label_ in RELEVANT_ENTITY_TYPES:
            key = (ent.text.strip(), ent.label_)
            if key not in seen:
                entities.append(key)
                seen.add(key)
    return entities


def extract_relations(text: str, entities: List[Tuple[str, str]]) -> List[Tuple[str, str, str]]:
    """
    Heuristic co-occurrence relation extraction.
    Two entities in the same sentence → lightweight 'related_to' edge.
    """
    nlp = _get_nlp()
    doc = nlp(text[:100_000])
    entity_texts = {e[0] for e in entities}
    relations = []

    for sent in doc.sents:
        sent_entities = []
        for ent in sent.ents:
            if ent.text.strip() in entity_texts:
                sent_entities.append(ent.text.strip())

        # Connect every pair in the same sentence
        for i in range(len(sent_entities)):
            for j in range(i + 1, len(sent_entities)):
                relations.append((sent_entities[i], "related_to", sent_entities[j]))

    return relations


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_knowledge_graph(documents: List[Document]) -> nx.Graph:
    """
    Build a NetworkX graph from a list of Documents.
    Nodes = entities (labelled with NER type + source).
    Edges = co-occurrence within same sentence.
    """
    G = nx.Graph()

    for doc in documents:
        text = doc.page_content
        source = doc.metadata.get("source", "unknown")

        entities = extract_entities(text)
        for ent_text, ent_label in entities:
            if not G.has_node(ent_text):
                G.add_node(ent_text, label=ent_label, sources=set())
            G.nodes[ent_text]["sources"].add(source)

        relations = extract_relations(text, entities)
        for head, relation, tail in relations:
            if G.has_edge(head, tail):
                G[head][tail]["weight"] = G[head][tail].get("weight", 1) + 1
            else:
                G.add_edge(head, tail, relation=relation, weight=1)

    return G


# ---------------------------------------------------------------------------
# Graph querying
# ---------------------------------------------------------------------------

def query_graph(
    G: nx.Graph,
    query: str,
    top_k: int = 5,
    hop_depth: int = 2,
) -> Dict[str, Any]:
    """
    Find the most relevant subgraph for a query.

    Strategy:
    1. Extract entities from the query text.
    2. Seed from those entities (or fall back to degree-ranked nodes).
    3. BFS-expand up to `hop_depth` hops.
    4. Return nodes, edges, and a textual reasoning trace.
    """
    # Entities from query
    query_entities = [e[0] for e in extract_entities(query)]

    # Seed nodes: query entities that exist in graph, else top-degree nodes
    seed_nodes = [n for n in query_entities if G.has_node(n)]
    if not seed_nodes:
        seed_nodes = sorted(G.nodes, key=lambda n: G.degree(n), reverse=True)[:top_k]

    # BFS expansion
    subgraph_nodes = set(seed_nodes)
    frontier = set(seed_nodes)
    for _ in range(hop_depth):
        next_frontier = set()
        for node in frontier:
            neighbors = list(G.neighbors(node))
            # Sort by edge weight → take strongest connections
            neighbors_sorted = sorted(
                neighbors,
                key=lambda n: G[node][n].get("weight", 1),
                reverse=True,
            )[:top_k]
            next_frontier.update(neighbors_sorted)
        subgraph_nodes.update(next_frontier)
        frontier = next_frontier

    subgraph = G.subgraph(subgraph_nodes)

    # Build reasoning trace
    trace_lines = [
        f"**Graph RAG reasoning trace**",
        f"- Query entities detected: {query_entities if query_entities else 'none (using top-degree fallback)'}",
        f"- Seed nodes: {seed_nodes}",
        f"- BFS expansion depth: {hop_depth} hops",
        f"- Subgraph size: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges",
    ]

    # Summarise top connections
    if subgraph.number_of_edges() > 0:
        top_edges = sorted(
            subgraph.edges(data=True),
            key=lambda e: e[2].get("weight", 1),
            reverse=True,
        )[:5]
        trace_lines.append("- Strongest connections:")
        for u, v, data in top_edges:
            trace_lines.append(f"  • {u} ↔ {v} (weight={data.get('weight', 1)})")

    return {
        "subgraph": subgraph,
        "seed_nodes": seed_nodes,
        "nodes": list(subgraph.nodes(data=True)),
        "edges": list(subgraph.edges(data=True)),
        "trace": "\n".join(trace_lines),
    }


def graph_context_to_text(graph_result: Dict[str, Any]) -> str:
    """
    Convert a subgraph result into a plain-text context string for the LLM.
    """
    lines = ["Entities and relationships from the knowledge graph:\n"]

    for node, data in graph_result["nodes"]:
        label = data.get("label", "")
        sources = ", ".join(data.get("sources", set()))
        lines.append(f"- {node} [{label}] (from: {sources})")

    lines.append("\nRelationships:")
    for u, v, data in graph_result["edges"]:
        weight = data.get("weight", 1)
        lines.append(f"  {u} ↔ {v} (co-occurrence strength: {weight})")

    return "\n".join(lines)
