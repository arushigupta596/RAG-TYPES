# backend/graph/traversal.py
# 2-hop BFS path finding at query time
from __future__ import annotations

from typing import List, Dict, Any, Set

import networkx as nx

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


def _extract_query_entities(query: str) -> List[str]:
    nlp = _get_nlp()
    doc = nlp(query)
    return [
        ent.text.strip()
        for ent in doc.ents
        if ent.label_ in RELEVANT_LABELS and ent.text.strip()
    ]


def _fuzzy_match_nodes(G: nx.DiGraph, query_entities: List[str]) -> List[str]:
    """
    Match query entities to graph nodes.
    Strategy: exact match first, then substring match (case-insensitive).
    """
    graph_nodes = set(G.nodes())
    matched: List[str] = []

    for qe in query_entities:
        qe_lower = qe.lower()
        # Exact
        if qe in graph_nodes:
            matched.append(qe)
            continue
        # Case-insensitive exact
        ci_match = next(
            (n for n in graph_nodes if n.lower() == qe_lower), None
        )
        if ci_match:
            matched.append(ci_match)
            continue
        # Substring containment
        sub_matches = [n for n in graph_nodes if qe_lower in n.lower()]
        if sub_matches:
            matched.extend(sub_matches[:2])

    return list(dict.fromkeys(matched))  # deduplicate, preserve order


def traverse(
    G: nx.DiGraph,
    query: str,
    max_hops: int = 2,
    max_seed_nodes: int = 5,
) -> Dict[str, Any]:
    """
    1. Extract query entities via spaCy.
    2. Fuzzy-match to graph nodes → seed_nodes.
    3. BFS up to max_hops from each seed.
    4. Collect all chunk_ids from traversed edges.
    5. Return: seed_nodes, paths, chunk_ids, trace_summary.
    """
    query_entities = _extract_query_entities(query)
    seed_nodes = _fuzzy_match_nodes(G, query_entities)

    # Fallback: use highest-degree nodes
    if not seed_nodes:
        seed_nodes = sorted(
            G.nodes(), key=lambda n: G.degree(n), reverse=True
        )[:max_seed_nodes]

    seed_nodes = seed_nodes[:max_seed_nodes]

    # BFS expansion
    visited: Set[str] = set(seed_nodes)
    frontier: Set[str] = set(seed_nodes)
    paths: List[tuple[str, str, str]] = []  # (from_node, edge_label, to_node)
    chunk_ids: Set[str] = set()

    for _ in range(max_hops):
        next_frontier: Set[str] = set()
        for node in frontier:
            for neighbour in G.successors(node):
                edge_data = G[node][neighbour]
                rel = edge_data.get("relation", "related-to")
                paths.append((node, rel, neighbour))
                # Collect chunk_ids from this edge
                for cid in edge_data.get("chunk_ids", set()):
                    chunk_ids.add(cid)
                if neighbour not in visited:
                    next_frontier.add(neighbour)
                    visited.add(neighbour)
        frontier = next_frontier

    trace_summary = (
        f"Query entities detected: {query_entities or ['none — using degree fallback']}\n"
        f"Seed nodes matched: {seed_nodes}\n"
        f"BFS depth: {max_hops} hops\n"
        f"Nodes visited: {len(visited)}\n"
        f"Chunk IDs collected: {len(chunk_ids)}\n"
        f"Top paths: {paths[:10]}"
    )

    return {
        "query_entities": query_entities,
        "seed_nodes": seed_nodes,
        "visited_nodes": list(visited),
        "paths": paths,
        "chunk_ids": list(chunk_ids),
        "trace_summary": trace_summary,
    }
