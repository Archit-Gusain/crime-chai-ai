"""
graph_analytics.py

Step 4 of the pipeline: Graph Analytics (algorithmic, no training).

Consumes exactly what your existing notebook already produces:
  - entities: list of dicts like {"text": "Rahul Sharma", "label": "PERSON"}
              (from spaCy)
  - triples:  list of dicts like {"subject": "Rahul", "relation": "met",
              "object": "Imran Khan"}  (from REBEL's extract_triplets())

Builds a NetworkX graph, then runs:
  - Degree, Betweenness, Eigenvector, PageRank centrality -> "who's influential"
  - Louvain community detection                            -> "who's in the same cell"

No fuzzy entity resolution wired in yet (you said that's coming later) —
for now, node identity is decided by normalized exact text match. Once you
add step 3's resolver, swap `normalize()` calls for a lookup into your
cluster map (see the NOTE near build_graph()).

Usage:
    from graph_analytics import build_graph, analyze_graph, print_report

    G = build_graph(entities, triples)
    report = analyze_graph(G)
    print_report(report)
"""

from collections import defaultdict
from typing import List, Dict, Any
import networkx as nx


# ---------------------------------------------------------------------------
# Normalization (placeholder for entity resolution — replace later)
# ---------------------------------------------------------------------------
def normalize(name: str) -> str:
    """
    Minimal normalization so 'Rahul' and 'Rahul Sharma' don't immediately
    collide, but casing/whitespace differences do collapse.

    NOTE: This is intentionally dumb — it's a stand-in for step 3's fuzzy
    entity resolution. Once that module is ready, replace this function's
    body with a lookup into the resolved-cluster map so "Rahul" and
    "Rahul Sharma" (same person, different mention) land on the same node.
    """
    if not name:
        return ""
    return " ".join(name.strip().split()).title()


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_graph(entities: List[Dict[str, Any]], triples: List[Dict[str, str]]) -> nx.Graph:
    """
    Builds an undirected graph:
      - one node per entity (attribute: label, e.g. PERSON/ORG/GPE)
      - one edge per REBEL triple (attribute: relation text)

    Entities not mentioned in any triple still get added as isolated nodes
    so they show up in the report (useful for spotting entities REBEL missed
    a relation for).
    """
    G = nx.Graph()

    # Add all NER entities as nodes first, so we retain their labels
    # (PERSON/ORG/GPE/etc) even for ones with no relation.
    for ent in entities:
        node = normalize(ent.get("text", ""))
        if not node:
            continue
        label = ent.get("label", "UNKNOWN")
        if G.has_node(node):
            # keep the first label seen; note if conflicting labels appear
            existing = G.nodes[node].get("label")
            if existing and existing != label:
                G.nodes[node]["label"] = f"{existing}/{label}"
        else:
            G.add_node(node, label=label)

    # Add edges from REBEL triples
    for t in triples:
        subj = normalize(t.get("subject", ""))
        obj = normalize(t.get("object", ""))
        relation = t.get("relation", "").strip()

        if not subj or not obj:
            continue  # skip malformed triples (empty subject/object)

        # Nodes may not have come from spaCy (REBEL sometimes extracts
        # entities NER missed) — add them so no relation gets silently dropped.
        if not G.has_node(subj):
            G.add_node(subj, label="UNKNOWN")
        if not G.has_node(obj):
            G.add_node(obj, label="UNKNOWN")

        if G.has_edge(subj, obj):
            # multiple relations between the same pair -> keep all of them
            G[subj][obj].setdefault("relations", set()).add(relation)
        else:
            G.add_edge(subj, obj, relations={relation})

    return G


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
def analyze_graph(G: nx.Graph, top_n: int = 10) -> Dict[str, Any]:
    """
    Runs the standard set of "who's influential / who's clustered together"
    algorithms. All deterministic, no training.
    """
    if G.number_of_nodes() == 0:
        return {"error": "empty graph — no entities/triples to analyze"}

    report: Dict[str, Any] = {}

    # --- Centrality measures ---
    report["degree_centrality"] = _top(nx.degree_centrality(G), top_n)
    report["betweenness_centrality"] = _top(nx.betweenness_centrality(G), top_n)

    # Eigenvector centrality can fail to converge on tiny/disconnected
    # graphs — degrade gracefully instead of crashing the whole pipeline.
    try:
        report["eigenvector_centrality"] = _top(
            nx.eigenvector_centrality(G, max_iter=1000), top_n
        )
    except (nx.PowerIterationFailedConvergence, nx.NetworkXException):
        report["eigenvector_centrality"] = {"note": "did not converge (graph too small/sparse)"}

    report["pagerank"] = _top(nx.pagerank(G), top_n)

    # --- Community detection (Louvain) ---
    if G.number_of_edges() > 0:
        communities = nx.community.louvain_communities(G, seed=42)
        report["communities"] = [sorted(c) for c in communities]
    else:
        report["communities"] = [[n] for n in G.nodes()]  # no edges -> every node alone

    report["node_count"] = G.number_of_nodes()
    report["edge_count"] = G.number_of_edges()

    return report


def _top(scores: Dict[str, float], n: int) -> Dict[str, float]:
    """Returns the top-n entries of a {node: score} dict, sorted descending."""
    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n])


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_report(report: Dict[str, Any]) -> None:
    if "error" in report:
        print(report["error"])
        return

    print(f"Graph: {report['node_count']} entities, {report['edge_count']} relations\n")

    print("Key individuals (PageRank — overall influence):")
    for node, score in report["pagerank"].items():
        print(f"  {node:25s} {score:.3f}")

    print("\nKey individuals (Betweenness — bridges/go-betweens):")
    for node, score in report["betweenness_centrality"].items():
        print(f"  {node:25s} {score:.3f}")

    print("\nClusters / suspected cells (Louvain communities):")
    for i, community in enumerate(report["communities"], 1):
        print(f"  Cluster {i}: {community}")


# ---------------------------------------------------------------------------
# Demo using YOUR notebook's exact FIR example + REBEL-style triples
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    entities = [
        {"text": "Rahul Sharma", "label": "PERSON"},
        {"text": "Imran Khan", "label": "PERSON"},
        {"text": "Connaught Place", "label": "GPE"},
        {"text": "Delhi", "label": "GPE"},
        {"text": "₹2,50,000", "label": "MONEY"},
        {"text": "Mohd Arif", "label": "PERSON"},
        {"text": "SBI Bank", "label": "ORG"},
        {"text": "Rafiq Sheikh", "label": "PERSON"},
    ]

    # This is the shape extract_triplets() returns in your notebook
    triples = [
        {"subject": "Rahul Sharma", "relation": "located in", "object": "Connaught Place"},
        {"subject": "Connaught Place", "relation": "located in", "object": "Delhi"},
        {"subject": "Rahul Sharma", "relation": "met", "object": "Imran Khan"},
        {"subject": "Rahul Sharma", "relation": "sent money to", "object": "Mohd Arif"},
        {"subject": "Rahul Sharma", "relation": "employer", "object": "SBI Bank"},
        {"subject": "Imran Khan", "relation": "contacted", "object": "Rafiq Sheikh"},
    ]

    G = build_graph(entities, triples)
    report = analyze_graph(G)
    print_report(report)
