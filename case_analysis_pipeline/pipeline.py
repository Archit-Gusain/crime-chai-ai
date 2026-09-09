"""
pipeline.py

FINAL PROJECT PIPELINE — ties together every step built so far:

  PDF upload
      |
      v
  [pdf_ingest.py]      -> raw case text
      |
      v
  [extraction.py]      -> entities (spaCy NER) + triples (REBEL relations)
      |
      v
  [entity_resolution.py] -> dedups entity mentions (fuzzy matching) and
      |                      re-points REBEL triples at the merged names
      v
  [graph_analytics.py] -> builds the case graph, computes centrality +
      |                    Louvain communities
      v
  [anomaly_detection.py] -> (optional) flags suspicious transactions, IF
      |                      transaction records are supplied separately
      v
  [nl_query.py]         -> investigator asks a question ->
                            graph facts (exact, computed) +
                            AI reasoning (grounded explanation of those facts)

Run end-to-end:
    python pipeline.py path/to/case.pdf

Or import and drive it yourself:
    from pipeline import run_pipeline
    result = run_pipeline("case.pdf")
    result["ask"]("who are the key individuals in this case?")
"""

import sys
from typing import Dict, Any, List, Optional, Callable, Tuple

import pdf_ingest
import extraction
from graph_analytics import build_graph, analyze_graph, print_report
from nl_query import answer_question

# entity_resolution.py isn't wired into the graph-building step yet (fuzzy
# matching is still a TODO on your end) — imported here so it's a one-line
# change to plug in once it's ready. See the NOTE in run_pipeline() below.
import entity_resolution


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------
def ingest(pdf_path: str) -> str:
    """Stage 1: PDF -> text."""
    print(f"[1/5] Reading {pdf_path} ...")
    text = pdf_ingest.load_pdf_text(pdf_path)
    print(f"      Extracted {len(text)} characters.")
    return text


def extract(text: str) -> Dict[str, List[Dict[str, str]]]:
    """Stage 2: text -> entities + relation triples."""
    print("[2/5] Running NER + relation extraction ...")
    entities, triples = extraction.extract_entities_and_relations(text)
    print(f"      Found {len(entities)} entities, {len(triples)} relations.")
    return {"entities": entities, "triples": triples}


def resolve(entities: List[Dict[str, str]],
            triples: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Stage 3: entity resolution / deduplication.

    Fuzzy-merges near-duplicate entity mentions ("Rahul Sharma" / "R. Sharma")
    into one canonical name using entity_resolution.py, THEN re-points every
    triple's subject/object at that same canonical name — so REBEL mentions
    that spaCy phrased slightly differently don't create a stray extra node
    when the graph is built.

    NOTE: entity_resolution.py's fuzzy backend uses `rapidfuzz` if installed,
    falling back to Python's built-in `difflib` otherwise. difflib is
    noticeably weaker on partial-name matches (e.g. "Rahul Sharma" vs
    "Rahul S." scores ~74%, under the 85% merge threshold, with difflib but
    would merge with rapidfuzz). Run `pip install rapidfuzz` in your actual
    (networked) environment for meaningfully better dedup accuracy.
    """
    print("[3/5] Entity resolution (fuzzy-merging near-duplicate mentions) ...")
    merged_entities = entity_resolution.resolve_and_merge_entities(entities)
    canonical_names = sorted({e["text"] for e in merged_entities if e.get("text")})
    resolved_triples = entity_resolution.apply_resolution_to_triples(triples, canonical_names)

    before, after = len(entities), len(canonical_names)
    print(f"      {before} raw entity mentions -> {after} distinct entities after merging.")
    return merged_entities, resolved_triples


def build_case_graph(entities: List[Dict[str, str]], triples: List[Dict[str, str]]):
    """Stage 4: build the graph + run centrality/community analysis."""
    print("[4/5] Building case graph + running centrality/community analysis ...")
    G = build_graph(entities, triples)
    report = analyze_graph(G)
    print(f"      Graph: {report.get('node_count', 0)} entities, "
          f"{report.get('edge_count', 0)} relations, "
          f"{len(report.get('communities', []))} clusters.")
    return G, report


def make_query_fn(G) -> Callable[[str], Dict[str, str]]:
    """Stage 5: returns a ready-to-call `ask(question)` function bound to this case's graph."""
    def ask(question: str) -> Dict[str, str]:
        return answer_question(G, question, verbose=True, with_reasoning=True)
    return ask


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_pipeline(pdf_path: str, run_anomaly: bool = False,
                  transactions: Optional[list] = None) -> Dict[str, Any]:
    """
    Runs the full pipeline on one uploaded PDF and returns everything
    downstream code (a notebook cell, a Streamlit UI, etc.) needs:
      - text, entities, triples: intermediate outputs, useful for debugging
      - G: the NetworkX case graph
      - report: centrality + community results
      - ask(question): a ready-to-use NL query function with AI reasoning
      - anomaly_report: only populated if you pass in `transactions`
        (anomaly_detection.py needs structured transaction records — amount
        + timestamp + actor — which typically come from bank/financial
        records rather than the FIR PDF itself, so it's opt-in here rather
        than assumed to run on every PDF)
    """
    text = ingest(pdf_path)
    extracted = extract(text)
    entities, triples = resolve(extracted["entities"], extracted["triples"])

    G, report = build_case_graph(entities, triples)

    print("[5/5] Query layer ready.")
    ask = make_query_fn(G)

    result = {
        "text": text,
        "entities": entities,
        "triples": triples,
        "graph": G,
        "report": report,
        "ask": ask,
    }

    if run_anomaly and transactions:
        from anomaly_detection import run_anomaly_detection
        result["anomaly_report"] = run_anomaly_detection(transactions)

    return result


# ---------------------------------------------------------------------------
# CLI entry point — run the whole thing end to end on a PDF, then drop into
# an interactive query loop.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py path/to/case.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    result = run_pipeline(pdf_path)

    print("\n" + "=" * 60)
    print("CASE GRAPH SUMMARY")
    print("=" * 60)
    print_report(result["report"])

    print("\n" + "=" * 60)
    print("INTERACTIVE QUERY  (type a question, or 'quit' to exit)")
    print("=" * 60)
    while True:
        try:
            question = input("\n> ").strip()
        except EOFError:
            break
        if not question or question.lower() in {"quit", "exit"}:
            break
        result["ask"](question)
