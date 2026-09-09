"""
nl_query.py

Step 6 of the pipeline: Natural-Language Query Layer (bonus feature).

Lets an investigator type a question like:
  "who are the top 3 most connected people?"
  "how is Rahul Sharma connected to Rafiq Sheikh?"
  "show me everyone linked to Imran Khan"

...and get an answer computed directly from the graph you already built in
graph_analytics.py.

Design choice: instead of asking the LLM to write raw Cypher/NetworkX code
and eval-ing it (dangerous — arbitrary code execution on user input, and
overkill since you're on NetworkX not Neo4j), the LLM's ONLY job is to
translate the question into a small structured JSON "filter spec" from a
fixed set of safe, pre-built operations. This module then executes that
spec against the graph in Python. Zero training, careful prompting only —
and it can't be used to inject arbitrary graph-traversal code.

Two modes:
  - LLM mode (uses the Anthropic API) — handles open-ended phrasing.
  - Fallback keyword mode (no API key / no network needed) — handles the
    common question shapes with simple keyword matching, so your demo
    still works even if the venue wifi dies.

Usage:
    from nl_query import answer_question
    from graph_analytics import build_graph

    G = build_graph(entities, triples)
    answer_question(G, "who are the top 3 most connected people?")
"""

import json
import os
import re
from typing import Dict, Any, Optional, List

import networkx as nx


# ---------------------------------------------------------------------------
# The fixed set of safe operations the LLM is allowed to request.
# Nothing outside this list can ever run, no matter what the LLM returns.
# ---------------------------------------------------------------------------
SUPPORTED_ACTIONS = {
    "top_n_by_centrality",  # {"metric": "pagerank"|"degree"|"betweenness"|"eigenvector", "n": int}
    "neighbors",             # {"node": str}                -> who is X directly connected to
    "shortest_path",         # {"source": str, "target": str} -> how is X connected to Y
    "filter_by_label",       # {"label": str}                -> all entities of a given NER type
    "node_summary",          # {"node": str}                  -> everything known about one entity
}


# ---------------------------------------------------------------------------
# LLM mode: question -> filter spec (JSON)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You convert an investigator's natural-language question about a case graph into a JSON filter spec.

Return ONLY a JSON object, no other text, no markdown fences. It must have this shape:
{"action": "<one of: top_n_by_centrality, neighbors, shortest_path, filter_by_label, node_summary>", ...params}

Action params:
- top_n_by_centrality: {"metric": "pagerank" | "degree" | "betweenness" | "eigenvector", "n": <int>}
- neighbors: {"node": "<entity name, matched from the graph node list below>"}
- shortest_path: {"source": "<entity name>", "target": "<entity name>"}
- filter_by_label: {"label": "PERSON" | "ORG" | "GPE" | "MONEY" | ...}
- node_summary: {"node": "<entity name>"}

If the question doesn't map cleanly to one of these, pick the closest one.
Match entity names against the exact node list you're given — don't invent new names.
"""


def question_to_spec_llm(question: str, node_list: List[str]) -> Optional[Dict[str, Any]]:
    """
    Calls the Anthropic API to translate a question into a filter spec.
    Requires `pip install anthropic` and an ANTHROPIC_API_KEY environment
    variable. Returns None (triggering fallback mode) if the SDK/key
    aren't available, or if the model's output can't be parsed as JSON.
    """
    try:
        import anthropic
    except ImportError:
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Graph nodes: {node_list}\n\nQuestion: {question}",
            }],
        )
        raw = response.content[0].text.strip()
        raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
        spec = json.loads(raw)
        if spec.get("action") in SUPPORTED_ACTIONS:
            return spec
        return None
    except Exception:
        # Any API/network/parsing failure -> fall back rather than crash the demo
        return None


# ---------------------------------------------------------------------------
# Fallback mode: simple keyword matching, zero dependencies, zero network.
# Covers the common investigator question shapes without needing an API key.
# ---------------------------------------------------------------------------
def question_to_spec_fallback(question: str, node_list: List[str]) -> Dict[str, Any]:
    q = question.lower()

    # "how is X connected to Y" / "path between X and Y"
    if any(kw in q for kw in ["connected to", "path between", "link between", "how is"]):
        found = [n for n in node_list if n.lower() in q]
        if len(found) >= 2:
            return {"action": "shortest_path", "source": found[0], "target": found[1]}

    # "who is connected to X" / "linked to X" / "neighbors of X"
    if any(kw in q for kw in ["connected to", "linked to", "neighbors of", "related to"]):
        found = [n for n in node_list if n.lower() in q]
        if found:
            return {"action": "neighbors", "node": found[0]}

    # "top N ..." / "most influential" / "key individuals" / "most connected"
    if any(kw in q for kw in ["top", "most influential", "key individual", "most connected", "important"]):
        n_match = re.search(r"\btop\s+(\d+)\b", q)
        n = int(n_match.group(1)) if n_match else 5
        metric = "pagerank"
        if "between" in q or "bridge" in q or "go-between" in q:
            metric = "betweenness"
        elif "degree" in q or "connected" in q:
            metric = "degree"
        return {"action": "top_n_by_centrality", "metric": metric, "n": n}

    # "all people" / "all organizations" / "show me the locations"
    label_map = {"people": "PERSON", "person": "PERSON", "organi": "ORG",
                 "location": "GPE", "place": "GPE", "money": "MONEY"}
    for keyword, label in label_map.items():
        if keyword in q:
            return {"action": "filter_by_label", "label": label}

    # "tell me about X" / "who is X"
    found = [n for n in node_list if n.lower() in q]
    if found:
        return {"action": "node_summary", "node": found[0]}

    # default: just show top individuals, better than returning nothing
    return {"action": "top_n_by_centrality", "metric": "pagerank", "n": 5}


# ---------------------------------------------------------------------------
# Spec execution against the graph — this is the only code that actually
# touches G, and it only runs the fixed operations above.
# ---------------------------------------------------------------------------
def execute_spec(G: nx.Graph, spec: Dict[str, Any]) -> str:
    action = spec.get("action")

    if action == "top_n_by_centrality":
        metric = spec.get("metric", "pagerank")
        n = spec.get("n", 5)
        metric_fn = {
            "pagerank": nx.pagerank,
            "degree": nx.degree_centrality,
            "betweenness": nx.betweenness_centrality,
            "eigenvector": lambda g: nx.eigenvector_centrality(g, max_iter=1000),
        }.get(metric, nx.pagerank)
        try:
            scores = metric_fn(G)
        except (nx.PowerIterationFailedConvergence, nx.NetworkXException):
            return f"Could not compute {metric} centrality — graph too small/sparse."
        top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n]
        lines = [f"Top {n} entities by {metric}:"]
        lines += [f"  {node} ({score:.3f})" for node, score in top]
        return "\n".join(lines)

    elif action == "neighbors":
        node = _resolve_node(G, spec.get("node", ""))
        if node is None:
            return f"'{spec.get('node')}' not found in the graph."
        neighbors = list(G.neighbors(node))
        if not neighbors:
            return f"{node} has no known connections."
        lines = [f"{node} is directly connected to:"]
        for nb in neighbors:
            relations = G[node][nb].get("relations", set())
            rel_str = ", ".join(relations) if relations else "unknown relation"
            lines.append(f"  {nb}  ({rel_str})")
        return "\n".join(lines)

    elif action == "shortest_path":
        source = _resolve_node(G, spec.get("source", ""))
        target = _resolve_node(G, spec.get("target", ""))
        if source is None or target is None:
            return "Could not find one or both entities in the graph."
        try:
            path = nx.shortest_path(G, source, target)
        except nx.NetworkXNoPath:
            return f"No connection found between {source} and {target}."
        return f"Path: {' -> '.join(path)}  ({len(path) - 1} hops)"

    elif action == "filter_by_label":
        label = spec.get("label", "")
        matches = [n for n, d in G.nodes(data=True) if label in d.get("label", "")]
        if not matches:
            return f"No entities found with label '{label}'."
        return f"Entities labeled {label}:\n  " + "\n  ".join(matches)

    elif action == "node_summary":
        node = _resolve_node(G, spec.get("node", ""))
        if node is None:
            return f"'{spec.get('node')}' not found in the graph."
        data = G.nodes[node]
        neighbors = list(G.neighbors(node))
        lines = [
            f"Entity: {node}",
            f"  Type: {data.get('label', 'UNKNOWN')}",
            f"  Direct connections: {len(neighbors)}",
        ]
        for nb in neighbors:
            relations = G[node][nb].get("relations", set())
            lines.append(f"    -> {nb} ({', '.join(relations) or 'unknown relation'})")
        return "\n".join(lines)

    return f"Unrecognized action: {action}"


REASONING_SYSTEM_PROMPT = """You are an investigative analysis assistant helping a police investigator interpret facts pulled from a case graph (built from FIRs, call records, and financial records via NER + relation extraction).

You will be given:
- The investigator's original question
- Raw facts computed directly from the graph (centrality scores, connections, paths, etc.)

Write a short (3-6 sentence) plain-language explanation of what these facts suggest, IN THE CONTEXT OF the question asked.

Hard rules:
- Only reason about entities, relations, and numbers that appear in the provided facts. Never invent a name, connection, amount, or date that isn't in the facts.
- Do not assert guilt or make a legal conclusion ("X is guilty/a criminal") — describe patterns and connections for the investigator's own judgment, using cautious language ("this suggests", "worth investigating further", "notable because").
- If the facts are thin (e.g. just one connection), say so plainly rather than over-interpreting.
- Be concrete: reference the actual names/numbers from the facts, not generic statements.
"""


def generate_reasoning(question: str, raw_facts: str) -> str:
    """
    Second LLM call: takes the RAW, already-computed graph facts (never the
    graph itself, and never anything the model could hallucinate structure
    from) and asks the model to explain what they mean for the investigator's
    question. Grounding the prompt in precomputed facts — rather than asking
    the model to reason over the graph freeform — is what keeps this from
    inventing connections that aren't actually in your data.

    Falls back to a plain templated explanation if no API key/package is
    available, so the pipeline still produces *something* offline.
    """
    try:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("no api key")

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            system=REASONING_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Question: {question}\n\nFacts from the graph:\n{raw_facts}",
            }],
        )
        return response.content[0].text.strip()

    except Exception:
        # Offline/no-key fallback: a plain, non-hallucinated restatement.
        return (
            "(AI reasoning unavailable — no ANTHROPIC_API_KEY set, or the "
            "`anthropic` package isn't installed. Showing raw facts only; "
            "set the key and re-run for a reasoned explanation.)"
        )


def _resolve_node(G: nx.Graph, name: str) -> Optional[str]:
    """Case-insensitive node lookup, since LLM/keyword matching won't always hit exact casing."""
    if not name:
        return None
    for n in G.nodes():
        if n.lower() == name.lower():
            return n
    # loose fallback: substring match
    for n in G.nodes():
        if name.lower() in n.lower() or n.lower() in name.lower():
            return n
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def answer_question(G: nx.Graph, question: str, verbose: bool = True,
                     with_reasoning: bool = True) -> Dict[str, str]:
    """
    Full query pipeline: question -> spec -> raw facts -> (optional) reasoning.

    Returns {"spec": ..., "facts": ..., "reasoning": ...} so a UI can show
    the raw computed facts (auditable, exact) alongside the AI's plain-
    language interpretation (helpful, but always secondary to the facts).
    """
    node_list = list(G.nodes())

    spec = question_to_spec_llm(question, node_list)
    mode = "LLM"
    if spec is None:
        spec = question_to_spec_fallback(question, node_list)
        mode = "keyword fallback"

    facts = execute_spec(G, spec)
    reasoning = generate_reasoning(question, facts) if with_reasoning else None

    if verbose:
        print(f"[{mode}] spec: {spec}")
        print("Facts (computed directly from the graph):")
        print(facts)
        if with_reasoning:
            print("\nReasoning:")
            print(reasoning)
        print()

    return {"spec": spec, "facts": facts, "reasoning": reasoning}


# ---------------------------------------------------------------------------
# Demo — reuses the exact FIR graph from graph_analytics.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from graph_analytics import build_graph

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

    triples = [
        {"subject": "Rahul Sharma", "relation": "located in", "object": "Connaught Place"},
        {"subject": "Connaught Place", "relation": "located in", "object": "Delhi"},
        {"subject": "Rahul Sharma", "relation": "met", "object": "Imran Khan"},
        {"subject": "Rahul Sharma", "relation": "sent money to", "object": "Mohd Arif"},
        {"subject": "Rahul Sharma", "relation": "employer", "object": "SBI Bank"},
        {"subject": "Imran Khan", "relation": "contacted", "object": "Rafiq Sheikh"},
    ]

    G = build_graph(entities, triples)

    # No ANTHROPIC_API_KEY set in this environment -> runs in fallback mode.
    # Set the env var (and `pip install anthropic`) to switch to LLM mode
    # automatically, no code changes needed.
    questions = [
        "who are the top 3 most connected people?",
        "who is linked to Rahul Sharma?",
        "how is Imran Khan connected to SBI Bank?",
        "show me all the people",
        "tell me about Mohd Arif",
    ]

    for q in questions:
        print(f"Q: {q}")
        answer_question(G, q, with_reasoning=True)
