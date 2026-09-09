"""
entity_resolution.py

Step 3 of the pipeline: Entity Resolution / Deduplication.

Takes entity records pulled from different sources (FIR text, CDR/call
records, bank/financial records) and figures out which ones refer to the
SAME real-world person, so they can be collapsed into a single graph node.

Two-tier matching:
  1. Exact-key matching  -> phone number, account number, vehicle plate
  2. Fuzzy name matching  -> handles spelling/format variation in names

Usage:
    from entity_resolution import resolve_entities, EntityRecord

    records = [
        EntityRecord(id="fir_1", name="Ramesh Kumar", phone="9876543210", source="FIR"),
        EntityRecord(id="cdr_5", name="R. Kumar", phone="9876543210", source="CDR"),
        EntityRecord(id="bank_2", name="Ramesh K.", account="AC1234", source="BANK"),
    ]
    matches = resolve_entities(records)
    for m in matches:
        print(m)
"""

from dataclasses import dataclass, field
from itertools import combinations
from typing import List, Optional
import re

# ---------------------------------------------------------------------------
# Similarity backend: use rapidfuzz if available (faster, better tuned for
# this use case), otherwise fall back to Python's built-in difflib so the
# script still runs with zero extra installs.
# ---------------------------------------------------------------------------
try:
    from rapidfuzz import fuzz

    def name_similarity(a: str, b: str) -> float:
        """Returns 0-100 similarity score using rapidfuzz."""
        return fuzz.token_sort_ratio(a, b)

    SIMILARITY_BACKEND = "rapidfuzz"

except ImportError:
    from difflib import SequenceMatcher

    def name_similarity(a: str, b: str) -> float:
        """Returns 0-100 similarity score using difflib (fallback)."""
        # token_sort_ratio equivalent: sort words, then compare
        a_sorted = " ".join(sorted(a.split()))
        b_sorted = " ".join(sorted(b.split()))
        return SequenceMatcher(None, a_sorted, b_sorted).ratio() * 100

    SIMILARITY_BACKEND = "difflib (rapidfuzz not installed — pip install rapidfuzz for better accuracy)"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class EntityRecord:
    id: str                      # unique id you assign, e.g. "fir_1"
    name: Optional[str] = None
    phone: Optional[str] = None
    account: Optional[str] = None
    vehicle: Optional[str] = None
    address: Optional[str] = None
    source: Optional[str] = None  # "FIR", "CDR", "BANK", "SOCIAL", etc.


@dataclass
class MatchResult:
    id_a: str
    id_b: str
    score: float                 # 0.0 - 1.0 confidence
    reason: str                  # why they were matched, for explainability
    match_type: str              # "exact" or "fuzzy"

    def __repr__(self):
        return (f"MatchResult({self.id_a} <-> {self.id_b}, "
                f"score={self.score:.2f}, reason='{self.reason}')")


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------
HONORIFICS = {"mr", "mrs", "ms", "shri", "smt", "dr", "sir"}


def normalize_name(name: str) -> str:
    if not name:
        return ""
    name = name.lower().strip()
    name = re.sub(r"[.,]", "", name)          # strip periods/commas
    words = [w for w in name.split() if w not in HONORIFICS]
    return " ".join(words)


def normalize_phone(phone: str) -> str:
    if not phone:
        return ""
    digits = re.sub(r"\D", "", phone)          # keep digits only
    return digits[-10:] if len(digits) >= 10 else digits  # last 10 digits


def normalize_key(value: str) -> str:
    if not value:
        return ""
    return re.sub(r"[\s\-]", "", value).upper()


# ---------------------------------------------------------------------------
# Core matching logic
# ---------------------------------------------------------------------------
FUZZY_NAME_THRESHOLD = 85  # tune this: lower = more matches, more false positives


def compare_pair(a: EntityRecord, b: EntityRecord) -> Optional[MatchResult]:
    """Compares two records, returns a MatchResult if they look like a match."""

    # --- Tier 1: exact-key matches (high confidence) ---
    if a.phone and b.phone:
        if normalize_phone(a.phone) == normalize_phone(b.phone) and normalize_phone(a.phone):
            return MatchResult(a.id, b.id, 1.0, "exact phone match", "exact")

    if a.account and b.account:
        if normalize_key(a.account) == normalize_key(b.account) and normalize_key(a.account):
            return MatchResult(a.id, b.id, 1.0, "exact account match", "exact")

    if a.vehicle and b.vehicle:
        if normalize_key(a.vehicle) == normalize_key(b.vehicle) and normalize_key(a.vehicle):
            return MatchResult(a.id, b.id, 1.0, "exact vehicle plate match", "exact")

    # --- Tier 2: fuzzy name match (lower confidence, needs a threshold) ---
    if a.name and b.name:
        na, nb = normalize_name(a.name), normalize_name(b.name)
        if na and nb:
            score = name_similarity(na, nb)
            if score >= FUZZY_NAME_THRESHOLD:
                return MatchResult(
                    a.id, b.id,
                    round(score / 100, 2),
                    f"fuzzy name match ('{a.name}' ~ '{b.name}', {score:.0f}%)",
                    "fuzzy",
                )

    return None


def resolve_entities(records: List[EntityRecord]) -> List[MatchResult]:
    """
    Compares every pair of records and returns all matches found.
    O(n^2) — fine for hackathon-scale data (hundreds of records).
    For thousands+, add blocking (e.g. only compare records sharing a
    source pair or a name first-letter) before doing pairwise comparison.
    """
    matches = []
    for a, b in combinations(records, 2):
        result = compare_pair(a, b)
        if result:
            matches.append(result)
    return matches


def build_clusters(records: List[EntityRecord], matches: List[MatchResult]) -> List[List[str]]:
    """
    Groups matched record ids into clusters (connected components), so you
    get one cluster per real-world entity, ready to become a single graph node.
    """
    parent = {r.id: r.id for r in records}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for m in matches:
        union(m.id_a, m.id_b)

    clusters = {}
    for r in records:
        root = find(r.id)
        clusters.setdefault(root, []).append(r.id)

    return list(clusters.values())


# ---------------------------------------------------------------------------
# Pipeline integration
#
# The functions above operate on EntityRecord/MatchResult, which is the
# right shape when you have structured fields (phone/account/vehicle) from
# CDRs and bank records. But extraction.py's spaCy output is just
# {"text": ..., "label": ...} — no phone/account/vehicle. These two
# functions bridge that gap so pipeline.py can call entity resolution
# directly on raw NER + REBEL output, collapsing near-duplicate mentions
# ("Rahul Sharma" / "Rahul S." / "R. Sharma") into one canonical name
# BEFORE the graph is built — this is what turns resolve() in pipeline.py
# from a pass-through into the real thing.
# ---------------------------------------------------------------------------
def resolve_and_merge_entities(entities: List[dict]) -> List[dict]:
    """
    Takes spaCy-style entities [{"text": ..., "label": ...}, ...], fuzzy-
    matches near-duplicate names against each other, and replaces each
    mention with its cluster's canonical name (the longest/most complete
    mention in the cluster — "Rahul Sharma" wins over "Rahul S.").

    Returns the same shape it was given, so it's a drop-in replacement in
    the pipeline: entities go in, entities (deduped) come out.
    """
    records = [
        EntityRecord(id=f"e{i}", name=e.get("text"), source="NER")
        for i, e in enumerate(entities)
    ]

    matches = resolve_entities(records)
    clusters = build_clusters(records, matches)

    id_to_record = {r.id: r for r in records}
    id_to_canonical: dict = {}
    for cluster in clusters:
        names = [id_to_record[rid].name for rid in cluster if id_to_record[rid].name]
        canonical = max(names, key=len) if names else None
        for rid in cluster:
            id_to_canonical[rid] = canonical

    merged = []
    for i, e in enumerate(entities):
        rid = f"e{i}"
        canonical = id_to_canonical.get(rid) or e.get("text")
        merged.append({"text": canonical, "label": e.get("label", "UNKNOWN")})
    return merged


def apply_resolution_to_triples(triples: List[dict], canonical_names: List[str]) -> List[dict]:
    """
    REBEL sometimes extracts a shorter/different mention of an entity than
    spaCy did ("Rahul" in one relation, "Rahul Sharma" as the NER entity).
    This fuzzy-matches each triple's subject/object against the resolved
    canonical names from resolve_and_merge_entities() and swaps in the
    canonical form where confident, so the graph doesn't end up with a
    stray extra node for the same person.

    Falls back to leaving the original text untouched if nothing matches
    closely enough (safer than forcing a bad merge).
    """
    def best_canonical(mention: str) -> str:
        if not mention or not canonical_names:
            return mention
        best_name, best_score = mention, 0.0
        for name in canonical_names:
            score = name_similarity(normalize_name(mention), normalize_name(name))
            if score > best_score:
                best_name, best_score = name, score
        return best_name if best_score >= FUZZY_NAME_THRESHOLD else mention

    resolved = []
    for t in triples:
        resolved.append({
            "subject": best_canonical(t.get("subject", "")),
            "relation": t.get("relation", ""),
            "object": best_canonical(t.get("object", "")),
        })
    return resolved


# ---------------------------------------------------------------------------
# Demo / smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Similarity backend in use: {SIMILARITY_BACKEND}\n")

    records = [
        EntityRecord(id="fir_1", name="Ramesh Kumar", phone="+91-98765-43210", source="FIR"),
        EntityRecord(id="cdr_5", name="R. Kumar", phone="9876543210", source="CDR"),
        EntityRecord(id="bank_2", name="Ramesh K.", account="AC-1234-5678", source="BANK"),
        EntityRecord(id="fir_9", name="Ramesh Kumar", account="AC12345678", source="FIR"),
        EntityRecord(id="social_3", name="Suresh Yadav", phone="9123456780", source="SOCIAL"),
        EntityRecord(id="cdr_11", name="Sooresh Yadav", phone="9123456780", source="CDR"),
        EntityRecord(id="fir_20", name="Priya Sharma", vehicle="DL-3C-AB-1234", source="FIR"),
        EntityRecord(id="cctv_4", name=None, vehicle="DL3CAB1234", source="CCTV"),
    ]

    matches = resolve_entities(records)
    print(f"Found {len(matches)} pairwise matches:\n")
    for m in matches:
        print(" ", m)

    clusters = build_clusters(records, matches)
    print(f"\nResolved into {len(clusters)} distinct entities:\n")
    for i, cluster in enumerate(clusters, 1):
        print(f"  Entity {i}: {cluster}")
