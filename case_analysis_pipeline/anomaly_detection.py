"""
anomaly_detection.py

Step 5 of the pipeline: Anomaly / Suspicious Pattern Detection.

Two complementary layers, as discussed:
  1. Isolation Forest (unsupervised, sklearn) — catches statistical outliers
     in transaction amount/frequency/timing that don't fit an obvious rule.
  2. Rule-based flags (no ML) — catches known patterns investigators
     specifically look for (structuring, off-hours activity, round-number
     transfers). These are fast, deterministic, and — importantly for
     judges — fully explainable, unlike the Isolation Forest score alone.

Designed to consume a flat list of transaction/record dicts — e.g. the
kind of thing you'd get out of bank/financial records once entities are
resolved (step 3) and linked in the graph (step 4). Each transaction just
needs: an id, an actor (the resolved entity name), an amount, and a
timestamp. Extra fields are ignored, so you can pass richer records
straight through.

Usage:
    from anomaly_detection import run_anomaly_detection, Transaction

    txns = [
        Transaction(id="t1", actor="Rahul Sharma", amount=250000,
                    timestamp="2025-03-03T21:10:00"),
        ...
    ]
    result = run_anomaly_detection(txns)
    print_report(result)
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict

import numpy as np
from sklearn.ensemble import IsolationForest


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Transaction:
    id: str
    actor: str                 # resolved entity name (person/account holder)
    amount: float
    timestamp: str              # ISO format, e.g. "2025-03-03T21:10:00"
    counterparty: Optional[str] = None
    channel: Optional[str] = None   # "bank", "upi", "cash", etc.


@dataclass
class Flag:
    txn_id: str
    actor: str
    flag_type: str              # "isolation_forest" or a specific rule name
    severity: float             # 0-1, higher = more suspicious
    reason: str                 # human-readable explanation for the UI


# ---------------------------------------------------------------------------
# Reporting-threshold constant used by the structuring rule below.
# India's cash transaction reporting threshold is commonly cited around
# ₹10,00,000 (Rs 10 lakh) for certain instruments; adjust to whatever
# threshold your problem statement specifies.
# ---------------------------------------------------------------------------
REPORTING_THRESHOLD = 1_000_000
STRUCTURING_MARGIN = 0.10       # "just under" = within 10% of the threshold


# ---------------------------------------------------------------------------
# Feature engineering for Isolation Forest
# ---------------------------------------------------------------------------
def build_feature_table(transactions: List[Transaction]) -> Dict[str, Any]:
    """
    Builds a per-actor feature matrix:
      - total transaction amount
      - transaction count
      - average amount
      - max single amount
      - std dev of amounts (irregularity)
      - count of off-hours transactions (11pm-5am)

    Returns the raw matrix plus the actor order, so results can be mapped
    back to actors after Isolation Forest scores them.
    """
    by_actor: Dict[str, List[Transaction]] = defaultdict(list)
    for t in transactions:
        by_actor[t.actor].append(t)

    actors = list(by_actor.keys())
    rows = []

    for actor in actors:
        txns = by_actor[actor]
        amounts = np.array([t.amount for t in txns], dtype=float)

        off_hours_count = 0
        for t in txns:
            try:
                hour = datetime.fromisoformat(t.timestamp).hour
                if hour >= 23 or hour < 5:
                    off_hours_count += 1
            except (ValueError, TypeError):
                pass  # skip malformed timestamps rather than crash

        rows.append([
            amounts.sum(),
            len(txns),
            amounts.mean(),
            amounts.max(),
            amounts.std() if len(txns) > 1 else 0.0,
            off_hours_count,
        ])

    return {"actors": actors, "by_actor": by_actor, "matrix": np.array(rows)}


# ---------------------------------------------------------------------------
# Layer 1: Isolation Forest
# ---------------------------------------------------------------------------
def run_isolation_forest(feature_table: Dict[str, Any], contamination: float = 0.1) -> List[Flag]:
    """
    Trains an Isolation Forest on the per-actor feature matrix and flags
    actors whose overall transaction behavior is an outlier.

    contamination: expected proportion of anomalies (0.1 = assume ~10% of
    actors look unusual). Tune this down for larger/cleaner datasets.
    """
    matrix = feature_table["matrix"]
    actors = feature_table["actors"]

    if len(actors) < 2:
        return []  # not enough data to find outliers relative to a baseline

    clf = IsolationForest(contamination=contamination, random_state=42)
    clf.fit(matrix)

    raw_scores = clf.decision_function(matrix)   # higher = more normal
    predictions = clf.predict(matrix)              # -1 = anomaly, 1 = normal

    flags = []
    for actor, score, pred in zip(actors, raw_scores, predictions):
        if pred == -1:
            # Normalize decision_function's score into a 0-1 severity,
            # where more-negative raw scores (more anomalous) -> higher severity.
            severity = min(1.0, max(0.0, -score + 0.5))
            flags.append(Flag(
                txn_id="(actor-level)",
                actor=actor,
                flag_type="isolation_forest",
                severity=round(severity, 2),
                reason="Overall transaction pattern (volume/frequency/timing) "
                       "statistically unusual compared to other actors in this dataset.",
            ))
    return flags


# ---------------------------------------------------------------------------
# Layer 2: Rule-based flags (explainable, no training)
# ---------------------------------------------------------------------------
def check_structuring(transactions: List[Transaction]) -> List[Flag]:
    """
    Flags transactions just under the reporting threshold — a classic
    money-laundering pattern ("structuring" / "smurfing").
    """
    lower_bound = REPORTING_THRESHOLD * (1 - STRUCTURING_MARGIN)
    flags = []
    for t in transactions:
        if lower_bound <= t.amount < REPORTING_THRESHOLD:
            flags.append(Flag(
                txn_id=t.id,
                actor=t.actor,
                flag_type="structuring",
                severity=0.8,
                reason=f"Amount ₹{t.amount:,.0f} is just under the ₹{REPORTING_THRESHOLD:,.0f} "
                       f"reporting threshold — possible structuring.",
            ))
    return flags


def check_repeated_structuring(transactions: List[Transaction]) -> List[Flag]:
    """
    Flags actors with MULTIPLE near-threshold transactions — a single one
    could be coincidence, three in a short window is a strong signal.
    """
    lower_bound = REPORTING_THRESHOLD * (1 - STRUCTURING_MARGIN)
    by_actor: Dict[str, List[Transaction]] = defaultdict(list)
    for t in transactions:
        if lower_bound <= t.amount < REPORTING_THRESHOLD:
            by_actor[t.actor].append(t)

    flags = []
    for actor, txns in by_actor.items():
        if len(txns) >= 3:
            flags.append(Flag(
                txn_id=", ".join(t.id for t in txns),
                actor=actor,
                flag_type="repeated_structuring",
                severity=0.95,
                reason=f"{len(txns)} separate near-threshold transactions — "
                       f"strong structuring pattern, not a one-off.",
            ))
    return flags


def check_off_hours(transactions: List[Transaction]) -> List[Flag]:
    """Flags individual transactions happening late night / early morning."""
    flags = []
    for t in transactions:
        try:
            hour = datetime.fromisoformat(t.timestamp).hour
        except (ValueError, TypeError):
            continue
        if hour >= 23 or hour < 5:
            flags.append(Flag(
                txn_id=t.id,
                actor=t.actor,
                flag_type="off_hours",
                severity=0.5,
                reason=f"Transaction at {hour:02d}:00 — outside normal banking hours.",
            ))
    return flags


def check_round_number(transactions: List[Transaction]) -> List[Flag]:
    """
    Flags suspiciously round large amounts (e.g. exactly ₹5,00,000) — often
    associated with informal/hawala-style transfers rather than genuine
    invoiced payments, which tend to have odd amounts (tax, fees, etc).
    """
    flags = []
    for t in transactions:
        if t.amount >= 100_000 and t.amount % 100_000 == 0:
            flags.append(Flag(
                txn_id=t.id,
                actor=t.actor,
                flag_type="round_number",
                severity=0.3,
                reason=f"₹{t.amount:,.0f} is a suspiciously round large amount.",
            ))
    return flags


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_anomaly_detection(transactions: List[Transaction], contamination: float = 0.1) -> Dict[str, Any]:
    feature_table = build_feature_table(transactions)

    all_flags = []
    all_flags += run_isolation_forest(feature_table, contamination=contamination)
    all_flags += check_structuring(transactions)
    all_flags += check_repeated_structuring(transactions)
    all_flags += check_off_hours(transactions)
    all_flags += check_round_number(transactions)

    # sort by severity, most suspicious first
    all_flags.sort(key=lambda f: f.severity, reverse=True)

    return {
        "flags": all_flags,
        "actor_count": len(feature_table["actors"]),
        "transaction_count": len(transactions),
    }


def print_report(result: Dict[str, Any]) -> None:
    print(f"Analyzed {result['transaction_count']} transactions across "
          f"{result['actor_count']} actors.\n")

    if not result["flags"]:
        print("No anomalies flagged.")
        return

    print(f"{len(result['flags'])} flags raised (sorted by severity):\n")
    for f in result["flags"]:
        print(f"  [{f.severity:.2f}] {f.flag_type:20s} actor={f.actor:15s} "
              f"txn={f.txn_id}\n         -> {f.reason}")


# ---------------------------------------------------------------------------
# Demo — extends your Rahul Sharma / Imran Khan FIR scenario with a
# synthetic transaction history so the flags have something to catch.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    transactions = [
        # Rahul: several near-threshold transfers to Mohd Arif -> structuring
        Transaction(id="t1", actor="Rahul Sharma", amount=950000,
                    timestamp="2025-02-20T14:00:00", counterparty="Mohd Arif"),
        Transaction(id="t2", actor="Rahul Sharma", amount=920000,
                    timestamp="2025-02-25T23:40:00", counterparty="Mohd Arif"),
        Transaction(id="t3", actor="Rahul Sharma", amount=980000,
                    timestamp="2025-03-01T02:15:00", counterparty="Mohd Arif"),
        Transaction(id="t4", actor="Rahul Sharma", amount=250000,
                    timestamp="2025-03-03T21:10:00", counterparty="Mohd Arif"),

        # Imran: one big round-number transfer
        Transaction(id="t5", actor="Imran Khan", amount=500000,
                    timestamp="2025-02-18T11:00:00", counterparty="Rafiq Sheikh"),

        # A handful of ordinary, unremarkable actors for the model to learn
        # a "normal" baseline from.
        Transaction(id="t6", actor="Suresh Yadav", amount=15000,
                    timestamp="2025-02-19T10:30:00"),
        Transaction(id="t7", actor="Priya Sharma", amount=22000,
                    timestamp="2025-02-21T16:00:00"),
        Transaction(id="t8", actor="Anil Verma", amount=8000,
                    timestamp="2025-02-22T09:15:00"),
        Transaction(id="t9", actor="Neha Gupta", amount=30000,
                    timestamp="2025-02-23T13:45:00"),
        Transaction(id="t10", actor="Sanjay Rao", amount=12500,
                    timestamp="2025-02-24T18:20:00"),
    ]

    result = run_anomaly_detection(transactions)
    print_report(result)
