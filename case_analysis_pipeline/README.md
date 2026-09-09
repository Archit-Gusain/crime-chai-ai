# Case Analysis Pipeline

Turns an uploaded case PDF (FIR, case file, etc.) into a queryable
knowledge graph, with AI-generated investigative reasoning on top.

```
PDF upload
    -> pdf_ingest.py        text extraction (+ OCR fallback for scans)
    -> extraction.py        NER (spaCy) + relation extraction (REBEL)
    -> entity_resolution.py fuzzy dedup of entity mentions
    -> graph_analytics.py   graph build + centrality + Louvain communities
    -> anomaly_detection.py (optional) Isolation Forest + rule-based flags
    -> nl_query.py          NL question -> graph facts -> AI reasoning
```

All orchestrated by `pipeline.py`.

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

For the AI reasoning layer in `nl_query.py` (optional but recommended):
```bash
export ANTHROPIC_API_KEY=your_key_here
```
Without a key, the pipeline still runs end-to-end — the query layer falls
back to keyword-based question parsing and returns raw graph facts without
the AI-generated explanation.

For OCR support on scanned PDFs (optional):
```bash
pip install pytesseract pdf2image
# + system packages: tesseract-ocr, poppler-utils
```

## Run it

```bash
python pipeline.py path/to/case.pdf
```

Prints extraction stats, the full centrality/community report, then drops
into an interactive prompt:

```
> who are the top 3 most connected people?
> how is Rahul Sharma connected to Rafiq Sheikh?
> tell me about Mohd Arif
```

Or drive it from your own code/notebook:

```python
from pipeline import run_pipeline

result = run_pipeline("case.pdf")
result["ask"]("why is Rahul Sharma the most central figure in this case?")

# result also contains: text, entities, triples, graph, report
```

### With transaction data (anomaly detection)

Anomaly detection is opt-in since it needs structured transaction records
(amount + timestamp + actor), which usually come from bank data rather
than the FIR PDF text itself:

```python
from anomaly_detection import Transaction

txns = [
    Transaction(id="t1", actor="Rahul Sharma", amount=950000,
                timestamp="2025-02-20T14:00:00"),
    # ...
]
result = run_pipeline("case.pdf", run_anomaly=True, transactions=txns)
print(result["anomaly_report"])
```

## File-by-file

| File | Purpose |
|---|---|
| `pipeline.py` | Master orchestrator — run this |
| `pdf_ingest.py` | PDF -> text, with OCR fallback for scanned documents |
| `extraction.py` | spaCy NER + REBEL relation extraction (refactored from notebook) |
| `entity_resolution.py` | Fuzzy dedup of entity mentions + merges triples onto canonical names |
| `graph_analytics.py` | Builds the NetworkX case graph, runs centrality + Louvain communities |
| `anomaly_detection.py` | Isolation Forest + explainable rule-based transaction flags |
| `nl_query.py` | NL question -> safe filter spec -> graph facts -> AI reasoning |

## Known limitations / next steps

- **Fuzzy matching accuracy** depends on `rapidfuzz` being installed —
  without it, `entity_resolution.py` falls back to Python's built-in
  `difflib`, which is noticeably weaker on partial-name matches (e.g.
  "Rahul Sharma" vs "Rahul S." may not merge). Always install `rapidfuzz`
  for real runs.
- **REBEL's 512-token window**: `extraction.py` chunks long documents on
  sentence boundaries before running REBEL, but very long/dense case files
  may still benefit from a smarter chunking strategy (e.g. paragraph-aware).
- **The AI reasoning layer never sees the graph directly** — it only
  receives pre-computed facts from `execute_spec()`, to prevent it from
  inventing connections that aren't actually in your data. Keep this
  separation if you extend `nl_query.py`.
- **Anomaly detection thresholds** (`REPORTING_THRESHOLD` in
  `anomaly_detection.py`) are placeholders — set them to whatever your
  problem statement specifies.
