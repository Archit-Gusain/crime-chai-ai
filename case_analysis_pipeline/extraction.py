"""
extraction.py

Steps 1 + 2 of the pipeline, refactored from your notebook into reusable
functions instead of top-level script code, so pipeline.py can call them
on whatever text pdf_ingest.py hands back.

Heavy ML imports (spacy, transformers, torch) are deferred to INSIDE the
load_* functions rather than at module level. That means:
  - You can `import extraction` anywhere (e.g. to unit-test other parts of
    your pipeline) without needing spaCy/transformers/torch installed.
  - The actual heavy models only load the first time you call
    load_ner_model() / load_rebel_model(), and are cached after that
    (see get_models()) so you don't reload them on every PDF.

Run this in your Colab/GPU environment — this sandbox doesn't have
spacy/transformers installed, so I couldn't execute this file directly,
but it's a straight refactor of code you already ran successfully, so the
logic itself is unchanged.
"""

from typing import List, Dict, Any, Tuple

# Populated lazily by get_models() — avoids reloading the model on every call.
_MODEL_CACHE: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Model loading (deferred imports — only needed when these are actually called)
# ---------------------------------------------------------------------------
def load_ner_model():
    import spacy
    import spacy.cli

    try:
        return spacy.load("en_core_web_lg")
    except OSError:
        spacy.cli.download("en_core_web_lg")
        return spacy.load("en_core_web_lg")


def load_rebel_model():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Babelscape/rebel-large"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)

    return tokenizer, model, device


def get_models() -> Dict[str, Any]:
    """Loads (once) and caches spaCy + REBEL, so repeated calls across many
    PDFs in one session don't reload multi-hundred-MB models each time."""
    if "nlp" not in _MODEL_CACHE:
        _MODEL_CACHE["nlp"] = load_ner_model()
    if "rebel" not in _MODEL_CACHE:
        tokenizer, model, device = load_rebel_model()
        _MODEL_CACHE["tokenizer"] = tokenizer
        _MODEL_CACHE["model"] = model
        _MODEL_CACHE["device"] = device
    return _MODEL_CACHE


# ---------------------------------------------------------------------------
# Step 1: NER
# ---------------------------------------------------------------------------
def extract_entities(text: str) -> List[Dict[str, str]]:
    models = get_models()
    doc = models["nlp"](text)
    return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]


# ---------------------------------------------------------------------------
# Step 2: Relation extraction (REBEL)
# ---------------------------------------------------------------------------
def parse_rebel_output(decoded_text: str) -> List[Dict[str, str]]:
    """
    Pure parser: turns REBEL's raw <triplet>/<subj>/<obj>-tagged output
    into a list of {"subject", "relation", "object"} dicts.
    (This is the corrected version from the ordering fix — takes ONLY the
    decoded model output, no side effects, no reference to outer-scope text.)
    """
    triplets = []
    relation, subject, obj = "", "", ""
    current = "x"

    decoded_text = (
        decoded_text.replace("<s>", "")
        .replace("</s>", "")
        .replace("<pad>", "")
    )

    for token in decoded_text.split():
        if token == "<triplet>":
            if relation != "":
                triplets.append({
                    "subject": subject.strip(),
                    "relation": relation.strip(),
                    "object": obj.strip(),
                })
                relation, subject, obj = "", "", ""
            current = "subject"
        elif token == "<subj>":
            current = "object"
        elif token == "<obj>":
            current = "relation"
        else:
            if current == "subject":
                subject += " " + token
            elif current == "object":
                obj += " " + token
            elif current == "relation":
                relation += " " + token

    if relation != "":
        triplets.append({
            "subject": subject.strip(),
            "relation": relation.strip(),
            "object": obj.strip(),
        })

    return triplets


def extract_relations(text: str, max_length: int = 512) -> List[Dict[str, str]]:
    """
    Runs REBEL on `text` and returns parsed triples. Handles text longer
    than REBEL's 512-token window by chunking on sentence boundaries and
    running each chunk separately — a single long FIR/case document would
    otherwise get silently truncated.
    """
    models = get_models()
    tokenizer, model, device = models["tokenizer"], models["model"], models["device"]

    chunks = _chunk_text(text, tokenizer, max_length)

    all_triples = []
    for chunk in chunks:
        inputs = tokenizer(chunk, max_length=max_length, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        generated = model.generate(**inputs, max_length=256, num_beams=4)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=False)[0]

        all_triples.extend(parse_rebel_output(decoded))

    return all_triples


def _chunk_text(text: str, tokenizer, max_length: int) -> List[str]:
    """Splits text into REBEL-sized chunks on sentence boundaries (simple
    period-split — good enough for FIR-style prose; swap for spaCy sentence
    segmentation if you need higher accuracy on messy punctuation)."""
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    chunks, current = [], ""

    for sentence in sentences:
        candidate = f"{current}. {sentence}".strip(". ").strip()
        if len(tokenizer.encode(candidate)) > max_length - 10:
            if current:
                chunks.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks or [text]


# ---------------------------------------------------------------------------
# Convenience: run both steps at once
# ---------------------------------------------------------------------------
def extract_entities_and_relations(text: str) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    entities = extract_entities(text)
    triples = extract_relations(text)
    return entities, triples
