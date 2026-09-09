import spacy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import spacy.cli

spacy.cli.download("en_core_web_lg")

# -------------------- Device (must be defined BEFORE anything uses it) --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------- Load spaCy --------------------
nlp = spacy.load("en_core_web_lg")

# -------------------- Load REBEL --------------------
MODEL_NAME = "Babelscape/rebel-large"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.to(device)


# -------------------- Pure parser: text -> triplets --------------------
def extract_triplets(decoded_text):
    """
    Parses REBEL's raw decoded output (with <triplet>/<subj>/<obj> markers)
    into a list of {"subject", "relation", "object"} dicts.

    Takes ONLY the decoded model output as input — does not call the model
    itself and does not touch the original FIR text. This was the bug in
    the original version: `text` was being reassigned to the FIR string
    before this function ran, and REBEL's <subj>/<obj> tag order was
    inverted relative to how the fields were being filled in.
    """
    triplets = []

    relation = ""
    subject = ""
    obj = ""
    current = "x"

    decoded_text = (
        decoded_text.replace("<s>", "")
        .replace("</s>", "")
        .replace("<pad>", "")
    )

    tokens = decoded_text.split()

    for token in tokens:
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


# -------------------- Run on the FIR text --------------------
fir_text = """
Rahul Sharma met Imran Khan near Connaught Place, Delhi.
Later Rahul transferred ₹2,50,000 to Mohd Arif through SBI Bank.
Police suspect Imran contacted Rafiq Sheikh before the robbery.
"""

# ----------- spaCy Entity Extraction -----------
doc = nlp(fir_text)

entities = []
for ent in doc.ents:
    entities.append({
        "text": ent.text,
        "label": ent.label_,
    })

print("\nEntities")
print(entities)

# ----------- REBEL Relationship Extraction -----------
inputs = tokenizer(
    fir_text,
    max_length=512,
    truncation=True,
    return_tensors="pt",
)
inputs = {k: v.to(device) for k, v in inputs.items()}

generated = model.generate(
    **inputs,
    max_length=256,
    num_beams=4,
)

decoded = tokenizer.batch_decode(
    generated,
    skip_special_tokens=False,
)[0]

# Parse triples using extract_triplets() — now a pure parser, called once
triples = extract_triplets(decoded)

print("\nRelationships")
for t in triples:
    print(t)
