"""
pdf_ingest.py

Entry point of the FINAL pipeline: takes a PDF the user uploads (an FIR,
a case file, a financial statement, etc.) and returns clean text ready
to feed into extraction.py (NER + relation extraction).

Two-tier extraction:
  1. pdfplumber — works for normal digitally-generated/typed PDFs (most FIRs
     typed into a system will be this).
  2. OCR fallback (pytesseract + pdf2image) — kicks in automatically if
     pdfplumber gets little/no text back, which happens with SCANNED PDFs
     (a photographed or scanned physical FIR, common in real police records).

This matters because a hackathon demo where the "upload PDF" feature
silently returns an empty string on a scanned document looks broken —
this module tries to fail informatively instead.
"""

from typing import Optional
import re

try:
    import pdfplumber
    HAVE_PDFPLUMBER = True
except ImportError:
    HAVE_PDFPLUMBER = False


MIN_CHARS_PER_PAGE_BEFORE_OCR = 20  # below this, assume the page is scanned/image-only


def extract_text_pdfplumber(pdf_path: str) -> str:
    """Primary path: extracts text from a normal (non-scanned) PDF."""
    if not HAVE_PDFPLUMBER:
        raise ImportError("pdfplumber not installed — run: pip install pdfplumber")

    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages_text.append(text)
    return "\n\n".join(pages_text)


def extract_text_ocr(pdf_path: str) -> str:
    """
    Fallback path for scanned/image PDFs. Requires:
        pip install pytesseract pdf2image
        + poppler and tesseract installed on the system
    """
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError:
        raise ImportError(
            "OCR fallback needs: pip install pytesseract pdf2image "
            "(plus poppler-utils and tesseract-ocr installed on the system)"
        )

    images = convert_from_path(pdf_path)
    pages_text = []
    for image in images:
        pages_text.append(pytesseract.image_to_string(image))
    return "\n\n".join(pages_text)


def clean_text(text: str) -> str:
    """Light cleanup — collapses excess whitespace without mangling sentence structure."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf_text(pdf_path: str, force_ocr: bool = False) -> str:
    """
    Main entry point. Tries pdfplumber first; automatically falls back to
    OCR if the result looks suspiciously empty (a strong sign the PDF is
    scanned images rather than real text) or if force_ocr=True.
    """
    if not force_ocr:
        try:
            text = extract_text_pdfplumber(pdf_path)
        except ImportError:
            text = ""
    else:
        text = ""

    # Rough heuristic: fewer than MIN_CHARS_PER_PAGE_BEFORE_OCR non-whitespace
    # chars in the whole doc is a strong signal this was a scanned PDF.
    if force_ocr or len(text.strip()) < MIN_CHARS_PER_PAGE_BEFORE_OCR:
        try:
            ocr_text = extract_text_ocr(pdf_path)
            if len(ocr_text.strip()) > len(text.strip()):
                text = ocr_text
        except ImportError as e:
            if not text.strip():
                # Nothing worked at all — surface a clear error rather than
                # silently returning an empty string into the rest of the pipeline.
                raise RuntimeError(
                    f"Could not extract any text from {pdf_path}. "
                    f"It may be a scanned document; OCR fallback unavailable ({e})."
                )

    return clean_text(text)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pdf_ingest.py <path_to_pdf>")
    else:
        text = load_pdf_text(sys.argv[1])
        print(f"Extracted {len(text)} characters:\n")
        print(text[:2000])
