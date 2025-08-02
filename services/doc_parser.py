import requests
import fitz  # PyMuPDF
import re
from utils.chunker import chunk_text_by_tokens

def process_document(blob_url: str):
    resp = requests.get(blob_url)
    pdf = fitz.open(stream=resp.content, filetype="pdf")
    doc_text = ""
    for page in pdf:
        doc_text += page.get_text()
    meta = {"num_pages": pdf.page_count}
    return doc_text, meta

def split_into_clauses(text: str):
    """
    Split text by legal style headings (e.g., "Section 1", "Section 1.1 ...").
    Returns list of {"section": str, "text": str}.
    """
    sections = re.split(r'(Section\s+\d+[\.\d]*\s*[\w\s]*)', text)
    return [{"section": s.strip(), "text": t.strip()} for s, t in zip(sections[1::2], sections[2::2])]

def split_clauses_into_token_chunks(clauses: list[dict], max_tokens: int = 2048):
    """
    For each clause dict, split text further into token-limited chunks.
    Return list of dicts: [{"section": ..., "text": ...}]
    """
    chunked_clauses = []
    for clause in clauses:
        if not isinstance(clause, dict) or "text" not in clause or "section" not in clause:
            raise ValueError(f"Invalid clause dict: {clause}")
        text_chunks = chunk_text_by_tokens(clause["text"], max_tokens=max_tokens)
        for i, chunk in enumerate(text_chunks):
            chunked_clauses.append({
                "section": f"{clause['section']} (Part {i+1})" if len(text_chunks) > 1 else clause['section'],
                "text": chunk
            })
    return chunked_clauses
