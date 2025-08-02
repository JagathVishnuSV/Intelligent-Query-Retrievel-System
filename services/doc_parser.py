import requests
import fitz  # PyMuPDF
import re
import asyncio

import tiktoken

async def process_document(blob_url: str) -> tuple[str, dict]:
    """Download PDF from URL and extract full text asynchronously."""
    loop = asyncio.get_event_loop()
    resp = await loop.run_in_executor(None, lambda: requests.get(blob_url))
    resp.raise_for_status()
    pdf = fitz.open(stream=resp.content, filetype="pdf")
    doc_text = ""
    for page in pdf:
        doc_text += page.get_text()
    meta = {"num_pages": pdf.page_count}
    return doc_text, meta

def split_into_clauses(text: str) -> list[dict]:
    """
    Split text by legal style headings (e.g., "Section 1", "Section 1.1 ...").
    Returns list of {"section": str, "text": str}.
    """
    sections = re.split(r'(Section\s+\d+[\.\d]*\s*[\w\s]*)', text)
    clauses = [{"section": s.strip(), "text": t.strip()} for s, t in zip(sections[1::2], sections[2::2])]
    if not clauses:
        # Fallback if no section headers found
        return [{"section": "Entire Document", "text": text.strip()}]
    return clauses

def chunk_text_by_tokens(text: str, max_tokens: int = 2048, overlap: int = 50) -> list[str]:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []
    start = 0
    length = len(tokens)
    while start < length:
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += max_tokens - overlap
    return chunks
