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
    Tries to split text by legal-style headings like "Section 1", "Section 1.1 ...".
    Falls back to paragraph-level splitting if no such headers are found.
    Returns list of {"section": str, "text": str}.
    """
    # Try multiple patterns for section headers
    patterns = [
        r'(Section\s+\d+[\.\d]*\s*[\w\s]*)',
        r'(Clause\s+\d+[\.\d]*\s*[\w\s]*)',
        r'([A-Z][\w\s]*:\s*)',  # Capitalized headers followed by colon
        r'(\d+\.\s*[A-Z][\w\s]*)'  # Numbered sections
    ]
    
    for pattern in patterns:
        sections = re.split(pattern, text)
        if len(sections) > 3:  # Found meaningful sections
            clauses = [{"section": s.strip(), "text": t.strip()} for s, t in zip(sections[1::2], sections[2::2])]
            if clauses:
                filtered_clauses = [c for c in clauses if c["text"]]  # filter empty
                if filtered_clauses:
                    return filtered_clauses
    
    # 🔁 Fallback to paragraph-level chunks if no sections found
    paragraphs = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 50]
    fallback_clauses = [{"section": f"Paragraph {i+1}", "text": p} for i, p in enumerate(paragraphs)]
    
    # Further fallback to sentence-level if paragraphs are too long
    if not fallback_clauses or any(len(c["text"]) > 2000 for c in fallback_clauses):
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        fallback_clauses = [{"section": f"Sentence {i+1}", "text": s} for i, s in enumerate(sentences) if len(s) > 20]
    
    # Final fallback if even that fails
    if not fallback_clauses:
        return [{"section": "Entire Document", "text": text.strip()}]
    
    return fallback_clauses

def chunk_text_by_tokens(text: str, max_tokens: int = 1000, overlap: int = 100) -> list[str]:
    """
    Splits text into token-based chunks with optional overlap.
    """
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
