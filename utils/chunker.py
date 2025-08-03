import tiktoken

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
        start += max_tokens - overlap  # move forward by chunk size minus overlap

    return chunks

def semantic_chunk_text(text: str, max_chunk_size: int = 1000) -> list[str]:
    # Split by paragraphs first
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) < max_chunk_size:
            current_chunk += paragraph + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # If no chunks were created, return the original text as a single chunk
    if not chunks:
        return [text]
    
    return chunks
