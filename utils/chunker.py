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
