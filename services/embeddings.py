# services/embeddings.py
"""
from sentence_transformers import SentenceTransformer

# Better model for retrieval
model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")
VECTOR_DIM = model.get_sentence_embedding_dimension()

embedding_cache = {}

async def embed_text_async(text: str) -> list[float]:
    if text in embedding_cache:
        return embedding_cache[text]
    loop = asyncio.get_event_loop()
    vector = await loop.run_in_executor(None, lambda: model.encode(text, normalize_embeddings=True).tolist())
    embedding_cache[text] = vector
    return vector
"""
import asyncio
from sentence_transformers import SentenceTransformer
from functools import lru_cache

# Load once
model = SentenceTransformer("intfloat/e5-small-v2")
VECTOR_DIM = model.get_sentence_embedding_dimension()

@lru_cache(maxsize=10000)
def embed_text_cached(text: str) -> list[float]:
    return model.encode(text, normalize_embeddings=True).tolist()

async def embed_text_async(text: str) -> list[float]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: embed_text_cached(text))

