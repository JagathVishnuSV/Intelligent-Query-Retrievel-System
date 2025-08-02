import asyncio
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer('all-MiniLM-L6-v2')  # or your choice

async def embed_text_async(text: str) -> list[float]:
    loop = asyncio.get_event_loop()
    vector = await loop.run_in_executor(None, lambda: _model.encode(text).tolist())
    return vector
