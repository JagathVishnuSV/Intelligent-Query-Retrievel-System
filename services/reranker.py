from sentence_transformers.util import cos_sim
from services.embeddings import embed_text_async  # same async embedder used elsewhere
import asyncio


async def rerank_by_cosine_similarity(query: str, candidates: list[dict]) -> list[dict]:
    query_vec = await embed_text_async(query)
    clause_vecs = await asyncio.gather(*[embed_text_async(c["text"]) for c in candidates])
    
    sims = cos_sim(query_vec, clause_vecs)[0]
    scored_candidates = list(zip(sims, candidates))
    sorted_candidates = sorted(scored_candidates, key=lambda x: x[0], reverse=True)
    
    return [candidate for _, candidate in sorted_candidates]
