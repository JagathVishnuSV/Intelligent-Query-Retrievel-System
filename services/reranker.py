from sentence_transformers import CrossEncoder
from services.llm_service import gemini_invoke_with_retry

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_crossencoder(query: str, candidates: list[dict]) -> list[dict]:
    pairs = [(query, c['text']) for c in candidates]
    scores = reranker.predict(pairs)
    return [c for _, c in sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)]

async def rerank_with_llm(query: str, candidates: list[dict]) -> list[dict]:
    numbered = "\n\n".join([f"Clause {i+1}:\n{c['text']}" for i, c in enumerate(candidates)])
    prompt = f"""You are an expert insurance policy analyst. Rank the following clauses based on their relevance to the question.
Question: "{query}"

Clauses:
{numbered}

Instructions:
1. Score each clause from 1-10 based on relevance (10 being most relevant)
2. Return ONLY a comma-separated list of scores in order
3. Example: 8,3,10,1,5

Scores:"""

    result = await gemini_invoke_with_retry(prompt)
    try:
        scores = [int(x.strip()) for x in result.strip().split(",") if x.strip().isdigit()]
        # Combine scores with original candidates
        scored_candidates = list(zip(scores, candidates))
        # Sort by scores in descending order
        sorted_candidates = sorted(scored_candidates, key=lambda x: x[0], reverse=True)
        return [candidate for score, candidate in sorted_candidates]
    except:
        # Fallback to original order if parsing fails
        return candidates
