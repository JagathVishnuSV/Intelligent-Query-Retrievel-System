from services.embeddings import embed_text_async
from services.vector_store import QdrantIndexer, QdrantRetriever, doc_hash
from services.bm25_retriever import BM25Retriever
from services.llm_service import gemini_invoke_with_retry
from services.explain import make_explanation
from services.reranker import rerank_by_cosine_similarity
from utils.chunker import chunk_text_by_tokens
from typing import List
from collections import defaultdict

bm25_retriever = BM25Retriever()

def compose_prompt_multi(questions: List[str], contexts: List[List[str]]) -> str:
    prompt = (
        "You are a helpful assistant answering questions based strictly on the given excerpts from an insurance policy document.\n"
        "If the answer is clearly found, provide it concisely.\n"
        "If the answer is not present, say exactly: 'Not mentioned in the policy document.'\n\n"
    )
    for i, (q, ctx_chunks) in enumerate(zip(questions, contexts), start=1):
        combined_context = "\n\n".join(ctx_chunks)
        prompt += (
            f"Question {i}: {q}\nContext:\n{combined_context}\n"
            "Answer:\n"
        )
    prompt += "\nProvide each answer starting with 'Answer 1:', 'Answer 2:', etc."
    return prompt

async def answer_query(query: str, clauses: list[dict], document_url: str, top_k=5, rerank_llm=True):
    doc_id = doc_hash(document_url)

    chunked_clauses = []
    for clause in clauses:
        chunks = chunk_text_by_tokens(clause["text"], max_tokens=1000, overlap=100)
        for idx, chunk in enumerate(chunks):
            chunked_clauses.append({
                "section": f"{clause['section']} (Part {idx+1})" if len(chunks) > 1 else clause['section'],
                "text": chunk
            })

    chunk_vectors = [await embed_text_async(c["text"]) for c in chunked_clauses]

    indexer = QdrantIndexer(doc_id)
    await indexer.upsert_vectors(chunk_vectors, chunked_clauses)

    retriever = QdrantRetriever(doc_id)
    query_vec = await embed_text_async(query)
    qdrant_hits = await retriever.search(query_vec, top_k=top_k*2)
    qdrant_clauses = [r.payload for r in qdrant_hits]

    bm25_retriever.index(chunked_clauses)
    bm25_clauses = bm25_retriever.search(query, top_k=top_k*2)

    # Hybrid scoring approach
    #combined_scores = defaultdict(float)
    #qdrant_dict = {hit.payload["text"]: hit for hit in qdrant_hits}

    scored = {}
    for clause in qdrant_clauses:
        scored[clause["text"]] = {"score": 0.6, "data": clause}
    for clause in bm25_clauses:
        if clause["text"] in scored:
            scored[clause["text"]]["score"] += 0.4
        else:
            scored[clause["text"]] = {"score": 0.4, "data": clause}

    all_candidates = sorted(scored.values(), key=lambda x: -x["score"])
    all_candidates = [x["data"] for x in all_candidates]

    reranked = await rerank_by_cosine_similarity(query, all_candidates)

    top_clauses = reranked[:10]
    if not top_clauses:
        # Fallback mechanism
        top_clauses = all_candidates[:10]
        if not top_clauses:
            return make_explanation("No relevant information was found in the policy document.", [], "")

    rationale = "\n".join([f"{c['section']}: {c['text']}" for c in top_clauses])
    prompt = f"""You are an expert insurance policy analyst. Answer the following question using ONLY the information provided in the clauses below.

Question: {query}

Relevant Policy Clauses:
{rationale}

Instructions:
1. If the answer is clearly found in the clauses, provide it concisely
2. If the information is partially available, state what is available
3. If the answer is not present at all, say exactly: "Not mentioned in the policy document."
4. Do not make up information not in the clauses

Answer:"""

    answer = await gemini_invoke_with_retry(prompt)
    return make_explanation(answer.strip(), top_clauses, rationale)
