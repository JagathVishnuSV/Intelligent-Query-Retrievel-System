from fastapi import APIRouter, HTTPException, Header, Depends
from pydantic import BaseModel
from typing import List, Optional, Tuple
import asyncio
import os
import logging
from dotenv import load_dotenv
load_dotenv()
from google.api_core.exceptions import ResourceExhausted

from services import doc_parser, embeddings, llm_service, vector_store

router = APIRouter()

API_KEY = "714c3fdb7fd84d510e3b5d4a0e21cc85a9a323700c63fad79fcd234ea93b99d5"

def verify_token(authorization: Optional[str] = Header(None)) -> str:
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization.removeprefix("Bearer ").strip()
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return token

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

# Simple in-memory cache (replace with Redis or database in production)
embedding_cache = {}
answer_cache = {}

def compose_prompt_multi(questions: List[str], contexts: List[List[str]]) -> str:
    prompt = (
        "You are a helpful assistant answering questions based on the following insurance policy excerpts.\n\n"
    )
    for i, (q, ctx_chunks) in enumerate(zip(questions, contexts), start=1):
        combined_context = "\n\n".join(ctx_chunks)
        prompt += (
            f"Question {i}: {q}\nContext:\n{combined_context}\n"
            "Answer concisely in plain English. If the answer is not found, say 'Information not available in the policy.'\n\n"
        )
    prompt += "Provide answers in order, separated clearly by 'Answer 1:', 'Answer 2:', etc."
    return prompt

@router.post("/hackrx/run", response_model=QueryResponse)
async def run_hackrx(request: QueryRequest, token: str = Depends(verify_token)):
    logging.info("Started /hackrx/run request")

    # 1. Download and parse document
    doc_text, meta = await doc_parser.process_document(request.documents)
    if not doc_text.strip():
        raise HTTPException(status_code=400, detail="Extracted document text is empty.")

    # 2. Split into clauses
    clauses = doc_parser.split_into_clauses(doc_text)
    if not clauses:
        raise HTTPException(status_code=400, detail="No clauses found in document.")

    # 3. Token chunk clauses with overlap
    max_tokens = 2048
    overlap = 50
    chunked_clauses = []
    for clause in clauses:
        chunks = doc_parser.chunk_text_by_tokens(clause["text"], max_tokens=max_tokens, overlap=overlap)
        for idx, chunk in enumerate(chunks):
            section_name = f"{clause['section']} (Part {idx+1})" if len(chunks) > 1 else clause['section']
            chunked_clauses.append({"section": section_name, "text": chunk})
    if not chunked_clauses:
        raise HTTPException(status_code=400, detail="No chunks generated from document.")

    # 4. Embed chunks with caching
    chunk_texts = [c["text"] for c in chunked_clauses]
    chunk_vectors = []
    for text in chunk_texts:
        if text in embedding_cache:
            vector = embedding_cache[text]
        else:
            vector = await embeddings.embed_text_async(text)
            embedding_cache[text] = vector
        chunk_vectors.append(vector)

    # 5. Index chunks with Qdrant
    await vector_store.indexer.upsert_vectors(chunk_vectors, chunked_clauses)

    # 6. For each question: embed, retrieve top_k chunks
    top_k = 5
    contexts_per_question = []
    for question in request.questions:
        if question in embedding_cache:
            q_vector = embedding_cache[question]
        else:
            q_vector = await embeddings.embed_text_async(question)
            embedding_cache[question] = q_vector
        results = await vector_store.retriever.search(q_vector, top_k=top_k)
        contexts_per_question.append([hit.payload['text'] for hit in results])

    # 7. Check answer cache to avoid duplicate LLM calls
    cache_key = (request.documents, tuple(request.questions))
    if cache_key in answer_cache:
        logging.info("Returning cached answers")
        return QueryResponse(answers=answer_cache[cache_key])

    # 8. Compose batch prompt for questions + contexts
    prompt = compose_prompt_multi(request.questions, contexts_per_question)

    # 9. Call Gemini with retry/backoff
    answer_text = await llm_service.gemini_invoke_with_retry(prompt)

    # 10. Parse answers
    parsed_answers = []
    for i in range(len(request.questions)):
        marker = f"Answer {i+1}:"
        start = answer_text.find(marker)
        if start == -1:
            # Fallback: return full text repeated if markers missing
            parsed_answers = [answer_text.strip()] * len(request.questions)
            break
        end = answer_text.find(f"Answer {i+2}:", start + 1)
        ans = answer_text[start + len(marker):end].strip() if end != -1 else answer_text[start + len(marker):].strip()
        parsed_answers.append(ans)

    # 11. Cache answers
    answer_cache[cache_key] = parsed_answers

    logging.info("Completed /hackrx/run request")
    return QueryResponse(answers=parsed_answers)
