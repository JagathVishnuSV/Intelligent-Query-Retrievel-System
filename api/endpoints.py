from fastapi import APIRouter, HTTPException, Header, Depends
from pydantic import BaseModel
from typing import List, Optional
import asyncio

from services import doc_parser
from services.logic import answer_query

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

@router.post("/hackrx/run", response_model=QueryResponse)
async def run_hackrx(request: QueryRequest, token: str = Depends(verify_token)):
    # Step 1: Download and parse
    doc_text, _ = await doc_parser.process_document(request.documents)
    if not doc_text.strip():
        raise HTTPException(status_code=400, detail="Document text is empty.")
    
    # Step 2: Split into clauses
    clauses = doc_parser.split_into_clauses(doc_text)
    if not clauses:
        raise HTTPException(status_code=400, detail="No clauses found in document.")

    # Step 3: Answer each question concurrently using asyncio.gather
    try:
        results = await asyncio.gather(*[
            answer_query(q, clauses, request.documents) for q in request.questions
        ])
        answers = [r.answer.strip() for r in results]
    except Exception:
        answers = ["Error: Failed to retrieve answer."] * len(request.questions)

    return QueryResponse(answers=answers)
