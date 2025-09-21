from fastapi import APIRouter, HTTPException, Header, Depends, UploadFile, File, Form
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
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

@router.post("/hackrx/run", response_model=QueryResponse)
async def run_hackrx(
    questions: str = Form(...),
    document_url: Optional[str] = Form(None),
    document_file: Optional[UploadFile] = File(None),
    token: str = Depends(verify_token)
):
    if not document_url and not document_file:
        raise HTTPException(status_code=400, detail="Either a document URL or a file must be provided.")
    if document_url and document_file:
        raise HTTPException(status_code=400, detail="Provide either a document URL or a file, not both.")

    doc_text = ""
    source_identifier = ""

    if document_url:
        doc_text, _ = await doc_parser.process_document_from_url(document_url)
        source_identifier = document_url
    elif document_file:
        content = await document_file.read()
        doc_text, _ = await doc_parser.process_document_from_content(content)
        source_identifier = document_file.filename

    if not doc_text.strip():
        raise HTTPException(status_code=400, detail="Document text is empty or could not be processed.")
    
    clauses = doc_parser.split_into_clauses(doc_text)
    if not clauses:
        raise HTTPException(status_code=400, detail="No clauses found in document.")

    try:
        question_list = questions.splitlines()
        results = await asyncio.gather(*[
            answer_query(q, clauses, source_identifier) for q in question_list
        ])
        answers = [r.answer.strip() for r in results]
    except Exception:
        answers = ["Error: Failed to retrieve answer."] * len(question_list)

    return QueryResponse(answers=answers)
