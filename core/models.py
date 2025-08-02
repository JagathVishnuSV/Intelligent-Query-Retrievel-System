from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    documents: str            # URL or path for uploaded document
    questions: List[str]

class ClauseInfo(BaseModel):
    section: str
    text: str
    page: Optional[int]

class AnswerDetail(BaseModel):
    answer: str
    clauses: List[ClauseInfo]
    explanation: str

class QueryResponse(BaseModel):
    answers: List[AnswerDetail]
