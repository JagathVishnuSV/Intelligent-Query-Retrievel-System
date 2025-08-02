from fastapi import APIRouter,Request
from core.models import QueryRequest, QueryResponse, AnswerDetail, ClauseInfo
from services import doc_parser, retriever, llm_service, embeddings, explain
from services.llm_service import aggregate_answers
from fastapi.responses import JSONResponse

router = APIRouter()

@router.post("/webhook/run_hackrx")
async def webhook_run_hackrx(request: Request):
    data = await request.json()
    
    # Extract expected data (document url and questions) from the webhook payload
    documents = data.get("documents")
    questions = data.get("questions")

    if not documents or not questions:
        return JSONResponse(status_code=400, content={"error": "Missing documents or questions in payload"})

    # Call your existing function or method that runs your pipeline
    # For example, assume you have a synchronous call 'run_query_logic' you want to wrap
    from api.endpoints import run_query
    from core.models import QueryRequest
    
    # Prepare the request model
    req = QueryRequest(documents=documents, questions=questions)
    
    # If run_query is synchronous (not async), run in threadpool or make it async
    response = run_query(req)

    # Return the result as JSON
    return response

def compose_prompt(question: str, retrieved_chunks: list[dict]):
    context = "\n\n".join(f"{c['section']}: {c['text']}" for c in retrieved_chunks)
    prompt = (
        f"Using the following excerpts from the insurance policy, answer clearly and concisely:\n"
        f"{context}\n\n"
        f"Question: {question}\n"
        f"Answer in a concise summary in plain English. Do not copy verbatim except key terms. "
        f"If the answer is not found, state 'Information not available in the policy.'"
    )
    return prompt


@router.post("/hackrx/run", response_model=QueryResponse)
def run_query(request: QueryRequest):
    # Step 1: Download and parse document
    doc_text, _ = doc_parser.process_document(request.documents)
    
    # Step 2: Semantic clauses splitting
    clauses = doc_parser.split_into_clauses(doc_text)
    
    # Step 3: Token-aware chunking of clauses
    chunked_clauses = doc_parser.split_clauses_into_token_chunks(clauses, max_tokens=2048)
    
    # Step 4: Extract texts for embedding
    chunk_texts = [c["text"] for c in chunked_clauses]
    
    # Step 5: Batch embed with cache and delay, tune batch_size/delay as per quota
    chunk_vectors = embeddings.embed_text_list(chunk_texts)
    
    #chunk_vectors = batch_embed_texts(chunk_texts, batch_size=20, delay=1, use_cache=True)
    
    # Step 6: Index chunks & embeddings
    retriever.index_clauses(chunk_vectors, chunked_clauses)
    
    # Step 7: For each question, embed query and retrieve
    answers = []
    for question in request.questions:
        query_vec = embeddings.embed_query(question)
        top_k = 10
        top_chunks = retriever.search(query_vec, top_k=top_k)

        #context = "\n\n".join(f"{c['section']}: {c['text']}" for c in top_chunks)
        prompt = compose_prompt(question, top_chunks)  # your improved prompt function
        answer_text = llm_service.aggregate_answers(question,[c["text"] for c in top_chunks])
        
        explanation = "Answer summarized from multiple relevant policy excerpts using Gemini LLM reasoning."
        
        answer_detail = AnswerDetail(
            answer=answer_text,
            clauses=[ClauseInfo(section=c["section"], text=c["text"], page=c.get("page")) for c in top_chunks],
            explanation=explanation,
        )
        answers.append(answer_detail)
    
    return QueryResponse(answers=answers)
