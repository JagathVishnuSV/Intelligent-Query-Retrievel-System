from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
# Initialize Gemini chat LLM with desired model and temperature
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.0)  # 'gemini-pro' or other available model

def get_llm_response(prompt: str) -> str:
    response = llm.invoke(prompt)
    return response.content

def aggregate_answers(question: str, chunk_texts: list[str]):
    # Compose combined context string
    combined_context = "\n\n".join(chunk_texts)

    prompt = (
        f"Given the following excerpts from an insurance policy:\n{combined_context}\n\n"
        f"Provide a clear, concise answer to the question:\n{question}\n"
        f"If the answer is not found, say 'Information not available in the policy.'"
    )
    return get_llm_response(prompt)