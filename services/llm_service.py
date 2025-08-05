import asyncio
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.exceptions import ResourceExhausted

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize LLM only if API key is provided
if GOOGLE_API_KEY:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.0,
        google_api_key=GOOGLE_API_KEY
    )
else:
    llm = None
    print("Warning: GOOGLE_API_KEY not set. LLM functionality will be disabled.")

async def get_llm_response_async(prompt: str) -> str:
    if llm is None:
        return "LLM functionality is disabled due to missing GOOGLE_API_KEY."
    
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, lambda: llm.invoke(prompt))
    return response.content

async def gemini_invoke_with_retry(prompt: str, max_retries=3, base_wait=2) -> str:
    if llm is None:
        return "LLM functionality is disabled due to missing GOOGLE_API_KEY."
    
    for attempt in range(max_retries):
        try:
            return await get_llm_response_async(prompt)
        except ResourceExhausted:
            wait_time = base_wait * (2 ** attempt)
            print(f"Gemini API quota exceeded. Retrying in {wait_time} seconds...")
            await asyncio.sleep(wait_time)
    return "Quota limit exceeded. Please try again later."

