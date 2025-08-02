from fastapi import FastAPI
from dotenv import load_dotenv
import os
from api.endpoints import router as api_router

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    raise EnvironmentError("GOOGLE_API_KEY not set. Please add it in your .env file.")

app = FastAPI(title="Gemini-powered Query-Retrieval System")
app.include_router(api_router, prefix="/api/v1")
