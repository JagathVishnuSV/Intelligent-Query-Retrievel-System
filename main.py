import uvicorn
from fastapi import FastAPI
from api.endpoints import router

app = FastAPI(title="Insurance Policy Q&A API")

app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
