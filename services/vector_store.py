# services/vector_store.py
import asyncio
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance
import hashlib
from services.embeddings import VECTOR_DIM

collection_name = "policy_chunks"
_qdrant_client = QdrantClient(host="localhost", port=6333)

def doc_hash(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:8]

class QdrantIndexer:
    def __init__(self, doc_id: str):
        self.collection_name = f"policy_chunks_{doc_id}"
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        existing = _qdrant_client.get_collections().collections
        if not any(c.name == self.collection_name for c in existing):
            _qdrant_client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )

    async def upsert_vectors(self, vectors: list[list[float]], chunked_clauses: list[dict]):
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={"text": clause["text"], "section": clause["section"]}
            )
            for vec, clause in zip(vectors, chunked_clauses)
        ]
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: _qdrant_client.upsert(collection_name=self.collection_name, points=points)
        )

class QdrantRetriever:
    def __init__(self, doc_id: str):
        self.collection_name = f"policy_chunks_{doc_id}"

    async def search(self, query_vector: list[float], top_k: int):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: _qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True
            )
        )

from rank_bm25 import BM25Okapi
import nltk
nltk.download('punkt')

class HybridRetriever:
    def __init__(self, vector_retriever, chunked_clauses):
        self.vector_retriever = vector_retriever
        self.chunked_clauses = chunked_clauses
        self.tokenized_clauses = [nltk.word_tokenize(c["text"].lower()) for c in chunked_clauses]
        self.bm25 = BM25Okapi(self.tokenized_clauses)

    def keyword_search(self, query: str, top_k=5):
        tokens = nltk.word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.chunked_clauses[i] for i in top_indices]

    async def hybrid_search(self, query_vector, query_text: str, top_k=5):
        vector_results = await self.vector_retriever.search(query_vector, top_k=top_k)
        bm25_results = self.keyword_search(query_text, top_k=top_k)

        merged = {c["text"]: c for c in bm25_results}
        for hit in vector_results:
            merged[hit.payload["text"]] = {
                "text": hit.payload["text"],
                "section": hit.payload.get("section", "Unknown"),
            }
        return list(merged.values())[:top_k]
