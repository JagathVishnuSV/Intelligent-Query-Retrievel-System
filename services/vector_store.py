import asyncio
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.http import models

# Initialize Qdrant client (adjust host/port as per deployment)
_qdrant_client = QdrantClient(host="localhost", port=6333)

collection_name = "policy_chunks"
VECTOR_DIM = 384  # Adjust this to your embedding dimension

_qdrant_client = QdrantClient(host="localhost", port=6333)

def ensure_collection_exists():
    try:
        collections = _qdrant_client.get_collections()
        existing = any(c.name == collection_name for c in collections.collections)
        if not existing:
            _qdrant_client.recreate_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=VECTOR_DIM, distance=models.Distance.COSINE),
            )
            print(f"Collection '{collection_name}' created.")
        else:
            print(f"Collection '{collection_name}' already exists.")
    except Exception as e:
        print(f"Error ensuring collection exists: {e}")

# Call this once before indexing
ensure_collection_exists()

# Then proceed with upserting points as usual...

class QdrantIndexer:
    def __init__(self, collection_name=collection_name):
        self.collection_name = collection_name

    async def upsert_vectors(self, vectors: list[list[float]], chunked_clauses: list[dict]):
        points = []
        for idx, (vec, clause) in enumerate(zip(vectors, chunked_clauses)):
            points.append(PointStruct(
                id=idx,
                vector=vec,
                payload={
                    "text": clause["text"],
                    "section": clause["section"],
                }
            ))
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, 
            lambda: _qdrant_client.upsert(collection_name=self.collection_name, points=points)
        )

class QdrantRetriever:
    def __init__(self, collection_name=collection_name):
        self.collection_name = collection_name

    async def search(self, query_vector: list[float], top_k: int):
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: _qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True
            )
        )
        return response

indexer = QdrantIndexer()
retriever = QdrantRetriever()
