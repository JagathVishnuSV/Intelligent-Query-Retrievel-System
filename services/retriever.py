import faiss
import numpy as np

index = None
clause_store = []

def index_clauses(vectors: list, chunks: list[dict]):
    global index, clause_store
    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(vectors).astype('float32'))
    clause_store = chunks

def search(query_vec, top_k=3):
    global index, clause_store
    if index is None:
        raise ValueError("Index not initialized")
    D, I = index.search(np.array([query_vec]).astype('float32'), top_k)
    return [clause_store[i] for i in I[0]]
