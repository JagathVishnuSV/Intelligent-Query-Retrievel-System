# services/bm25_retriever.py
from rank_bm25 import BM25Okapi
from typing import List

class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.corpus = []
        self.clauses = []

    def index(self, clauses: List[dict]):
        self.clauses = clauses
        self.corpus = [clause["text"].lower().split() for clause in clauses]
        self.bm25 = BM25Okapi(self.corpus)

    def search(self, query: str, top_k=5) -> List[dict]:
        if not self.bm25:
            return []
        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.clauses[i] for i in top_indices]
