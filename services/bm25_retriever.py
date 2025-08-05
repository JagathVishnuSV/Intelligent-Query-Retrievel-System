from typing import List
import numpy as np
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self):
        self.clauses: List[dict] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25 = None

    def index(self, clauses: List[dict]):
        self.clauses = clauses
        self.tokenized_corpus = [self.tokenize(clause["text"]) for clause in clauses]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        if not self.bm25:
            return []

        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [self.clauses[i] for i in top_indices if scores[i] > 0.0]

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return text.lower().split()
