from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

# Load the model once globally (do this once to avoid loading repeatedly)
model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_text_list(texts: List[str]) -> List[np.ndarray]:
    """
    Embed a list of texts using Sentence Transformers locally.
    Returns list of numpy arrays (embeddings).
    """
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.tolist()  # or keep as numpy arrays if your retriever supports that

def embed_query(query: str):
    """
    Embed a query string to vector using Sentence Transformers.
    """
    emb = model.encode(query, convert_to_numpy=True)
    return emb
