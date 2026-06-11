import numpy as np
from sentence_transformers import SentenceTransformer, util

class Reranker:
    def __init__(self):
        self.model = SentenceTransformer("BAAI/bge-small-en")

    def rerank(self, query: str, docs, top_k=3):
        # --- SAFETY CHECK: no docs retrieved ---
        if not docs:
            return []

        # Encode query
        query_emb = self.model.encode(query, normalize_embeddings=True)

        # Encode documents
        doc_embs = self.model.encode(docs, normalize_embeddings=True)

        # --- SAFETY CHECK: empty embeddings (rare but possible) ---
        if doc_embs is None or len(doc_embs) == 0:
            return []

        # Compute cosine similarity
        scores = util.cos_sim(query_emb, doc_embs)[0]

        # Pair docs with scores
        scored_docs = list(zip(docs, scores))

        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Return top-k docs only
        return [d for d, _ in scored_docs[:top_k]]
