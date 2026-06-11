import os
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

class Retriever:
    def __init__(self, persist_dir=CHROMA_DIR, collection_name="phone_docs"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.embedder = SentenceTransformer("BAAI/bge-small-en")

    def retrieve(self, query: str, top_k: int = 5):
        # Encode query
        query_embedding = self.embedder.encode(
            [query],
            normalize_embeddings=True
        )[0]

        # Query Chroma
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # Extract docs + metadata
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        return list(zip(docs, metas))
