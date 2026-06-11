import chromadb

class VectorStore:
    def __init__(self, persist_dir="../chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)

        self.collection = self.client.get_or_create_collection(
            name="phone_docs",
            metadata={"hnsw:space": "cosine"}
        )

    def add(self, ids, embeddings, documents, metadata):
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadata
        )

    def query(self, text_embedding, top_k=5):
        return self.collection.query(
            query_embeddings=[text_embedding],
            n_results=top_k
        )
