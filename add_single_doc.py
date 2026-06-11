# ingestion/add_single_doc.py
import os
import logging
from sentence_transformers import SentenceTransformer
import chromadb

# Configure paths (adjust if needed)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "phone_docs"
EMBED_MODEL = "BAAI/bge-small-en"

# File to add (change to your tiger 800 manual path)
DOC_PATH = os.path.join(BASE_DIR, "data", "tiger800", "tiger800_manual.txt")

# Chunking params (match ingest.py)
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBED_BATCH = 64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("add_single_doc")

def chunk_text(s: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    if not s:
        return []
    s = s.strip()
    chunks = []
    i = 0
    L = len(s)
    while i < L:
        end = min(i + chunk_size, L)
        chunk = s[i:end].strip()
        if chunk:
            chunks.append(chunk)
        i = end - overlap
        if i < 0:
            i = 0
    return chunks

def main():
    if not os.path.exists(DOC_PATH):
        logger.error("Document not found: %s", DOC_PATH)
        return

    # Read file
    with open(DOC_PATH, "r", encoding="utf-8", errors="ignore") as fh:
        text = fh.read()

    chunks = chunk_text(text)
    if not chunks:
        logger.info("No chunks produced for %s", DOC_PATH)
        return

    # Embed
    embedder = SentenceTransformer(EMBED_MODEL)
    embeddings = []
    for i in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[i : i + EMBED_BATCH]
        emb = embedder.encode(batch, batch_size=len(batch), normalize_embeddings=True, show_progress_bar=False)
        embeddings.extend([list(map(float, e)) for e in emb])

    # Prepare ids and metadata
    ids = [f"{os.path.relpath(DOC_PATH, BASE_DIR)}::{i}" for i in range(len(chunks))]
    metadatas = [{"source": os.path.relpath(DOC_PATH, BASE_DIR)} for _ in chunks]

    # Add to Chroma
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    col = client.get_or_create_collection(name=COLLECTION_NAME)
    try:
        col.add(ids=ids, documents=chunks, metadatas=metadatas, embeddings=embeddings)
        logger.info("Added %d chunks for %s", len(chunks), DOC_PATH)
    except Exception as e:
        logger.warning("Add failed, attempting upsert: %s", e)
        col.upsert(ids=ids, documents=chunks, metadatas=metadatas, embeddings=embeddings)
        logger.info("Upserted %d chunks for %s", len(chunks), DOC_PATH)

if __name__ == "__main__":
    main()
