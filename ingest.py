# ingestion/ingest.py
import os
import uuid
import logging
from typing import List

from chunker import chunk_text
from embedder import Embedder
from vectorstore import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingest")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# Batch size for embeddings (CRITICAL for stability)
EMBED_BATCH = 32


def load_all_files():
    file_paths = []
    for root, _, files in os.walk(DATA_DIR):
        for f in files:
            if f.endswith((".txt", ".md")):
                file_paths.append(os.path.join(root, f))
    return sorted(file_paths)


def read_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def embed_in_batches(embedder, chunks, batch_size=EMBED_BATCH):
    """Yield embeddings in safe batches."""
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        yield embedder.embed(batch)


def main():
    logger.info("Loading files...")
    files = load_all_files()

    store = VectorStore(persist_dir=CHROMA_DIR)
    embedder = Embedder()

    for file_path in files:
        logger.info(f"Processing: {file_path}")

        text = read_file(file_path)
        chunks = chunk_text(text)

        if not chunks:
            logger.warning(f"No chunks produced for {file_path}")
            continue

        # Generate deterministic IDs
        ids = [f"{os.path.relpath(file_path, BASE_DIR)}::{i}" for i in range(len(chunks))]
        metadata = [{"source": file_path} for _ in chunks]

        # Embed in batches
        all_embeddings = []
        for batch_emb in embed_in_batches(embedder, chunks):
            all_embeddings.extend(batch_emb)

        # Add to vector store in batches
        for i in range(0, len(chunks), EMBED_BATCH):
            store.add(
                ids[i:i + EMBED_BATCH],
                all_embeddings[i:i + EMBED_BATCH],
                chunks[i:i + EMBED_BATCH],
                metadata[i:i + EMBED_BATCH]
            )

        logger.info(f"Added {len(chunks)} chunks from {file_path}")

    logger.info("Ingestion complete!")


if __name__ == "__main__":
    main()
