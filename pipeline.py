"""
pipeline.py — Document ingestion, chunking, embedding, and ChromaDB indexing.

Usage:
    python ingestion/pipeline.py

Supported file types:
  .pdf  -- assignment briefs, user guides
  .txt  -- notes, specs, plain text
  .csv  -- task lists (each row = one chunk)
  .xlsx -- task sheets  (each row = one chunk, requires openpyxl)
"""

import os
import csv
import pickle

import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

# Config
DOCUMENTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "documents")
OUTPUT_DIR     = os.path.join(os.path.dirname(__file__), "..")
CHROMA_DIR     = os.path.join(OUTPUT_DIR, "chroma_db")
COLLECTION     = "rag_documents"
CHUNK_SIZE     = 128
OVERLAP        = 16
EMBED_MODEL    = "all-MiniLM-L6-v2"


def row_to_sentence(headers, row):
    """Convert a CSV/XLSX row dict into a readable labelled sentence."""
    parts = []
    for h in headers:
        val = str(row.get(h, "")).strip()
        if val and val.lower() not in ("none", "n/a", "-", ""):
            parts.append(f"{h}: {val}")
    return " | ".join(parts)


def load_csv_as_chunks(path):
    """Load CSV — each row becomes one chunk."""
    chunks = []
    try:
        with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
            reader = csv.DictReader(f)
            headers = [h.strip() for h in (reader.fieldnames or []) if h and h.strip()]
            for row in reader:
                values = [str(v).strip() for v in row.values() if v and str(v).strip()]
                if not values:
                    continue
                sentence = row_to_sentence(headers, row)
                if sentence:
                    chunks.append(sentence)
    except Exception as e:
        print(f"  [X] CSV parse error: {e}")
    return chunks


def load_xlsx_as_chunks(path):
    """Load XLSX — each row becomes one chunk."""
    try:
        import openpyxl
    except ImportError:
        print("  [!] openpyxl not installed. Run: pip install openpyxl")
        return []

    chunks = []
    try:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows = list(ws.iter_rows(values_only=True))
            if not rows:
                continue
            headers = None
            data_start = 0
            for i, row in enumerate(rows):
                non_empty = [c for c in row if c is not None and str(c).strip()]
                if non_empty:
                    headers = [str(c).strip() if c else "" for c in row]
                    data_start = i + 1
                    break
            if not headers:
                continue
            for row in rows[data_start:]:
                row_dict = {h: (str(v).strip() if v is not None else "") for h, v in zip(headers, row)}
                if not any(row_dict.values()):
                    continue
                sentence = row_to_sentence(headers, row_dict)
                if sentence:
                    chunks.append(sentence)
        wb.close()
    except Exception as e:
        print(f"  [X] XLSX parse error: {e}")
    return chunks


def load_documents(source_dir):
    """Load all supported files. Returns (text_docs, row_chunks)."""
    text_docs  = []
    row_chunks = []
    source_dir = os.path.abspath(source_dir)

    if not os.path.isdir(source_dir):
        print(f"[WARNING] Documents directory not found: {source_dir}")
        return text_docs, row_chunks

    files = os.listdir(source_dir)
    if not files:
        print(f"[WARNING] No files found in {source_dir}.")
        return text_docs, row_chunks

    for fname in files:
        path = os.path.join(source_dir, fname)
        try:
            if fname.lower().endswith(".pdf"):
                pdf  = fitz.open(path)
                text = " ".join(page.get_text() for page in pdf)
                text_docs.append(text)
                print(f"  [ok] Loaded PDF:  {fname} ({len(text):,} chars)")

            elif fname.lower().endswith(".txt"):
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                text_docs.append(text)
                print(f"  [ok] Loaded TXT:  {fname} ({len(text):,} chars)")

            elif fname.lower().endswith(".csv"):
                chunks = load_csv_as_chunks(path)
                row_chunks.extend(chunks)
                print(f"  [ok] Loaded CSV:  {fname} ({len(chunks)} rows as chunks)")

            elif fname.lower().endswith(".xlsx"):
                chunks = load_xlsx_as_chunks(path)
                row_chunks.extend(chunks)
                print(f"  [ok] Loaded XLSX: {fname} ({len(chunks)} rows as chunks)")

            else:
                print(f"  [~]  Skipped (unsupported): {fname}")

        except Exception as e:
            print(f"  [X]  Failed to load {fname}: {e}")

    return text_docs, row_chunks


def chunk_text_documents(docs, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Split PDF/TXT documents into overlapping word-level chunks."""
    chunks = []
    for doc in docs:
        words = doc.split()
        step  = chunk_size - overlap
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + chunk_size])
            if len(chunk.strip()) > 50:
                chunks.append(chunk)
    return chunks


def build_chroma(chunks):
    """Embed chunks and store in ChromaDB. Returns the collection."""
    print(f"\n[ChromaDB] Initialising persistent store at: {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Drop old collection if it exists so we start fresh
    try:
        client.delete_collection(COLLECTION)
        print(f"[ChromaDB] Dropped old collection '{COLLECTION}'")
    except Exception:
        pass

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    collection = client.create_collection(
        name=COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    # Upsert in batches of 100
    batch_size = 100
    total = len(chunks)
    for start in range(0, total, batch_size):
        batch = chunks[start : start + batch_size]
        ids   = [str(i) for i in range(start, start + len(batch))]
        collection.add(documents=batch, ids=ids)
        print(f"  [ChromaDB] Indexed {min(start + batch_size, total)}/{total} chunks")

    return collection


def build_bm25(chunks):
    """Build a BM25 sparse index over the same chunks."""
    print("[BM25] Building sparse index ...")
    return BM25Okapi([c.split() for c in chunks])


def save_bm25(chunks, bm25, output_dir=OUTPUT_DIR):
    """Save chunks list and BM25 index (ChromaDB persists itself)."""
    import pickle
    with open(os.path.join(output_dir, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)
    with open(os.path.join(output_dir, "bm25.pkl"), "wb") as f:
        pickle.dump(bm25, f)
    print(f"[Saved] chunks.pkl | bm25.pkl -> {os.path.abspath(output_dir)}")
    print(f"[Saved] ChromaDB   -> {os.path.abspath(CHROMA_DIR)}")


if __name__ == "__main__":
    print("=" * 60)
    print("RAG Project -- Ingestion Pipeline (ChromaDB)")
    print("=" * 60)

    print(f"\n[1/4] Loading documents from: {os.path.abspath(DOCUMENTS_DIR)}")
    text_docs, row_chunks = load_documents(DOCUMENTS_DIR)

    if not text_docs and not row_chunks:
        print("\n[ERROR] No documents loaded.")
        print("        Supported: .pdf  .txt  .csv  .xlsx")
        exit(1)

    print(f"\n[2/4] Chunking ...")
    text_chunks = chunk_text_documents(text_docs)
    all_chunks  = row_chunks + text_chunks
    print(f"      CSV/XLSX row chunks : {len(row_chunks)}")
    print(f"      PDF/TXT text chunks : {len(text_chunks)}")
    print(f"      Total chunks        : {len(all_chunks)}")

    print("\n[3/4] Embedding and storing in ChromaDB ...")
    build_chroma(all_chunks)

    print("\n[4/4] Building BM25 index and saving ...")
    bm25 = build_bm25(all_chunks)
    save_bm25(all_chunks, bm25)

    print("\n[ok] Pipeline complete. Run: streamlit run app.py")
    print("=" * 60)
