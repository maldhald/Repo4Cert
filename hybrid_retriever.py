"""
hybrid_retriever.py -- Hybrid BM25 + ChromaDB retrieval with RRF fusion.

Three modes:
  - retrieve()        : precise top-k retrieval with cross-encoder reranking
  - retrieve_all()     : fetch ALL chunks matching a keyword filter (e.g. "list all UI tasks")
  - retrieve_delayed() : date-aware filter for "delayed/overdue" queries -- computed
                          directly from End Target Date vs today, not guessed by the LLM
"""

import os
import re
import pickle
import datetime
import numpy as np
from sentence_transformers import CrossEncoder
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DIR  = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
COLLECTION  = "rag_documents"
EMBED_MODEL = "all-MiniLM-L6-v2"
TODAY       = datetime.date.today()

# Keywords that signal a "list all" type query
LIST_TRIGGERS = [
    "list all", "list me all", "show all", "show me all",
    "all tasks", "all ui", "all be", "all backend", "all frontend",
    "give me all", "what are all", "every task",
]

# Keywords that signal a date-aware "delayed/overdue" query --
# these must be computed from real dates, never guessed by the LLM
DELAYED_TRIGGERS = [
    "delayed", "overdue", "late", "behind schedule",
    "past due", "missed deadline", "running late",
]

DONE_STATUSES = ("done", "completed")


def rrf_fusion(ranked_lists, k=60):
    """Merge multiple ranked lists using Reciprocal Rank Fusion."""
    scores = {}
    for ranked in ranked_lists:
        for rank, idx in enumerate(ranked):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


def is_list_query(query: str) -> bool:
    """Return True if the query is asking to list all items of a type."""
    q = query.lower()
    return any(trigger in q for trigger in LIST_TRIGGERS)


def is_delayed_query(query: str) -> bool:
    """Return True if the query is asking about delayed/overdue tasks."""
    q = query.lower()
    return any(trigger in q for trigger in DELAYED_TRIGGERS)


def parse_chunk_field(chunk: str, field_name: str) -> str:
    """
    Extract a field value from a 'Field: value | Field2: value2' style chunk.
    Returns '' if not found.
    """
    pattern = rf"{re.escape(field_name)}\s*:\s*(.*?)(?:\s*\||$)"
    match = re.search(pattern, chunk)
    return match.group(1).strip() if match else ""


def parse_chunk_date(value: str):
    """Parse a date string like '2026-07-11 00:00:00' or '2026-07-11' into a date object."""
    if not value:
        return None
    value = value.strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


class HybridRetriever:
    """
    Retrieval pipeline:
      1. Date-aware queries (delayed/overdue) -> computed directly from real dates
      2. "List all X" queries -> scans all chunks directly
      3. Otherwise -> BM25 + ChromaDB semantic search -> RRF fusion -> cross-encoder reranking
    """

    RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, chunks, bm25, candidate_pool=20):
        self.chunks         = chunks
        self.bm25           = bm25
        self.candidate_pool = candidate_pool

        self.client = chromadb.PersistentClient(path=os.path.abspath(CHROMA_DIR))
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        self.collection = self.client.get_collection(
            name=COLLECTION,
            embedding_function=ef,
        )
        print(f"[ChromaDB] Connected -- {self.collection.count()} chunks in store")

        print(f"[Reranker] Loading '{self.RERANKER_MODEL}' ...")
        self.reranker = CrossEncoder(self.RERANKER_MODEL)

    def retrieve_delayed(self) -> list[str]:
        """
        Return chunks for tasks that are genuinely overdue:
        End Target Date < today AND Status is not Done/Completed.
        This is computed directly from data -- never left to the LLM to infer.
        """
        delayed = []
        for chunk in self.chunks:
            end_str = parse_chunk_field(chunk, "End Target Date") or parse_chunk_field(chunk, "End Target")
            status  = parse_chunk_field(chunk, "Status").lower()

            end_date = parse_chunk_date(end_str)
            if end_date and end_date < TODAY and status not in DONE_STATUSES:
                delayed.append(chunk)

        return delayed

    def retrieve_all(self, query: str) -> list[str]:
        """Scan ALL chunks and return those containing the filter keyword."""
        q = query.lower()

        filter_terms = []
        if "ui" in q:
            filter_terms = ["Task ID: UI-", "Owner: UI"]
        elif "backend" in q or " be" in q:
            filter_terms = ["Task ID: BE-", "Owner: Backend"]
        else:
            words = [w for w in query.split() if len(w) > 2]
            filter_terms = words

        matched = []
        for chunk in self.chunks:
            if any(term.lower() in chunk.lower() for term in filter_terms):
                matched.append(chunk)

        return matched

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        """
        Smart retrieval:
        - Delayed/overdue queries -> computed directly from dates (most accurate)
        - 'List all X' queries -> scan all chunks directly
        - Otherwise -> hybrid BM25 + ChromaDB + reranking
        """
        if is_delayed_query(query):
            return self.retrieve_delayed()

        if is_list_query(query):
            return self.retrieve_all(query)

        pool = self.candidate_pool

        bm25_scores = self.bm25.get_scores(query.split())
        bm25_ranks  = np.argsort(bm25_scores)[::-1][:pool].tolist()

        results      = self.collection.query(
            query_texts=[query],
            n_results=min(pool, self.collection.count()),
        )
        chroma_ranks = [int(i) for i in results["ids"][0]]

        fused      = rrf_fusion([bm25_ranks, chroma_ranks])[:pool]
        candidates = [self.chunks[i] for i in fused if i < len(self.chunks)]

        pairs         = [(query, c) for c in candidates]
        rerank_scores = self.reranker.predict(pairs)
        ranked        = sorted(zip(rerank_scores, candidates), key=lambda x: x[0], reverse=True)

        return [chunk for _, chunk in ranked[:top_k]]


# Smoke test
if __name__ == "__main__":
    chunks_path = os.path.join(os.path.dirname(__file__), "..", "chunks.pkl")
    bm25_path   = os.path.join(os.path.dirname(__file__), "..", "bm25.pkl")

    for p in [chunks_path, bm25_path]:
        if not os.path.exists(p):
            print(f"[ERROR] Missing: {p} -- run pipeline.py first.")
            exit(1)

    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)

    retriever = HybridRetriever(chunks, bm25)

    print(f"\n[Test 1] Delayed query (today={TODAY}):")
    results = retriever.retrieve("show me delayed tasks")
    print(f"Found {len(results)} delayed tasks")
    for r in results[:5]:
        print(" -", r[:100])

    print("\n[Test 2] List query:")
    results = retriever.retrieve("list me all UI tasks")
    print(f"Found {len(results)} UI tasks")

    print("\n[Test 3] Precise query:")
    results = retriever.retrieve("end date of UI-1.3", top_k=3)
    for i, r in enumerate(results, 1):
        print(f"\n-- Result {i} --")
        print(r)

    print("\n[ok] Smoke-test passed.")
