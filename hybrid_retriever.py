# ingestion/hybrid_retriever.py
import os
import re
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import chromadb
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# simple token regex for BM25
_TOKEN_RE = re.compile(r"\w+")


def _tokenize_text(s: str) -> List[str]:
    if not s or not isinstance(s, str):
        return []
    return _TOKEN_RE.findall(s.lower())


def _row_normalize(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize rows to unit length and return float32 array."""
    if a is None:
        return np.zeros((0, 0), dtype=np.float32)
    if a.size == 0:
        return a.astype(np.float32)
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (a / norms).astype(np.float32)


class HybridRetriever:
    """
    Hybrid retriever that fuses BM25 keyword scores with vector similarity.
    """

    def __init__(self,
                 collection_name: str = "phone_docs",
                 embed_model: str = "BAAI/bge-small-en",
                 batch_size: int = 64):
        logger.info("Initializing HybridRetriever")
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.client.get_or_create_collection(name=collection_name)

        self.embedder = SentenceTransformer(embed_model)
        self.batch_size = batch_size

        self.docs: List[str] = []
        self.metas: List[dict] = []
        self.ids: List[str] = []
        self.tokenized: List[List[str]] = []
        self.doc_embeddings: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        self.bm25 = None

        self.refresh()

    def _flatten_field(self, field):
        """
        Flatten possible nested list shapes returned by Chroma.
        Accepts: [], ['doc1','doc2'], [['doc1','doc2']], [[['doc1'],...]], numpy arrays.
        Returns a flat Python list.
        """
        if field is None:
            return []
        # numpy array -> list
        if hasattr(field, "tolist"):
            try:
                field = field.tolist()
            except Exception:
                pass
        # if empty
        if not field:
            return []
        # if first element is a list, flatten one level
        if isinstance(field[0], list):
            return [item for sub in field for item in sub]
        return list(field)

    def _load_from_chroma(self) -> Tuple[List[str], List[dict], List]:
        """
        Load documents, metadatas, embeddings from Chroma collection.
        Uses only allowed include fields and flattens nested shapes.
        """
        data = self.collection.get(include=["documents", "metadatas", "embeddings"])

        raw_docs = data.get("documents", [])
        raw_metas = data.get("metadatas", [])
        raw_embs = data.get("embeddings", [])

        docs = self._flatten_field(raw_docs)
        metas = self._flatten_field(raw_metas)
        embs = self._flatten_field(raw_embs)

        # Ensure docs are strings
        docs = [d if isinstance(d, str) else "" for d in docs]

        # If metas shorter than docs, pad with placeholders
        if len(metas) < len(docs):
            metas = metas + [{"source": None}] * (len(docs) - len(metas))

        return docs, metas, embs

    def refresh(self):
        """
        Reload documents from Chroma, rebuild BM25 index and embeddings.
        Call this after re-ingestion.
        """
        logger.info("Refreshing HybridRetriever index from ChromaDB: %s", CHROMA_DIR)
        self.docs, self.metas, embs = self._load_from_chroma()
        logger.info("Loaded %d docs from Chroma", len(self.docs))

        # Tokenize safely for BM25
        self.tokenized = [_tokenize_text(d) for d in self.docs]
        nonempty_tokens = sum(1 for t in self.tokenized if t)
        logger.info("Tokenized docs: %d non-empty token lists", nonempty_tokens)

        if nonempty_tokens > 0:
            self.bm25 = BM25Okapi(self.tokenized)
        else:
            self.bm25 = None
            logger.info("BM25 index not built (no tokenized docs)")

        # Build or load embeddings (batch encode) only if docs exist
        if self.docs:
            try:
                # sentence-transformers may expose different methods for dim
                emb_dim = self.embedder.get_embedding_dimension()
            except Exception:
                try:
                    emb_dim = self.embedder.get_sentence_embedding_dimension()
                except Exception:
                    emb_dim = None

            # If Chroma returned embeddings, try to use them to avoid re-encoding
            if embs and len(embs) == len(self.docs):
                try:
                    arr = np.asarray(embs, dtype=np.float32)
                    # If arr is 1D (single vector), reshape
                    if arr.ndim == 1:
                        arr = arr.reshape(1, -1)
                    self.doc_embeddings = _row_normalize(arr)
                    logger.info("Loaded embeddings from Chroma, shape: %s", self.doc_embeddings.shape)
                except Exception:
                    # fallback to encoding
                    enc = self.embedder.encode(
                        self.docs,
                        batch_size=self.batch_size,
                        normalize_embeddings=True,
                        show_progress_bar=False
                    )
                    arr = np.asarray(enc, dtype=np.float32)
                    self.doc_embeddings = _row_normalize(arr)
                    logger.info("Encoded docs into embeddings, shape: %s", self.doc_embeddings.shape)
            else:
                # encode documents
                enc = self.embedder.encode(
                    self.docs,
                    batch_size=self.batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                arr = np.asarray(enc, dtype=np.float32)
                self.doc_embeddings = _row_normalize(arr)
                logger.info("Encoded docs into embeddings, shape: %s", self.doc_embeddings.shape)
        else:
            # no docs: create empty embeddings with known dim if possible
            try:
                emb_dim = self.embedder.get_embedding_dimension()
            except Exception:
                try:
                    emb_dim = self.embedder.get_sentence_embedding_dimension()
                except Exception:
                    emb_dim = 0
            if emb_dim and emb_dim > 0:
                self.doc_embeddings = np.zeros((0, emb_dim), dtype=np.float32)
            else:
                self.doc_embeddings = np.zeros((0, 0), dtype=np.float32)
            logger.info("No docs found; created empty embeddings array with dim %s", self.doc_embeddings.shape[1] if self.doc_embeddings.ndim == 2 else 0)

    def retrieve(self, query: str, top_k: int = 5, bm25_weight: float = 0.4, vector_weight: float = 0.6):
        if not query:
            return []

        n_docs = len(self.docs)
        if n_docs == 0:
            logger.info("No documents available in retriever")
            return []

        # BM25 scores
        if self.bm25 is not None:
            token_q = _tokenize_text(query)
            bm25_scores = np.array(self.bm25.get_scores(token_q), dtype=np.float32)
        else:
            bm25_scores = np.zeros(n_docs, dtype=np.float32)

        # Vector scores
        q_emb = self.embedder.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        q_emb = np.asarray(q_emb, dtype=np.float32)
        # ensure query vector normalized
        q_norm = np.linalg.norm(q_emb)
        if q_norm > 0:
            q_emb = q_emb / (q_norm + 1e-12)
        else:
            q_emb = q_emb.astype(np.float32)

        if self.doc_embeddings.size == 0:
            vec_scores = np.zeros(n_docs, dtype=np.float32)
        else:
            # ensure shapes align: (n_docs, dim) dot (dim,) -> (n_docs,)
            try:
                vec_scores = np.dot(self.doc_embeddings, q_emb).astype(np.float32)
            except Exception:
                # fallback to inner product per row
                vec_scores = np.array([float(np.inner(d, q_emb)) for d in self.doc_embeddings], dtype=np.float32)

        def _normalize(arr: np.ndarray) -> np.ndarray:
            if arr.size == 0:
                return arr
            minv = float(np.min(arr))
            maxv = float(np.max(arr))
            span = maxv - minv
            if span <= 1e-9:
                return np.zeros_like(arr, dtype=np.float32)
            return ((arr - minv) / (span + 1e-12)).astype(np.float32)

        bm25_norm = _normalize(bm25_scores)
        vec_norm = _normalize(vec_scores)

        fused = bm25_weight * bm25_norm + vector_weight * vec_norm

        top_k = min(top_k, n_docs)
        if top_k <= 0:
            return []

        top_idx = np.argsort(fused)[-top_k:][::-1]

        results = []
        for i in top_idx:
            results.append((self.docs[i], self.metas[i], float(fused[i])))

        logger.info("Retrieve: query='%s' returned %d results", query, len(results))
        return results
