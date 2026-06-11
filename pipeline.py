# pipeline.py
import os
import re
import textwrap
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

# Local imports (same package)
from .hybrid_retriever import HybridRetriever

# Simple tokenizer for short extractive fallback
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_TOKEN_RE = re.compile(r"\w+")


def _tokenize_for_bm25(s: str) -> List[str]:
    if not s or not isinstance(s, str):
        return []
    return _TOKEN_RE.findall(s.lower())


def _first_sentences(text: str, n: int = 2) -> str:
    if not text:
        return ""
    parts = _SENTENCE_SPLIT_RE.split(text.strip())
    return " ".join(p.strip() for p in parts[:n] if p).strip()


# --- External LLM callers (OpenAI) ---
def _call_openai_chat(prompt: str, max_tokens: int = 250, temperature: float = 0.2) -> Optional[str]:
    """
    Try to call OpenAI ChatCompletion if OPENAI_API_KEY is set.
    Returns the model text or None if not available or on error.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        import openai

        openai.api_key = api_key
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.debug("OpenAI call failed: %s", e)
        return None


# --- Ollama HTTP streaming-aware caller (top-level helper) ---
def _call_ollama(prompt: str, model: str = "llama3:latest", max_tokens: int = 250, temperature: float = 0.2, timeout: int = 60) -> Optional[str]:
    """
    Call Ollama HTTP API and assemble streaming frames into a single string.
    Returns the full text or None on error.
    """
    try:
        import requests
        import json
        import logging as _logging
        from typing import List

        url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        with requests.post(url, json=payload, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            text_parts: List[str] = []
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not raw_line:
                    continue
                try:
                    frame = json.loads(raw_line)
                except Exception:
                    # ignore non-json lines
                    continue
                if isinstance(frame, dict):
                    fragment = frame.get("response") or frame.get("text") or ""
                    if fragment:
                        text_parts.append(fragment)
                    if frame.get("done") is True:
                        break
            final_text = "".join(text_parts).strip()
            if not final_text:
                # try non-streaming JSON body fallback
                try:
                    data = resp.json()
                    if isinstance(data, dict):
                        if "text" in data:
                            final_text = data["text"].strip()
                        elif "response" in data:
                            final_text = data["response"].strip()
                        elif "outputs" in data and isinstance(data["outputs"], list):
                            out = []
                            for o in data["outputs"]:
                                if isinstance(o, dict) and "content" in o:
                                    out.append(o["content"])
                            final_text = "\n".join(out).strip()
                except Exception:
                    pass
            return final_text or None
    except Exception as e:
        try:
            logging.getLogger(__name__).debug("Ollama call error: %s", e)
        except Exception:
            pass
        return None


# --- Deterministic fallback synthesizer (no LLM) ---
_SENTENCE_SPLIT_RE_SIMPLE = re.compile(r"(?<=[.!?])\s+")
import html

_GREETING_RE = re.compile(r"^(hi|hello|hey|good (morning|afternoon|evening)|yo)[\!\.\s]*$", re.I)
_CLEAN_TITLE_RE = re.compile(r"^\s*(#{1,6}\s*|={2,}\s*|-{2,}\s*|\*{3,}\s*|\[.*\]:\s*).*", re.M)


def _is_greeting(q: str) -> bool:
    if not q or not isinstance(q, str):
        return False
    return bool(_GREETING_RE.match(q.strip()))


def _clean_snippet(text: str, max_chars: int = 800) -> str:
    if not text:
        return ""
    cleaned = _CLEAN_TITLE_RE.sub("", text)
    cleaned = " ".join(cleaned.split())
    cleaned = html.unescape(cleaned)
    return cleaned.strip()[:max_chars]


def _first_sentence(text: str) -> str:
    if not text:
        return ""
    parts = _SENTENCE_SPLIT_RE_SIMPLE.split(text.strip())
    return parts[0].strip() if parts else text.strip()


def _synth_fallback(query: str, top_items: List[Tuple[str, dict]]) -> Tuple[str, str]:
    """
    Build a short, natural answer from top_items (list of (text, meta)).
    Strips headings and title lines, dedupes, and composes a friendly sentence.
    """
    seen = set()
    sentences = []
    for text, meta in top_items:
        cleaned = _clean_snippet(text)
        if not cleaned:
            continue
        parts = _SENTENCE_SPLIT_RE.split(cleaned)
        for p in parts:
            s = p.strip()
            if not s:
                continue
            if len(s.split()) <= 6 and (s.isupper() or s.startswith("—") or ("—" in s and len(s) < 60)):
                continue
            s_norm = s.lower()
            if s_norm in seen:
                continue
            seen.add(s_norm)
            sentences.append(s)
            break

    if not sentences:
        return ("I don't know; I couldn't find that in the sources.", "I don't know; I couldn't find that in the sources.")

    lead = sentences[0]
    if len(sentences) > 1:
        extra = " ".join(sentences[1:3])
        detailed = f"{lead} {extra}"
    else:
        detailed = lead

    short = lead.split(".")[0].strip()
    if not short.endswith("."):
        short = short + "."

    return (short, detailed)


class RAGPipeline:
    """
    Minimal pipeline wrapper that uses HybridRetriever for retrieval.
    answer_from_docs returns (short_answer, detailed_answer, sources).
    """

    def __init__(
        self,
        collection_name: str = "phone_docs",
        embed_model: str = "BAAI/bge-small-en",
        batch_size: int = 64,
    ):
        self.retriever = HybridRetriever(
            collection_name=collection_name, embed_model=embed_model, batch_size=batch_size
        )

    def refresh_index(self):
        """Call after re-ingest to reload BM25 and embeddings."""
        self.retriever.refresh()

    def _build_context_blocks(self, docs: List[str], metas: List[dict], max_chars: int = 800, max_sources: int = 4):
        """
        Build compact context blocks from top docs for LLM prompt.
        Returns (context_text, top_items) where top_items is list of tuples (text, meta).
        """
        top_items = list(zip(docs[:max_sources], metas[:max_sources]))
        parts = []
        for i, (d, m) in enumerate(top_items, start=1):
            src = m.get("source") if isinstance(m, dict) else (m or "unknown")
            snippet = (d or "").replace("\n", " ").strip()[:max_chars]
            parts.append(f"[SRC{i}] {snippet}\nSource: {src}")
        context = "\n\n".join(parts)
        return context, top_items

    def llm_call(self, system_prompt: str, user_prompt: str) -> str:
        """
        Attempt to call an external chat model (if configured). Tries OpenAI, then Ollama.
        If no model is configured or calls fail, returns an empty string to signal fallback.
        """
        prompt = f"{system_prompt}\n\n{user_prompt}"
        logger.info("llm_call invoked; trying OpenAI then Ollama")
        # Try OpenAI
        out = _call_openai_chat(prompt, max_tokens=300, temperature=0.2)
        if out:
            logger.info("llm_call: OpenAI returned non-empty output")
            return out
        # Try Ollama (local) using environment model name if set
        ollama_model = os.environ.get("OLLAMA_MODEL", "llama3:latest")
        logger.info("llm_call: calling Ollama model=%s", ollama_model)
        out = _call_ollama(prompt, model=ollama_model, max_tokens=300, temperature=0.2)
        if out:
            logger.info("llm_call: Ollama returned non-empty output (len=%d)", len(out))
            return out
        logger.info("llm_call: no LLM available, falling back")
        # No external LLM available
        return ""

    def answer_from_docs(self, query: str, docs: List[str], metas: List[dict]) -> Tuple[str, str, List[str]]:
        """
        Returns (short_answer, detailed_answer, sources).
        - short_answer: 1-2 sentences, concise, direct answer.
        - detailed_answer: fuller explanation with citations (for expand).
        - sources: list of source strings (metadatas).
        This function will attempt to synthesize a natural answer using an LLM if configured.
        If no LLM is available, it falls back to a deterministic, human-sounding summary.
        """
        # Build context and sources
        context, top_items = self._build_context_blocks(docs, metas, max_chars=800, max_sources=4)
        sources = [m.get("source") if isinstance(m, dict) else (m or "unknown") for m in metas[: len(docs)]]

        system_prompt = (
            "You are a concise, helpful assistant. Use ONLY the context below to answer the user's question. "
            "Write a short, natural, conversational answer (1-3 sentences). Do not copy long verbatim passages; summarize in your own words. "
            "If the sources do not contain the answer, say: \"I don't know; I couldn't find that in the sources.\" "
            "At the end, include a compact Sources section listing the source identifiers."
        )

        user_prompt = textwrap.dedent(f"""
        Context:
        {context}

        Question: {query}

        Provide:
        1) A one-sentence concise answer.
        2) A short 1-2 sentence explanation (optional).
        3) A compact Sources list.
        """)

        # Try LLM synthesis
        detailed_answer = self.llm_call(system_prompt, user_prompt)

        if detailed_answer:
            # Ensure Sources section exists; if not, append compact attributions
            if "Sources:" not in detailed_answer and "Source:" not in detailed_answer:
                atts = "\n\nSources:\n" + "\n".join(
                    [f"[SRC{i+1}] { (item[1].get('source') if isinstance(item[1], dict) else 'unknown') }" for i, item in enumerate(top_items)]
                )
                detailed_answer = detailed_answer.strip() + atts
            # Short answer: first sentence or first line
            short_answer = detailed_answer.strip().split("\n")[0].strip()
            # Truncate to first sentence if it's long
            if "." in short_answer:
                short_answer = short_answer.split(".")[0].strip() + "."
            return short_answer, detailed_answer.strip(), sources

        # Fallback: deterministic, human-sounding summary (no external LLM)
        short, detailed = _synth_fallback(query, top_items)
        atts = "\n\nSources:\n" + "\n".join(
            [f"[SRC{i+1}] { (m.get('source') if isinstance(m, dict) else 'unknown') }" for i, (_, m) in enumerate(top_items)]
        )
        detailed_with_atts = detailed + atts
        return short, detailed_with_atts, sources

    def simple_rerank(self, query: str, docs: List[str], top_n: int = 3) -> Tuple[List[str], List[int]]:
        """
        Optional simple TF-IDF reranker stub. Returns (reordered_docs, indices).
        If sklearn is not available, this function returns the original order.
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            import numpy as np

            vec = TfidfVectorizer(stop_words="english")
            doc_vecs = vec.fit_transform(docs)
            q_vec = vec.transform([query])
            scores = (doc_vecs @ q_vec.T).toarray().ravel()
            idx = scores.argsort()[::-1][:top_n]
            reordered = [docs[i] for i in idx]
            return reordered, idx.tolist()
        except Exception:
            return docs[:top_n], list(range(min(top_n, len(docs))))

    def query(
        self,
        user_query: str,
        top_k: int = 3,
        bm25_weight: float = 0.4,
        vector_weight: float = 0.6,
        reranker_on: bool = True,
    ) -> Tuple[str, str, List[str]]:
        """
        Run hybrid retrieval, optionally rerank, and produce (short_answer, detailed_answer, sources).
        """
        # Greeting shortcut: handle simple conversational inputs without retrieval
        if _is_greeting(user_query):
            logger.info("Greeting shortcut triggered for query: %s", user_query)
            short = "Hello! How can I help you with your phone today?"
            detailed = "Hi there — I can answer questions about phone specs, troubleshooting, or battery tips. What would you like to know?"
            return short, detailed, []

        # Retrieve fused results
        results = self.retriever.retrieve(
            user_query, top_k=top_k, bm25_weight=bm25_weight, vector_weight=vector_weight
        )

        docs = [r[0] for r in results]
        metas = [r[1] for r in results]

        # Optional rerank
        if reranker_on and docs:
            try:
                reordered_docs, idxs = self.simple_rerank(user_query, docs, top_n=min(len(docs), top_k))
                reordered_metas = [metas[i] for i in idxs]
                docs, metas = reordered_docs, reordered_metas
            except Exception:
                pass

        short_ans, detailed_ans, sources = self.answer_from_docs(user_query, docs, metas)
        return short_ans, detailed_ans, sources
