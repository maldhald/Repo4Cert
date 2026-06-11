# app.py
import os
import sys
import streamlit as st
from typing import List

# Ensure the directory containing this file is on sys.path so local modules import reliably
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

st.set_page_config(page_title="Phone RAG Assistant", layout="wide")

# Sidebar controls
st.sidebar.header("Search settings")
bm25_weight = st.sidebar.slider("BM25 weight", 0.0, 1.0, 0.4, step=0.05)
vector_weight = 1.0 - bm25_weight
top_k = st.sidebar.slider("Top K retrieval", 1, 10, 3)
reranker_on = st.sidebar.checkbox("Enable reranker", value=True)

# Lazy import and cached pipeline factory to avoid import-time issues
@st.cache_resource
def get_pipeline():
    # Import inside function so Streamlit's working dir doesn't break imports
    try:
        from pipeline import RAGPipeline
    except Exception:
        from ingestion.pipeline import RAGPipeline
    return RAGPipeline(collection_name="phone_docs", embed_model="BAAI/bge-small-en", batch_size=64)


pipeline = get_pipeline()

# Page header
st.title("📱 Phone RAG Assistant")
st.write("Ask anything about the phones in your dataset. Use the sidebar to tune retrieval behavior.")

# Query form
with st.form("query_form", clear_on_submit=False):
    user_query = st.text_input("Enter your question:", "")
    submitted = st.form_submit_button("Search")

if submitted and user_query:
    with st.spinner("Retrieving and composing answer..."):
        short_ans, detailed_ans, sources = pipeline.query(
            user_query,
            top_k=top_k,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
            reranker_on=reranker_on,
        )

    st.subheader("💬 Answer")
    st.write(short_ans)

    with st.expander("Show detailed explanation and sources"):
        st.write(detailed_ans)
        st.markdown("**Sources**")
        if sources:
            for i, s in enumerate(sources, 1):
                st.write(f"{i}. {s}")
        else:
            st.write("No sources returned.")

    st.subheader("📚 Retrieved Context")
    if sources:
        # Show previews for each source (best-effort)
        for i, src in enumerate(sources, start=1):
            with st.expander(f"Source {i}: {src}"):
                try:
                    docs = pipeline.retriever.docs
                    metas = pipeline.retriever.metas
                    preview = None
                    for d, m in zip(docs, metas):
                        msrc = m.get("source") if isinstance(m, dict) else m
                        if msrc == src or (isinstance(src, str) and src in str(msrc)):
                            preview = (d or "")[:1000].replace("\n", " ")
                            break
                    if preview:
                        st.write(preview)
                    else:
                        st.write("Preview not available.")
                except Exception:
                    st.write("Preview not available.")
    else:
        st.write("No sources returned.")

# Refresh index button
if st.sidebar.button("Refresh index"):
    with st.spinner("Refreshing index..."):
        pipeline.refresh_index()
    st.sidebar.success("Index refreshed.")
