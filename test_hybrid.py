# ingestion/test_hybrid.py
import os
import sys

# Ensure project root is on sys.path so 'ingestion' package imports work
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ingestion.hybrid_retriever import HybridRetriever

def main():
    r = HybridRetriever()
    queries = [
        "battery life of iPhone 15",
        "does Galaxy S24 support DeX",
        "Pixel 8 update policy",
        "OnePlus 12 charging speed",
        "iPhone 14 warranty coverage"
    ]
    for q in queries:
        print("\n=== QUERY:", q, "===\n")
        res = r.retrieve(q, top_k=3, bm25_weight=0.4, vector_weight=0.6)
        for i, (doc, meta, score) in enumerate(res, 1):
            src = meta.get("source") if isinstance(meta, dict) else meta
            snippet = (doc or "")[:300].replace("\n", " ")
            print(f"Result {i}:\nSource: {src}\nScore: {score:.4f}\nSnippet: {snippet}\n")

if __name__ == "__main__":
    main()
