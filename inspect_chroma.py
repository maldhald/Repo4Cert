# ingestion/inspect_chroma.py
import os
import chromadb
import json
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

print("Chroma DB path:", CHROMA_DIR)

client = chromadb.PersistentClient(path=CHROMA_DIR)
cols = client.list_collections()
print("Collections:", cols)

for c in cols:
    try:
        name = c.name
    except Exception:
        try:
            name = c["name"]
        except Exception:
            name = str(c)

    print(f"\n--- Collection: {name} ---")
    col = client.get_collection(name)

    # Request allowed fields
    data = col.get(include=["documents", "metadatas", "embeddings"])

    # documents
    docs = data.get("documents", [])
    if docs and isinstance(docs[0], list):
        docs = [item for sub in docs for item in sub]
    docs = [d if isinstance(d, str) else "" for d in docs]

    # metadatas
    metas = data.get("metadatas", [])
    if metas and isinstance(metas[0], list):
        metas = [item for sub in metas for item in sub]

    # embeddings: could be list, nested list, or numpy array
    embs = data.get("embeddings", [])
    if isinstance(embs, np.ndarray):
        embs_list = embs.tolist()
    elif embs and isinstance(embs[0], list):
        embs_list = [item for sub in embs for item in sub]
    else:
        embs_list = embs

    print("doc count:", len(docs))
    print("meta count:", len(metas))
    print("embeddings count:", len(embs_list))

    if docs:
        first_doc = docs[0] or ""
        print("first doc length:", len(first_doc))
        print("first meta sample:", json.dumps(metas[0], ensure_ascii=False) if metas else "NO META")
        print("first doc preview:", first_doc[:400].replace("\n", " "))
    else:
        print("No documents found in this collection.")
