# inspect_db.py
from pipeline import RAGPipeline
import statistics
from collections import Counter

p = RAGPipeline()

docs = p.retriever.docs
metas = p.retriever.metas

print("====================================")
print("📦 CHROMA DB — COLLECTION STATS")
print("====================================")

# Basic counts
print(f"Total chunks: {len(docs)}")
print(f"Total metadata entries: {len(metas)}")

# Chunk lengths
lengths = [len(d) for d in docs if isinstance(d, str)]
if lengths:
    print("\n--- Chunk Lengths (characters) ---")
    print(f"Min: {min(lengths)}")
    print(f"Median: {statistics.median(lengths)}")
    print(f"Mean: {round(statistics.mean(lengths), 1)}")
    print(f"Max: {max(lengths)}")
else:
    print("No chunk lengths available.")

# Count chunks per source file
sources = [m.get("source") for m in metas if isinstance(m, dict)]
counts = Counter(sources)

print("\n--- Chunks per Source File ---")
for src, count in counts.items():
    print(f"{src}: {count}")

# Show first few docs + metas
print("\n--- Sample Chunks (first 3) ---")
for i in range(min(3, len(docs))):
    print(f"\nChunk {i}:")
    print(docs[i][:200].replace("\n", " ") + "...")
    print("Meta:", metas[i])

print("\nDone.")
