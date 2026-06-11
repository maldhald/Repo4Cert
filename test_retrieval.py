from retriever import Retriever

def main():
    r = Retriever()

    queries = [
        "battery life of iPhone 15",
        "does Galaxy S24 support DeX",
        "Pixel 8 update policy",
        "OnePlus 12 charging speed",
        "iPhone 14 warranty coverage"
    ]

    for q in queries:
        print(f"\n=== QUERY: {q} ===")
        results = r.retrieve(q, top_k=3)
        for i, (doc, meta) in enumerate(results, start=1):
            print(f"\nResult {i}:")
            print(f"Source: {meta.get('source')}")
            print(f"Snippet: {doc[:300]}...")

if __name__ == "__main__":
    main()
