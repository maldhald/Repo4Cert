from pipeline import RAGPipeline

def main():
    rag = RAGPipeline()

    queries = [
        "Which phone has the best battery life?",
        "Does the Galaxy S24 support DeX?",
        "How fast does the OnePlus 12 charge?",
        "What is the warranty on the iPhone 14?",
        "How long will the Pixel 8 receive updates?"
    ]

    for q in queries:
        print(f"\n=== QUERY: {q} ===")
        answer, docs = rag.query(q)

        print("\n--- ANSWER ---")
        print(answer)

        print("\n--- CONTEXT USED ---")
        for d in docs:
            print("\n", d[:200], "...")
        
if __name__ == "__main__":
    main()
