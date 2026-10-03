from src.retrieval.vector_store import VectorStore

vs = VectorStore.load()
print("Type a question (or 'q' to quit)\n")

while True:
    query = input("> ").strip()
    if query.lower() in ("q", "quit", ""):
        break
    for r in vs.search(query, k=5):
        print(f"\n[{r['score']:.3f}] {r['paper_id']} | {r['section']} | "
              f"p.{r['page_start']}-{r['page_end']}")
        print("   ", r["title"][:80])
        print("   ", r["text"][:220].replace("\n", " "), "...")
    print()