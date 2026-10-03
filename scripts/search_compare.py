from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid import Retriever

ret = Retriever(VectorStore.load())
print("Query (blank to quit)")
while True:
    q = input("> ").strip()
    if not q:
        break
    for mode in ("vector", "bm25", "hybrid", "hybrid_rerank"):
        print(f"\n=== {mode.upper()} ===")
        for r in ret.search(q, k=5, mode=mode):
            print(f"  {r['paper_id']} | {r['section']} | p.{r['page_start']} | {r['title'][:50]}")
    print()