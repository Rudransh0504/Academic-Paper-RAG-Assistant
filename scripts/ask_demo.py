from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid import Retriever
from src.rag.pipeline import ask

vs = Retriever(VectorStore.load())
print("Ask a question (or 'q' to quit)\n")

while True:
    q = input("> ").strip()
    if q.lower() in ("q", "quit", ""):
        break
    out = ask(vs, q)
    print("\nANSWER:\n", out["answer"])
    print("\nSOURCES:")
    for s in out["sources"]:
        print(f"  [{s['ref']}] {s['title'][:70]} | {s['section']} | p.{s['page_start']}-{s['page_end']} | {s['paper_id']}")
    print()