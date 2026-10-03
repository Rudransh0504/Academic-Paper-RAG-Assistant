from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid import Retriever

ret = Retriever(VectorStore.load())
q = input("Query: ")
for r in ret.search(q, k=5, mode="hybrid_rerank"):
    print(f"\n{r['paper_id']} p.{r['page_start']} | {r['section']}")
    print(r["text"][:500])