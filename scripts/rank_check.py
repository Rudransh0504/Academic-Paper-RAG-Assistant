from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid import Retriever

ret = Retriever(VectorStore.load())
q = "How does Ragas compute the faithfulness score?"
gold = {"2309.15217v2_9", "2309.15217v2_10"}

for mode in ("vector", "bm25", "hybrid", "hybrid_rerank"):
    res = ret.search(q, k=50, mode=mode)
    ranks = [i for i, r in enumerate(res, 1) if r["chunk_id"] in gold]
    print(f"{mode:14} first gold rank:", ranks[0] if ranks else "not found")