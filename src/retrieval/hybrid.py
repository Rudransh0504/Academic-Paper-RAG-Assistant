from src.retrieval.bm25 import BM25Store
from src.retrieval.reranker import Reranker


def rrf(rank_lists, k=60):
    """Reciprocal Rank Fusion: combine rankings by position, not raw score."""
    fused = {}
    for lst in rank_lists:
        for rank, r in enumerate(lst, 1):
            entry = fused.setdefault(r["chunk_id"], {**r, "score": 0.0})
            entry["score"] += 1.0 / (k + rank)
    return sorted(fused.values(), key=lambda r: r["score"], reverse=True)


class Retriever:
    def __init__(self, vs, mode="hybrid_rerank", pool=50, rerank_pool=30):
        self.vs = vs
        self.bm25 = BM25Store(vs.chunks)
        self.mode, self.pool, self.rerank_pool = mode, pool, rerank_pool
        self._reranker = None

    def search(self, query, k=30, mode=None):
        mode = mode or self.mode
        if mode == "vector":
            return self.vs.search(query, k)
        if mode == "bm25":
            return self.bm25.search(query, k)

        fused = rrf([self.vs.search(query, self.pool),
                     self.bm25.search(query, self.pool)])
        if mode == "hybrid":
            return fused[:k]

        # hybrid_rerank
        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker.rerank(query, fused[:self.rerank_pool], k=k)