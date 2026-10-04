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

    def refresh(self):
        """Call after chunks are added or removed so BM25 sees the change."""
        self.bm25 = BM25Store(self.vs.chunks)

    def search(self, query, k=30, mode=None, paper_ids=None):
        mode = mode or self.mode
        ids = set(paper_ids) if paper_ids else None

        def fetch(store, n):
            if ids is None:
                return store.search(query, n)
            hits = store.search(query, len(self.vs.chunks))
            return [r for r in hits if r["paper_id"] in ids][:n]

        if mode == "vector":
            return fetch(self.vs, k)
        if mode == "bm25":
            return fetch(self.bm25, k)

        fused = rrf([fetch(self.vs, self.pool), fetch(self.bm25, self.pool)])
        if mode == "hybrid" or not fused:
            return fused[:k]

        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker.rerank(query, fused[:self.rerank_pool], k=k)