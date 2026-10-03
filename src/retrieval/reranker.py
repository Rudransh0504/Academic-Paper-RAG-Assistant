from sentence_transformers import CrossEncoder

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name=RERANK_MODEL):
        self.model = CrossEncoder(model_name, max_length=512)

    def rerank(self, query, results, k=10):
        scores = self.model.predict([(query, r["text"]) for r in results])
        ranked = sorted(zip(scores, results), key=lambda x: x[0], reverse=True)
        return [{**r, "score": float(s)} for s, r in ranked[:k]]