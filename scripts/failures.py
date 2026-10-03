import json
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid import Retriever

qs = json.load(open("data/eval/questions.json"))
ret = Retriever(VectorStore.load())
for q in qs:
    res = ret.search(q["question"], k=5, mode="hybrid_rerank")
    if q["gold_chunk_id"] not in [r["chunk_id"] for r in res]:
        print(q["qid"], "|", q["question"])
        print("   gold:", q["gold_chunk_id"], "| got:", [r["chunk_id"] for r in res[:3]])