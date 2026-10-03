import json
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid import Retriever

QIDS = ["q029", "q050", "q051"]
qs = {q["qid"]: q for q in json.load(open("data/eval/questions.json"))}
chunks = {c["chunk_id"]: c for c in json.load(open("data/processed/chunks.json"))}
ret = Retriever(VectorStore.load())

for qid in QIDS:
    q = qs[qid]
    print("\n" + "=" * 70)
    print(qid, q["question"])
    print("\nGOLD", q["gold_chunk_id"], ":\n", chunks[q["gold_chunk_id"]]["text"][:700])
    for r in ret.search(q["question"], k=3, mode="hybrid_rerank"):
        print("\nGOT", r["chunk_id"], ":\n", r["text"][:700])