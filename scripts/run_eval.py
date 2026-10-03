import json
import pandas as pd
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid import Retriever

qs = json.load(open("data/eval/questions.json"))
ret = Retriever(VectorStore.load())
KS = [1, 3, 5, 10]
MODES = ["vector", "bm25", "hybrid", "hybrid_rerank"]
DEPTH = 30   # each mode returns its top 30; MRR is capped at this depth


def first_rank(results, match):
    for i, r in enumerate(results, 1):
        if match(r):
            return i
    return None


def recall(ranks, k):
    return sum(1 for r in ranks if r and r <= k) / len(ranks)


def mrr(ranks):
    return sum(1 / r for r in ranks if r) / len(ranks)


rows = []
for mode in MODES:
    chunk_ranks, paper_ranks = [], []
    for q in qs:
        res = ret.search(q["question"], k=DEPTH, mode=mode)
        chunk_ranks.append(first_rank(res, lambda r: r["chunk_id"] == q["gold_chunk_id"]))
        paper_ranks.append(first_rank(res, lambda r: r["paper_id"] == q["gold_paper_id"]))

    row = {"mode": mode}
    for k in KS:
        row[f"chunk_R@{k}"] = round(recall(chunk_ranks, k), 3)
    row["chunk_MRR"] = round(mrr(chunk_ranks), 3)
    row["paper_R@5"] = round(recall(paper_ranks, 5), 3)
    rows.append(row)
    print("done", mode)

df = pd.DataFrame(rows)
print("\n", df.to_string(index=False))
df.to_csv("data/eval/results.csv", index=False)