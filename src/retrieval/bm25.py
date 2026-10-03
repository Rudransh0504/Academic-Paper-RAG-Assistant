import re
from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP = set("the a an of and or to in is are for on with by as at from that this "
           "we our be it its which can was were has have not but also".split())


def tokenize(text):
    # splitting on non-alphanumerics also splits "retrieval-augmented" and "unmodi-fied"
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP]


class BM25Store:
    def __init__(self, chunks):
        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    def search(self, query, k=5):
        scores = self.bm25.get_scores(tokenize(query))
        top = scores.argsort()[::-1][:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]