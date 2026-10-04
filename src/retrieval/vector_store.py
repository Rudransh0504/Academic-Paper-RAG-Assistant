import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class VectorStore:
    def __init__(self, model_name=MODEL):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def build(self, chunks):
        self.chunks = chunks
        emb = self.model.encode(
            [c["text"] for c in chunks],
            batch_size=32, normalize_embeddings=True, show_progress_bar=True)
        emb = np.asarray(emb, dtype="float32")
        self.index = faiss.IndexFlatIP(emb.shape[1])   # inner product = cosine (vectors normalized)
        self.index.add(emb)

    def search(self, query, k=5):
        q = self.model.encode([QUERY_PREFIX + query], normalize_embeddings=True)
        scores, ids = self.index.search(np.asarray(q, dtype="float32"), k)
        return [{**self.chunks[i], "score": float(s)}
                for s, i in zip(scores[0], ids[0])]

    def save(self, folder="data/index"):
        faiss.write_index(self.index, f"{folder}/faiss.index")
        json.dump(self.chunks, open(f"{folder}/chunks.json", "w"))

    def add(self, new_chunks):
        emb = self.model.encode([c["text"] for c in new_chunks],
                                batch_size=32, normalize_embeddings=True)
        self.index.add(np.asarray(emb, dtype="float32"))
        self.chunks.extend(new_chunks)

    def remove_paper(self, paper_id):
        keep = [i for i, c in enumerate(self.chunks) if c["paper_id"] != paper_id]
        vecs = self.index.reconstruct_n(0, self.index.ntotal)[keep]
        self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(np.ascontiguousarray(vecs, dtype="float32"))
        self.chunks = [self.chunks[i] for i in keep]

    @classmethod
    def load(cls, folder="data/index"):
        vs = cls()
        vs.index = faiss.read_index(f"{folder}/faiss.index")
        vs.chunks = json.load(open(f"{folder}/chunks.json"))
        return vs

    