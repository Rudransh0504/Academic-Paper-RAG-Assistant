import json, os
from src.retrieval.vector_store import VectorStore

os.makedirs("data/index", exist_ok=True)
chunks = json.load(open("data/processed/chunks.json"))

vs = VectorStore()
vs.build(chunks)
vs.save()
print("Indexed", vs.index.ntotal, "chunks")