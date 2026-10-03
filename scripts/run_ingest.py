import glob, json, os
from tqdm import tqdm
from src.ingest.parser import parse_pdf
from src.ingest.chunker import chunk_paper

os.makedirs("data/processed", exist_ok=True)
papers, all_chunks = [], []

for path in tqdm(sorted(glob.glob("data/raw_pdfs/*.pdf"))):
    try:
        p = parse_pdf(path)
        chunks = chunk_paper(p)
    except Exception as e:
        print("FAILED:", path, "->", e)
        continue
    if not chunks:
        print("NO TEXT (maybe scanned):", path)
        continue
    papers.append({"paper_id": p.paper_id, "title": p.title,
                   "abstract": p.abstract, "n_pages": max(b[0] for b in p.blocks),
                   "n_chunks": len(chunks)})
    all_chunks.extend(chunks)

json.dump(papers, open("data/processed/papers.json", "w"), indent=2)
json.dump(all_chunks, open("data/processed/chunks.json", "w"), indent=2)

words = [len(c["text"].split()) for c in all_chunks]
print(f"\nPapers: {len(papers)} | Chunks: {len(all_chunks)} | "
      f"Avg words/chunk: {sum(words)/len(words):.0f}")

from collections import Counter
print("Top sections:", Counter(c["section"] for c in all_chunks).most_common(8))
print("\nSAMPLE CHUNK:\n", json.dumps(all_chunks[15], indent=2)[:900])