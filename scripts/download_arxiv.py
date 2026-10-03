import arxiv, os, sys

QUERY = sys.argv[1] if len(sys.argv) > 1 else "retrieval augmented generation"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 40
OUT = "data/raw_pdfs"
os.makedirs(OUT, exist_ok=True)

client = arxiv.Client(page_size=50, delay_seconds=3, num_retries=3)
search = arxiv.Search(query=QUERY, max_results=N,
                      sort_by=arxiv.SortCriterion.Relevance)

for r in client.results(search):
    arxiv_id = r.get_short_id().replace("/", "_")
    path = os.path.join(OUT, f"{arxiv_id}.pdf")
    if os.path.exists(path):
        continue
    r.download_pdf(dirpath=OUT, filename=f"{arxiv_id}.pdf")
    print("downloaded", arxiv_id, "-", r.title[:70])