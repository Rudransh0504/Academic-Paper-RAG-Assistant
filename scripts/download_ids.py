import arxiv, os

IDS = [l.strip() for l in open("data/eval/corpus_ids.txt") if l.strip()]
OUT = "data/raw_pdfs"
os.makedirs(OUT, exist_ok=True)

client = arxiv.Client(page_size=50, delay_seconds=3, num_retries=3)
for r in client.results(arxiv.Search(id_list=IDS)):
    name = r.get_short_id().replace("/", "_")
    if not os.path.exists(f"{OUT}/{name}.pdf"):
        r.download_pdf(dirpath=OUT, filename=f"{name}.pdf")
        print("downloaded", name)