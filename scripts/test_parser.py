import glob
from src.ingest.parser import parse_pdf

for path in glob.glob("data/raw_pdfs/*.pdf")[:6]:
    p = parse_pdf(path)
    print("-" * 60)
    print("ID:      ", p.paper_id)
    print("TITLE:   ", p.title[:100])
    print("ABSTRACT:", len(p.abstract), "chars |", p.abstract[:120])
    print("SECTIONS:", sorted(set(b[1] for b in p.blocks)))