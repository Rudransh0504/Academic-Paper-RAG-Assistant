import json, time
from collections import defaultdict
from src.analysis.summarize import analyze_paper

papers = json.load(open("data/processed/papers.json"))
by_paper = defaultdict(list)
for c in json.load(open("data/processed/chunks.json")):
    by_paper[c["paper_id"]].append(c)

for i, p in enumerate(papers, 1):
    try:
        analyze_paper(p, by_paper[p["paper_id"]])
        print(i, "ok", p["paper_id"])
    except Exception as e:
        print(i, "FAILED", p["paper_id"], str(e)[:80])
    time.sleep(2)