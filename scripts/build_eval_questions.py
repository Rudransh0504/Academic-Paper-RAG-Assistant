import json, os, random, time
from src.rag.generate import generate

random.seed(7)
os.makedirs("data/eval", exist_ok=True)
OUT = "data/eval/candidates.json"

chunks = json.load(open("data/processed/chunks.json"))
def usable(c):
    t = c["text"]
    body = t.replace(" ", "")
    numeric = sum(ch.isdigit() or ch == "." for ch in body) / max(len(body), 1)
    letters = sum(ch.isalpha() for ch in body) / max(len(body), 1)
    return (t[0].isupper()                      # not a mid-sentence start
            and numeric < 0.15 and letters > 0.65
            and "N/A" not in t and t.count("{") < 2)
pool = [c for c in chunks
        if usable(c)
        and c["section"] in ("introduction", "method", "results", "experiment",
                             "limitations", "discussion", "related work")
        and len(c["text"].split()) >= 150]
random.shuffle(pool)

picked, per_paper = [], {}
for c in pool:
    if per_paper.get(c["paper_id"], 0) < 3:       # spread across papers
        picked.append(c)
        per_paper[c["paper_id"]] = per_paper.get(c["paper_id"], 0) + 1
    if len(picked) == 100:
        break

done = json.load(open(OUT)) if os.path.exists(OUT) else []
done_ids = {d["gold_chunk_id"] for d in done}

SYSTEM = "You write evaluation questions for a search system over research papers."
RULES = ("Write ONE specific question that the passage below answers.\n"
         "Rules:\n"
         "- The question MUST contain a specific proper name that appears in the passage: "
         "a method, framework, dataset, benchmark, or system introduced or described there.\n"
         "- It must make sense on its own. Never say 'this paper', 'the authors', 'the study', "
         "or refer to unnamed things like 'the model' or 'the framework'.\n"
         "- Do not copy phrases of 4 or more words from the passage.\n"
         "- It must be fully answerable from the passage text alone. Not yes/no.\n"
         "- If the passage has no such named item, or is mostly a table, equation, prompt "
         "template or reference list, reply with exactly: SKIP\n"
         "Reply with only the question.\n\nPassage:\n")

for c in picked:
    if c["chunk_id"] in done_ids:
        continue
    try:
        q = generate(SYSTEM, RULES + c["text"]).strip().strip('"')
        if q.upper().startswith("SKIP"):
            print("  skipped (no named item):", c["chunk_id"])
            continue
    except Exception as e:
        print("skip", c["chunk_id"], e)
        continue
    done.append({"qid": f"q{len(done)+1:03d}", "question": q,
                 "gold_chunk_id": c["chunk_id"], "gold_paper_id": c["paper_id"],
                 "gold_pages": [c["page_start"], c["page_end"]],
                 "section": c["section"]})
    json.dump(done, open(OUT, "w"), indent=2)     # save after every question
    print(len(done), q)
    time.sleep(2)                                 # stay under rate limits