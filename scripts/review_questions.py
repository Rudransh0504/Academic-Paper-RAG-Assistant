import json, os

cands = json.load(open("data/eval/candidates.json"))
chunks = {c["chunk_id"]: c for c in json.load(open("data/processed/chunks.json"))}
OUT = "data/eval/questions.json"
SEEN = "data/eval/reviewed_ids.json"
kept = json.load(open(OUT)) if os.path.exists(OUT) else []
reviewed = set(json.load(open(SEEN))) if os.path.exists(SEEN) else set()


def bad_chunk(text):
    body = text.replace(" ", "")
    numeric = sum(ch.isdigit() or ch == "." for ch in body) / max(len(body), 1)
    return numeric > 0.2 or text.count("{") >= 2    # tables, prompt templates


for q in cands:
    if q["qid"] in reviewed:
        continue
    c = chunks[q["gold_chunk_id"]]
    if bad_chunk(c["text"]):
        print(q["qid"], "auto-dropped (table or prompt template)")
        reviewed.add(q["qid"])
        json.dump(sorted(reviewed), open(SEEN, "w"))
        continue

    print("\n" + "=" * 70)
    print(q["qid"], "|", c["title"][:70], "| p.", c["page_start"])
    print("\nQUESTION:", q["question"])
    print("\nFULL PASSAGE:\n" + c["text"])
    ans = input("\nkeep (k) / drop (d) / quit (q)? ").strip().lower()
    if ans == "q":
        break
    if ans == "k":
        kept.append(q)
    reviewed.add(q["qid"])
    json.dump(kept, open(OUT, "w"), indent=2)
    json.dump(sorted(reviewed), open(SEEN, "w"))

print("Kept so far:", len(kept))