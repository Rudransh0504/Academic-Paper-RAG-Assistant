import json, sys
from src.rag.generate import generate

FIELDS = ["problem", "methodology", "dataset", "results", "key_findings", "limitations"]
ids = sys.argv[1:]          # e.g. 2309.15217v2 2404.07220v2
papers = [json.load(open(f"data/analysis/{i}.json")) for i in ids]

short = lambda p: p["title"][:30].replace("|", "/")
header = "| Aspect | " + " | ".join(short(p) for p in papers) + " |"
lines = [header, "|---" * (len(papers) + 1) + "|"]
for f in FIELDS:
    cells = [p[f].replace("|", "/").replace("\n", " ") for p in papers]
    lines.append(f"| {f} | " + " | ".join(cells) + " |")
table = "\n".join(lines)
print(table)

summary = generate(
    "You compare research papers using only the structured notes provided.",
    "Using ONLY these notes, write under 150 words: how the papers relate "
    "(if they address different tasks, say so plainly), the key differences, "
    "and which fits which situation. Only state a similarity if both sets of "
    "notes support it. Do not add facts that are not in the notes.\n\n"
    + json.dumps(papers, indent=1))
print("\nCOMPARISON:\n", summary)