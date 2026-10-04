import json, os, re
from src.rag.generate import generate

FIELDS = ["problem", "methodology", "dataset", "results", "key_findings", "limitations"]
KEEP_SECTIONS = {"introduction", "method", "experiment", "results",
                 "discussion", "limitations", "conclusion", "related work"}
OUT_DIR = "data/analysis"

SYSTEM = "You extract structured information from research papers. Use only the text provided."
PROMPT = """From the paper text below, return ONLY a JSON object with these keys:
problem, methodology, dataset, results, key_findings, limitations.
Rules:
- Each value is 1-3 sentences, using only the text provided.
- If the text does not state something, use exactly "Not stated".
- Do not guess or use outside knowledge. Add page numbers like (p.5) where possible.
- No markdown and no code fences.

Title: {title}
Abstract: {abstract}

Text:
{body}
"""


def build_body(chunks, max_words=5000):
    parts, words = [], 0
    for c in chunks:
        if c["section"] not in KEEP_SECTIONS:
            continue
        words += len(c["text"].split())
        if words > max_words:
            break
        parts.append(f"[p.{c['page_start']}] {c['text']}")
    return "\n".join(parts)


def parse_json(text):
    text = re.sub(r"```(?:json)?", "", text)
    return json.loads(text[text.index("{"): text.rindex("}") + 1])


def analyze_paper(paper, chunks):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = f"{OUT_DIR}/{paper['paper_id']}.json"
    if os.path.exists(path):
        return json.load(open(path))
        parsed = None
    for attempt in range(3):
        raw = generate(SYSTEM, PROMPT.format(title=paper["title"],
                                             abstract=paper["abstract"],
                                             body=build_body(chunks)),
                       temperature=0.2 * attempt)
        try:
            parsed = parse_json(raw)
            break
        except Exception:
            print("   bad JSON, retrying...")
    if parsed is None:
        raise ValueError("model did not return valid JSON")
    data = {k: str(parsed.get(k, "Not stated")) for k in FIELDS}
    data.update({"paper_id": paper["paper_id"], "title": paper["title"]})
    json.dump(data, open(path, "w"), indent=2)
    return data