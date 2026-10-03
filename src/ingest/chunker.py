import re

REF_HINT = re.compile(r"doi\.org|arxiv:\s?\d{4}\.\d+|https?://|\[\d+\]\s+[A-Z]", re.I)


def looks_like_references(text):
    return len(REF_HINT.findall(text)) >= 4

SECTION_MAP = {
    "experiments": "experiment", "conclusions": "conclusion",
    "methods": "method", "methodology": "method",
    "materials and methods": "method", "result": "results",
}


def chunk_paper(paper, max_words=220, overlap=40, min_words=30):
    chunks, buf, pages = [], [], []
    state = {"sec": None, "fresh": 0}   # fresh = new words since last flush

    def flush():
        if state["fresh"] >= min_words and not looks_like_references(" ".join(buf)):
            chunks.append({
                "chunk_id": f"{paper.paper_id}_{len(chunks)}",
                "paper_id": paper.paper_id,
                "title": paper.title,
                "section": state["sec"],
                "page_start": min(pages),
                "page_end": max(pages),
                "text": " ".join(buf),
            })
        state["fresh"] = 0

    for page, section, text in paper.blocks:
        if section == "front_matter":       # authors, affiliations: noise
            continue
        section = SECTION_MAP.get(section, section)

        if section != state["sec"]:         # never mix two sections in one chunk
            flush()
            buf.clear(); pages.clear()
            state["sec"] = section

        for word in text.split():
            buf.append(word); pages.append(page)
            state["fresh"] += 1
            if len(buf) >= max_words:
                flush()
                buf[:] = buf[-overlap:]     # keep a little overlap for context
                pages[:] = pages[-overlap:]
    flush()
    return chunks