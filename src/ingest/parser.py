import re
import fitz  # PyMuPDF
import unicodedata
from dataclasses import dataclass, field

SECTION_RE = re.compile(
    r"^((\d+(\.\d+)*|[IVX]+|[A-Z])\.?\s+)?"
    r"(?P<name>abstract|introduction|related work|background|"
    r"methods?|methodology|materials and methods|experiments?|results?|"
    r"discussion|conclusions?|limitations|references|bibliography|"
    r"acknowledg(?:e)?ments?|appendix|appendices)\b", re.I)

ARXIV_RE = re.compile(r"^arxiv:", re.I)
ABSTRACT_RE = re.compile(r"^abstract\b[\s\.\:\-–—]*(.*)", re.I | re.S)


@dataclass
class ParsedPaper:
    paper_id: str
    title: str = ""
    abstract: str = ""
    blocks: list = field(default_factory=list)      # (page, section, text)
    references: list = field(default_factory=list)


def parse_pdf(path: str) -> ParsedPaper:
    doc = fitz.open(path)
    name = path.replace("\\", "/").split("/")[-1]
    paper = ParsedPaper(paper_id=name.removesuffix(".pdf"))

    raw = []  # (page_number, text, font_size)
    for pno, page in enumerate(doc, 1):
        for b in page.get_text("dict")["blocks"]:
            if b["type"] != 0:      # skip images
                continue
            spans = [s for l in b["lines"] for s in l["spans"]]
            text = " ".join(
                "".join(s["text"] for s in line["spans"]) for line in b["lines"]
            ).strip()
            text = re.sub(r"-\s+", "-", text)   # keep hyphens: retrieval- augmented -> retrieval-augmented
            text = unicodedata.normalize("NFKC", text)   # fix ligatures like ﬀ
            if text:
                raw.append((pno, text, max(s["size"] for s in spans)))

       # Title guess: biggest font on page 1, ignoring the arXiv stamp
    page1 = [r for r in raw
             if r[0] == 1 and not ARXIV_RE.match(r[1]) and len(r[1]) > 10]
    if page1:
        biggest = max(r[2] for r in page1)
        paper.title = " ".join(r[1] for r in page1 if abs(r[2] - biggest) < 0.5)

    section = "front_matter"
    for pno, text, _ in raw:
        t = text.strip()

        # "Abstract" heading, possibly with the abstract text in the same block
        a = ABSTRACT_RE.match(t)
        if a and section == "front_matter":
            section = "abstract"
            rest = a.group(1).strip()
            if rest:
                paper.abstract += " " + rest
                paper.blocks.append((pno, "abstract", rest))
            continue

        m = SECTION_RE.match(t)
        if len(t) < 80 and m:
            section = m.group("name").lower()
            continue

        if section in ("references", "bibliography"):
            paper.references.append(text)
        elif section.startswith("acknowledg") or section.startswith("appendi"):
            pass    # skip these sections entirely
        else:
            paper.blocks.append((pno, section, text))
        if section == "abstract":
            paper.abstract += " " + text

    paper.abstract = paper.abstract.strip()[:3000]   # safety cap

    # Fallback: no "Abstract" heading found, so guess the first long front-matter block
    if not paper.abstract:
        long_fm = [b for b in paper.blocks if b[1] == "front_matter" and len(b[2]) > 400]
        if long_fm:
            paper.abstract = long_fm[0][2]

    return paper