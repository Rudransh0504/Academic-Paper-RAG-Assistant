import csv, json, os, re, threading
from typing import List, Optional
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.analysis.summarize import analyze_paper
from src.ingest.chunker import chunk_paper
from src.ingest.parser import parse_pdf
from src.rag.generate import generate
from src.rag.pipeline import ask
from src.retrieval.hybrid import Retriever
from src.retrieval.vector_store import VectorStore

FIELDS = ["problem", "methodology", "dataset", "results", "key_findings", "limitations"]
MODES = {"vector", "bm25", "hybrid", "hybrid_rerank"}
UPLOAD_DIR = "data/uploads"
PDF_DIR = f"{UPLOAD_DIR}/pdfs"
os.makedirs(PDF_DIR, exist_ok=True)
lock = threading.Lock()

app = FastAPI(title="Academic Paper Intelligence Platform")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

retriever = Retriever(VectorStore.load())
base_papers = json.load(open("data/processed/papers.json"))


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []


uploaded_papers = load_json(f"{UPLOAD_DIR}/papers.json")
uploaded_chunks = load_json(f"{UPLOAD_DIR}/chunks.json")
if uploaded_chunks:                      # re-add earlier uploads on startup
    retriever.vs.add(uploaded_chunks)
    retriever.refresh()


def save_uploads():
    with open(f"{UPLOAD_DIR}/papers.json", "w") as f:
        json.dump(uploaded_papers, f, indent=2)
    with open(f"{UPLOAD_DIR}/chunks.json", "w") as f:
        json.dump(uploaded_chunks, f)


class SearchReq(BaseModel):
    query: str
    mode: str = "hybrid_rerank"
    k: int = 5
    paper_ids: Optional[List[str]] = None


class AskReq(BaseModel):
    question: str
    mode: str = "hybrid_rerank"
    k: int = 6
    paper_ids: Optional[List[str]] = None


class CompareReq(BaseModel):
    paper_ids: List[str]
    with_summary: bool = False


def check_mode(mode):
    if mode not in MODES:
        raise HTTPException(400, f"mode must be one of {sorted(MODES)}")


def get_analysis(paper_id):
    path = f"data/analysis/{paper_id}.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    meta = next((p for p in uploaded_papers if p["paper_id"] == paper_id), None)
    if meta is None:
        raise HTTPException(404, "no analysis for this paper")
    chunks = [c for c in uploaded_chunks if c["paper_id"] == paper_id]
    try:
        return analyze_paper(meta, chunks)       # summarizes new uploads on demand
    except Exception as e:
        raise HTTPException(503, f"Analysis failed: {str(e)[:100]}")


@app.get("/papers")
def list_papers():
    return [dict(p, uploaded=False) for p in base_papers] + uploaded_papers


@app.post("/upload")
def upload(file: UploadFile = File(...)):
    name = file.filename or "paper.pdf"
    if not name.lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF file.")
    data = file.file.read()
    if len(data) > 25 * 1024 * 1024:
        raise HTTPException(400, "File is larger than 25 MB.")
    base = re.sub(r"[^A-Za-z0-9_-]+", "_", os.path.splitext(name)[0])[:40].strip("_") or "paper"
    with lock:
        taken = {p["paper_id"] for p in base_papers + uploaded_papers}
        pid, n = f"upload_{base}", 2
        while pid in taken:
            pid, n = f"upload_{base}_{n}", n + 1
        path = f"{PDF_DIR}/{pid}.pdf"
        with open(path, "wb") as f:
            f.write(data)
        try:
            parsed = parse_pdf(path)
            if len(parsed.title.strip()) < 8:
                parsed.title = base.replace("_", " ")
            chunks = chunk_paper(parsed)
        except Exception as e:
            os.remove(path)
            raise HTTPException(400, f"Could not read this PDF: {str(e)[:100]}")
        if not chunks:
            os.remove(path)
            raise HTTPException(400, "No readable text found (is it a scanned PDF?).")
        retriever.vs.add(chunks)
        retriever.refresh()
        meta = {"paper_id": pid, "title": parsed.title, "abstract": parsed.abstract,
                "n_pages": max(b[0] for b in parsed.blocks),
                "n_chunks": len(chunks), "uploaded": True}
        uploaded_papers.append(meta)
        uploaded_chunks.extend(chunks)
        save_uploads()
    return meta


@app.delete("/papers/{paper_id}")
def delete_paper(paper_id: str):
    with lock:
        if not any(p["paper_id"] == paper_id for p in uploaded_papers):
            raise HTTPException(404, "Only uploaded papers can be removed.")
        retriever.vs.remove_paper(paper_id)
        retriever.refresh()
        uploaded_papers[:] = [p for p in uploaded_papers if p["paper_id"] != paper_id]
        uploaded_chunks[:] = [c for c in uploaded_chunks if c["paper_id"] != paper_id]
        save_uploads()
        for f in (f"{PDF_DIR}/{paper_id}.pdf", f"data/analysis/{paper_id}.json"):
            if os.path.exists(f):
                os.remove(f)
    return {"deleted": paper_id}


@app.post("/search")
def search(req: SearchReq):
    check_mode(req.mode)
    return [{"paper_id": r["paper_id"], "title": r["title"], "section": r["section"],
             "page_start": r["page_start"], "page_end": r["page_end"],
             "score": r["score"], "text": r["text"]}
            for r in retriever.search(req.query, k=req.k, mode=req.mode, paper_ids=req.paper_ids)]


@app.post("/ask")
def ask_endpoint(req: AskReq):
    check_mode(req.mode)
    try:
        out = ask(retriever, req.question, k=req.k, mode=req.mode, paper_ids=req.paper_ids)
    except Exception as e:
        raise HTTPException(503, f"LLM unavailable: {str(e)[:120]}")
    keep = ("ref", "paper_id", "title", "section", "page_start", "page_end", "text")
    rkeep = ("paper_id", "title", "section", "page_start", "page_end", "score", "text")
    return {"question": out["question"], "answer": out["answer"], "mode": req.mode,
            "sources": [{k: s[k] for k in keep} for s in out["sources"]],
            "retrieved": [{"ref": i, **{k: c[k] for k in rkeep}}
                          for i, c in enumerate(out["retrieved"], 1)]}


@app.get("/analysis/{paper_id}")
def analysis(paper_id: str):
    return get_analysis(paper_id)


@app.post("/compare")
def compare(req: CompareReq):
    if not 2 <= len(req.paper_ids) <= 4:
        raise HTTPException(400, "choose 2 to 4 papers")
    notes = [get_analysis(i) for i in req.paper_ids]
    out = {"papers": [{"paper_id": p["paper_id"], "title": p["title"]} for p in notes],
           "rows": [{"aspect": f, "values": [p[f] for p in notes]} for f in FIELDS]}
    if req.with_summary:
        try:
            out["summary"] = generate(
                "You compare research papers using only the structured notes provided.",
                "Using ONLY these notes, write under 150 words: how the papers relate "
                "(if they address different tasks, say so plainly), the key differences, "
                "and which fits which situation. Only state a similarity if both sets of "
                "notes support it. Do not add facts that are not in the notes.\n\n"
                + json.dumps(notes, indent=1))
        except Exception:
            out["summary"] = None
    return out


@app.get("/eval")
def eval_results():
    path = "data/eval/results.csv"
    if not os.path.exists(path):
        raise HTTPException(404, "Run python -m scripts.run_eval first.")
    with open(path) as f:
        return list(csv.DictReader(f))


app.mount("/", StaticFiles(directory="web", html=True), name="web")   # keep this last