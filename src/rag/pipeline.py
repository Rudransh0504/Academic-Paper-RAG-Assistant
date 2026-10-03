import re
from src.rag.generate import generate

SYSTEM = """You are a research assistant answering questions about academic papers.
Rules:
1. Use ONLY the numbered context passages provided. Do not use outside knowledge.
2. After every claim, cite the passage number(s) in square brackets, like [1] or [2][3].
3. If the passages do not contain the answer, reply exactly: "The provided papers do not contain enough information to answer this."
4. Be concise and precise. Do not cite a passage that does not support the claim."""


def diversify(results, k=6, per_paper=2):
    """Keep the best chunks but cap how many come from one paper."""
    picked, count = [], {}
    for r in results:
        if count.get(r["paper_id"], 0) < per_paper:
            picked.append(r)
            count[r["paper_id"]] = count.get(r["paper_id"], 0) + 1
        if len(picked) == k:
            break
    return picked


def build_context(chunks):
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] ({c['title'][:70]}, {c['section']}, p.{c['page_start']}-{c['page_end']})\n{c['text']}")
    return "\n\n".join(parts)


def ask(vs, question, k=6):
    candidates = vs.search(question, k=30)
    chunks = diversify(candidates, k=k)
    user_prompt = f"Context passages:\n\n{build_context(chunks)}\n\nQuestion: {question}\n\nAnswer:"
    answer = generate(SYSTEM, user_prompt)

    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
    sources = [{"ref": n, **chunks[n - 1]} for n in cited if 1 <= n <= len(chunks)]
    return {"question": question, "answer": answer, "sources": sources, "retrieved": chunks}