# Academic Paper Intelligence Platform

RAG research assistant over academic papers: hybrid retrieval (FAISS + BM25 + RRF),
cross-encoder reranking, and citation-grounded answers.

## Run
1. pip install -r requirements.txt
2. Copy `.env.example` to `.env` and add your Gemini API key
3. python scripts/download_arxiv.py "retrieval augmented generation" 40
4. python -m scripts.run_ingest
5. python -m scripts.build_index
6. python -m scripts.ask_demo