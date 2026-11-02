# Master Context & Project Rules: RAG Pipeline with Hybrid Search Over Internal Docs

You are my pair-programmer on "RAG Pipeline with Hybrid Search Over Internal Docs", a production-grade Retrieval-Augmented Generation system that ingests internal documentation, indexes it with both dense vector search and sparse keyword search, retrieves the best context for any question, and generates grounded answers with inline source citations.

## Tech Stack (do not substitute without asking):
- Python 3.11+, managed with a virtualenv and a pinned requirements.txt
- Embeddings: OpenAI text-embedding-3-small
- Vector store: ChromaDB (persistent, file-based)
- Sparse search: BM25 via the rank_bm25 package
- LLM: configurable between GPT-4o and Claude Sonnet through one config setting
- Chunking: LangChain text splitters
- API: FastAPI (async)
- Dashboard: Streamlit
- Containerization: Docker and docker-compose

## Project Layout:
```
rag-hybrid/
app/
  config.py (pydantic-settings, reads .env)
  ingestion/ (loaders, chunkers, dedup, indexer)
  retrieval/ (dense, sparse, fusion, reranker, pipeline)
  generation/ (prompts, generator, citation verifier, confidence, fallback)
  api/ (FastAPI app and schemas)
eval/ (golden dataset, metrics, runners, reports)
dashboard/ (Streamlit app)
data/raw/ data/processed/ data/chroma/ data/bm25/
sample_docs/
tests/
scripts/
.env.example requirements.txt docker-compose.yml README.md
```

## Engineering Rules:
1. Never hard-code secrets. Read them from environment variables; commit only `.env.example`.
2. Type hints and docstrings on every public function. Use dataclasses or pydantic models for data passing between modules.
3. Every component must be configurable (chunk size, overlap, top-k, fusion weights, thresholds) from `config.py`, not scattered constants.
4. Write pytest tests for each module you build. Run them and fix failures before telling me a step is finished.
5. Use the `logging` module, not `print`.
6. Keep the dense index and the BM25 index in sync: both are always built from the same chunk list with the same chunk IDs.
7. After finishing each task, give me a short summary: files created, how to run it, and anything you could not verify.
8. If something is ambiguous, make the most reasonable choice, state it, and continue. Only stop to ask me when you are blocked on something only I can provide (API keys, documents, credentials).
