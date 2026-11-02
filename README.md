# RAG Pipeline with Hybrid Search Over Internal Docs

A production-grade Retrieval-Augmented Generation (RAG) system that ingests internal documentation, indexes it with both dense vector search and sparse keyword search (BM25), retrieves the best context for any question using Reciprocal Rank Fusion (RRF) and reranking, and generates grounded answers with inline citations verified by an LLM judge.

## Tech Stack
- **Python 3.11+**
- **Embeddings:** OpenAI `text-embedding-3-small`
- **Vector Store:** ChromaDB (persistent, file-based)
- **Sparse Search:** BM25 (`rank_bm25`)
- **LLM:** Configurable between GPT-4o and Claude Sonnet via `LLM_PROVIDER`
- **Chunking:** LangChain text splitters (Fixed size, Structure-aware, Semantic)
- **API:** FastAPI (async)
- **Dashboard:** Streamlit
- **Containerization:** Docker & Docker Compose

## Project Layout
```
app/
  config.py               # Central pydantic-settings configuration
  ingestion/              # Loaders, chunkers, deduplication, indexer
  retrieval/              # Dense, BM25 sparse, RRF fusion, reranker, pipeline
  generation/             # Prompts, generator, citation verifier, confidence scorer, fallback
  api/                    # FastAPI routes and schemas
eval/                     # Golden dataset, automated metrics, evaluation runners, reports
dashboard/                # Streamlit user interface
data/
  raw/                    # Raw uploaded/ingested files
  processed/              # Parsed document JSONs
  chroma/                 # Persistent ChromaDB collections
  bm25/                   # Persisted BM25 indexes
sample_docs/              # 15-30 internal documents (.md, .txt, .html, .pdf)
tests/                    # Pytest test suite
scripts/                  # Helper & seeding scripts
```

## Setup Instructions

### 1. Create and Activate Virtual Environment
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
```bash
cp .env.example .env
# Edit .env and supply your OPENAI_API_KEY (and optionally ANTHROPIC_API_KEY)
```

### 4. Sample Documents
Place your 15 to 30 internal documents into `sample_docs/`.
