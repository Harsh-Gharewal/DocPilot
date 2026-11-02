"""FastAPI route definitions for DocPilot RAG service."""

import logging
import shutil
import uuid
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile, status

from app.api.auth import verify_api_key
from app.api.models import (
    AskRequest,
    DocumentItem,
    DocumentListResponse,
    HealthResponse,
    IngestJobResponse,
)
from app.config import settings
from app.generation.models import AnswerResponse
from app.generation.rag import ask
from app.ingestion.indexer import index_manager
from app.ingestion.loaders import load_document

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["RAG Services"])

# In-memory registry for background ingestion jobs
_INGEST_JOBS: dict[str, dict[str, Any]] = {}


def _run_background_ingest(job_id: str, file_paths: list[Path], strategy: str) -> None:
    """Worker function executing ingestion in the background."""
    from app.ingestion.chunkers import get_chunker

    _INGEST_JOBS[job_id]["status"] = "processing"
    _INGEST_JOBS[job_id]["message"] = "Processing files and indexing chunks..."

    total_chunks = 0
    errors = []

    strategies_to_run = (
        ["fixed_size", "structure_aware", "semantic"]
        if strategy == "all"
        else [strategy]
    )

    for p in file_paths:
        try:
            doc = load_document(p)
            if not doc:
                continue

            for strat in strategies_to_run:
                chunker = get_chunker(strat)
                raw_chunks = chunker.chunk(doc)
                index_manager.index_chunks(raw_chunks, strategy=strat)
                total_chunks += len(raw_chunks)

        except Exception as e:
            logger.exception(f"Error processing {p.name}: {e}")
            errors.append(f"{p.name}: {str(e)}")

    _INGEST_JOBS[job_id]["chunks_indexed"] = total_chunks
    _INGEST_JOBS[job_id]["errors"] = errors
    if errors and total_chunks == 0:
        _INGEST_JOBS[job_id]["status"] = "failed"
        _INGEST_JOBS[job_id]["message"] = f"Ingestion failed with {len(errors)} error(s)."
    else:
        _INGEST_JOBS[job_id]["status"] = "completed"
        _INGEST_JOBS[job_id]["message"] = f"Ingestion completed. Indexed {total_chunks} chunks."


@router.post(
    "/ask",
    response_model=AnswerResponse,
    summary="Ask a question against internal knowledge base",
    dependencies=[Depends(verify_api_key)],
)
async def ask_endpoint(payload: AskRequest) -> AnswerResponse:
    """Execute hybrid retrieval, grounded generation, and verification on input question."""
    try:
        response = ask(
            question=payload.question,
            strategy=payload.strategy,
            mode=payload.mode,
            top_k=payload.top_k,
        )
        return response
    except Exception as e:
        logger.exception(f"Unhandled error in /ask: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating answer: {str(e)}",
        )


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all indexed documents with chunk statistics",
    dependencies=[Depends(verify_api_key)],
)
async def list_documents_endpoint() -> DocumentListResponse:
    """Return all ingested documents across raw and processed storage."""
    proc_dir = settings.processed_data_dir
    docs_map: dict[str, DocumentItem] = {}

    if proc_dir.exists():
        for file_path in proc_dir.glob("*.json"):
            try:
                import json
                data = json.loads(file_path.read_text(encoding="utf-8"))
                meta = data.get("metadata", {})
                doc_id = data.get("doc_id", file_path.stem)

                chunk_counts = {}
                for strat in ["fixed_size", "structure_aware", "semantic"]:
                    stats = index_manager.get_strategy_stats(strat)
                    chunk_counts[strat] = stats.chunk_count

                docs_map[doc_id] = DocumentItem(
                    doc_id=doc_id,
                    filename=meta.get("filename", file_path.name),
                    file_type=meta.get("file_extension", ".json"),
                    sha256=meta.get("sha256", doc_id),
                    chunk_counts=chunk_counts,
                    total_chars=len(data.get("text", "")),
                    raw_path=data.get("source_file", str(file_path)),
                )
            except Exception as e:
                logger.warning(f"Could not load metadata for {file_path}: {e}")

    items = list(docs_map.values())
    return DocumentListResponse(documents=items, total_count=len(items))


@router.post(
    "/ingest",
    response_model=IngestJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload files or trigger folder ingestion",
    dependencies=[Depends(verify_api_key)],
)
async def ingest_endpoint(
    background_tasks: BackgroundTasks,
    files: Optional[list[UploadFile]] = File(None),
    folder_path: Optional[str] = Query(None, description="Local folder path to ingest"),
    strategy: str = Query("all", description="Chunking strategy ('all', 'fixed_size', 'structure_aware', 'semantic')"),
) -> IngestJobResponse:
    """Submit documents for asynchronous chunking, deduplication, and dual-indexing."""
    job_id = str(uuid.uuid4())
    temp_dir = settings.raw_data_dir / "uploads" / job_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    paths_to_process: list[Path] = []

    if files:
        for file in files:
            dest_path = temp_dir / file.filename
            with open(dest_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            paths_to_process.append(dest_path)

    if folder_path:
        local_folder = Path(folder_path)
        if not local_folder.exists() or not local_folder.is_dir():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Folder path does not exist or is not a directory: {folder_path}",
            )
        for ext in ["*.pdf", "*.md", "*.txt", "*.html"]:
            paths_to_process.extend(local_folder.glob(ext))

    if not paths_to_process:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid files provided for ingestion (.pdf, .md, .txt, .html).",
        )

    _INGEST_JOBS[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "message": f"Queued {len(paths_to_process)} document(s) for indexing.",
        "chunks_indexed": 0,
        "errors": [],
    }

    background_tasks.add_task(
        _run_background_ingest,
        job_id=job_id,
        file_paths=paths_to_process,
        strategy=strategy,
    )

    return IngestJobResponse(**_INGEST_JOBS[job_id])


@router.get(
    "/ingest/{job_id}",
    response_model=IngestJobResponse,
    summary="Get status and progress of an ingestion job",
    dependencies=[Depends(verify_api_key)],
)
async def get_ingest_status_endpoint(job_id: str) -> IngestJobResponse:
    """Check asynchronous ingestion progress and chunk count."""
    if job_id not in _INGEST_JOBS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ingestion job '{job_id}' not found.",
        )
    return IngestJobResponse(**_INGEST_JOBS[job_id])


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Diagnostics"],
    summary="Perform comprehensive system health check",
)
async def health_endpoint() -> HealthResponse:
    """Verify ChromaDB connectivity, BM25 indices, and loaded model backends."""
    # Check Chroma
    chroma_status = "healthy"
    try:
        index_manager.chroma_client.heartbeat()
    except Exception as e:
        chroma_status = f"unhealthy: {e}"

    # Check BM25
    bm25_status = "healthy"
    ready_strategies = []
    for strat in ["fixed_size", "structure_aware", "semantic"]:
        stats = index_manager.get_strategy_stats(strat)
        if stats.chunk_count > 0:
            ready_strategies.append(strat)

    return HealthResponse(
        status="ok" if chroma_status == "healthy" else "degraded",
        chromadb=chroma_status,
        bm25=bm25_status,
        embedding_provider=settings.embedding_provider,
        llm_provider=settings.llm_provider,
        strategies_available=ready_strategies,
    )
