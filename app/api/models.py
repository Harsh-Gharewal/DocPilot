"""Pydantic request and response models for DocPilot FastAPI service."""

from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

from app.generation.models import AnswerResponse


class AskRequest(BaseModel):
    """Payload for question answering endpoint."""

    question: str = Field(
        ...,
        min_length=2,
        description="Natural language question to answer",
        json_schema_extra={"example": "What is the minimum medical cover in the health policy?"},
    )
    strategy: Literal["fixed_size", "structure_aware", "semantic"] = Field(
        default="structure_aware",
        description="Chunking strategy to retrieve against",
    )
    mode: Literal["dense_only", "sparse_only", "hybrid", "hybrid_rerank"] = Field(
        default="hybrid_rerank",
        description="Retrieval engine execution mode",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of candidate chunks to retrieve")


class DocumentItem(BaseModel):
    """Metadata summary of an ingested document."""

    doc_id: str = Field(..., description="Unique document ID (hash)")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="Document extension: .pdf, .md, .txt, .html")
    sha256: str = Field(..., description="SHA-256 fingerprint")
    chunk_counts: dict[str, int] = Field(default_factory=dict, description="Number of chunks indexed per strategy")
    total_chars: int = Field(default=0, description="Total characters extracted")
    raw_path: str = Field(..., description="Relative raw file path")


class DocumentListResponse(BaseModel):
    """List of all registered documents in the knowledge base."""

    documents: list[DocumentItem]
    total_count: int


class IngestJobResponse(BaseModel):
    """Status or submission confirmation of an ingestion job."""

    job_id: str = Field(..., description="Unique ingestion job identifier")
    status: Literal["pending", "processing", "completed", "failed"] = Field(
        ..., description="Current status of the ingestion pipeline"
    )
    message: str = Field(default="", description="Descriptive status message")
    chunks_indexed: int = Field(default=0, description="Total chunks indexed across strategies")
    errors: list[str] = Field(default_factory=list, description="List of any errors encountered during processing")


class HealthResponse(BaseModel):
    """System health check and diagnostic status."""

    status: str = Field(default="ok", description="Overall service status")
    chromadb: str = Field(default="healthy", description="ChromaDB vector store connection status")
    bm25: str = Field(default="healthy", description="BM25 index status")
    embedding_provider: str = Field(..., description="Active embedding provider ('openai' or 'local')")
    llm_provider: str = Field(..., description="Configured LLM provider ('openai' or 'anthropic')")
    strategies_available: list[str] = Field(
        default_factory=list, description="Chunking strategies ready for search"
    )
