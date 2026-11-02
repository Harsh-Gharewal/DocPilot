"""Pydantic data models for the ingestion and chunking pipeline."""

from typing import Any, Optional
from pydantic import BaseModel, Field


class Section(BaseModel):
    """Represents a structural section within a document."""

    section_heading: Optional[str] = Field(
        default=None, description="Heading title of the section if detected"
    )
    page_number: Optional[int] = Field(
        default=None, description="1-based page number for PDF documents, None otherwise"
    )
    text: str = Field(..., description="Clean plaintext content of the section")


class Document(BaseModel):
    """Normalized representation of an ingested document."""

    doc_id: str = Field(
        ..., description="Deterministic hash (SHA-256) of raw file content"
    )
    source_file: str = Field(..., description="Relative or absolute path to the original source file")
    text: str = Field(..., description="Full normalized plaintext of the document")
    sections: list[Section] = Field(
        default_factory=list, description="List of document sections with headings and page numbers"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary file metadata (file size, extension, etc.)"
    )


class Chunk(BaseModel):
    """Discrete text chunk ready for dense embedding and sparse indexing."""

    chunk_id: str = Field(
        ..., description="Deterministic unique ID: {doc_id}_{strategy}_{chunk_index:04d}"
    )
    doc_id: str = Field(..., description="Source document identifier")
    source_file: str = Field(..., description="Source file path")
    chunk_index: int = Field(..., description="0-based index of this chunk within document and strategy")
    text: str = Field(..., description="Chunk text content")
    section_heading: Optional[str] = Field(
        default=None, description="Section heading context for this chunk"
    )
    page_number: Optional[int] = Field(
        default=None, description="1-based page number for PDF documents, None otherwise"
    )
    strategy: str = Field(..., description="Chunking strategy used: fixed_size, structure_aware, semantic")
    char_count: int = Field(..., description="Number of characters in the chunk text")


class DedupReport(BaseModel):
    """Report detailing deduplication outcomes during an indexing operation."""

    chunks_seen: int = Field(default=0, description="Total chunks submitted for indexing")
    chunks_inserted: int = Field(default=0, description="Chunks successfully added to indexes")
    duplicates_skipped: int = Field(default=0, description="Chunks skipped due to high cosine similarity")
    duplicate_pairs: list[dict[str, Any]] = Field(
        default_factory=list, description="Pairs of (new_chunk_id, existing_chunk_id, similarity)"
    )


class StrategyStats(BaseModel):
    """Statistics for an indexed chunking strategy."""

    strategy: str = Field(..., description="Strategy name")
    chunk_count: int = Field(default=0, description="Total chunks currently in the index")
    avg_char_count: float = Field(default=0.0, description="Average character count per chunk")
    duplicates_skipped: int = Field(default=0, description="Total duplicates skipped during ingestion")

