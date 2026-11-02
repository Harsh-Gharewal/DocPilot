"""Ingestion package for multi-format document loading, chunking, and dual-indexing."""

from app.ingestion.chunkers import (
    Chunker,
    FixedSizeChunker,
    SemanticChunker,
    StructureAwareChunker,
    get_chunker,
)
from app.ingestion.dedup import Deduplicator
from app.ingestion.embeddings import EmbeddingClient, embedding_client
from app.ingestion.indexer import IndexManager, index_chunks, index_manager, verify_index_sync
from app.ingestion.loaders import load_document, load_documents_from_directory
from app.ingestion.models import Chunk, DedupReport, Document, Section, StrategyStats

__all__ = [
    "Chunk",
    "Chunker",
    "DedupReport",
    "Deduplicator",
    "Document",
    "EmbeddingClient",
    "FixedSizeChunker",
    "IndexManager",
    "Section",
    "SemanticChunker",
    "StrategyStats",
    "StructureAwareChunker",
    "embedding_client",
    "get_chunker",
    "index_chunks",
    "index_manager",
    "load_document",
    "load_documents_from_directory",
    "verify_index_sync",
]
