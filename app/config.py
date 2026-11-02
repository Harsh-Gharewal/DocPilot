"""Application configuration loaded from environment variables and .env file.

Uses pydantic-settings to manage and validate all pipeline parameters.
"""

from pathlib import Path
import os
from typing import Literal

# Ensure offline operation for downloaded huggingface models
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration class for the RAG Hybrid pipeline."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys & LLM Provider
    api_key: str = Field(default="docpilot-secret-key", description="DocPilot service API authorization key")
    openai_api_key: str = Field(default="", description="OpenAI API Key for embeddings and GPT-4o")
    anthropic_api_key: str = Field(default="", description="Anthropic API Key for Claude Sonnet")
    llm_provider: Literal["openai", "anthropic"] = Field(
        default="openai", description="Active LLM provider: 'openai' or 'anthropic'"
    )

    # Models
    openai_chat_model: str = Field(default="gpt-4o", description="OpenAI chat model name")
    anthropic_chat_model: str = Field(
        default="claude-3-5-sonnet-20241022", description="Anthropic chat model name"
    )
    embedding_provider: Literal["openai", "local"] = Field(
        default="openai", description="Embedding provider: 'openai' or 'local'"
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", description="OpenAI embedding model name"
    )
    local_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace model identifier for local sentence-transformers",
    )

    # Storage Paths
    raw_data_dir: Path = Field(default=Path("data/raw"), description="Directory for original files")
    processed_data_dir: Path = Field(
        default=Path("data/processed"), description="Directory for parsed documents JSON"
    )
    chroma_dir: Path = Field(
        default=Path("data/chroma"), description="Persistent ChromaDB storage directory"
    )
    bm25_dir: Path = Field(
        default=Path("data/bm25"), description="Persistent BM25 storage directory"
    )
    sample_docs_dir: Path = Field(
        default=Path("sample_docs"), description="Directory holding documents to index"
    )

    # Ingestion & Chunking Defaults
    chunk_size: int = Field(default=1000, description="Default character chunk size")
    chunk_overlap: int = Field(default=200, description="Default character overlap size")
    default_chunking_strategy: Literal["fixed_size", "structure_aware", "semantic"] = Field(
        default="structure_aware", description="Default chunking strategy"
    )
    dedup_similarity_threshold: float = Field(
        default=0.95, description="Cosine similarity threshold above which a chunk is marked duplicate"
    )
    semantic_similarity_percentile: float = Field(
        default=25.0, description="Percentile threshold for semantic chunking sentence boundary splits"
    )

    # Retrieval & RRF
    top_k_retrieval: int = Field(default=10, description="Number of chunks retrieved per retriever")
    top_k_fused: int = Field(default=20, description="Number of fused candidate chunks passed to reranker")
    top_k_final: int = Field(default=5, description="Final number of chunks returned after reranking")
    rrf_k: int = Field(default=60, description="Reciprocal Rank Fusion smoothing constant")
    dense_weight: float = Field(default=0.7, description="Default dense retrieval weight for RRF")
    sparse_weight: float = Field(default=0.3, description="Default sparse retrieval weight for RRF")

    # Reranker
    reranker_backend: Literal["cross-encoder", "llm"] = Field(
        default="cross-encoder", description="Reranker backend: 'cross-encoder' or 'llm'"
    )
    cross_encoder_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="HuggingFace model identifier for cross-encoder reranking",
    )

    # Confidence & Fallback Thresholds
    confidence_threshold_no_answer: float = Field(
        default=0.5, description="Composite confidence score below which fallback response is triggered"
    )
    confidence_weight_retrieval: float = Field(
        default=0.4, description="Weight of retrieval confidence in composite score"
    )
    confidence_weight_citation: float = Field(
        default=0.4, description="Weight of citation coverage in composite score"
    )
    confidence_weight_completeness: float = Field(
        default=0.2, description="Weight of answer completeness in composite score"
    )

    # API Server & Auth
    api_host: str = Field(default="0.0.0.0", description="FastAPI host")
    api_port: int = Field(default=8000, description="FastAPI port")
    api_key: str = Field(default="rag-internal-dev-key", description="Internal API key for authentication")


# Singleton instance
settings = Settings()
