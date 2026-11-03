"""Application configuration loaded from environment variables and .env file.

Uses pydantic-settings to manage and validate all pipeline parameters.
"""

from pathlib import Path
import os
from typing import Literal, Optional

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
    openai_api_key: str = Field(default="", description="Primary OpenAI API Key")
    anthropic_api_key: str = Field(default="", description="Primary Anthropic API Key")
    groq_api_key: str = Field(default="", description="Groq API Key for fast inference")

    # Task-Specific API Keys (fallback to primary provider keys if not explicitly set)
    embedding_api_key: str = Field(default="", description="Dedicated API key for embedding generation")
    generation_api_key: str = Field(default="", description="Dedicated API key for answer generation")
    verification_api_key: str = Field(default="", description="Dedicated API key for citation verification")
    judge_api_key: str = Field(default="", description="Dedicated API key for evaluation benchmark judge")

    # Multi-Key Rotation Pools (comma-separated list of keys for auto-failover)
    openai_api_keys: str = Field(default="", description="Comma-separated OpenAI keys for auto-failover pool")
    anthropic_api_keys: str = Field(default="", description="Comma-separated Anthropic keys for auto-failover pool")

    llm_provider: Literal["openai", "anthropic", "groq"] = Field(
        default="openai", description="Active LLM provider: 'openai', 'anthropic', or 'groq'"
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

    def get_key_pool(self, provider: str = "openai") -> list[str]:
        """Return all configured API keys for a provider in priority order."""
        prov = provider.lower()
        keys: list[str] = []
        if prov == "openai":
            if self.openai_api_keys:
                keys.extend([k.strip() for k in self.openai_api_keys.split(",") if k.strip()])
            if self.openai_api_key and self.openai_api_key not in keys:
                keys.append(self.openai_api_key.strip())
        elif prov == "anthropic":
            if self.anthropic_api_keys:
                keys.extend([k.strip() for k in self.anthropic_api_keys.split(",") if k.strip()])
            if self.anthropic_api_key and self.anthropic_api_key not in keys:
                keys.append(self.anthropic_api_key.strip())
        elif prov == "groq":
            if self.groq_api_key:
                keys.append(self.groq_api_key.strip())
        return keys

    def get_key_for_task(self, task: str, provider: Optional[str] = None) -> str:
        """Resolve the active API key for a specific task with smart fallbacks."""
        task_lower = task.lower()
        if task_lower in ("embedding", "embeddings"):
            if self.embedding_api_key:
                return self.embedding_api_key.strip()
            pool = self.get_key_pool("openai")
            return pool[0] if pool else self.openai_api_key.strip()

        if task_lower == "verification":
            if self.verification_api_key:
                return self.verification_api_key.strip()
            if self.generation_api_key:
                return self.generation_api_key.strip()
            prov = provider or self.llm_provider
            pool = self.get_key_pool(prov)
            return pool[0] if pool else (self.openai_api_key if prov == "openai" else self.anthropic_api_key)

        if task_lower == "judge":
            if self.judge_api_key:
                return self.judge_api_key.strip()
            if self.generation_api_key:
                return self.generation_api_key.strip()
            prov = provider or self.llm_provider
            pool = self.get_key_pool(prov)
            return pool[0] if pool else (self.openai_api_key if prov == "openai" else self.anthropic_api_key)

        # Default: generation / synthesis
        if self.generation_api_key:
            return self.generation_api_key.strip()
        prov = provider or self.llm_provider
        pool = self.get_key_pool(prov)
        return pool[0] if pool else (self.openai_api_key if prov == "openai" else self.anthropic_api_key)


# Singleton instance
settings = Settings()
