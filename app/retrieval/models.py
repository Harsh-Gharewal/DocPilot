"""Data models for retrieval, fusion, reranking, and stage tracing."""

from typing import Any, Optional
from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """Represents a chunk retrieved by any retrieval, fusion, or reranking stage."""

    chunk_id: str = Field(..., description="Unique chunk ID")
    text: str = Field(..., description="Chunk content text")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Metadata including source_file, section, page, etc."
    )
    score: float = Field(..., description="Score assigned at the current retrieval/ranking stage")
    rank: int = Field(..., description="1-based rank within this stage")
    retriever: str = Field(
        ..., description="Stage name: 'dense', 'sparse', 'fused', 'reranked'"
    )
    raw_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Dictionary of scores from all contributing retrievers (e.g. dense, sparse, rrf, cross_encoder)",
    )
    pre_rerank_rank: Optional[int] = Field(
        default=None, description="Original rank before reranking was applied"
    )


class RetrievalTrace(BaseModel):
    """Complete trace capturing every stage of the retrieval pipeline."""

    question: str = Field(..., description="User query or question")
    strategy: str = Field(..., description="Chunking strategy used for index lookup")
    mode: str = Field(..., description="Retrieval mode: dense_only, sparse_only, hybrid, hybrid_rerank")
    weights: dict[str, float] = Field(
        default_factory=dict, description="Dense and sparse weights used for fusion"
    )
    dense_results: list[RetrievedChunk] = Field(
        default_factory=list, description="Top candidates from dense retrieval"
    )
    sparse_results: list[RetrievedChunk] = Field(
        default_factory=list, description="Top candidates from sparse BM25 retrieval"
    )
    fused_results: list[RetrievedChunk] = Field(
        default_factory=list, description="Candidates after Reciprocal Rank Fusion"
    )
    reranked_results: list[RetrievedChunk] = Field(
        default_factory=list, description="Candidates after cross-encoder or LLM reranking"
    )
    final_chunks: list[RetrievedChunk] = Field(
        default_factory=list, description="Final top-k chunks returned to user or downstream generator"
    )

