"""Data models for answer generation, citation verification, confidence scoring, and fallback handling."""

from typing import Any, Optional
from pydantic import BaseModel, Field

from app.retrieval.models import RetrievalTrace


class Citation(BaseModel):
    """Represents a validated citation reference within generated text."""

    citation_number: int = Field(..., description="1-based citation index corresponding to context block [n]")
    chunk_id: str = Field(..., description="Unique chunk ID cited")
    source_file: str = Field(..., description="Source document file path")
    section_heading: Optional[str] = Field(default=None, description="Section heading context")
    page_number: Optional[int] = Field(default=None, description="1-based page number if PDF")
    chunk_text: str = Field(..., description="Excerpt from the source chunk text")
    verified: bool = Field(default=False, description="Whether an LLM judge verified this citation supports the claim")
    verification_reason: Optional[str] = Field(
        default=None, description="Judge rationale explaining verification status"
    )


class ClaimCitation(BaseModel):
    """A claim sentence paired with its cited context blocks."""

    claim_text: str = Field(..., description="Extracted sentence or claim from the generated answer")
    citation_numbers: list[int] = Field(
        default_factory=list, description="Citation marker indices attached to this claim"
    )
    supported: bool = Field(default=False, description="Whether all cited sources support the claim")
    reason: Optional[str] = Field(default=None, description="LLM judge rationale")


class VerificationResult(BaseModel):
    """Aggregate result from the citation verification stage."""

    claims: list[ClaimCitation] = Field(default_factory=list, description="List of parsed claims and verdicts")
    citation_coverage: float = Field(
        default=0.0, description="Fraction of claims that are supported by verified citations (0.0 to 1.0)"
    )
    flagged_citations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of detected anomalies (e.g. non-existent citations, unsupported claims)",
    )
    is_fully_grounded: bool = Field(
        default=False, description="True if 100% of claims are properly verified"
    )


class ConfidenceScore(BaseModel):
    """Detailed three-part answer confidence breakdown."""

    retrieval_confidence: float = Field(
        ..., description="Score based on reranker top scores and confidence separation gap (0.0 to 1.0)"
    )
    citation_coverage: float = Field(
        ..., description="Score based on percentage of verified claims (0.0 to 1.0)"
    )
    completeness: float = Field(
        ..., description="LLM judge score on whether the answer addresses the full question (0.0 to 1.0)"
    )
    composite_score: float = Field(
        ..., description="Weighted composite confidence score (0.0 to 1.0)"
    )
    label: str = Field(..., description="'HIGH' (>=0.75), 'MEDIUM' (>=0.50), or 'LOW' (<0.50)")


class NoAnswerResponse(BaseModel):
    """Structured response returned when retrieval confidence is too low or context is insufficient."""

    is_fallback: bool = Field(default=True, description="Flag indicating fallback response was triggered")
    message: str = Field(
        default="I could not find a reliable answer to your question in the available documentation.",
        description="User-facing refusal statement",
    )
    retrieval_confidence: float = Field(..., description="The low retrieval confidence score that triggered fallback")
    top_chunks_summary: list[dict[str, str]] = Field(
        default_factory=list,
        description="Summaries and source locations of the closest candidate chunks found",
    )
    missing_information: str = Field(
        default="Specific facts directly addressing the user inquiry were absent from indexed docs.",
        description="Explanation of what information was not located",
    )
    recommended_manual_checks: list[str] = Field(
        default_factory=list, description="List of source documents that may be worth checking manually"
    )


class AnswerResponse(BaseModel):
    """Complete response returned by the RAG generation pipeline."""

    question: str = Field(..., description="User query")
    answer: str = Field(..., description="Generated answer text (or refusal message if fallback)")
    citations: list[Citation] = Field(
        default_factory=list, description="List of bracketed citations with source context"
    )
    confidence: ConfidenceScore = Field(..., description="Confidence breakdown and composite score")
    verification: VerificationResult = Field(..., description="Citation verification report")
    no_answer: Optional[NoAnswerResponse] = Field(
        default=None, description="Detailed refusal payload if fallback occurred"
    )
    retrieval_trace: Optional[RetrievalTrace] = Field(
        default=None, description="Audit trace from the retrieval stages"
    )
    latency_by_stage: dict[str, float] = Field(
        default_factory=dict, description="Latency in milliseconds for each stage"
    )
    token_usage: dict[str, int] = Field(
        default_factory=dict, description="Tokens consumed (prompt, completion, total)"
    )

