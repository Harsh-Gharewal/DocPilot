"""Generation package integrating grounded synthesis, citation verification, and confidence scoring."""

from app.generation.citation_verifier import CitationVerifier, citation_verifier
from app.generation.confidence import ConfidenceScorer, compute_retrieval_confidence, confidence_scorer
from app.generation.fallback import build_fallback_response
from app.generation.generator import GroundedGenerator, generate_answer, generator
from app.generation.llm_client import LLMClient, LLMResponse, llm_client
from app.generation.models import (
    AnswerResponse,
    Citation,
    ClaimCitation,
    ConfidenceScore,
    NoAnswerResponse,
    VerificationResult,
)
from app.generation.rag import ask

__all__ = [
    "AnswerResponse",
    "Citation",
    "CitationVerifier",
    "ClaimCitation",
    "ConfidenceScore",
    "ConfidenceScorer",
    "GroundedGenerator",
    "LLMClient",
    "LLMResponse",
    "NoAnswerResponse",
    "VerificationResult",
    "ask",
    "build_fallback_response",
    "citation_verifier",
    "compute_retrieval_confidence",
    "confidence_scorer",
    "generate_answer",
    "generator",
    "llm_client",
]
