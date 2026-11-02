"""Unit and integration tests for generation, citation verification, confidence scoring, and fallback handling."""

from unittest.mock import MagicMock
import pytest

from app.generation.citation_verifier import CitationVerifier
from app.generation.confidence import ConfidenceScorer, compute_retrieval_confidence
from app.generation.fallback import build_fallback_response
from app.generation.generator import GroundedGenerator
from app.generation.llm_client import LLMResponse
from app.generation.models import Citation, VerificationResult
from app.retrieval.models import RetrievedChunk


def make_chunk(cid: str, text: str, score: float, rank: int) -> RetrievedChunk:
    """Helper to construct RetrievedChunk."""
    return RetrievedChunk(
        chunk_id=cid,
        text=text,
        metadata={"source_file": "doc.md", "section_heading": "Section A"},
        score=score,
        rank=rank,
        retriever="reranked",
    )


# --- 1. Generation & Citation Extraction Tests ---

def test_grounded_generator_citations_extraction():
    """Verify generator formats context blocks and maps bracketed citations."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = LLMResponse(
        content="The system validates JWT tokens [1]. It uses PostgreSQL connection pooling [2].",
        prompt_tokens=150,
        completion_tokens=25,
        total_tokens=175,
    )

    c1 = make_chunk("c1", "The API Gateway validates JWT tokens.", 2.5, 1)
    c2 = make_chunk("c2", "PostgreSQL connection pool max size is 20.", 1.8, 2)

    gen = GroundedGenerator(client=mock_llm)
    ans, citations_map, resp = gen.generate_answer("How is auth handled?", [c1, c2])

    assert "[1]" in ans
    assert "[2]" in ans
    assert 1 in citations_map
    assert 2 in citations_map
    assert citations_map[1].chunk_id == "c1"
    assert citations_map[2].chunk_id == "c2"


# --- 2. Citation Verification Tests ---

def test_citation_verifier_supported_claim():
    """Verify accurate claim supported by context is marked verified."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = LLMResponse(
        content='{"supported": true, "reason": "Directly stated in cited chunk."}'
    )

    verifier = CitationVerifier(client=mock_llm)
    c1 = make_chunk("c1", "The API Gateway validates JWT tokens using auth.jwt_secret.", 2.0, 1)

    citations_map = {
        1: Citation(
            citation_number=1,
            chunk_id="c1",
            source_file="auth.md",
            chunk_text=c1.text,
        )
    }

    ans = "The gateway validates JWT tokens using the secret [1]."
    result, updated_map = verifier.verify_answer(ans, citations_map, [c1])

    assert result.citation_coverage == 1.0
    assert result.is_fully_grounded is True
    assert len(result.claims) == 1
    assert result.claims[0].supported is True
    assert updated_map[1].verified is True


def test_citation_verifier_deliberately_wrong_citation_flagged():
    """Verify that a deliberately false/unsupported citation is flagged by judge."""
    mock_llm = MagicMock()
    # LLM judge determines claim contradicts text
    mock_llm.generate.return_value = LLMResponse(
        content='{"supported": false, "reason": "Chunk specifies max 20 connections, but claim asserts 1000."}'
    )

    verifier = CitationVerifier(client=mock_llm)
    c1 = make_chunk("c1", "The pool size is set to a maximum of 20 connections per pod.", 1.5, 1)

    citations_map = {
        1: Citation(
            citation_number=1,
            chunk_id="c1",
            source_file="db.md",
            chunk_text=c1.text,
        )
    }

    # Deliberately untrue claim
    ans = "The database pool accommodates up to 1000 simultaneous connections [1]."
    result, updated_map = verifier.verify_answer(ans, citations_map, [c1])

    assert result.citation_coverage == 0.0
    assert result.is_fully_grounded is False
    assert len(result.flagged_citations) == 1
    assert result.flagged_citations[0]["type"] == "unsupported_citation"
    assert updated_map[1].verified is False


def test_citation_verifier_flags_non_existent_citation_number():
    """Verify citations pointing to numbers outside retrieved context are caught."""
    verifier = CitationVerifier(client=MagicMock())
    c1 = make_chunk("c1", "Some context.", 1.0, 1)
    citations_map = {1: Citation(citation_number=1, chunk_id="c1", source_file="doc.md", chunk_text=c1.text)}

    # Answer hallucinates citation [99]
    ans = "This fact comes from an unindexed document [99]."
    result, _ = verifier.verify_answer(ans, citations_map, [c1])

    assert len(result.flagged_citations) == 1
    assert result.flagged_citations[0]["type"] == "hallucinated_citation_number"
    assert result.flagged_citations[0]["citation_number"] == 99


# --- 3. Confidence Math & Scoring Tests ---

def test_confidence_math():
    """Verify retrieval confidence, completeness, and composite score calculation math."""
    c1 = make_chunk("c1", "Top match", 3.0, 1)
    c2 = make_chunk("c2", "Second match", 1.0, 2)

    retrieval_conf = compute_retrieval_confidence([c1, c2])
    assert retrieval_conf > 0.80  # Positive top score yields high retrieval confidence

    mock_llm = MagicMock()
    mock_llm.generate.return_value = LLMResponse(
        content='{"completeness_score": 0.90, "reason": "Fully answers question."}'
    )

    scorer = ConfidenceScorer(client=mock_llm)
    v_res = VerificationResult(
        claims=[], citation_coverage=0.85, flagged_citations=[], is_fully_grounded=False
    )

    score = scorer.score_confidence(
        question="What is the pool size?",
        answer="Pool size is 20 [1].",
        retrieved_chunks=[c1, c2],
        verification_result=v_res,
    )

    assert score.citation_coverage == 0.85
    assert score.completeness == 0.90
    # Expected: 0.4 * retrieval + 0.4 * 0.85 + 0.2 * 0.90
    expected_composite = (0.4 * retrieval_conf) + (0.4 * 0.85) + (0.2 * 0.90)
    assert pytest.approx(score.composite_score, 0.01) == expected_composite
    assert score.label in {"HIGH", "MEDIUM"}


# --- 4. Fallback Trigger Tests ---

def test_low_retrieval_confidence_triggers_fallback():
    """Verify that strongly negative cross-encoder scores trigger NoAnswerResponse."""
    # Chunk with strongly negative score (typical for off-topic query)
    c_off_topic = make_chunk("c_off", "Unrelated text", -10.5, 1)
    retrieval_conf = compute_retrieval_confidence([c_off_topic])

    assert retrieval_conf < 0.20  # Well below 0.5 threshold

    fallback = build_fallback_response(
        question="What is the lifespan of a Galapagos tortoise?",
        retrieval_confidence=retrieval_conf,
        candidate_chunks=[c_off_topic],
    )

    assert fallback.is_fallback is True
    assert "could not locate a sufficiently grounded answer" in fallback.message
    assert len(fallback.top_chunks_summary) == 1
    assert fallback.top_chunks_summary[0]["source"] == "doc.md"

