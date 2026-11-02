"""End-to-end RAG orchestrator coordinating retrieval, confidence gating, generation, verification, and scoring."""

import logging
import time
from typing import Any, Optional

from app.config import settings
from app.generation.citation_verifier import citation_verifier
from app.generation.confidence import (
    compute_retrieval_confidence,
    confidence_scorer,
)
from app.generation.fallback import build_fallback_response
from app.generation.generator import generator
from app.generation.models import (
    AnswerResponse,
    Citation,
    ConfidenceScore,
    NoAnswerResponse,
    VerificationResult,
)
from app.retrieval.pipeline import hybrid_retriever

logger = logging.getLogger(__name__)


def ask(
    question: str,
    strategy: str = "structure_aware",
    mode: str = "hybrid_rerank",
    top_k: Optional[int] = None,
    dense_weight: Optional[float] = None,
    sparse_weight: Optional[float] = None,
) -> AnswerResponse:
    """Execute end-to-end RAG pipeline for a user question.

    Stages:
        1. Multi-stage Hybrid Retrieval (dense, sparse, fusion, reranking)
        2. Confidence Gate (blocks generation if retrieval confidence is low)
        3. Grounded Answer Generation (with bracketed citations)
        4. Citation Verification (LLM judge claims vs context chunks)
        5. Answer Confidence Scoring (three-part breakdown)
    """
    total_start = time.perf_counter()
    latency_by_stage: dict[str, float] = {}
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    # 1. Retrieval stage
    ret_start = time.perf_counter()
    weights = {}
    if dense_weight is not None:
        weights["dense"] = dense_weight
    if sparse_weight is not None:
        weights["sparse"] = sparse_weight

    chunks, trace = hybrid_retriever.retrieve(
        question=question,
        strategy=strategy,
        mode=mode,
        top_k_final=top_k,
        weights=weights if weights else None,
    )
    latency_by_stage["retrieval_ms"] = round((time.perf_counter() - ret_start) * 1000, 2)

    # 2. Confidence Gate check on retrieved candidates
    retrieval_conf = compute_retrieval_confidence(chunks)
    threshold = settings.confidence_threshold_no_answer

    if retrieval_conf < threshold or not chunks:
        logger.info(
            f"Retrieval confidence {retrieval_conf:.2f} is below threshold {threshold:.2f}. Bypassing generator to fallback response."
        )
        fallback = build_fallback_response(
            question=question,
            retrieval_confidence=retrieval_conf,
            candidate_chunks=chunks,
        )

        confidence_obj = ConfidenceScore(
            retrieval_confidence=retrieval_conf,
            citation_coverage=0.0,
            completeness=0.0,
            composite_score=round(retrieval_conf * settings.confidence_weight_retrieval, 4),
            label="LOW",
        )

        verification_obj = VerificationResult(
            claims=[],
            citation_coverage=0.0,
            flagged_citations=[],
            is_fully_grounded=False,
        )

        latency_by_stage["total_ms"] = round((time.perf_counter() - total_start) * 1000, 2)

        return AnswerResponse(
            question=question,
            answer=fallback.message,
            citations=[],
            confidence=confidence_obj,
            verification=verification_obj,
            no_answer=fallback,
            retrieval_trace=trace,
            latency_by_stage=latency_by_stage,
            token_usage=token_usage,
        )

    # 3. Grounded Generation
    gen_start = time.perf_counter()
    answer_text, citations_map, llm_resp = generator.generate_answer(
        question=question, chunks=chunks
    )
    latency_by_stage["generation_ms"] = round((time.perf_counter() - gen_start) * 1000, 2)

    token_usage["prompt_tokens"] += llm_resp.prompt_tokens
    token_usage["completion_tokens"] += llm_resp.completion_tokens
    token_usage["total_tokens"] += llm_resp.total_tokens

    # Check if generator itself indicated insufficient context
    if "INSUFFICIENT_CONTEXT" in answer_text:
        fallback = build_fallback_response(
            question=question,
            retrieval_confidence=retrieval_conf,
            candidate_chunks=chunks,
            reason="The language model determined the retrieved context was insufficient to answer.",
        )
        confidence_obj = ConfidenceScore(
            retrieval_confidence=retrieval_conf,
            citation_coverage=1.0,
            completeness=0.0,
            composite_score=round(retrieval_conf * settings.confidence_weight_retrieval, 4),
            label="LOW",
        )
        verification_obj = VerificationResult(
            claims=[], citation_coverage=1.0, flagged_citations=[], is_fully_grounded=True
        )
        latency_by_stage["total_ms"] = round((time.perf_counter() - total_start) * 1000, 2)

        return AnswerResponse(
            question=question,
            answer=answer_text,
            citations=[],
            confidence=confidence_obj,
            verification=verification_obj,
            no_answer=fallback,
            retrieval_trace=trace,
            latency_by_stage=latency_by_stage,
            token_usage=token_usage,
        )

    # 4. Citation Verification
    ver_start = time.perf_counter()
    verification_res, updated_citations_map = citation_verifier.verify_answer(
        answer=answer_text,
        citations_map=citations_map,
        retrieved_chunks=chunks,
    )
    latency_by_stage["verification_ms"] = round((time.perf_counter() - ver_start) * 1000, 2)

    # 5. Answer Confidence Scoring
    score_start = time.perf_counter()
    confidence_res = confidence_scorer.score_confidence(
        question=question,
        answer=answer_text,
        retrieved_chunks=chunks,
        verification_result=verification_res,
    )
    latency_by_stage["scoring_ms"] = round((time.perf_counter() - score_start) * 1000, 2)

    latency_by_stage["total_ms"] = round((time.perf_counter() - total_start) * 1000, 2)

    final_citations = list(updated_citations_map.values())

    return AnswerResponse(
        question=question,
        answer=answer_text,
        citations=final_citations,
        confidence=confidence_res,
        verification=verification_res,
        no_answer=None,
        retrieval_trace=trace,
        latency_by_stage=latency_by_stage,
        token_usage=token_usage,
    )

