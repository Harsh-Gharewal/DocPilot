"""Answer confidence scorer evaluating retrieval quality, citation support, and answer completeness."""

import json
import logging
import math
from typing import Optional

from app.config import settings
from app.generation.llm_client import LLMClient, llm_client
from app.generation.models import ConfidenceScore, VerificationResult
from app.generation.prompts import (
    COMPLETENESS_SYSTEM_PROMPT,
    build_completeness_prompt,
)
from app.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)


def compute_retrieval_confidence(chunks: list[RetrievedChunk]) -> float:
    """Compute normalized retrieval confidence in the range [0.0, 1.0].

    Formula & Rationale:
        Cross-encoder relevance scores range from ~ -11.0 (completely irrelevant)
        to ~ +5.0 (highly relevant).
        1. Sigmoid Base:
           norm_top = 1 / (1 + exp(-(top_score + 1.5)))
           Maps positive scores (> 0) to >= 0.82, and strongly negative scores (<-5) to <= 0.03.
        2. Separation Gap:
           gap = max(0.0, top_score - second_score)
           gap_factor = min(1.0, gap / 2.0)
        3. Blended Score:
           retrieval_confidence = 0.8 * norm_top + 0.2 * gap_factor
    """
    if not chunks:
        return 0.0

    top_score = chunks[0].score

    # Sigmoid normalization
    try:
        norm_top = 1.0 / (1.0 + math.exp(-(top_score + 1.5)))
    except OverflowError:
        norm_top = 0.0 if top_score < 0 else 1.0

    # Calculate separation gap if multiple candidates exist
    if len(chunks) > 1:
        gap = max(0.0, top_score - chunks[1].score)
        gap_factor = min(1.0, gap / 2.0)
    else:
        gap_factor = 0.5

    confidence = round(0.8 * norm_top + 0.2 * gap_factor, 4)
    return max(0.0, min(1.0, confidence))


class ConfidenceScorer:
    """Computes a three-part confidence breakdown and composite confidence score."""

    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or llm_client

    def score_confidence(
        self,
        question: str,
        answer: str,
        retrieved_chunks: list[RetrievedChunk],
        verification_result: VerificationResult,
    ) -> ConfidenceScore:
        """Calculate retrieval, citation, and completeness scores, and produce composite score."""
        # 1. Retrieval confidence
        retrieval_conf = compute_retrieval_confidence(retrieved_chunks)

        # 2. Citation coverage
        citation_cov = verification_result.citation_coverage

        # 3. Completeness evaluation
        completeness = self._judge_completeness(question, answer)

        # 4. Composite weighted score
        w_ret = settings.confidence_weight_retrieval
        w_cit = settings.confidence_weight_citation
        w_comp = settings.confidence_weight_completeness

        composite = round(
            (w_ret * retrieval_conf) + (w_cit * citation_cov) + (w_comp * completeness), 4
        )
        composite = max(0.0, min(1.0, composite))

        # Categorize label
        if composite >= 0.75:
            label = "HIGH"
        elif composite >= 0.50:
            label = "MEDIUM"
        else:
            label = "LOW"

        score = ConfidenceScore(
            retrieval_confidence=retrieval_conf,
            citation_coverage=citation_cov,
            completeness=completeness,
            composite_score=composite,
            label=label,
        )

        logger.debug(
            f"Confidence calculated: composite={composite:.2f} ({label}) "
            f"[retrieval={retrieval_conf:.2f}, citation={citation_cov:.2f}, completeness={completeness:.2f}]"
        )
        return score

    def _judge_completeness(self, question: str, answer: str) -> float:
        """Evaluate completeness using LLM judge."""
        if "INSUFFICIENT_CONTEXT" in answer or not answer.strip():
            return 0.0

        prompt = build_completeness_prompt(question=question, answer=answer)
        resp = self.client.generate(
            prompt=prompt,
            system_prompt=COMPLETENESS_SYSTEM_PROMPT,
            temperature=0.0,
            json_mode=True,
        )

        try:
            data = json.loads(resp.content)
            return max(0.0, min(1.0, round(float(data.get("completeness_score", 0.8)), 4)))
        except Exception:
            return 0.80


# Singleton instance
confidence_scorer = ConfidenceScorer()

