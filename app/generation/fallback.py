"""Graceful fallback handler for out-of-corpus questions and insufficient context."""

import logging
from typing import Optional

from app.config import settings
from app.generation.models import NoAnswerResponse
from app.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)


def build_fallback_response(
    question: str,
    retrieval_confidence: float,
    candidate_chunks: list[RetrievedChunk],
    reason: Optional[str] = None,
) -> NoAnswerResponse:
    """Build a structured refusal response detailing what was found and recommended manual checks."""
    top_chunks = candidate_chunks[:3]

    chunk_summaries = []
    recommended_sources = set()

    for idx, c in enumerate(top_chunks, start=1):
        source = c.metadata.get("source_file", "unknown")
        heading = c.metadata.get("section_heading", "Overview")
        recommended_sources.add(source)

        preview = c.text.strip().replace("\n", " ")
        if len(preview) > 120:
            preview = preview[:117] + "..."

        chunk_summaries.append(
            {
                "rank": str(idx),
                "source": source,
                "section": heading,
                "score": f"{c.score:.4f}",
                "snippet": preview,
            }
        )

    explanation = (
        reason
        or f"The retrieval confidence ({retrieval_confidence:.2f}) fell below the reliability threshold "
        f"({settings.confidence_threshold_no_answer:.2f}), indicating the query is likely out-of-corpus."
    )

    return NoAnswerResponse(
        is_fallback=True,
        message=(
            "I could not locate a sufficiently grounded answer to your inquiry within the indexed documentation."
        ),
        retrieval_confidence=retrieval_confidence,
        top_chunks_summary=chunk_summaries,
        missing_information=explanation,
        recommended_manual_checks=sorted(list(recommended_sources)),
    )

