"""Grounded answer generator formatting numbered context blocks and capturing bracketed citations."""

import logging
import re
from typing import Optional

from app.generation.llm_client import LLMClient, LLMResponse, llm_client
from app.generation.models import Citation
from app.generation.prompts import (
    GROUNDED_SYSTEM_PROMPT,
    build_generation_user_prompt,
    format_context_blocks,
)
from app.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)


class GroundedGenerator:
    """Generates answers strictly grounded in retrieved context with bracketed citations."""

    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or llm_client

    def generate_answer(
        self,
        question: str,
        chunks: list[RetrievedChunk],
    ) -> tuple[str, dict[int, Citation], LLMResponse]:
        """Generate answer with bracketed citations and build the citation mapping.

        Returns:
            (answer_text, citation_map, raw_llm_response)
        """
        if not chunks:
            refusal = (
                "INSUFFICIENT_CONTEXT: The provided internal documentation does not contain "
                "enough information to reliably answer this question."
            )
            dummy_resp = LLMResponse(content=refusal, prompt_tokens=0, completion_tokens=0, total_tokens=0)
            return refusal, {}, dummy_resp

        # Format context blocks
        context_blocks = format_context_blocks(chunks)
        user_prompt = build_generation_user_prompt(question=question, context_blocks=context_blocks)

        # Generate answer from LLM
        response = self.client.generate(
            prompt=user_prompt,
            system_prompt=GROUNDED_SYSTEM_PROMPT,
            temperature=0.0,
        )

        answer_text = response.content.strip()

        # Build map of available citations from chunks
        available_citations: dict[int, Citation] = {}
        for idx, chunk in enumerate(chunks, start=1):
            available_citations[idx] = Citation(
                citation_number=idx,
                chunk_id=chunk.chunk_id,
                source_file=chunk.metadata.get("source_file", "unknown"),
                section_heading=chunk.metadata.get("section_heading"),
                page_number=chunk.metadata.get("page_number"),
                chunk_text=chunk.text,
            )

        # Find which citations were actually referenced in the answer
        referenced_numbers = {
            int(m) for m in re.findall(r"\[(\d+)\]", answer_text)
        }

        # Keep citation mapping for cited references
        active_citations: dict[int, Citation] = {}
        for num in referenced_numbers:
            if num in available_citations:
                active_citations[num] = available_citations[num]

        logger.debug(
            f"Generated answer referencing {len(active_citations)} context citations out of {len(chunks)} provided."
        )
        return answer_text, active_citations, response


# Singleton instance
generator = GroundedGenerator()


def generate_answer(
    question: str, chunks: list[RetrievedChunk]
) -> tuple[str, dict[int, Citation], LLMResponse]:
    """Convenience functional wrapper for answer generation."""
    return generator.generate_answer(question, chunks)

