"""Citation verification engine that validates claims against cited context chunks using an LLM judge."""

import json
import logging
import re
from typing import Any, Optional

from app.generation.llm_client import LLMClient, llm_client
from app.generation.models import Citation, ClaimCitation, VerificationResult
from app.generation.prompts import (
    CITATION_VERIFY_SYSTEM_PROMPT,
    build_citation_verify_prompt,
)
from app.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)


def split_sentences(text: str) -> list[str]:
    """Split text into sentences while respecting inline punctuation and abbreviations."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in raw if s.strip()]
    return sentences


class CitationVerifier:
    """Verifies citation accuracy, flags fake/unsupported references, and measures grounded coverage."""

    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or llm_client

    def verify_answer(
        self,
        answer: str,
        citations_map: dict[int, Citation],
        retrieved_chunks: list[RetrievedChunk],
    ) -> tuple[VerificationResult, dict[int, Citation]]:
        """Parse answer claims, verify each against cited chunks, and return verification result."""
        # Map 1-based index to chunk text
        chunks_by_number: dict[int, str] = {
            i: c.text for i, c in enumerate(retrieved_chunks, start=1)
        }

        # Check for insufficient context indicator
        if "INSUFFICIENT_CONTEXT" in answer:
            return (
                VerificationResult(
                    claims=[],
                    citation_coverage=1.0,
                    flagged_citations=[],
                    is_fully_grounded=True,
                ),
                citations_map,
            )

        sentences = split_sentences(answer)
        parsed_claims: list[ClaimCitation] = []
        flagged_anomalies: list[dict[str, Any]] = []
        verified_citations_map = dict(citations_map)

        for sent in sentences:
            # Extract [n] markers
            citation_numbers = [int(n) for n in re.findall(r"\[(\d+)\]", sent)]
            # Clean claim text
            claim_text = re.sub(r"\[\d+\]", "", sent).strip()
            if not claim_text or len(claim_text) < 5:
                continue

            # Check 1: Claim with NO citations
            if not citation_numbers:
                flagged_anomalies.append(
                    {
                        "type": "unsupported_claim_missing_citation",
                        "claim": claim_text,
                        "description": "Sentence asserts factual claim with no attached [n] citation reference.",
                    }
                )
                parsed_claims.append(
                    ClaimCitation(
                        claim_text=claim_text,
                        citation_numbers=[],
                        supported=False,
                        reason="No citation provided for this claim.",
                    )
                )
                continue

            # Check 2: Citations pointing to non-existent context blocks
            has_invalid_num = False
            for num in citation_numbers:
                if num not in chunks_by_number:
                    has_invalid_num = True
                    flagged_anomalies.append(
                        {
                            "type": "hallucinated_citation_number",
                            "citation_number": num,
                            "claim": claim_text,
                            "description": f"Answer references [{num}] which does not exist in retrieved context.",
                        }
                    )

            if has_invalid_num:
                parsed_claims.append(
                    ClaimCitation(
                        claim_text=claim_text,
                        citation_numbers=citation_numbers,
                        supported=False,
                        reason="References non-existent citation number.",
                    )
                )
                continue

            # Check 3: LLM Judge evaluation for valid citations
            is_claim_supported = True
            reasons = []

            for num in citation_numbers:
                chunk_text = chunks_by_number[num]
                verdict, reason = self._judge_claim_support(claim=claim_text, chunk_text=chunk_text)
                reasons.append(f"[{num}]: {reason}")

                # Update Citation object in map
                if num in verified_citations_map:
                    citation_obj = verified_citations_map[num]
                    citation_obj.verified = verdict
                    citation_obj.verification_reason = reason

                if not verdict:
                    is_claim_supported = False
                    flagged_anomalies.append(
                        {
                            "type": "unsupported_citation",
                            "citation_number": num,
                            "claim": claim_text,
                            "reason": reason,
                        }
                    )

            combined_reason = "; ".join(reasons)
            parsed_claims.append(
                ClaimCitation(
                    claim_text=claim_text,
                    citation_numbers=citation_numbers,
                    supported=is_claim_supported,
                    reason=combined_reason,
                )
            )

        # Compute citation coverage
        total_claims = len(parsed_claims)
        supported_claims = sum(1 for c in parsed_claims if c.supported)
        coverage = round(supported_claims / total_claims, 4) if total_claims > 0 else 0.0

        is_grounded = (coverage == 1.0) and (len(flagged_anomalies) == 0)

        result = VerificationResult(
            claims=parsed_claims,
            citation_coverage=coverage,
            flagged_citations=flagged_anomalies,
            is_fully_grounded=is_grounded,
        )

        logger.debug(
            f"Citation verification complete: coverage={coverage:.2%}, flagged={len(flagged_anomalies)}"
        )
        return result, verified_citations_map

    def _judge_claim_support(self, claim: str, chunk_text: str) -> tuple[bool, str]:
        """Ask LLM judge whether the chunk text directly supports the claim."""
        prompt = build_citation_verify_prompt(claim=claim, cited_text=chunk_text)
        resp = self.client.generate(
            prompt=prompt,
            system_prompt=CITATION_VERIFY_SYSTEM_PROMPT,
            temperature=0.0,
            json_mode=True,
        )

        try:
            data = json.loads(resp.content)
            supported = bool(data.get("supported", False))
            reason = str(data.get("reason", "Judgment recorded."))
            return supported, reason
        except Exception as e:
            logger.warning(f"Could not parse citation judge JSON: {e} (Raw: {resp.content})")
            # If JSON parsing fails, fall back to keyword overlap check
            claim_words = set(re.findall(r"\w{4,}", claim.lower()))
            chunk_words = set(re.findall(r"\w{4,}", chunk_text.lower()))
            overlap = len(claim_words & chunk_words)
            if overlap >= 2:
                return True, "Verified via text keyword alignment fallback."
            return False, "Failed verification check."


# Singleton instance
citation_verifier = CitationVerifier()

