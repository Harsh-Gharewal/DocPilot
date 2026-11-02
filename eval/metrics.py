"""Automated evaluation metrics: Correctness, Faithfulness, Retrieval Relevance, Citation Accuracy, Latency & Cost."""

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from app.generation.citation_verifier import CitationVerifier, citation_verifier, split_sentences
from app.generation.llm_client import LLMClient, llm_client
from app.retrieval.models import RetrievedChunk
from eval.judge_cache import JudgeCache, judge_cache

logger = logging.getLogger(__name__)


# --- 1. Retrieval Relevance (Hit Rate @ K, Recall, MRR) ---

def compute_retrieval_relevance(
    retrieved_chunks: list[RetrievedChunk],
    expected_source_docs: list[str],
    expected_sections: Optional[list[str]] = None,
    k: int = 5,
) -> dict[str, float]:
    """Compute hit rate @ k, recall of expected sources, and MRR.

    Returns:
        dict with keys: 'hit_rate_at_k', 'recall', 'mrr'
    """
    if not expected_source_docs:
        # For no_answer test cases, expected sources is empty
        return {"hit_rate_at_k": 1.0, "recall": 1.0, "mrr": 1.0}

    # Normalize expected names to lowercase basenames
    expected_basenames = {Path(s).name.lower() for s in expected_source_docs}

    top_chunks = retrieved_chunks[:k]
    retrieved_basenames = [
        Path(c.metadata.get("source_file", "")).name.lower() for c in top_chunks
    ]

    # Hit rate @ k: 1.0 if ANY expected document is retrieved in top-k
    hit = 1.0 if any(doc in expected_basenames for doc in retrieved_basenames) else 0.0

    # Recall of expected sources: fraction of expected documents retrieved in top-k
    matched_sources = set(retrieved_basenames) & expected_basenames
    recall = round(len(matched_sources) / len(expected_basenames), 4) if expected_basenames else 1.0

    # MRR (Mean Reciprocal Rank): 1 / rank of the first relevant document
    mrr = 0.0
    for rank_idx, doc in enumerate(retrieved_basenames, start=1):
        if doc in expected_basenames:
            mrr = round(1.0 / rank_idx, 4)
            break

    return {
        "hit_rate_at_k": hit,
        "recall": recall,
        "mrr": mrr,
    }


# --- 2. Answer Correctness (LLM Judge) ---

CORRECTNESS_SYSTEM_PROMPT = """You are a strict technical evaluation judge measuring Answer Correctness.
Compare the System Answer against the Golden Answer.

RUBRIC:
- 1.0: Factually equivalent to the golden answer.
- 0.7 - 0.9: Mostly correct, but misses minor non-critical details.
- 0.4 - 0.6: Partially correct with notable omissions or inaccuracies.
- 0.0 - 0.3: Contradicts golden answer, hallucinates, or wrong facts.

Return ONLY a valid JSON object:
{
  "correctness_score": float,
  "reason": "concise explanation of score"
}
"""

def compute_answer_correctness(
    question: str,
    system_answer: str,
    golden_answer: str,
    category: str,
    is_fallback: bool = False,
    client: Optional[LLMClient] = None,
    cache: Optional[JudgeCache] = None,
) -> dict[str, Any]:
    """Judge correctness against the golden answer, with strict handling for NO_ANSWER cases."""
    active_client = client or llm_client
    active_cache = cache or judge_cache

    # Special handling for no_answer cases
    if category == "no_answer":
        refusal_indicators = [
            "could not find",
            "insufficient",
            "not found",
            "no reliable answer",
            "could not locate",
        ]
        is_refusal = is_fallback or any(ind in system_answer.lower() for ind in refusal_indicators)
        if is_refusal:
            return {
                "score": 1.0,
                "reason": "Correctly refused to answer out-of-corpus question.",
            }
        else:
            return {
                "score": 0.0,
                "reason": "Failed to refuse out-of-corpus question; attempted to hallucinate an answer.",
            }

    # If system gave a fallback refusal when an answer existed
    if is_fallback or "insufficient" in system_answer.lower() or "could not locate" in system_answer.lower():
        return {
            "score": 0.0,
            "reason": "System falsely claimed insufficient context for an answerable question.",
        }

    # Check cache
    cache_key = active_cache.compute_key("correctness", question, system_answer, golden_answer)
    cached = active_cache.get(cache_key)
    if cached:
        return cached

    prompt = f"""Question: {question}

Golden Answer:
\"\"\"{golden_answer}\"\"\"

System Answer:
\"\"\"{system_answer}\"\"\"

Evaluate the correctness:"""

    resp = active_client.generate(
        prompt=prompt,
        system_prompt=CORRECTNESS_SYSTEM_PROMPT,
        temperature=0.0,
        json_mode=True,
    )

    try:
        data = json.loads(resp.content)
        score = max(0.0, min(1.0, round(float(data.get("correctness_score", 0.8)), 4)))
        reason = str(data.get("reason", "Evaluated correctness."))
    except Exception:
        # Heuristic fallback: check keyword overlap
        gold_words = set(re.findall(r"\w{3,}", golden_answer.lower()))
        sys_words = set(re.findall(r"\w{3,}", system_answer.lower()))
        overlap = len(gold_words & sys_words) / len(gold_words) if gold_words else 1.0
        score = round(max(0.0, min(1.0, overlap)), 4)
        reason = f"Heuristic word overlap match ({score:.2f})."

    result = {"score": score, "reason": reason}
    active_cache.set(cache_key, "correctness", {"q": question, "sys": system_answer}, result)
    return result


# --- 3. Faithfulness (Claims Grounded in Retrieved Context) ---

FAITHFULNESS_SYSTEM_PROMPT = """You are a strict faithfulness judge.
Given an answer claim and the combined retrieved context, determine if the claim is fully grounded.
Return ONLY a valid JSON object:
{
  "grounded": boolean,
  "reason": "concise rationale"
}
"""

def compute_faithfulness(
    system_answer: str,
    retrieved_chunks: list[RetrievedChunk],
    client: Optional[LLMClient] = None,
    cache: Optional[JudgeCache] = None,
) -> dict[str, Any]:
    """Calculate the fraction of claims in the system answer grounded in retrieved context."""
    active_client = client or llm_client
    active_cache = cache or judge_cache

    sentences = split_sentences(system_answer)
    if not sentences or "could not locate" in system_answer.lower():
        return {"faithfulness_score": 1.0, "claims_grounded": 0, "total_claims": 0}

    combined_context = "\n".join(c.text for c in retrieved_chunks[:5])
    grounded_count = 0
    total_claims = 0

    for sent in sentences:
        claim_text = re.sub(r"\[\d+\]", "", sent).strip()
        if len(claim_text) < 10:
            continue
        total_claims += 1

        cache_key = active_cache.compute_key("faithfulness", claim_text, combined_context[:500])
        cached = active_cache.get(cache_key)
        if cached:
            if cached.get("grounded"):
                grounded_count += 1
            continue

        prompt = f"Claim: \"{claim_text}\"\n\nContext:\n\"\"\"{combined_context[:1500]}\"\"\""
        resp = active_client.generate(
            prompt=prompt,
            system_prompt=FAITHFULNESS_SYSTEM_PROMPT,
            temperature=0.0,
            json_mode=True,
        )

        try:
            data = json.loads(resp.content)
            is_grounded = bool(data.get("grounded", False))
            reason = str(data.get("reason", ""))
        except Exception:
            # Word overlap fallback
            claim_words = set(re.findall(r"\w{4,}", claim_text.lower()))
            ctx_words = set(re.findall(r"\w{4,}", combined_context.lower()))
            is_grounded = len(claim_words & ctx_words) >= 2
            reason = "Keyword overlap fallback."

        active_cache.set(cache_key, "faithfulness", {"claim": claim_text}, {"grounded": is_grounded, "reason": reason})
        if is_grounded:
            grounded_count += 1

    score = round(grounded_count / total_claims, 4) if total_claims > 0 else 1.0
    return {
        "faithfulness_score": score,
        "claims_grounded": grounded_count,
        "total_claims": total_claims,
    }


# --- 4. Citation Accuracy ---

def compute_citation_accuracy(
    answer: str,
    citations_map: dict[int, Any],
    retrieved_chunks: list[RetrievedChunk],
    verifier: Optional[CitationVerifier] = None,
) -> float:
    """Calculate the fraction of cited context blocks that actually support their claims."""
    active_verifier = verifier or citation_verifier
    result, _ = active_verifier.verify_answer(
        answer=answer,
        citations_map=citations_map,
        retrieved_chunks=retrieved_chunks,
    )
    return result.citation_coverage


# --- 5. Estimated Token Cost Calculation ---

def calculate_token_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate token cost in USD using standard GPT-4o pricing ($5/1M in, $15/1M out)."""
    in_cost = (prompt_tokens / 1_000_000.0) * 5.0
    out_cost = (completion_tokens / 1_000_000.0) * 15.0
    return round(in_cost + out_cost, 6)

