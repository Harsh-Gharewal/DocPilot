"""Unit tests for the evaluation metrics, dataset validation, and scoring calculations."""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from eval.dataset import GoldenTestCase, load_golden_dataset
from eval.metrics import (
    calculate_token_cost,
    compute_answer_correctness,
    compute_faithfulness,
    compute_retrieval_relevance,
)
from app.retrieval.models import RetrievedChunk


def make_chunk(cid: str, source_file: str, text: str = "some text") -> RetrievedChunk:
    """Helper to construct RetrievedChunk for evaluation metric tests."""
    return RetrievedChunk(
        chunk_id=cid,
        text=text,
        metadata={"source_file": source_file},
        score=1.0,
        rank=1,
        retriever="reranked",
    )


# --- 1. Metric Math: Hit Rate, Recall, MRR ---

def test_retrieval_relevance_hit_rate_and_mrr():
    """Verify hit rate @ k, source recall, and MRR calculations with known values."""
    # Top 3 chunks retrieved
    chunks = [
        make_chunk("c1", "docs/other.md"),
        make_chunk("c2", "docs/architecture.md"),  # First match at rank 2 -> MRR = 1/2 = 0.5
        make_chunk("c3", "docs/api.md"),           # Second match at rank 3
    ]
    expected_sources = ["architecture.md", "api.md"]

    rel = compute_retrieval_relevance(
        retrieved_chunks=chunks,
        expected_source_docs=expected_sources,
        k=3,
    )

    # Both expected docs are in top 3 -> recall = 2/2 = 1.0
    assert rel["hit_rate_at_k"] == 1.0
    assert rel["recall"] == 1.0
    assert rel["mrr"] == 0.5  # First match was at rank 2


def test_retrieval_relevance_partial_recall():
    """Verify partial recall when only 1 of 2 expected sources is in top-k."""
    chunks = [
        make_chunk("c1", "docs/unrelated.md"),
        make_chunk("c2", "docs/architecture.md"),
    ]
    expected_sources = ["architecture.md", "api.md"]

    rel = compute_retrieval_relevance(
        retrieved_chunks=chunks,
        expected_source_docs=expected_sources,
        k=2,
    )

    assert rel["hit_rate_at_k"] == 1.0
    assert rel["recall"] == 0.5  # Only 1 out of 2 found
    assert rel["mrr"] == 0.5     # First match at rank 2


def test_retrieval_relevance_zero_hit():
    """Verify zero hit rate and zero MRR when none of the expected sources are retrieved."""
    chunks = [
        make_chunk("c1", "docs/unrelated1.md"),
        make_chunk("c2", "docs/unrelated2.md"),
    ]
    expected_sources = ["target_doc.md"]

    rel = compute_retrieval_relevance(
        retrieved_chunks=chunks,
        expected_source_docs=expected_sources,
        k=2,
    )

    assert rel["hit_rate_at_k"] == 0.0
    assert rel["recall"] == 0.0
    assert rel["mrr"] == 0.0


# --- 2. Metric Math: Answer Correctness ---

def test_correctness_no_answer_category():
    """Verify no_answer category correctly rewards refusal and penalizes hallucinated answer."""
    # Proper refusal -> 1.0
    res_correct = compute_answer_correctness(
        question="What is the Arctic Tern lifespan?",
        system_answer="I could not locate a sufficiently grounded answer to your inquiry.",
        golden_answer="NO_ANSWER",
        category="no_answer",
        is_fallback=True,
    )
    assert res_correct["score"] == 1.0

    # Hallucinated answer -> 0.0
    res_hallucinated = compute_answer_correctness(
        question="What is the Arctic Tern lifespan?",
        system_answer="The average lifespan of an Arctic Tern is 30 years in the wild.",
        golden_answer="NO_ANSWER",
        category="no_answer",
        is_fallback=False,
    )
    assert res_hallucinated["score"] == 0.0


def test_correctness_standard_category():
    """Verify standard correctness judging against golden answer."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = MagicMock(
        content='{"correctness_score": 0.95, "reason": "Accurately conveys facts."}'
    )
    mock_cache = MagicMock()
    mock_cache.get.return_value = None

    res = compute_answer_correctness(
        question="What is the port?",
        system_answer="The port is 8000.",
        golden_answer="Port 8000.",
        category="lookup",
        is_fallback=False,
        client=mock_llm,
        cache=mock_cache,
    )
    assert res["score"] == 0.95


# --- 3. Metric Math: Faithfulness & Token Cost ---

def test_token_cost_calculation():
    """Verify token cost arithmetic."""
    # 10,000 prompt tokens ($0.05) + 1,000 completion tokens ($0.015) = $0.065
    cost = calculate_token_cost(prompt_tokens=10_000, completion_tokens=1_000)
    assert pytest.approx(cost, 0.0001) == 0.065


# --- 4. Dataset Loader & Strict Validator ---

def test_dataset_loader_catches_duplicate_ids(tmp_path: Path):
    """Verify dataset validator raises ValueError on duplicate IDs."""
    dataset_file = tmp_path / "dup_dataset.jsonl"
    content = (
        '{"id": "c1", "category": "lookup", "question": "Q1", "golden_answer": "A1", "source_docs": [], "source_sections": []}\n'
        '{"id": "c1", "category": "lookup", "question": "Q2", "golden_answer": "A2", "source_docs": [], "source_sections": []}\n'
    )
    dataset_file.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate test case ID 'c1'"):
        load_golden_dataset(dataset_file)


def test_dataset_loader_catches_malformed_json(tmp_path: Path):
    """Verify dataset validator raises ValueError on malformed JSON lines."""
    dataset_file = tmp_path / "bad_dataset.jsonl"
    content = '{"id": "c1", "category": "lookup", "question": NOT_VALID_JSON}\n'
    dataset_file.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed JSON on line 1"):
        load_golden_dataset(dataset_file)

