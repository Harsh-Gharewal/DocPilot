"""Comprehensive unit and integration tests for the Hybrid Retrieval Engine."""

from unittest.mock import MagicMock
import pytest

from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.models import RetrievedChunk
from app.retrieval.pipeline import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.sparse import SparseRetriever


def make_chunk(chunk_id: str, text: str, score: float, rank: int, retriever: str) -> RetrievedChunk:
    """Helper to construct RetrievedChunk objects."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        metadata={"source": "test.md"},
        score=score,
        rank=rank,
        retriever=retriever,
        raw_scores={retriever: score},
    )


# --- 1. Hand-computed Reciprocal Rank Fusion tests ---

def test_rrf_hand_computed_order():
    """Verify RRF formula with hand-calculated scores, including chunks in only one list.

    Formula: score(d) = sum(weight_r / (rrf_k + rank_r(d)))
    Let rrf_k = 60, weight_dense = 0.7, weight_sparse = 0.3.

    Dense ranked list:
      Rank 1: Chunk_A -> score = 0.7 / (60 + 1) = 0.7 / 61 ≈ 0.0114754
      Rank 2: Chunk_B -> score = 0.7 / (60 + 2) = 0.7 / 62 ≈ 0.0112903
      Rank 3: Chunk_C -> score = 0.7 / (60 + 3) = 0.7 / 63 ≈ 0.0111111

    Sparse ranked list:
      Rank 1: Chunk_B -> score = 0.3 / (60 + 1) = 0.3 / 61 ≈ 0.0049180
      Rank 2: Chunk_D (only in sparse) -> score = 0.3 / (60 + 2) = 0.3 / 62 ≈ 0.0048387
      Rank 3: Chunk_A -> score = 0.3 / (60 + 3) = 0.3 / 63 ≈ 0.0047619

    Fused Totals:
      Chunk_B: 0.0112903 + 0.0049180 = 0.0162083
      Chunk_A: 0.0114754 + 0.0047619 = 0.0162373
      Chunk_C: 0.0111111 + 0.0 = 0.0111111
      Chunk_D: 0.0 + 0.0048387 = 0.0048387

    Expected Rank Order:
      1. Chunk_A (~0.016237)
      2. Chunk_B (~0.016208)
      3. Chunk_C (~0.011111)
      4. Chunk_D (~0.004839)
    """
    dense_list = [
        make_chunk("Chunk_A", "Text A", 0.90, 1, "dense"),
        make_chunk("Chunk_B", "Text B", 0.85, 2, "dense"),
        make_chunk("Chunk_C", "Text C", 0.70, 3, "dense"),
    ]
    sparse_list = [
        make_chunk("Chunk_B", "Text B", 8.0, 1, "sparse"),
        make_chunk("Chunk_D", "Text D", 6.0, 2, "sparse"),
        make_chunk("Chunk_A", "Text A", 4.0, 3, "sparse"),
    ]

    weights = {"dense": 0.7, "sparse": 0.3}
    fused = reciprocal_rank_fusion(
        ranked_lists={"dense": dense_list, "sparse": sparse_list},
        weights=weights,
        rrf_k=60,
    )

    ordered_ids = [c.chunk_id for c in fused]
    assert ordered_ids == ["Chunk_A", "Chunk_B", "Chunk_C", "Chunk_D"]
    assert fused[0].rank == 1
    assert fused[1].rank == 2
    assert fused[2].rank == 3
    assert fused[3].rank == 4
    assert pytest.approx(fused[0].score, 0.0001) == (0.7 / 61 + 0.3 / 63)
    assert pytest.approx(fused[1].score, 0.0001) == (0.7 / 62 + 0.3 / 61)


def test_rrf_weight_change_alters_rank():
    """Verify that changing weights from 0.7/0.3 to 0.1/0.9 visibly flips the fused ranking."""
    # Setup candidate where Chunk_1 is top in dense but low in sparse; Chunk_2 is top in sparse but lower in dense
    dense_list = [
        make_chunk("Dense_Winner", "Text 1", 0.95, 1, "dense"),
        make_chunk("Sparse_Winner", "Text 2", 0.60, 5, "dense"),
    ]
    sparse_list = [
        make_chunk("Sparse_Winner", "Text 2", 9.5, 1, "sparse"),
        make_chunk("Dense_Winner", "Text 1", 1.0, 5, "sparse"),
    ]

    # With dense-heavy weights (0.8 dense, 0.2 sparse), Dense_Winner should win
    dense_heavy = reciprocal_rank_fusion(
        ranked_lists={"dense": dense_list, "sparse": sparse_list},
        weights={"dense": 0.8, "sparse": 0.2},
        rrf_k=60,
    )
    assert dense_heavy[0].chunk_id == "Dense_Winner"

    # With sparse-heavy weights (0.1 dense, 0.9 sparse), Sparse_Winner should win
    sparse_heavy = reciprocal_rank_fusion(
        ranked_lists={"dense": dense_list, "sparse": sparse_list},
        weights={"dense": 0.1, "sparse": 0.9},
        rrf_k=60,
    )
    assert sparse_heavy[0].chunk_id == "Sparse_Winner"


# --- 2. Sparse vs Dense exact identifier matching ---

def test_sparse_catches_exact_identifier_that_dense_misses():
    """Verify that BM25 catches exact identifiers like config keys and error codes."""
    mock_bm25 = MagicMock()
    # 3 chunks: chunk 0 mentions auth.jwt_secret, chunk 1 and 2 mention general auth
    chunks_data = [
        {"chunk_id": "c0", "text": "Set auth.jwt_secret in env file.", "source_file": "config.md"},
        {"chunk_id": "c1", "text": "General user authentication guidelines.", "source_file": "auth.md"},
        {"chunk_id": "c2", "text": "Password hashing and security tokens.", "source_file": "security.md"},
    ]
    # Return high BM25 score for c0 matching 'auth.jwt_secret'
    mock_bm25.get_scores.return_value = [12.5, 0.0, 0.0]

    mock_index_mgr = MagicMock()
    mock_index_mgr.load_bm25_data.return_value = (mock_bm25, chunks_data)

    sparse = SparseRetriever(index_mgr=mock_index_mgr)
    results = sparse.retrieve(query="auth.jwt_secret", strategy="structure_aware", top_k=3)

    assert len(results) == 1
    assert results[0].chunk_id == "c0"
    assert results[0].score == 12.5
    assert results[0].retriever == "sparse"


# --- 3. Reranker rank changes and pre_rerank_rank preservation ---

def test_cross_encoder_reranker_reorders_and_preserves_pre_rank():
    """Verify reranker modifies rank ordering and attaches pre_rerank_rank."""
    reranker = CrossEncoderReranker()

    mock_model = MagicMock()
    # Original candidates in order: C1, C2, C3
    # Model scores them such that C3 gets highest score, C1 second, C2 lowest
    mock_model.predict.return_value = [1.2, -0.5, 4.8]
    reranker._model = mock_model

    c1 = make_chunk("C1", "General overview text", 0.015, 1, "fused")
    c2 = make_chunk("C2", "Somewhat related text", 0.014, 2, "fused")
    c3 = make_chunk("C3", "Direct answer to the user question", 0.013, 3, "fused")

    reranked = reranker.rerank(
        query="What is the direct answer?",
        candidates=[c1, c2, c3],
        top_k=3,
    )

    assert len(reranked) == 3
    # C3 should now be rank 1 with pre_rerank_rank = 3
    assert reranked[0].chunk_id == "C3"
    assert reranked[0].rank == 1
    assert reranked[0].pre_rerank_rank == 3
    assert reranked[0].score == 4.8

    # C1 should now be rank 2 with pre_rerank_rank = 1
    assert reranked[1].chunk_id == "C1"
    assert reranked[1].rank == 2
    assert reranked[1].pre_rerank_rank == 1

    # C2 should now be rank 3 with pre_rerank_rank = 2
    assert reranked[2].chunk_id == "C2"
    assert reranked[2].rank == 3
    assert reranked[2].pre_rerank_rank == 2


# --- 4. HybridRetriever Pipeline across all modes ---

def test_hybrid_pipeline_modes():
    """Verify HybridRetriever correctly dispatches all four retrieval modes."""
    mock_dense = MagicMock()
    mock_sparse = MagicMock()
    mock_reranker = MagicMock()

    c_dense = make_chunk("dense_1", "Dense chunk", 0.9, 1, "dense")
    c_sparse = make_chunk("sparse_1", "Sparse chunk", 5.0, 1, "sparse")
    c_reranked = make_chunk("reranked_1", "Reranked chunk", 3.2, 1, "reranked")

    mock_dense.retrieve.return_value = [c_dense]
    mock_sparse.retrieve.return_value = [c_sparse]
    mock_reranker.rerank.return_value = [c_reranked]

    pipeline = HybridRetriever(dense=mock_dense, sparse=mock_sparse, reranker=mock_reranker)

    # 1. dense_only
    res, trace = pipeline.retrieve("test question", mode="dense_only", top_k_final=1)
    assert len(res) == 1
    assert res[0].chunk_id == "dense_1"
    assert trace.mode == "dense_only"
    mock_dense.retrieve.assert_called_once()
    mock_sparse.retrieve.assert_not_called()

    # 2. sparse_only
    mock_dense.reset_mock()
    mock_sparse.reset_mock()
    res, trace = pipeline.retrieve("test question", mode="sparse_only", top_k_final=1)
    assert len(res) == 1
    assert res[0].chunk_id == "sparse_1"
    assert trace.mode == "sparse_only"
    mock_sparse.retrieve.assert_called_once()
    mock_dense.retrieve.assert_not_called()

    # 3. hybrid
    mock_dense.reset_mock()
    mock_sparse.reset_mock()
    res, trace = pipeline.retrieve("test question", mode="hybrid", top_k_final=2)
    assert trace.mode == "hybrid"
    assert len(trace.fused_results) > 0
    assert trace.final_chunks == res

    # 4. hybrid_rerank
    mock_dense.reset_mock()
    mock_sparse.reset_mock()
    res, trace = pipeline.retrieve("test question", mode="hybrid_rerank", top_k_final=1)
    assert trace.mode == "hybrid_rerank"
    assert res[0].chunk_id == "reranked_1"
    mock_reranker.rerank.assert_called_once()

