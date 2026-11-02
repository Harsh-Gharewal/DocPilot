"""Unit tests for deduplication logic (intra-batch and cross-batch)."""

from unittest.mock import MagicMock
import pytest

from app.ingestion.dedup import Deduplicator, cosine_similarity
from app.ingestion.models import Chunk


def make_test_chunk(chunk_id: str, text: str) -> Chunk:
    """Helper to construct minimal test Chunk."""
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc1",
        source_file="doc1.txt",
        chunk_index=0,
        text=text,
        strategy="structure_aware",
        char_count=len(text),
    )


def test_cosine_similarity():
    """Verify cosine similarity calculation math."""
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    assert pytest.approx(cosine_similarity(v1, v2), 0.0001) == 1.0

    v3 = [0.0, 1.0, 0.0]
    assert pytest.approx(cosine_similarity(v1, v3), 0.0001) == 0.0

    v_zero = [0.0, 0.0, 0.0]
    assert cosine_similarity(v1, v_zero) == 0.0


def test_intra_batch_dedup():
    """Verify that near-identical chunks in the same batch are detected and skipped."""
    deduplicator = Deduplicator(threshold=0.95)

    c1 = make_test_chunk("c1", "The primary server runs on port 8000.")
    c2 = make_test_chunk("c2", "The primary server runs on port 8000 with TLS.")
    c3 = make_test_chunk("c3", "Something completely unrelated about databases.")

    # c1 and c2 have very high cosine similarity (>0.95), c3 has low similarity
    emb1 = [1.0, 0.0, 0.0]
    emb2 = [0.98, 0.02, 0.0]  # Cosine similarity is ~0.9998
    emb3 = [0.0, 1.0, 0.0]

    unique_chunks, unique_embs, report = deduplicator.filter_batch(
        chunks=[c1, c2, c3],
        embeddings=[emb1, emb2, emb3],
        chroma_collection=None,
    )

    assert len(unique_chunks) == 2
    assert unique_chunks[0].chunk_id == "c1"
    assert unique_chunks[1].chunk_id == "c3"
    assert report.chunks_seen == 3
    assert report.chunks_inserted == 2
    assert report.duplicates_skipped == 1
    assert len(report.duplicate_pairs) == 1
    assert report.duplicate_pairs[0]["candidate_chunk_id"] == "c2"
    assert report.duplicate_pairs[0]["duplicate_of_id"] == "c1"


def test_cross_batch_chroma_dedup():
    """Verify that chunks matching an existing chunk in ChromaDB are flagged and skipped."""
    deduplicator = Deduplicator(threshold=0.95)

    mock_collection = MagicMock()
    mock_collection.count.return_value = 1
    # In Chroma cosine space: distance = 1 - sim => distance = 0.02 means sim = 0.98 >= 0.95
    mock_collection.query.return_value = {
        "ids": [["existing_chunk_001"]],
        "distances": [[0.02]],
    }

    c_candidate = make_test_chunk("c_new", "The server configuration has been updated.")
    candidate_emb = [0.99, 0.01, 0.0]

    unique_chunks, _, report = deduplicator.filter_batch(
        chunks=[c_candidate],
        embeddings=[candidate_emb],
        chroma_collection=mock_collection,
    )

    assert len(unique_chunks) == 0
    assert report.duplicates_skipped == 1
    assert report.duplicate_pairs[0]["candidate_chunk_id"] == "c_new"
    assert report.duplicate_pairs[0]["duplicate_of_id"] == "existing_chunk_001"
    assert report.duplicate_pairs[0]["similarity"] >= 0.95

