"""Unit tests for persistent indexer and ChromaDB / BM25 index synchronization."""

from pathlib import Path
from unittest.mock import MagicMock
import pytest

from app.ingestion.indexer import IndexManager, bm25_tokenize
from app.ingestion.models import Chunk


def make_chunk(chunk_id: str, text: str, strategy: str = "structure_aware") -> Chunk:
    """Helper to create chunk for testing indexer."""
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc_test",
        source_file="test.md",
        chunk_index=0,
        text=text,
        strategy=strategy,
        char_count=len(text),
    )


def test_bm25_tokenize():
    """Verify tokenizer keeps identifiers, config keys, and error codes intact."""
    sample = "Call verify_user_token with auth.jwt_secret for error-503 or HTTP 404!"
    tokens = bm25_tokenize(sample)

    assert "verify_user_token" in tokens  # snake_case preserved
    assert "auth.jwt_secret" in tokens    # config.keys preserved
    assert "error-503" in tokens          # error codes preserved
    assert "404" in tokens
    assert "!" not in "".join(tokens)


def test_indexer_dual_storage_and_sync(tmp_path: Path):
    """Verify that chunks are indexed into both ChromaDB and BM25 and remain 100% in sync."""
    chroma_dir = tmp_path / "chroma"
    bm25_dir = tmp_path / "bm25"

    manager = IndexManager(chroma_dir=chroma_dir, bm25_dir=bm25_dir)
    strategy = "structure_aware"

    c1 = make_chunk("chunk_001", "FastAPI provides high-performance asynchronous endpoints.")
    c2 = make_chunk("chunk_002", "ChromaDB stores persistent dense vector embeddings.")

    # Distinct mock embeddings to avoid dedup collisions
    embs = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]

    report = manager.index_chunks(
        chunks=[c1, c2],
        strategy=strategy,
        custom_embeddings=embs,
    )

    assert report.chunks_seen == 2
    assert report.chunks_inserted == 2
    assert report.duplicates_skipped == 0

    # Verify sync passes
    assert manager.verify_index_sync(strategy) is True

    # Verify BM25 model and chunk metadata exist on disk
    bm25_model, bm25_chunks = manager.load_bm25_data(strategy)
    assert bm25_model is not None
    assert len(bm25_chunks) == 2
    assert {c["chunk_id"] for c in bm25_chunks} == {"chunk_001", "chunk_002"}

    # Verify Chroma collection count
    collection = manager.get_chroma_collection(strategy)
    assert collection.count() == 2

    # Verify stats
    stats = manager.get_strategy_stats(strategy)
    assert stats.chunk_count == 2
    assert stats.avg_char_count > 0.0


def test_indexer_sync_failure_detection(tmp_path: Path):
    """Verify verify_index_sync raises ValueError when an index discrepancy occurs."""
    chroma_dir = tmp_path / "chroma"
    bm25_dir = tmp_path / "bm25"

    manager = IndexManager(chroma_dir=chroma_dir, bm25_dir=bm25_dir)
    strategy = "fixed_size"

    c1 = make_chunk("chunk_a", "Chunk A content.")
    embs = [[1.0, 0.0, 0.0]]

    manager.index_chunks(chunks=[c1], strategy=strategy, custom_embeddings=embs)
    assert manager.verify_index_sync(strategy) is True

    # Artificially desynchronize Chroma by inserting an extra chunk directly
    collection = manager.get_chroma_collection(strategy)
    collection.add(
        ids=["chunk_rogue"],
        embeddings=[[0.0, 0.0, 1.0]],
        documents=["Rogue chunk content"],
    )

    with pytest.raises(ValueError, match="Index desynchronization detected"):
        manager.verify_index_sync(strategy)

