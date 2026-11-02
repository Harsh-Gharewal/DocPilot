"""Unit tests for chunking strategies (fixed_size, structure_aware, semantic)."""

from unittest.mock import MagicMock
import pytest

from app.ingestion.chunkers import (
    FixedSizeChunker,
    SemanticChunker,
    StructureAwareChunker,
    get_chunker,
)
from app.ingestion.models import Document, Section


@pytest.fixture
def sample_document():
    """Create a sample multi-section document."""
    sections = [
        Section(
            section_heading="Overview",
            page_number=1,
            text="Antigravity is a modern autonomous development platform designed for high performance.",
        ),
        Section(
            section_heading="Architecture",
            page_number=2,
            text=(
                "The core architecture is built upon a hybrid search mechanism that blends dense vector embeddings "
                "with sparse BM25 keyword matching. Results are then reranked using cross-encoders to ensure maximum "
                "precision and retrieval quality."
            ),
        ),
        Section(
            section_heading="Deployment",
            page_number=3,
            text="Deployments are automated through containerized Docker instances running on cloud infrastructure.",
        ),
    ]
    full_text = "\n\n".join(s.text for s in sections)
    return Document(
        doc_id="testdoc123",
        source_file="docs/architecture.md",
        text=full_text,
        sections=sections,
    )


def test_fixed_size_chunker(sample_document):
    """Verify fixed-size chunker splits according to size and overlap."""
    chunker = FixedSizeChunker(chunk_size=120, chunk_overlap=30)
    chunks = chunker.chunk(sample_document)

    assert len(chunks) > 1
    for i, chk in enumerate(chunks):
        assert chk.strategy == "fixed_size"
        assert chk.chunk_id == f"testdoc123_fixed_size_{i:04d}"
        assert chk.doc_id == "testdoc123"
        assert len(chk.text) <= 150  # Allows slight margin for word boundary preservation
        assert chk.char_count == len(chk.text)


def test_structure_aware_chunker(sample_document):
    """Verify structure-aware chunker preserves section headers and does not mix sections."""
    # Set chunk_size large enough to keep sections intact
    chunker = StructureAwareChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk(sample_document)

    # We had 3 sections, none exceeding 500 chars, so exactly 3 chunks should be created
    assert len(chunks) == 3

    assert chunks[0].section_heading == "Overview"
    assert chunks[0].page_number == 1
    assert "Overview" in chunks[0].section_heading

    assert chunks[1].section_heading == "Architecture"
    assert chunks[1].page_number == 2
    assert "hybrid search mechanism" in chunks[1].text

    assert chunks[2].section_heading == "Deployment"
    assert chunks[2].page_number == 3

    # Now test with small chunk_size to trigger sub-splitting within a section
    sub_chunker = StructureAwareChunker(chunk_size=100, chunk_overlap=20)
    sub_chunks = sub_chunker.chunk(sample_document)
    assert len(sub_chunks) > 3

    # All sub-chunks originating from Section 2 must retain Section 2's heading and page number
    arch_sub_chunks = [c for c in sub_chunks if c.section_heading == "Architecture"]
    assert len(arch_sub_chunks) >= 2
    for c in arch_sub_chunks:
        assert c.page_number == 2


def test_semantic_chunker():
    """Verify semantic chunker splits at topic shift boundaries using sentence similarity."""
    mock_embedder = MagicMock()

    # Create 4 sentences across 2 distinct topics: Topic A (sentences 0,1) and Topic B (sentences 2,3)
    sentences = [
        "First topic sentence one about neural networks.",
        "First topic sentence two about deep learning algorithms.",
        "Second topic sentence one about international economic trade.",
        "Second topic sentence two about monetary fiscal policy.",
    ]

    # Mock embeddings: topic A vectors are [1, 0], topic B vectors are [0, 1]
    mock_embedder.embed_texts.return_value = [
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ]

    doc = Document(
        doc_id="semdoc456",
        source_file="docs/topics.md",
        text=" ".join(sentences),
        sections=[Section(section_heading="Topics", page_number=None, text=" ".join(sentences))],
    )

    chunker = SemanticChunker(
        percentile_threshold=50.0,
        max_chunk_size=1000,
        min_sentences=1,
        embedder=mock_embedder,
    )
    chunks = chunker.chunk(doc)

    assert len(chunks) == 2
    assert "neural networks" in chunks[0].text
    assert "deep learning" in chunks[0].text
    assert "economic trade" in chunks[1].text
    assert "fiscal policy" in chunks[1].text
    assert chunks[0].chunk_id == "semdoc456_semantic_0000"
    assert chunks[1].chunk_id == "semdoc456_semantic_0001"


def test_get_chunker_factory():
    """Verify factory returns appropriate instances and catches invalid inputs."""
    fixed = get_chunker("fixed_size")
    assert isinstance(fixed, FixedSizeChunker)

    struct = get_chunker("structure_aware")
    assert isinstance(struct, StructureAwareChunker)

    sem = get_chunker("semantic")
    assert isinstance(sem, SemanticChunker)

    with pytest.raises(ValueError):
        get_chunker("invalid_strategy_name")

