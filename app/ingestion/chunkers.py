"""Configurable chunking strategies implementing a unified Chunker protocol.

Provides three switchable strategies:
1. fixed_size: Fixed-size character chunks with overlap.
2. structure_aware: Recursive character splitting respecting headers, paragraphs, and sentences.
3. semantic: Topic-boundary splitting using sentence embeddings and percentile similarity drops.
"""

import logging
import re
from typing import Optional, Protocol, runtime_checkable

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.ingestion.embeddings import EmbeddingClient, embedding_client
from app.ingestion.models import Chunk, Document, Section

logger = logging.getLogger(__name__)


@runtime_checkable
class Chunker(Protocol):
    """Protocol that all chunking strategies must implement."""

    strategy_name: str

    def chunk(self, document: Document) -> list[Chunk]:
        """Split a Document into an ordered list of deterministic Chunks."""
        ...


def make_chunk(
    document: Document,
    chunk_index: int,
    text: str,
    strategy: str,
    section_heading: Optional[str] = None,
    page_number: Optional[int] = None,
) -> Chunk:
    """Construct a Chunk instance with a deterministic chunk_id."""
    clean_text = text.strip()
    return Chunk(
        chunk_id=f"{document.doc_id}_{strategy}_{chunk_index:04d}",
        doc_id=document.doc_id,
        source_file=document.source_file,
        chunk_index=chunk_index,
        text=clean_text,
        section_heading=section_heading,
        page_number=page_number,
        strategy=strategy,
        char_count=len(clean_text),
    )


class FixedSizeChunker:
    """Fixed-size character chunking with configurable size and overlap."""

    strategy_name = "fixed_size"

    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def chunk(self, document: Document) -> list[Chunk]:
        """Split document into fixed-size chunks, inferring nearest section context."""
        raw_chunks = self.splitter.split_text(document.text)
        chunks: list[Chunk] = []

        # Find best-matching section for each chunk based on text containment
        for idx, text in enumerate(raw_chunks):
            heading: Optional[str] = None
            page_num: Optional[int] = None

            for sec in document.sections:
                if text[:60] in sec.text or sec.text[:60] in text:
                    heading = sec.section_heading
                    page_num = sec.page_number
                    break

            if not heading and document.sections:
                heading = document.sections[0].section_heading
                page_num = document.sections[0].page_number

            chunks.append(
                make_chunk(
                    document=document,
                    chunk_index=idx,
                    text=text,
                    strategy=self.strategy_name,
                    section_heading=heading,
                    page_number=page_num,
                )
            )

        return chunks


class StructureAwareChunker:
    """Splits text by respecting section headers first, then paragraphs, then sentences."""

    strategy_name = "structure_aware"

    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.sub_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk(self, document: Document) -> list[Chunk]:
        """Split each section independently so chunk boundaries never cross section headers."""
        chunks: list[Chunk] = []
        chunk_idx = 0

        sections = document.sections or [
            Section(section_heading="Overview", page_number=None, text=document.text)
        ]

        for sec in sections:
            sec_text = sec.text.strip()
            if not sec_text:
                continue

            if len(sec_text) <= self.chunk_size:
                chunks.append(
                    make_chunk(
                        document=document,
                        chunk_index=chunk_idx,
                        text=sec_text,
                        strategy=self.strategy_name,
                        section_heading=sec.section_heading,
                        page_number=sec.page_number,
                    )
                )
                chunk_idx += 1
            else:
                sub_texts = self.sub_splitter.split_text(sec_text)
                for st in sub_texts:
                    if st.strip():
                        chunks.append(
                            make_chunk(
                                document=document,
                                chunk_index=chunk_idx,
                                text=st,
                                strategy=self.strategy_name,
                                section_heading=sec.section_heading,
                                page_number=sec.page_number,
                            )
                        )
                        chunk_idx += 1

        return chunks


class SemanticChunker:
    """Splits on topic boundaries by embedding sentence windows and detecting similarity drops."""

    strategy_name = "semantic"

    def __init__(
        self,
        percentile_threshold: Optional[float] = None,
        max_chunk_size: Optional[int] = None,
        min_sentences: int = 2,
        embedder: Optional[EmbeddingClient] = None,
    ):
        self.percentile_threshold = (
            percentile_threshold
            if percentile_threshold is not None
            else settings.semantic_similarity_percentile
        )
        self.max_chunk_size = max_chunk_size or (settings.chunk_size * 2)
        self.min_sentences = min_sentences
        self.embedder = embedder or embedding_client

    @staticmethod
    def _split_into_sentences(text: str) -> list[str]:
        """Split text into sentences cleanly using regex punctuation boundaries."""
        raw_sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        return sentences

    @staticmethod
    def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
        """Compute cosine similarity between two numeric vectors."""
        a = np.array(v1, dtype=np.float32)
        b = np.array(v2, dtype=np.float32)
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def chunk(self, document: Document) -> list[Chunk]:
        """Split document semantically by sentence embeddings with fallback for short documents."""
        sentences = self._split_into_sentences(document.text)

        # If too few sentences, return whole text as a single chunk
        if len(sentences) <= self.min_sentences:
            return [
                make_chunk(
                    document=document,
                    chunk_index=0,
                    text=document.text,
                    strategy=self.strategy_name,
                    section_heading=document.sections[0].section_heading if document.sections else None,
                    page_number=document.sections[0].page_number if document.sections else None,
                )
            ]

        # Embed sentences (using local SQLite cache for cheap re-runs)
        sentence_embeddings = self.embedder.embed_texts(sentences)

        # Compute cosine similarities between adjacent sentences
        similarities: list[float] = []
        for i in range(len(sentence_embeddings) - 1):
            sim = self._cosine_similarity(sentence_embeddings[i], sentence_embeddings[i + 1])
            similarities.append(sim)

        # Determine drop threshold
        threshold = float(np.percentile(similarities, self.percentile_threshold))

        chunks: list[Chunk] = []
        current_chunk_sentences: list[str] = [sentences[0]]
        chunk_idx = 0

        for i, sim in enumerate(similarities):
            next_sentence = sentences[i + 1]
            current_text_len = sum(len(s) for s in current_chunk_sentences)

            # Split if similarity drops below percentile threshold or length exceeds maximum
            should_split = (sim < threshold and len(current_chunk_sentences) >= self.min_sentences) or (
                current_text_len + len(next_sentence) > self.max_chunk_size
            )

            if should_split:
                chunk_text = " ".join(current_chunk_sentences)
                # Map to section
                heading: Optional[str] = None
                page_num: Optional[int] = None
                for sec in document.sections:
                    if current_chunk_sentences[0][:40] in sec.text:
                        heading = sec.section_heading
                        page_num = sec.page_number
                        break

                chunks.append(
                    make_chunk(
                        document=document,
                        chunk_index=chunk_idx,
                        text=chunk_text,
                        strategy=self.strategy_name,
                        section_heading=heading,
                        page_number=page_num,
                    )
                )
                chunk_idx += 1
                current_chunk_sentences = [next_sentence]
            else:
                current_chunk_sentences.append(next_sentence)

        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            heading = None
            page_num = None
            for sec in document.sections:
                if current_chunk_sentences[0][:40] in sec.text:
                    heading = sec.section_heading
                    page_num = sec.page_number
                    break

            chunks.append(
                make_chunk(
                    document=document,
                    chunk_index=chunk_idx,
                    text=chunk_text,
                    strategy=self.strategy_name,
                    section_heading=heading,
                    page_number=page_num,
                )
            )

        return chunks


def get_chunker(strategy: Optional[str] = None) -> Chunker:
    """Factory creating a Chunker based on the requested strategy or config setting."""
    strat = strategy or settings.default_chunking_strategy
    strat = strat.lower()

    if strat == "fixed_size":
        return FixedSizeChunker()
    elif strat == "structure_aware":
        return StructureAwareChunker()
    elif strat == "semantic":
        return SemanticChunker()
    else:
        raise ValueError(
            f"Unknown chunking strategy '{strat}'. Valid strategies: fixed_size, structure_aware, semantic."
        )
