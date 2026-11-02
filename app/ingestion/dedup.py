"""Deduplication logic for chunk indexing.

Implements intra-batch and cross-batch nearest-neighbor deduplication using cosine similarity.
"""

import logging
from typing import Any, Optional

import numpy as np

from app.config import settings
from app.ingestion.models import Chunk, DedupReport

logger = logging.getLogger(__name__)


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute the cosine similarity between two float vectors."""
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class Deduplicator:
    """Deduplicates chunks intra-batch and against existing ChromaDB vectors."""

    def __init__(self, threshold: Optional[float] = None):
        self.threshold = (
            threshold if threshold is not None else settings.dedup_similarity_threshold
        )

    def filter_batch(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        chroma_collection: Optional[Any] = None,
    ) -> tuple[list[Chunk], list[list[float]], DedupReport]:
        """Filter out duplicates from candidate chunks both intra-batch and against ChromaDB.

        Returns:
            (unique_chunks, unique_embeddings, dedup_report)
        """
        report = DedupReport(chunks_seen=len(chunks))
        accepted_chunks: list[Chunk] = []
        accepted_embeddings: list[list[float]] = []

        for chunk, emb in zip(chunks, embeddings):
            is_dup = False
            dup_with_id = ""
            sim_score = 0.0

            # 1. Check against accepted items within the current batch
            for prev_chunk, prev_emb in zip(accepted_chunks, accepted_embeddings):
                sim = cosine_similarity(emb, prev_emb)
                if sim >= self.threshold:
                    is_dup = True
                    dup_with_id = prev_chunk.chunk_id
                    sim_score = sim
                    break

            # 2. Check against existing ChromaDB collection
            if not is_dup and chroma_collection is not None and chroma_collection.count() > 0:
                try:
                    res = chroma_collection.query(
                        query_embeddings=[emb],
                        n_results=1,
                        include=["distances"],
                    )
                    if res and res.get("distances") and res["distances"][0]:
                        dist = res["distances"][0][0]
                        # With cosine space in ChromaDB: dist = 1 - sim => sim = 1 - dist
                        existing_sim = 1.0 - dist
                        if existing_sim >= self.threshold:
                            is_dup = True
                            dup_with_id = res["ids"][0][0]
                            sim_score = existing_sim
                except Exception as e:
                    logger.warning(f"Error querying ChromaDB for deduplication: {e}")

            if is_dup:
                report.duplicates_skipped += 1
                logger.warning(
                    f"Duplicate detected: chunk '{chunk.chunk_id}' matches '{dup_with_id}' "
                    f"with cosine similarity {sim_score:.4f} >= threshold {self.threshold:.4f}. Skipping."
                )
                report.duplicate_pairs.append(
                    {
                        "candidate_chunk_id": chunk.chunk_id,
                        "duplicate_of_id": dup_with_id,
                        "similarity": round(sim_score, 4),
                        "text_preview": chunk.text[:80],
                    }
                )
            else:
                accepted_chunks.append(chunk)
                accepted_embeddings.append(emb)

        report.chunks_inserted = len(accepted_chunks)
        return accepted_chunks, accepted_embeddings, report

