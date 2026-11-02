"""Sparse keyword retrieval using persisted BM25Okapi indexes."""

import logging
from typing import Optional

import numpy as np

from app.config import settings
from app.ingestion.indexer import IndexManager, bm25_tokenize, index_manager
from app.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)


class SparseRetriever:
    """Retrieves top-k chunks ranked by BM25 score using persisted sparse indexes."""

    def __init__(
        self,
        index_mgr: Optional[IndexManager] = None,
        default_top_k: Optional[int] = None,
    ):
        self.index_mgr = index_mgr or index_manager
        self.default_top_k = default_top_k or settings.top_k_retrieval

    def retrieve(
        self,
        query: str,
        strategy: str = "structure_aware",
        top_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """Score all chunks against query using BM25Okapi and return top-k matches."""
        k = top_k or self.default_top_k
        bm25_model, chunks_data = self.index_mgr.load_bm25_data(strategy)

        if not bm25_model or not chunks_data:
            logger.warning(f"BM25 index for strategy '{strategy}' is empty or not found.")
            return []

        # Tokenize using the same identifier-preserving tokenizer as indexing
        tokenized_query = bm25_tokenize(query)
        if not tokenized_query:
            logger.warning(f"Sparse query tokenization resulted in empty tokens for query: '{query}'")
            return []

        # Calculate BM25 scores
        scores = bm25_model.get_scores(tokenized_query)

        # Get indices sorted by score descending
        sorted_indices = np.argsort(scores)[::-1]

        retrieved: list[RetrievedChunk] = []
        rank = 1

        for idx in sorted_indices:
            score = float(scores[idx])
            # Only include documents with non-zero BM25 score
            if score <= 0.0 and len(retrieved) > 0:
                break
            if len(retrieved) >= k:
                break

            chunk_info = chunks_data[idx]
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_info["chunk_id"],
                    text=chunk_info["text"],
                    metadata={
                        "source_file": chunk_info.get("source_file", ""),
                        "doc_id": chunk_info.get("doc_id", ""),
                        "chunk_index": chunk_info.get("chunk_index", 0),
                        "section_heading": chunk_info.get("section_heading", ""),
                        "page_number": chunk_info.get("page_number"),
                        "strategy": chunk_info.get("strategy", strategy),
                        "char_count": chunk_info.get("char_count", len(chunk_info["text"])),
                    },
                    score=round(score, 6),
                    rank=rank,
                    retriever="sparse",
                    raw_scores={"sparse": round(score, 6)},
                )
            )
            rank += 1

        logger.debug(f"SparseRetriever retrieved {len(retrieved)} chunk(s) for query: '{query}'")
        return retrieved


# Singleton instance
sparse_retriever = SparseRetriever()

