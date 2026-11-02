"""Persistent Dual-Index Storage (ChromaDB + BM25) and Synchronization.

Ensures dense vector store (ChromaDB) and sparse keyword index (BM25)
remain 100% synchronized with identical chunk sets and chunk IDs.
"""

import json
import logging
import pickle
import re
from pathlib import Path
from typing import Any, Optional

import chromadb
from rank_bm25 import BM25Okapi

from app.config import settings
from app.ingestion.dedup import Deduplicator
from app.ingestion.embeddings import EmbeddingClient, embedding_client
from app.ingestion.models import Chunk, DedupReport, StrategyStats

logger = logging.getLogger(__name__)


def bm25_tokenize(text: str) -> list[str]:
    """Tokenize text preserving identifiers (snake_case, config.keys, error-codes)."""
    tokens = re.findall(r"[a-zA-Z0-9_.-]+", text.lower())
    clean_tokens = [t.strip(".-") for t in tokens if len(t.strip(".-")) > 0]
    return clean_tokens


class IndexManager:
    """Manages synchronized ChromaDB dense collections and BM25 sparse indexes."""

    def __init__(
        self,
        chroma_dir: Optional[Path] = None,
        bm25_dir: Optional[Path] = None,
        embedder: Optional[EmbeddingClient] = None,
    ):
        self.chroma_dir = chroma_dir or settings.chroma_dir
        self.bm25_dir = bm25_dir or settings.bm25_dir
        self.embedder = embedder or embedding_client

        # Ensure directories exist
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.bm25_dir.mkdir(parents=True, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(path=str(self.chroma_dir))

    def get_chroma_collection(self, strategy: str):
        """Retrieve or create persistent Chroma collection for a strategy."""
        collection_name = f"chunks_{strategy}"
        return self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _get_bm25_paths(self, strategy: str) -> tuple[Path, Path]:
        """Paths to the pickled BM25 index and JSON chunk metadata store."""
        index_path = self.bm25_dir / f"{strategy}_bm25.pkl"
        chunks_path = self.bm25_dir / f"{strategy}_chunks.json"
        return index_path, chunks_path

    def load_bm25_data(self, strategy: str) -> tuple[Optional[BM25Okapi], list[dict[str, Any]]]:
        """Load the persisted BM25Okapi instance and its associated chunk data."""
        index_path, chunks_path = self._get_bm25_paths(strategy)
        if not (index_path.exists() and chunks_path.exists()):
            return None, []

        try:
            with open(index_path, "rb") as f:
                bm25_model = pickle.load(f)
            chunks_data = json.loads(chunks_path.read_text(encoding="utf-8"))
            return bm25_model, chunks_data
        except Exception as e:
            logger.warning(f"Failed to load BM25 index for strategy '{strategy}': {e}")
            return None, []

    def _save_bm25_data(
        self, strategy: str, bm25_model: BM25Okapi, chunks_data: list[dict[str, Any]]
    ) -> None:
        """Persist BM25 model and JSON chunk metadata to disk."""
        index_path, chunks_path = self._get_bm25_paths(strategy)
        with open(index_path, "wb") as f:
            pickle.dump(bm25_model, f)
        chunks_path.write_text(json.dumps(chunks_data, indent=2), encoding="utf-8")

    def index_chunks(
        self,
        chunks: list[Chunk],
        strategy: str,
        custom_embeddings: Optional[list[list[float]]] = None,
    ) -> DedupReport:
        """Embed, deduplicate, and index chunks into both ChromaDB and BM25.

        Maintains strict synchronization between both indexes.
        """
        if not chunks:
            logger.info(f"No chunks provided to index for strategy '{strategy}'.")
            return DedupReport()

        collection = self.get_chroma_collection(strategy)

        # 1. Compute or accept embeddings for candidate chunks
        if custom_embeddings is not None:
            candidate_embeddings = custom_embeddings
        else:
            texts = [c.text for c in chunks]
            candidate_embeddings = self.embedder.embed_texts(texts)

        # 2. Run Deduplication (intra-batch + cross-batch against Chroma)
        deduplicator = Deduplicator(threshold=settings.dedup_similarity_threshold)
        unique_chunks, unique_embeddings, report = deduplicator.filter_batch(
            chunks=chunks,
            embeddings=candidate_embeddings,
            chroma_collection=collection,
        )

        if not unique_chunks:
            logger.info(
                f"All {report.chunks_seen} candidate chunks were identified as duplicates for strategy '{strategy}'."
            )
            # Verify sync still holds
            self.verify_index_sync(strategy)
            return report

        # 3. Add to ChromaDB
        ids = [c.chunk_id for c in unique_chunks]
        documents = [c.text for c in unique_chunks]
        metadatas = [
            {
                "source_file": c.source_file,
                "doc_id": c.doc_id,
                "chunk_index": c.chunk_index,
                "section_heading": c.section_heading or "",
                "page_number": c.page_number if c.page_number is not None else -1,
                "strategy": c.strategy,
                "char_count": c.char_count,
            }
            for c in unique_chunks
        ]

        collection.upsert(
            ids=ids,
            embeddings=unique_embeddings,
            documents=documents,
            metadatas=metadatas,
        )

        # 4. Update BM25 Index
        _, existing_chunks_data = self.load_bm25_data(strategy)
        chunk_map = {item["chunk_id"]: item for item in existing_chunks_data}

        for c in unique_chunks:
            chunk_map[c.chunk_id] = {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "source_file": c.source_file,
                "chunk_index": c.chunk_index,
                "text": c.text,
                "section_heading": c.section_heading,
                "page_number": c.page_number,
                "strategy": c.strategy,
                "char_count": c.char_count,
            }

        all_active_chunks = list(chunk_map.values())
        tokenized_corpus = [bm25_tokenize(item["text"]) for item in all_active_chunks]
        new_bm25 = BM25Okapi(tokenized_corpus)

        self._save_bm25_data(strategy, new_bm25, all_active_chunks)

        logger.info(
            f"Indexed {len(unique_chunks)} chunks for strategy '{strategy}'. "
            f"(Total active in BM25: {len(all_active_chunks)}, Chroma: {collection.count()})"
        )

        # 5. Verify synchronization
        self.verify_index_sync(strategy)

        return report

    def verify_index_sync(self, strategy: str) -> bool:
        """Verify that the chunk_id sets in Chroma and BM25 match exactly.

        Raises ValueError if there is any discrepancy.
        """
        collection = self.get_chroma_collection(strategy)
        chroma_res = collection.get(include=[])
        chroma_ids = set(chroma_res["ids"]) if chroma_res and "ids" in chroma_res else set()

        _, bm25_chunks = self.load_bm25_data(strategy)
        bm25_ids = {c["chunk_id"] for c in bm25_chunks}

        if chroma_ids != bm25_ids:
            missing_in_bm25 = chroma_ids - bm25_ids
            missing_in_chroma = bm25_ids - chroma_ids
            err_msg = (
                f"Index desynchronization detected for strategy '{strategy}'! "
                f"Chroma has {len(chroma_ids)} chunks, BM25 has {len(bm25_ids)} chunks. "
                f"Missing in BM25: {len(missing_in_bm25)}, Missing in Chroma: {len(missing_in_chroma)}."
            )
            logger.error(err_msg)
            raise ValueError(err_msg)

        logger.info(
            f"Sync check PASSED for strategy '{strategy}'. Chroma and BM25 share identical {len(chroma_ids)} chunks."
        )
        return True

    def get_strategy_stats(self, strategy: str) -> StrategyStats:
        """Retrieve chunk counts and average character length for a given strategy."""
        _, bm25_chunks = self.load_bm25_data(strategy)
        if not bm25_chunks:
            # Check Chroma count as well
            collection = self.get_chroma_collection(strategy)
            return StrategyStats(
                strategy=strategy,
                chunk_count=collection.count(),
                avg_char_count=0.0,
                duplicates_skipped=0,
            )

        total_chars = sum(c.get("char_count", len(c.get("text", ""))) for c in bm25_chunks)
        avg_chars = round(total_chars / len(bm25_chunks), 2) if bm25_chunks else 0.0

        return StrategyStats(
            strategy=strategy,
            chunk_count=len(bm25_chunks),
            avg_char_count=avg_chars,
            duplicates_skipped=0,
        )


# Singleton instance
index_manager = IndexManager()


def index_chunks(
    chunks: list[Chunk],
    strategy: str,
    custom_embeddings: Optional[list[list[float]]] = None,
) -> DedupReport:
    """Convenience functional wrapper for index_chunks."""
    return index_manager.index_chunks(chunks, strategy, custom_embeddings=custom_embeddings)


def verify_index_sync(strategy: str) -> bool:
    """Convenience functional wrapper for verify_index_sync."""
    return index_manager.verify_index_sync(strategy)

