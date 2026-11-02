"""Embedding client wrapper with batching, retries, and local caching.

Uses OpenAI text-embedding-3-small with exponential backoff on rate limits
and local SQLite caching to keep re-runs cheap and fast.
"""

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

import numpy as np
from openai import OpenAI, RateLimitError, APIError

from app.config import settings

logger = logging.getLogger(__name__)


class EmbeddingClient:
    """Wrapper around OpenAI text-embedding-3-small with caching and retry logic."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        cache_path: Optional[Path] = None,
    ):
        self.api_key = api_key or settings.openai_api_key
        self.provider = settings.embedding_provider
        self.model = model or (
            settings.openai_embedding_model
            if self.provider == "openai"
            else settings.local_embedding_model
        )
        self.cache_path = cache_path or (settings.processed_data_dir / "embedding_cache.sqlite")
        self._client: Optional[OpenAI] = None
        self._local_model = None
        self._init_cache()

    def _get_local_model(self):
        """Lazily initialize sentence-transformers model for local embeddings."""
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading local embedding model: {self.model}")
            self._local_model = SentenceTransformer(self.model)
        return self._local_model

    def _get_client(self) -> OpenAI:
        """Lazily initialize OpenAI client."""
        if self._client is None:
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY is not configured in settings or environment.")
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def _init_cache(self) -> None:
        """Initialize the local SQLite embedding cache table."""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.cache_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sentence_embeddings (
                        text_hash TEXT PRIMARY KEY,
                        text TEXT,
                        embedding TEXT,
                        created_at REAL
                    )
                    """
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"Could not initialize embedding SQLite cache: {e}")

    def _get_cached_embeddings(self, texts: list[str]) -> tuple[dict[str, list[float]], list[str]]:
        """Look up existing embeddings from cache. Returns (found_map, missing_texts)."""
        cached: dict[str, list[float]] = {}
        missing: list[str] = []

        try:
            with sqlite3.connect(self.cache_path) as conn:
                for t in texts:
                    t_hash = hashlib.sha256(t.strip().encode("utf-8")).hexdigest()
                    cursor = conn.execute(
                        "SELECT embedding FROM sentence_embeddings WHERE text_hash = ?", (t_hash,)
                    )
                    row = cursor.fetchone()
                    if row:
                        cached[t] = json.loads(row[0])
                    else:
                        missing.append(t)
        except Exception as e:
            logger.warning(f"Embedding cache lookup failed: {e}")
            missing = texts

        return cached, missing

    def _save_cached_embeddings(self, text_embedding_pairs: list[tuple[str, list[float]]]) -> None:
        """Save newly computed embeddings into the SQLite cache."""
        try:
            with sqlite3.connect(self.cache_path) as conn:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO sentence_embeddings (text_hash, text, embedding, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    [
                        (
                            hashlib.sha256(t.strip().encode("utf-8")).hexdigest(),
                            t,
                            json.dumps(emb),
                            time.time(),
                        )
                        for t, emb in text_embedding_pairs
                    ],
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to persist embeddings to SQLite cache: {e}")

    def embed_texts(
        self, texts: list[str], batch_size: int = 64, max_retries: int = 5
    ) -> list[list[float]]:
        """Embed a list of text strings with batching, exponential backoff, and caching.

        Returns a list of embedding vectors matching the input order.
        """
        if not texts:
            return []

        cached_map, missing_texts = self._get_cached_embeddings(texts)

        # Remove duplicate missing texts for API call efficiency
        unique_missing = list(dict.fromkeys(missing_texts))

        newly_computed: dict[str, list[float]] = {}
        if unique_missing:
            if self.provider == "local":
                local_model = self._get_local_model()
                # Local encoding with sentence-transformers
                encoded = local_model.encode(unique_missing, show_progress_bar=False)
                for orig_text, emb_vec in zip(unique_missing, encoded):
                    newly_computed[orig_text] = emb_vec.tolist()
            else:
                client = self._get_client()

                for i in range(0, len(unique_missing), batch_size):
                    batch = unique_missing[i : i + batch_size]
                    # Replace empty strings or whitespace-only with placeholder to prevent API error
                    clean_batch = [t.strip() if t.strip() else " " for t in batch]

                    retry_count = 0
                    delay = 1.0
                    while True:
                        try:
                            response = client.embeddings.create(
                                input=clean_batch,
                                model=self.model,
                            )
                            for orig_text, item in zip(batch, response.data):
                                newly_computed[orig_text] = item.embedding
                            break
                        except RateLimitError as rle:
                            retry_count += 1
                            if retry_count > max_retries:
                                raise rle
                            logger.warning(
                                f"OpenAI rate limit hit. Retrying in {delay:.1f}s (attempt {retry_count}/{max_retries})."
                            )
                            time.sleep(delay)
                            delay *= 2.0
                        except APIError as apie:
                            retry_count += 1
                            if retry_count > max_retries:
                                raise apie
                            logger.warning(
                                f"OpenAI API error ({apie}). Retrying in {delay:.1f}s (attempt {retry_count}/{max_retries})."
                            )
                            time.sleep(delay)
                            delay *= 2.0

            # Persist newly computed embeddings
            self._save_cached_embeddings(list(newly_computed.items()))

        # Assemble final result in original order
        results: list[list[float]] = []
        for t in texts:
            if t in cached_map:
                results.append(cached_map[t])
            elif t in newly_computed:
                results.append(newly_computed[t])
            else:
                # Fallback if text was duplicate or edge case
                results.append(newly_computed.get(t, [0.0] * 1536))

        return results

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        vectors = self.embed_texts([query])
        return vectors[0] if vectors else [0.0] * 1536


# Singleton instance
embedding_client = EmbeddingClient()

