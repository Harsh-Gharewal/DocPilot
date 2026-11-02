"""Persistent SQLite cache for LLM judge evaluations to ensure fast, cheap reruns."""

import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

from app.config import settings

logger = logging.getLogger(__name__)


class JudgeCache:
    """Stores LLM judge responses keyed by SHA-256 of their inputs."""

    def __init__(self, cache_path: Optional[Path] = None):
        self.cache_path = cache_path or (settings.processed_data_dir / "judge_eval_cache.sqlite")
        self._init_db()

    def _init_db(self) -> None:
        """Create the cache table if it does not exist."""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.cache_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS judge_cache (
                        cache_key TEXT PRIMARY KEY,
                        judge_type TEXT,
                        input_payload TEXT,
                        output_payload TEXT,
                        created_at REAL
                    )
                    """
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"Could not initialize judge SQLite cache: {e}")

    @staticmethod
    def compute_key(judge_type: str, *inputs: Any) -> str:
        """Generate deterministic hash key for judge inputs."""
        combined = f"{judge_type}::" + "::".join(str(i) for i in inputs)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def get(self, cache_key: str) -> Optional[dict[str, Any]]:
        """Look up cached evaluation result."""
        try:
            with sqlite3.connect(self.cache_path) as conn:
                cursor = conn.execute(
                    "SELECT output_payload FROM judge_cache WHERE cache_key = ?",
                    (cache_key,),
                )
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
        except Exception as e:
            logger.warning(f"Judge cache read failed: {e}")
        return None

    def set(self, cache_key: str, judge_type: str, input_payload: Any, output_payload: dict[str, Any]) -> None:
        """Store evaluation result in SQLite cache."""
        try:
            with sqlite3.connect(self.cache_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO judge_cache
                    (cache_key, judge_type, input_payload, output_payload, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        cache_key,
                        judge_type,
                        json.dumps(input_payload),
                        json.dumps(output_payload),
                        time.time(),
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"Judge cache write failed: {e}")


# Singleton instance
judge_cache = JudgeCache()

