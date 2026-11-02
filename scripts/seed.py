"""Database seed script for DocPilot RAG Platform.

Loads all documents from sample_docs/ into the dual-indexing architecture
across fixed_size, structure_aware, and semantic strategies.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from app.ingestion.chunkers import get_chunker
from app.ingestion.indexer import index_manager
from app.ingestion.loaders import load_documents_from_directory

logger = logging.getLogger("docpilot_seed")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

STRATEGIES = ["fixed_size", "structure_aware", "semantic"]


def seed_corpus(source_dir: str = "sample_docs") -> None:
    """Index all documents in source_dir across all chunking strategies."""
    src = Path(source_dir)
    if not src.exists():
        logger.error(f"Source directory '{source_dir}' does not exist.")
        sys.exit(1)

    print(f"\n[DocPilot Seed] Scanning directory: {src.resolve()}")
    docs = load_documents_from_directory(src)
    if not docs:
        logger.warning(f"No supported documents found in {source_dir}.")
        return

    print(f"[DocPilot Seed] Found {len(docs)} document(s) to index.")

    for strat in STRATEGIES:
        print(f"\n[DocPilot Seed] Processing strategy: '{strat}'...")
        chunker = get_chunker(strat)
        all_chunks = []
        for d in docs:
            chunks = chunker.chunk(d)
            all_chunks.extend(chunks)

        print(f"  -> Generated {len(all_chunks)} raw chunks. Indexing into ChromaDB + BM25...")
        index_manager.index_chunks(all_chunks, strategy=strat)
        stats = index_manager.get_strategy_stats(strat)
        print(f"  -> Done! Total indexed chunks: {stats.chunk_count} (Avg size: {stats.avg_char_count:.1f} chars)")

    print("\n[DocPilot Seed] All strategies indexed and synchronized successfully!\n")


if __name__ == "__main__":
    seed_corpus()

