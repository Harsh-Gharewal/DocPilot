"""Command Line Interface for the Ingestion and Chunking pipeline.

Usage:
    python -m app.ingestion.cli ingest --path sample_docs --strategy structure_aware
    python -m app.ingestion.cli ingest --path sample_docs --strategy all
    python -m app.ingestion.cli stats
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from app.config import settings
from app.ingestion.chunkers import get_chunker
from app.ingestion.indexer import index_manager, verify_index_sync
from app.ingestion.loaders import load_document, load_documents_from_directory
from app.ingestion.models import Chunk, Document

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ingestion_cli")

AVAILABLE_STRATEGIES = ["fixed_size", "structure_aware", "semantic"]


def run_ingest(path_str: str, strategy: str) -> None:
    """Ingest documents from the specified file or folder into dense and sparse indexes."""
    path = Path(path_str)
    if not path.exists():
        logger.error(f"Path does not exist: {path_str}")
        sys.exit(1)

    # 1. Load documents
    documents: list[Document] = []
    if path.is_file():
        doc = load_document(path)
        if doc:
            documents.append(doc)
    else:
        documents = load_documents_from_directory(path)

    if not documents:
        logger.warning(f"No valid documents found to ingest from: {path_str}")
        return

    logger.info(f"Loaded {len(documents)} document(s) ready for chunking.")

    # 2. Determine strategies to run
    strategies_to_run = AVAILABLE_STRATEGIES if strategy.lower() == "all" else [strategy.lower()]

    results: dict[str, dict] = {}

    for strat in strategies_to_run:
        logger.info(f"--- Processing chunking strategy: {strat} ---")
        chunker = get_chunker(strat)
        strat_chunks: list[Chunk] = []

        for doc in documents:
            chunks = chunker.chunk(doc)
            strat_chunks.extend(chunks)

        logger.info(
            f"Generated {len(strat_chunks)} raw chunks across {len(documents)} document(s) for strategy '{strat}'."
        )

        # Index and deduplicate
        report = index_manager.index_chunks(strat_chunks, strategy=strat)

        # Verify sync
        sync_ok = False
        try:
            sync_ok = verify_index_sync(strat)
        except Exception as e:
            logger.error(f"Sync check failed for strategy {strat}: {e}")

        # Compute stats
        stats = index_manager.get_strategy_stats(strat)
        results[strat] = {
            "chunks_seen": report.chunks_seen,
            "chunks_inserted": report.chunks_inserted,
            "duplicates_skipped": report.duplicates_skipped,
            "total_in_index": stats.chunk_count,
            "avg_char_count": stats.avg_char_count,
            "sync_status": "PASSED" if sync_ok else "FAILED",
        }

    # Print summary table
    print("\n" + "=" * 80)
    print("INGESTION & CHUNKING SUMMARY REPORT")
    print("=" * 80)
    header = f"{'Strategy':<18} | {'Seen':<6} | {'Inserted':<8} | {'Dups':<6} | {'Total':<6} | {'Avg Chars':<10} | {'Sync'}"
    print(header)
    print("-" * 80)
    for strat, data in results.items():
        row = (
            f"{strat:<18} | "
            f"{data['chunks_seen']:<6} | "
            f"{data['chunks_inserted']:<8} | "
            f"{data['duplicates_skipped']:<6} | "
            f"{data['total_in_index']:<6} | "
            f"{data['avg_char_count']:<10.1f} | "
            f"{data['sync_status']}"
        )
        print(row)
    print("=" * 80 + "\n")


def run_stats() -> None:
    """Print current index statistics for all chunking strategies."""
    print("\n" + "=" * 65)
    print("RAG HYBRID INDEX STATISTICS")
    print("=" * 65)
    header = f"{'Strategy':<18} | {'Total Chunks':<14} | {'Avg Char Count':<16} | {'Sync'}"
    print(header)
    print("-" * 65)

    for strat in AVAILABLE_STRATEGIES:
        stats = index_manager.get_strategy_stats(strat)
        try:
            sync_ok = verify_index_sync(strat)
            sync_str = "PASSED" if sync_ok else "FAILED"
        except Exception:
            sync_str = "FAILED"

        row = (
            f"{strat:<18} | "
            f"{stats.chunk_count:<14} | "
            f"{stats.avg_char_count:<16.1f} | "
            f"{sync_str}"
        )
        print(row)

    print("=" * 65 + "\n")


def main() -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="app.ingestion.cli",
        description="Ingestion and chunking pipeline management CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Subcommand to execute")

    # Ingest subcommand
    ingest_parser = subparsers.add_parser("ingest", help="Ingest and index documents")
    ingest_parser.add_argument(
        "--path",
        type=str,
        default=str(settings.sample_docs_dir),
        help=f"Path to file or folder containing documents (default: {settings.sample_docs_dir})",
    )
    ingest_parser.add_argument(
        "--strategy",
        type=str,
        default="structure_aware",
        choices=["fixed_size", "structure_aware", "semantic", "all"],
        help="Chunking strategy to run (default: structure_aware, or 'all')",
    )

    # Stats subcommand
    subparsers.add_parser("stats", help="Show current index statistics for all strategies")

    args = parser.parse_args()

    if args.command == "ingest":
        run_ingest(path_str=args.path, strategy=args.strategy)
    elif args.command == "stats":
        run_stats()


if __name__ == "__main__":
    main()

