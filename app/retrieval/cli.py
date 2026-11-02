"""Command-line interface for testing and visualizing the multi-stage retrieval pipeline.

Usage:
    python -m app.retrieval.cli "your question" --mode hybrid_rerank --strategy structure_aware
"""

import argparse
import sys
from typing import Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from app.config import settings
from app.retrieval.models import RetrievedChunk
from app.retrieval.pipeline import hybrid_retriever


def print_stage_table(stage_name: str, chunks: list[RetrievedChunk], max_text_len: int = 75) -> None:
    """Pretty-print a stage's retrieved chunks as a formatted table."""
    print(f"\n--- Stage: {stage_name.upper()} (Count: {len(chunks)}) ---")
    if not chunks:
        print("  (No results retrieved at this stage)")
        return

    # Header
    has_pre_rank = any(c.pre_rerank_rank is not None for c in chunks)
    if has_pre_rank:
        header = f"{'Rank':<5} | {'Pre-Rank':<8} | {'Score':<10} | {'Chunk ID':<35} | {'Text Preview'}"
        sep_len = 110
    else:
        header = f"{'Rank':<5} | {'Score':<10} | {'Chunk ID':<35} | {'Text Preview'}"
        sep_len = 100

    print(header)
    print("-" * sep_len)

    for c in chunks:
        clean_text = c.text.replace("\n", " ").strip()
        preview = (clean_text[:max_text_len] + "...") if len(clean_text) > max_text_len else clean_text
        if has_pre_rank:
            pre_str = str(c.pre_rerank_rank) if c.pre_rerank_rank is not None else "-"
            print(f"{c.rank:<5} | {pre_str:<8} | {c.score:<10.6f} | {c.chunk_id:<35} | {preview}")
        else:
            print(f"{c.rank:<5} | {c.score:<10.6f} | {c.chunk_id:<35} | {preview}")


def run_cli_query(
    question: str,
    strategy: str = "structure_aware",
    mode: str = "hybrid_rerank",
    top_k: Optional[int] = None,
    dense_weight: Optional[float] = None,
    sparse_weight: Optional[float] = None,
) -> None:
    """Execute query and print comprehensive stage-by-stage results."""
    weights = {}
    if dense_weight is not None:
        weights["dense"] = dense_weight
    if sparse_weight is not None:
        weights["sparse"] = sparse_weight

    print("\n" + "=" * 90)
    print(f"HYBRID RETRIEVAL QUERY: \"{question}\"")
    print(f"Strategy: {strategy} | Mode: {mode}")
    if weights:
        print(f"Custom Weights: dense={weights.get('dense', settings.dense_weight)}, sparse={weights.get('sparse', settings.sparse_weight)}")
    print("=" * 90)

    try:
        final_chunks, trace = hybrid_retriever.retrieve(
            question=question,
            strategy=strategy,
            mode=mode,
            top_k_final=top_k,
            weights=weights if weights else None,
        )
    except Exception as exc:
        print(f"Error during retrieval: {exc}", file=sys.stderr)
        return

    # Print stages based on mode
    if trace.dense_results:
        print_stage_table("1. Dense Retrieval (Cosine Similarity)", trace.dense_results[:10])

    if trace.sparse_results:
        print_stage_table("2. Sparse Retrieval (BM25 Okapi)", trace.sparse_results[:10])

    if trace.fused_results:
        print_stage_table("3. Reciprocal Rank Fusion (RRF)", trace.fused_results[:10])

    if trace.reranked_results:
        print_stage_table("4. Reranked Results (Cross-Encoder)", trace.reranked_results)

    print("\n" + "=" * 90)
    print(f"FINAL RETURNED CONTEXT (Top {len(final_chunks)} chunks for generation)")
    print("=" * 90)
    for i, c in enumerate(final_chunks, start=1):
        heading = c.metadata.get("section_heading") or "General"
        source = c.metadata.get("source_file") or "unknown"
        page = c.metadata.get("page_number")
        page_str = f"Page {page}" if page else ""
        print(f"[{i}] {source} ({heading}{', ' + page_str if page_str else ''}) - Score: {c.score:.4f}")
        print(f"    {c.text[:200]}...")
        print()


def main() -> None:
    """Parse arguments and run retrieval CLI."""
    parser = argparse.ArgumentParser(
        prog="app.retrieval.cli",
        description="Inspect multi-stage hybrid retrieval results across dense, sparse, fusion, and reranking.",
    )
    parser.add_argument("question", type=str, help="Search question or keyword query")
    parser.add_argument(
        "--strategy",
        type=str,
        default="structure_aware",
        choices=["fixed_size", "structure_aware", "semantic"],
        help="Chunking strategy to retrieve from (default: structure_aware)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="hybrid_rerank",
        choices=["dense_only", "sparse_only", "hybrid", "hybrid_rerank"],
        help="Retrieval mode (default: hybrid_rerank)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=settings.top_k_final,
        help=f"Number of final chunks to return (default: {settings.top_k_final})",
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=None,
        help=f"Dense weight for RRF fusion (default: {settings.dense_weight})",
    )
    parser.add_argument(
        "--sparse-weight",
        type=float,
        default=None,
        help=f"Sparse weight for RRF fusion (default: {settings.sparse_weight})",
    )

    args = parser.parse_args()
    run_cli_query(
        question=args.question,
        strategy=args.strategy,
        mode=args.mode,
        top_k=args.top_k,
        dense_weight=args.dense_weight,
        sparse_weight=args.sparse_weight,
    )


if __name__ == "__main__":
    main()

