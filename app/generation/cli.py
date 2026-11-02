"""Command-line interface for the RAG generation and citation layer.

Usage:
    python -m app.generation.cli "your question"
"""

import argparse
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from app.generation.rag import ask


def main() -> None:
    """CLI entrypoint for generation and verification."""
    parser = argparse.ArgumentParser(
        prog="app.generation.cli",
        description="Query the RAG pipeline to generate grounded answers with verified citations and confidence scores.",
    )
    parser.add_argument("question", type=str, help="Question to ask the documentation")
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
        default=5,
        help="Number of chunks to feed to generator (default: 5)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 90)
    print(f"RAG QUESTION: \"{args.question}\"")
    print(f"Strategy: {args.strategy} | Mode: {args.mode}")
    print("=" * 90 + "\n")

    resp = ask(
        question=args.question,
        strategy=args.strategy,
        mode=args.mode,
        top_k=args.top_k,
    )

    # 1. Fallback or Answer
    if resp.no_answer:
        print("[FALLBACK: INSUFFICIENT CONTEXT / LOW RETRIEVAL CONFIDENCE]")
        print(f"Message: {resp.answer}\n")
        print(f"Retrieval Confidence: {resp.confidence.retrieval_confidence:.4f}")
        print(f"Reason: {resp.no_answer.missing_information}\n")
        print("Closest documentation found:")
        for chk in resp.no_answer.top_chunks_summary:
            print(f"  - [{chk['source']}] (Section: {chk['section']}) [Score: {chk['score']}]")
            print(f"    \"{chk['snippet']}\"")
        if resp.no_answer.recommended_manual_checks:
            print("\nRecommended manual checks in files:")
            for rec in resp.no_answer.recommended_manual_checks:
                print(f"  * {rec}")
    else:
        print("ANSWER:")
        print("-" * 90)
        print(resp.answer)
        print("-" * 90 + "\n")

        # 2. Citations & Verification
        print(f"CITATIONS ({len(resp.citations)} context references):")
        for cit in resp.citations:
            ver_status = "VERIFIED [OK]" if cit.verified else "UNVERIFIED / FLAGGED"
            page_str = f", Page {cit.page_number}" if cit.page_number and cit.page_number > 0 else ""
            print(f"[{cit.citation_number}] {cit.source_file} ({cit.section_heading or 'Overview'}{page_str}) -> {ver_status}")
            if cit.verification_reason:
                print(f"    Rationale: {cit.verification_reason}")
            print(f"    Excerpt: \"{cit.chunk_text[:120].strip()}...\"")
            print()

        # 3. Claims breakdown
        if resp.verification.claims:
            print("CLAIMS BREAKDOWN:")
            for idx, clm in enumerate(resp.verification.claims, start=1):
                status = "SUPPORTED" if clm.supported else "UNSUPPORTED"
                cits = ", ".join(f"[{c}]" for c in clm.citation_numbers) if clm.citation_numbers else "[None]"
                print(f"  {idx}. \"{clm.claim_text}\" -> {status} via {cits}")

        if resp.verification.flagged_citations:
            print("\nFLAGGED ANOMALIES:")
            for flag in resp.verification.flagged_citations:
                print(f"  ! {flag.get('type')}: {flag.get('claim', flag)}")

    # 4. Confidence Breakdown
    print("\n" + "=" * 90)
    print("CONFIDENCE BREAKDOWN & LATENCY")
    print("=" * 90)
    conf = resp.confidence
    print(f"Composite Score:      {conf.composite_score:.4f}  [{conf.label}]")
    print(f"  - Retrieval Score:  {conf.retrieval_confidence:.4f} (weight: 0.4)")
    print(f"  - Citation Coverage:{conf.citation_coverage:.4f} (weight: 0.4)")
    print(f"  - Completeness:     {conf.completeness:.4f} (weight: 0.2)")
    print()
    print("Latency by Stage (ms):")
    for stage, ms in resp.latency_by_stage.items():
        print(f"  - {stage:<18}: {ms:.1f} ms")
    print(f"Token Usage: Prompt={resp.token_usage['prompt_tokens']}, Completion={resp.token_usage['completion_tokens']}, Total={resp.token_usage['total_tokens']}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()

