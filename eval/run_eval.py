"""Evaluation harness executing golden dataset test cases and generating JSON and Markdown benchmark reports."""

import argparse
import concurrent.futures
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from app.generation.models import AnswerResponse
from app.generation.rag import ask
from eval.dataset import GoldenTestCase, load_golden_dataset
from eval.judge_cache import judge_cache
from eval.metrics import (
    calculate_token_cost,
    compute_answer_correctness,
    compute_citation_accuracy,
    compute_faithfulness,
    compute_retrieval_relevance,
)

logger = logging.getLogger("eval_runner")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def evaluate_single_case(
    case: GoldenTestCase,
    strategy: str,
    mode: str,
    top_k: int,
) -> dict[str, Any]:
    """Execute the pipeline on a single GoldenTestCase and compute all metrics."""
    t0 = time.perf_counter()

    # 1. Run RAG ask orchestrator
    resp: AnswerResponse = ask(
        question=case.question,
        strategy=strategy,
        mode=mode,
        top_k=top_k,
    )
    total_latency_ms = resp.latency_by_stage.get(
        "total_ms", round((time.perf_counter() - t0) * 1000, 2)
    )

    retrieved_chunks = resp.retrieval_trace.final_chunks if resp.retrieval_trace else []
    retrieved_chunk_ids = [c.chunk_id for c in retrieved_chunks]

    # 2. Retrieval Relevance
    relevance_res = compute_retrieval_relevance(
        retrieved_chunks=retrieved_chunks,
        expected_source_docs=case.source_docs,
        expected_sections=case.source_sections,
        k=top_k,
    )

    # 3. Answer Correctness
    is_fallback = resp.no_answer is not None
    correctness_res = compute_answer_correctness(
        question=case.question,
        system_answer=resp.answer,
        golden_answer=case.golden_answer,
        category=case.category,
        is_fallback=is_fallback,
        cache=judge_cache,
    )

    # 4. Faithfulness
    faithfulness_res = compute_faithfulness(
        system_answer=resp.answer,
        retrieved_chunks=retrieved_chunks,
        cache=judge_cache,
    )

    # 5. Citation Accuracy
    citation_map = {c.citation_number: c for c in resp.citations}
    citation_acc = compute_citation_accuracy(
        answer=resp.answer,
        citations_map=citation_map,
        retrieved_chunks=retrieved_chunks,
    )

    # 6. Cost calculation
    p_tokens = resp.token_usage.get("prompt_tokens", 0)
    c_tokens = resp.token_usage.get("completion_tokens", 0)
    cost_usd = calculate_token_cost(p_tokens, c_tokens)

    return {
        "id": case.id,
        "category": case.category,
        "question": case.question,
        "golden_answer": case.golden_answer,
        "system_answer": resp.answer,
        "is_fallback": is_fallback,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "metrics": {
            "correctness": correctness_res["score"],
            "correctness_reason": correctness_res["reason"],
            "faithfulness": faithfulness_res["faithfulness_score"],
            "hit_rate_at_k": relevance_res["hit_rate_at_k"],
            "recall": relevance_res["recall"],
            "mrr": relevance_res["mrr"],
            "citation_accuracy": citation_acc,
            "latency_ms": total_latency_ms,
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,
            "cost_usd": cost_usd,
        },
    }


def aggregate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate metric scores overall and broken down by category."""
    categories = sorted(list({r["category"] for r in results}))

    def calc_averages(items: list[dict[str, Any]]) -> dict[str, float]:
        if not items:
            return {}
        n = len(items)
        return {
            "correctness": round(sum(i["metrics"]["correctness"] for i in items) / n, 4),
            "faithfulness": round(sum(i["metrics"]["faithfulness"] for i in items) / n, 4),
            "hit_rate_at_k": round(sum(i["metrics"]["hit_rate_at_k"] for i in items) / n, 4),
            "recall": round(sum(i["metrics"]["recall"] for i in items) / n, 4),
            "mrr": round(sum(i["metrics"]["mrr"] for i in items) / n, 4),
            "citation_accuracy": round(sum(i["metrics"]["citation_accuracy"] for i in items) / n, 4),
            "avg_latency_ms": round(sum(i["metrics"]["latency_ms"] for i in items) / n, 1),
            "total_cost_usd": round(sum(i["metrics"]["cost_usd"] for i in items), 6),
            "count": n,
        }

    overall = calc_averages(results)
    by_category = {cat: calc_averages([r for r in results if r["category"] == cat]) for cat in categories}

    return {"overall": overall, "by_category": by_category}


def generate_markdown_report(
    label: str,
    strategy: str,
    mode: str,
    results: list[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
    """Generate GitHub-flavored Markdown evaluation summary report."""
    ov = summary["overall"]
    by_cat = summary["by_category"]

    md = []
    md.append(f"# RAG Benchmark Report: `{label}`\n")
    md.append(f"**Strategy:** `{strategy}` | **Retrieval Mode:** `{mode}` | **Total Cases:** {ov['count']}\n")

    md.append("## Overall Metrics Summary\n")
    md.append("| Metric | Score |")
    md.append("| :--- | :--- |")
    md.append(f"| **Answer Correctness** | **{ov['correctness']:.2%}** |")
    md.append(f"| **Faithfulness** | **{ov['faithfulness']:.2%}** |")
    md.append(f"| **Retrieval Hit Rate @ K** | **{ov['hit_rate_at_k']:.2%}** |")
    md.append(f"| **Source Recall** | **{ov['recall']:.2%}** |")
    md.append(f"| **Mean Reciprocal Rank (MRR)** | **{ov['mrr']:.4f}** |")
    md.append(f"| **Citation Accuracy** | **{ov['citation_accuracy']:.2%}** |")
    md.append(f"| Average Latency | {ov['avg_latency_ms']:.1f} ms |")
    md.append(f"| Total Token Cost | ${ov['total_cost_usd']:.4f} |\n")

    md.append("## Breakdown by Category\n")
    md.append("| Category | Count | Correctness | Faithfulness | Hit Rate @ K | Recall | MRR | Citations |")
    md.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for cat, stats in by_cat.items():
        md.append(
            f"| `{cat}` | {stats['count']} | {stats['correctness']:.2%} | {stats['faithfulness']:.2%} | "
            f"{stats['hit_rate_at_k']:.2%} | {stats['recall']:.2%} | {stats['mrr']:.4f} | {stats['citation_accuracy']:.2%} |"
        )
    md.append("")

    # Separate reporting of no_answer category
    if "no_answer" in by_cat:
        na = by_cat["no_answer"]
        md.append("### Out-of-Corpus (`no_answer`) Specific Evaluation")
        md.append(f"- **Refusal Correctness**: {na['correctness']:.2%} ({na['count']} cases)")
        md.append("- Evaluates whether the system properly triggers fallback without hallucinating.\n")

    # Top 10 worst cases by correctness score
    sorted_worst = sorted(results, key=lambda x: (x["metrics"]["correctness"], x["metrics"]["mrr"]))[:10]
    md.append("## Top 10 Challenging / Worst Cases\n")
    md.append("| ID | Category | Question | Score | Failure Reason |")
    md.append("| :--- | :--- | :--- | :--- | :--- |")
    for w in sorted_worst:
        clean_q = w["question"].replace("|", "\\|")
        reason = w["metrics"]["correctness_reason"].replace("|", "\\|")
        md.append(f"| `{w['id']}` | `{w['category']}` | {clean_q} | {w['metrics']['correctness']:.2f} | {reason} |")
    md.append("")

    return "\n".join(md)


def compare_runs(baseline_path: str, new_path: str) -> None:
    """Compare two evaluation runs and output improvements and regressions."""
    b_data = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    n_data = json.loads(Path(new_path).read_text(encoding="utf-8"))

    b_cases = {c["id"]: c for c in b_data["results"]}
    n_cases = {c["id"]: c for c in n_data["results"]}

    print("\n" + "=" * 80)
    print(f"EVALUATION COMPARISON: '{Path(baseline_path).stem}' vs '{Path(new_path).stem}'")
    print("=" * 80)

    improved = []
    regressed = []
    unchanged = 0

    for cid, n_case in n_cases.items():
        if cid in b_cases:
            b_sc = b_cases[cid]["metrics"]["correctness"]
            n_sc = n_case["metrics"]["correctness"]
            diff = round(n_sc - b_sc, 4)
            if diff > 0.05:
                improved.append((cid, b_sc, n_sc, diff, n_case["question"]))
            elif diff < -0.05:
                regressed.append((cid, b_sc, n_sc, diff, n_case["question"]))
            else:
                unchanged += 1

    print(f"Summary: {len(improved)} Improved, {len(regressed)} Regressed, {unchanged} Unchanged.\n")

    if improved:
        print("--- Top Improvements ---")
        for cid, b_sc, n_sc, diff, q in improved:
            print(f"  [+] {cid} (+{diff:.2f}): {q} ({b_sc:.2f} -> {n_sc:.2f})")

    if regressed:
        print("\n--- Regressions ---")
        for cid, b_sc, n_sc, diff, q in regressed:
            print(f"  [-] {cid} ({diff:.2f}): {q} ({b_sc:.2f} -> {n_sc:.2f})")

    print("=" * 80 + "\n")


def main() -> None:
    """Parse CLI options and run evaluation harness."""
    parser = argparse.ArgumentParser(
        prog="eval.run_eval",
        description="Run RAG benchmark evaluation suite across golden dataset.",
    )
    parser.add_argument("--strategy", type=str, default="structure_aware", choices=["fixed_size", "structure_aware", "semantic"])
    parser.add_argument("--mode", type=str, default="hybrid_rerank", choices=["dense_only", "sparse_only", "hybrid", "hybrid_rerank"])
    parser.add_argument("--label", type=str, default="baseline", help="Report label identifier (default: baseline)")
    parser.add_argument("--dataset", type=str, default="eval/golden_dataset.jsonl", help="Path to golden dataset")
    parser.add_argument("--concurrency", type=int, default=1, help="Evaluation concurrency limit")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved chunks")
    parser.add_argument("--compare", type=str, default=None, help="Path to previous JSON result file to diff against")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of test cases to run")

    args = parser.parse_args()

    if args.compare:
        compare_runs(args.compare, f"eval/results/{args.label}.json")
        return

    # Load dataset
    cases = load_golden_dataset(args.dataset)
    if args.limit:
        cases = cases[:args.limit]
    logger.info(f"Loaded {len(cases)} test cases from {args.dataset}")

    # Pre-warm reranker and embedding models
    from app.retrieval.pipeline import hybrid_retriever
    if hasattr(hybrid_retriever.reranker, "_get_model"):
        hybrid_retriever.reranker._get_model()

    results = []
    print(f"\nRunning benchmark '{args.label}' on {len(cases)} cases (Strategy: {args.strategy}, Mode: {args.mode})...", flush=True)

    if args.concurrency > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [
                executor.submit(evaluate_single_case, c, args.strategy, args.mode, args.top_k)
                for c in cases
            ]
            for idx, fut in enumerate(concurrent.futures.as_completed(futures), start=1):
                res = fut.result()
                results.append(res)
                if idx % 10 == 0 or idx == len(cases):
                    print(f"  Progress: {idx}/{len(cases)} cases evaluated...", flush=True)
    else:
        for idx, c in enumerate(cases, start=1):
            res = evaluate_single_case(c, args.strategy, args.mode, args.top_k)
            results.append(res)
            if idx % 10 == 0 or idx == len(cases):
                print(f"  Progress: {idx}/{len(cases)} cases evaluated...", flush=True)

    # Sort results by ID for deterministic reports
    results.sort(key=lambda x: x["id"])

    # Aggregate metrics
    summary = aggregate_metrics(results)

    # Persist JSON output
    out_dir = Path("eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.label}.json"
    md_path = out_dir / f"{args.label}.md"

    output_payload = {
        "label": args.label,
        "strategy": args.strategy,
        "mode": args.mode,
        "timestamp": time.time(),
        "summary": summary,
        "results": results,
    }
    json_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

    # Persist Markdown output
    md_content = generate_markdown_report(
        label=args.label,
        strategy=args.strategy,
        mode=args.mode,
        results=results,
        summary=summary,
    )
    md_path.write_text(md_content, encoding="utf-8")

    print(f"\nEvaluation complete!")
    print(f"  -> JSON result: {json_path}")
    print(f"  -> Markdown report: {md_path}\n")

    # Print summary table to console
    ov = summary["overall"]
    print("=" * 60)
    print(f"BENCHMARK SUMMARY: {args.label.upper()}")
    print("=" * 60)
    print(f"Answer Correctness:       {ov['correctness']:.2%}")
    print(f"Faithfulness:             {ov['faithfulness']:.2%}")
    print(f"Retrieval Hit Rate @ K:   {ov['hit_rate_at_k']:.2%}")
    print(f"Source Recall:            {ov['recall']:.2%}")
    print(f"Mean Reciprocal Rank:     {ov['mrr']:.4f}")
    print(f"Citation Accuracy:        {ov['citation_accuracy']:.2%}")
    print(f"Average Latency:          {ov['avg_latency_ms']:.1f} ms")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

