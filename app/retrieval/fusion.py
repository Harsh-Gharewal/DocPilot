"""Reciprocal Rank Fusion (RRF) algorithm for combining dense and sparse search results."""

import logging
from typing import Optional

from app.config import settings
from app.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[RetrievedChunk]],
    weights: Optional[dict[str, float]] = None,
    rrf_k: Optional[int] = None,
    top_k: Optional[int] = None,
) -> list[RetrievedChunk]:
    """Combine multiple ranked candidate lists using weighted Reciprocal Rank Fusion.

    Formula:
        score(d) = sum_{r in retrievers} (weight_r / (rrf_k + rank_r(d)))

    Args:
        ranked_lists: Mapping from retriever name (e.g. 'dense', 'sparse') to ranked chunks.
        weights: Optional weighting for each retriever (default: from settings).
        rrf_k: Smoothing constant (default: settings.rrf_k = 60).
        top_k: Maximum number of fused items to return.

    Returns:
        List of fused RetrievedChunk items ordered by descending RRF score.
    """
    k_const = rrf_k if rrf_k is not None else settings.rrf_k

    # Resolve weights
    default_weights = {
        "dense": settings.dense_weight,
        "sparse": settings.sparse_weight,
    }
    active_weights = dict(default_weights)
    if weights:
        active_weights.update(weights)

    # Accumulate RRF scores and record per-retriever details
    scores: dict[str, float] = {}
    chunk_objects: dict[str, RetrievedChunk] = {}
    contributions: dict[str, dict[str, float]] = {}
    per_retriever_ranks: dict[str, dict[str, int]] = {}

    for retriever_name, chunk_list in ranked_lists.items():
        weight = active_weights.get(retriever_name, 1.0)
        for chunk in chunk_list:
            cid = chunk.chunk_id
            if cid not in chunk_objects:
                chunk_objects[cid] = chunk
                contributions[cid] = {}
                per_retriever_ranks[cid] = {}

            # Calculate RRF increment
            rank_val = chunk.rank
            rrf_increment = weight / (k_const + rank_val)
            scores[cid] = scores.get(cid, 0.0) + rrf_increment
            contributions[cid][retriever_name] = chunk.score
            per_retriever_ranks[cid][retriever_name] = rank_val

    # Sort chunk IDs by fused RRF score descending
    sorted_cids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    if top_k is not None:
        sorted_cids = sorted_cids[:top_k]

    fused_results: list[RetrievedChunk] = []
    for rank_idx, cid in enumerate(sorted_cids, start=1):
        base_chunk = chunk_objects[cid]
        fused_score = round(scores[cid], 8)

        # Merge raw scores and record individual ranks in metadata
        merged_raw = dict(base_chunk.raw_scores)
        merged_raw.update(contributions[cid])
        merged_raw["rrf"] = fused_score

        meta = dict(base_chunk.metadata)
        meta["retriever_ranks"] = per_retriever_ranks[cid]

        fused_results.append(
            RetrievedChunk(
                chunk_id=cid,
                text=base_chunk.text,
                metadata=meta,
                score=fused_score,
                rank=rank_idx,
                retriever="fused",
                raw_scores=merged_raw,
            )
        )

    logger.debug(
        f"Reciprocal Rank Fusion merged {len(chunk_objects)} unique candidates into {len(fused_results)} results."
    )
    return fused_results

