"""Unified Hybrid Retrieval Pipeline coordinating dense, sparse, fusion, and reranking stages."""

import logging
from typing import Optional

from app.config import settings
from app.retrieval.dense import DenseRetriever, dense_retriever
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.models import RetrievalTrace, RetrievedChunk
from app.retrieval.reranker import Reranker, get_reranker
from app.retrieval.sparse import SparseRetriever, sparse_retriever

logger = logging.getLogger(__name__)

VALID_MODES = {"dense_only", "sparse_only", "hybrid", "hybrid_rerank"}


class HybridRetriever:
    """Orchestrates multi-stage retrieval across dense, sparse, fusion, and reranking."""

    def __init__(
        self,
        dense: Optional[DenseRetriever] = None,
        sparse: Optional[SparseRetriever] = None,
        reranker: Optional[Reranker] = None,
    ):
        self.dense = dense or dense_retriever
        self.sparse = sparse or sparse_retriever
        self.reranker = reranker or get_reranker()

    def retrieve(
        self,
        question: str,
        strategy: str = "structure_aware",
        mode: str = "hybrid_rerank",
        top_k_final: Optional[int] = None,
        weights: Optional[dict[str, float]] = None,
    ) -> tuple[list[RetrievedChunk], RetrievalTrace]:
        """Execute the retrieval pipeline in the requested mode and return (final_chunks, trace).

        Modes:
            - dense_only: Direct vector similarity query against ChromaDB
            - sparse_only: Keyword scoring using BM25Okapi
            - hybrid: Combines dense and sparse results using Reciprocal Rank Fusion (RRF)
            - hybrid_rerank: Blends dense + sparse candidates with RRF then reranks with Cross-Encoder
        """
        mode_clean = mode.lower()
        if mode_clean not in VALID_MODES:
            raise ValueError(
                f"Invalid retrieval mode '{mode}'. Choose from: {sorted(list(VALID_MODES))}"
            )

        k_final = top_k_final or settings.top_k_final
        active_weights = {
            "dense": weights.get("dense", settings.dense_weight) if weights else settings.dense_weight,
            "sparse": weights.get("sparse", settings.sparse_weight) if weights else settings.sparse_weight,
        }

        trace = RetrievalTrace(
            question=question,
            strategy=strategy,
            mode=mode_clean,
            weights=active_weights,
        )

        # 1. Mode: dense_only
        if mode_clean == "dense_only":
            dense_chunks = self.dense.retrieve(
                query=question, strategy=strategy, top_k=k_final
            )
            trace.dense_results = dense_chunks
            trace.final_chunks = dense_chunks
            return dense_chunks, trace

        # 2. Mode: sparse_only
        if mode_clean == "sparse_only":
            sparse_chunks = self.sparse.retrieve(
                query=question, strategy=strategy, top_k=k_final
            )
            trace.sparse_results = sparse_chunks
            trace.final_chunks = sparse_chunks
            return sparse_chunks, trace

        # 3. Modes: hybrid and hybrid_rerank
        # Fetch initial candidate pool for both dense and sparse
        fetch_k = settings.top_k_fused if mode_clean == "hybrid_rerank" else max(k_final * 2, settings.top_k_retrieval)

        dense_candidates = self.dense.retrieve(
            query=question, strategy=strategy, top_k=fetch_k
        )
        sparse_candidates = self.sparse.retrieve(
            query=question, strategy=strategy, top_k=fetch_k
        )

        trace.dense_results = dense_candidates
        trace.sparse_results = sparse_candidates

        # Apply Reciprocal Rank Fusion
        fused_pool_k = settings.top_k_fused if mode_clean == "hybrid_rerank" else k_final
        fused_candidates = reciprocal_rank_fusion(
            ranked_lists={"dense": dense_candidates, "sparse": sparse_candidates},
            weights=active_weights,
            rrf_k=settings.rrf_k,
            top_k=fused_pool_k,
        )
        trace.fused_results = fused_candidates

        if mode_clean == "hybrid":
            final_chunks = fused_candidates[:k_final]
            trace.final_chunks = final_chunks
            return final_chunks, trace

        # 4. Mode: hybrid_rerank
        reranked_chunks = self.reranker.rerank(
            query=question,
            candidates=fused_candidates,
            top_k=k_final,
        )
        trace.reranked_results = reranked_chunks
        trace.final_chunks = reranked_chunks

        return reranked_chunks, trace


# Singleton instance
hybrid_retriever = HybridRetriever()

