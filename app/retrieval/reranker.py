"""Reranking interface and implementations: Cross-Encoder and LLM-as-judge."""

import json
import logging
import threading
from typing import Any, Optional, Protocol, runtime_checkable

from app.config import settings
from app.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)

_cross_encoder_lock = threading.Lock()


@runtime_checkable
class Reranker(Protocol):
    """Protocol for rerankers evaluating question-chunk relevance."""

    backend_name: str

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """Rerank candidates against the query, preserving pre-rerank ranks."""
        ...


class CrossEncoderReranker:
    """Reranks candidate chunks using a cross-encoder model from sentence-transformers."""

    backend_name = "cross-encoder"

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.cross_encoder_model
        self._model = None

    def _get_model(self):
        """Lazily initialize the CrossEncoder model."""
        with _cross_encoder_lock:
            if self._model is None:
                import os
                # Prevent remote network probe timeouts if already cached locally
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                from sentence_transformers import CrossEncoder

                logger.info(f"Loading CrossEncoder model: {self.model_name}")
                self._model = CrossEncoder(self.model_name)
            return self._model

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """Score (query, chunk_text) pairs and return top candidates."""
        k = top_k or settings.top_k_final
        if not candidates:
            return []

        model = self._get_model()
        pairs = [(query, c.text) for c in candidates]
        with _cross_encoder_lock:
            scores = model.predict(pairs)

        scored_candidates: list[tuple[float, RetrievedChunk]] = []
        for score, candidate in zip(scores, candidates):
            scored_candidates.append((float(score), candidate))

        # Sort descending by cross-encoder score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        if k is not None:
            scored_candidates = scored_candidates[:k]

        reranked: list[RetrievedChunk] = []
        for rank_idx, (score, orig_chunk) in enumerate(scored_candidates, start=1):
            raw_sc = dict(orig_chunk.raw_scores)
            raw_sc["cross_encoder"] = round(score, 6)

            reranked.append(
                RetrievedChunk(
                    chunk_id=orig_chunk.chunk_id,
                    text=orig_chunk.text,
                    metadata=dict(orig_chunk.metadata),
                    score=round(score, 6),
                    rank=rank_idx,
                    retriever="reranked",
                    raw_scores=raw_sc,
                    pre_rerank_rank=orig_chunk.rank,  # Preserve pre-rerank rank
                )
            )

        logger.debug(f"CrossEncoderReranker reranked {len(candidates)} candidates into top {len(reranked)}")
        return reranked


class LLMJudgeReranker:
    """Reranks candidates by prompting an LLM to score relevance from 0 to 10."""

    backend_name = "llm"

    def __init__(self, model: Optional[str] = None):
        self.model = model or settings.openai_chat_model

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """Prompt LLM to score chunk relevance and sort accordingly."""
        k = top_k or settings.top_k_final
        if not candidates:
            return []

        # Prepare batched prompt
        chunk_descriptions = "\n\n".join(
            f"Chunk [{i}]:\n{c.text[:400]}" for i, c in enumerate(candidates, start=1)
        )
        system_prompt = (
            "You are a retrieval relevance judge. Evaluate how relevant each document chunk is "
            "to the user query. Output a JSON list of objects, each containing 'index' (1-based integer) "
            "and 'score' (float from 0.0 to 10.0, where 10.0 is directly answering the query)."
        )
        user_prompt = f"Query: {query}\n\nCandidate Chunks:\n{chunk_descriptions}"

        scores_map: dict[int, float] = {}

        try:
            from openai import OpenAI

            if not settings.openai_api_key:
                raise ValueError("OpenAI API key not configured for LLM judge reranker.")

            client = OpenAI(api_key=settings.openai_api_key)
            response = client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": user_prompt
                        + "\n\nReturn format: {\"scores\": [{\"index\": 1, \"score\": 8.5}, ...]}",
                    },
                ],
            )
            data = json.loads(response.choices[0].message.content or "{}")
            for item in data.get("scores", []):
                idx = int(item["index"])
                sc = float(item["score"])
                scores_map[idx] = sc
        except Exception as e:
            logger.warning(
                f"LLM-as-judge reranker call failed ({e}). Falling back to cross-encoder reranker."
            )
            fallback = CrossEncoderReranker()
            return fallback.rerank(query=query, candidates=candidates, top_k=top_k)

        # Build scored candidates
        scored: list[tuple[float, RetrievedChunk]] = []
        for i, c in enumerate(candidates, start=1):
            sc = scores_map.get(i, 0.0)
            scored.append((sc, c))

        scored.sort(key=lambda x: x[0], reverse=True)
        if k is not None:
            scored = scored[:k]

        reranked: list[RetrievedChunk] = []
        for rank_idx, (sc, orig_chunk) in enumerate(scored, start=1):
            raw_sc = dict(orig_chunk.raw_scores)
            raw_sc["llm_judge"] = round(sc, 2)

            reranked.append(
                RetrievedChunk(
                    chunk_id=orig_chunk.chunk_id,
                    text=orig_chunk.text,
                    metadata=dict(orig_chunk.metadata),
                    score=round(sc, 2),
                    rank=rank_idx,
                    retriever="reranked",
                    raw_scores=raw_sc,
                    pre_rerank_rank=orig_chunk.rank,
                )
            )

        return reranked


def get_reranker(backend: Optional[str] = None) -> Reranker:
    """Factory selecting the reranker backend configured or requested."""
    choice = backend or settings.reranker_backend
    choice = choice.lower()

    if choice in {"cross-encoder", "cross_encoder"}:
        return CrossEncoderReranker()
    elif choice in {"llm", "llm-judge"}:
        return LLMJudgeReranker()
    else:
        raise ValueError(
            f"Unknown reranker backend '{choice}'. Valid backends: 'cross-encoder', 'llm'."
        )

