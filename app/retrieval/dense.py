"""Dense vector retrieval using persistent ChromaDB collections."""

import logging
from typing import Optional

from app.config import settings
from app.ingestion.embeddings import EmbeddingClient, embedding_client
from app.ingestion.indexer import IndexManager, index_manager
from app.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Retrieves top-k chunks ranked by cosine similarity from ChromaDB."""

    def __init__(
        self,
        index_mgr: Optional[IndexManager] = None,
        embedder: Optional[EmbeddingClient] = None,
        default_top_k: Optional[int] = None,
    ):
        self.index_mgr = index_mgr or index_manager
        self.embedder = embedder or embedding_client
        self.default_top_k = default_top_k or settings.top_k_retrieval

    def retrieve(
        self,
        query: str,
        strategy: str = "structure_aware",
        top_k: Optional[int] = None,
    ) -> list[RetrievedChunk]:
        """Query ChromaDB with embedded question vector and return top-k matches."""
        k = top_k or self.default_top_k
        collection = self.index_mgr.get_chroma_collection(strategy)

        if collection.count() == 0:
            logger.warning(f"Dense collection for strategy '{strategy}' is empty.")
            return []

        # Generate query vector
        query_vector = self.embedder.embed_query(query)

        # Query collection
        fetch_k = min(k, collection.count())
        res = collection.query(
            query_embeddings=[query_vector],
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: list[RetrievedChunk] = []
        if not res or not res.get("ids") or not res["ids"][0]:
            return []

        ids = res["ids"][0]
        docs = res["documents"][0] if res.get("documents") else [""] * len(ids)
        metas = res["metadatas"][0] if res.get("metadatas") else [{}] * len(ids)
        dists = res["distances"][0] if res.get("distances") else [1.0] * len(ids)

        for i, (cid, text, meta, dist) in enumerate(zip(ids, docs, metas, dists)):
            # With cosine metric in ChromaDB, cosine_sim = 1.0 - distance
            cos_sim = round(float(1.0 - dist), 6)
            retrieved.append(
                RetrievedChunk(
                    chunk_id=cid,
                    text=text or "",
                    metadata=dict(meta) if meta else {},
                    score=cos_sim,
                    rank=i + 1,
                    retriever="dense",
                    raw_scores={"dense": cos_sim},
                )
            )

        logger.debug(f"DenseRetriever retrieved {len(retrieved)} chunk(s) for query: '{query}'")
        return retrieved


# Singleton instance
dense_retriever = DenseRetriever()

