from __future__ import annotations

import logging
from typing import Any, List

from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

from backend.config import (
    QDRANT_HOST,
    QDRANT_PORT,
    RERANK_TOP_K,
    SCORE_THRESHOLD,
    TOP_K_RETRIEVAL,
)
from backend.rag.embedder import MultilingualEmbedder

logger = logging.getLogger(__name__)

# Multilingual cross-encoder — trained on mMARCO (MS-MARCO translated to 26 langs)
_RERANKER_MODEL: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class MultilingualRetriever:
    """
    Retrieves and reranks document chunks for a given tenant.

    Instantiation loads both LaBSE (~471MB) and the reranker (~120MB).
    Create once per process and reuse across queries.
    """

    def __init__(self) -> None:
        self._client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            timeout=10,
        )
        self._embedder = MultilingualEmbedder()
        logger.info("Loading multilingual cross-encoder: %s", _RERANKER_MODEL)
        self._reranker = CrossEncoder(_RERANKER_MODEL, device="cpu")
        logger.info("Cross-encoder loaded.")

    def retrieve(
        self,
        query: str,
        tenant_id: str,
        preferred_language: str | None = None,
    ) -> List[dict[str, Any]]:
        """
        Retrieve top-K most relevant chunks for a query from a tenant's collection.

        Args:
            query:              Raw query string (Hindi, English, or Hinglish).
            tenant_id:          e.g. "tenant_hdfc_bank"
            preferred_language: BCP-47 code hint (e.g. "hi-IN"). Currently logged
                                for observability; not used to filter results
                                (cross-lingual retrieval is intended).

        Returns:
            List of chunk dicts (up to RERANK_TOP_K), sorted by rerank_score DESC:
                {
                    "text": str,
                    "language": str,           # "hi-IN" or "en-IN"
                    "score": float,            # Qdrant cosine similarity
                    "rerank_score": float,     # cross-encoder relevance score
                    "tenant_id": str,
                    "source": str,             # source filename
                    "chunk_index": int,
                    "char_count": int,
                }

        Raises:
            ValueError: if query is empty.
            RuntimeError: if Qdrant is unreachable.
        """
        if not query or not query.strip():
            raise ValueError("retrieve() received an empty query string.")

        if preferred_language:
            logger.debug(
                "retrieve | tenant=%s | lang=%s | query=%r",
                tenant_id,
                preferred_language,
                query[:80],
            )

        collection = f"koyalai_{tenant_id}"
        query_vector = self._embedder.embed_single(query)

        # ── Stage 1: ANN search via query_points (Qdrant 1.18.0+) ──
        response = self._client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=TOP_K_RETRIEVAL,
            score_threshold=SCORE_THRESHOLD,
            with_payload=True,
        )

        scored_points = response.points
        if not scored_points:
            logger.info(
                "No results above threshold=%.2f for query=%r in %s",
                SCORE_THRESHOLD,
                query[:60],
                collection,
            )
            return []

        chunks: List[dict[str, Any]] = [
            {
                "text": pt.payload["text"],
                "language": pt.payload.get("language", "und"),
                "score": float(pt.score),
                "tenant_id": pt.payload.get("tenant_id", tenant_id),
                "source": pt.payload.get("source", ""),
                "chunk_index": pt.payload.get("chunk_index", -1),
                "char_count": pt.payload.get("char_count", 0),
                "rerank_score": 0.0,
            }
            for pt in scored_points
        ]

        # ── Stage 2: Cross-encoder reranking ──
        if len(chunks) > 1:
            pairs = [[query, c["text"]] for c in chunks]
            rerank_scores: list[float] = self._reranker.predict(pairs).tolist()
            for chunk, rs in zip(chunks, rerank_scores):
                chunk["rerank_score"] = float(rs)
            chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        else:
            chunks[0]["rerank_score"] = chunks[0]["score"]

        top_chunks = chunks[:RERANK_TOP_K]
        logger.debug(
            "retrieve | returned %d chunks | top_rerank_score=%.3f",
            len(top_chunks),
            top_chunks[0].get("rerank_score", 0) if top_chunks else 0,
        )
        return top_chunks

    @staticmethod
    def format_context(chunks: List[dict[str, Any]]) -> str:
        """
        Format retrieved chunks into a numbered context block for the LLM prompt.

        Language label is human-readable (Hindi/English) — the LLM uses this
        as a cue to respond in the caller's language (Phase 2 response_agent).

        Args:
            chunks: Output of retrieve().

        Returns:
            Formatted string:
                [Source 1 — हिंदी | score=0.823]
                <chunk text>

                [Source 2 — English | score=0.756]
                <chunk text>
            Returns "No relevant information found." if chunks is empty.
        """
        if not chunks:
            return "No relevant information found in the knowledge base."

        lang_label_map: dict[str, str] = {
            "hi-IN": "हिंदी",
            "en-IN": "English",
            "hi-IN+en-IN": "हिंदी/English",
        }

        parts: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            lang_label = lang_label_map.get(chunk["language"], chunk["language"])
            score_info = f"score={chunk.get('rerank_score', chunk.get('score', 0)):.3f}"
            header = f"[Source {i} — {lang_label} | {score_info}]"
            parts.append(f"{header}\n{chunk['text']}")

        return "\n\n".join(parts)