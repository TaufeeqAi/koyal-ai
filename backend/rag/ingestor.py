from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from backend.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    EMBEDDING_DIMENSION,
    QDRANT_HOST,
    QDRANT_PORT,
    UPSERT_BATCH_SIZE,
    UPSERT_MAX_RETRIES,
    UPSERT_RETRY_DELAY_SECONDS,
)
from backend.rag.chunker import BilingualChunker
from backend.rag.embedder import MultilingualEmbedder

logger = logging.getLogger(__name__)

# Language file stem → BCP-47 language code
_LANG_CODE_MAP: dict[str, str] = {
    "hindi": "hi-IN",
    "english": "en-IN",
    "marathi": "mr-IN",
    "tamil": "ta-IN",
    "telugu": "te-IN",
    "kannada": "kn-IN",
    "bengali": "bn-IN",
}


class MultilingualIngestor:
    """
    Ingests multi-language documents for a tenant into an isolated Qdrant collection.

    Instantiation is expensive (loads LaBSE ~3s). Create once, call ingest_tenant()
    for each tenant in a loop (see scripts/ingest_all.py).
    """

    def __init__(self) -> None:
        self._client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            timeout=30,
        )
        self._chunker = BilingualChunker(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        self._embedder = MultilingualEmbedder()

    @staticmethod
    def collection_name(tenant_id: str) -> str:
        """
        Return the deterministic Qdrant collection name for a tenant.

        Static method ensures deterministic naming without instantiating
        the full ingestor (useful for Phase 8 K8s init containers).
        """
        return f"koyalai_{tenant_id}"

    def _recreate_collection(self, collection_name: str) -> None:
        """
        Drop the collection if it exists, then create fresh.
        Uses the current Qdrant API (recreate_collection removed).
        """
        if self._client.collection_exists(collection_name):
            logger.info("Dropping existing collection: %s", collection_name)
            self._client.delete_collection(collection_name)

        logger.info(
            "Creating collection: %s (dim=%d, distance=COSINE)",
            collection_name,
            EMBEDDING_DIMENSION,
        )
        self._client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=EMBEDDING_DIMENSION,
                distance=Distance.COSINE,
            ),
        )

    def _batch_upsert(
        self, collection_name: str, points: list[PointStruct]
    ) -> None:
        """
        Upload points in bounded batches with retry on transient failure.

        Args:
            collection_name: Target Qdrant collection.
            points:          All PointStructs to upsert.

        Raises:
            RuntimeError: if Qdrant upsert fails after max retries.
        """
        if not points:
            logger.warning(
                "No points to upsert for collection: %s", collection_name
            )
            return

        for batch_start in range(0, len(points), UPSERT_BATCH_SIZE):
            batch = points[batch_start : batch_start + UPSERT_BATCH_SIZE]
            batch_num = batch_start // UPSERT_BATCH_SIZE + 1
            total_batches = (len(points) + UPSERT_BATCH_SIZE - 1) // UPSERT_BATCH_SIZE

            for attempt in range(1, UPSERT_MAX_RETRIES + 1):
                try:
                    self._client.upsert(
                        collection_name=collection_name,
                        points=batch,
                        wait=True,     # block until Qdrant confirms persistence
                    )
                    logger.info(
                        "Upserted batch %d/%d (%d points) into %s",
                        batch_num,
                        total_batches,
                        len(batch),
                        collection_name,
                    )
                    break
                except Exception as exc:
                    if attempt == UPSERT_MAX_RETRIES:
                        raise RuntimeError(
                            f"Qdrant upsert failed after {UPSERT_MAX_RETRIES} attempts "
                            f"for batch {batch_num}: {exc}"
                        ) from exc
                    logger.warning(
                        "Qdrant upsert attempt %d failed (%s). "
                        "Retrying in %.1fs …",
                        attempt,
                        exc,
                        UPSERT_RETRY_DELAY_SECONDS,
                    )
                    time.sleep(UPSERT_RETRY_DELAY_SECONDS)

    def ingest_tenant(self, tenant_id: str) -> dict[str, Any]:
        """
        Ingest all language guideline files for a tenant.

        Discovers files matching `data/{tenant_id}/guidelines_*.txt`
        and derives the language code from the filename stem.

        Returns:
            {
                "tenant_id": str,
                "collection": str,
                "total_chunks": int,
                "total_vectors": int,
                "per_language": dict[str, int],
            }

        Raises:
            FileNotFoundError: if tenant data directory does not exist.
            RuntimeError: if no guideline files found for tenant or no points generated.
        """
        tenant_dir: Path = DATA_DIR / tenant_id
        if not tenant_dir.is_dir():
            raise FileNotFoundError(
                f"Tenant data directory not found: {tenant_dir}\n"
                f"Expected: data/{tenant_id}/ with guidelines_hindi.txt "
                f"and guidelines_english.txt"
            )

        lang_files = sorted(tenant_dir.glob("guidelines_*.txt"))
        if not lang_files:
            raise RuntimeError(
                f"No guideline files found in {tenant_dir}. "
                f"Expected files matching guidelines_*.txt"
            )

        collection_name = self.collection_name(tenant_id)
        self._recreate_collection(collection_name)

        all_points: list[PointStruct] = []
        total_chunks: int = 0
        lang_stats: dict[str, int] = {}

        for lang_file in lang_files:
            # Derive language code from filename: guidelines_hindi.txt → hi-IN
            stem = lang_file.stem.replace("guidelines_", "")
            lang_code = _LANG_CODE_MAP.get(stem, "en-IN")

            text = lang_file.read_text(encoding="utf-8").strip()
            if not text:
                logger.warning("Skipping empty file: %s", lang_file.name)
                continue

            chunks = self._chunker.chunk(
                text,
                metadata={
                    "tenant_id": tenant_id,
                    "language": lang_code,
                    "source": lang_file.name,
                },
            )

            if not chunks:
                logger.warning("No chunks produced from: %s", lang_file.name)
                continue

            texts = [c["text"] for c in chunks]
            embeddings = self._embedder.embed(texts, show_progress=False)

            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=emb.tolist(),
                    payload={
                        "text": chunk["text"],
                        "tenant_id": tenant_id,
                        "language": lang_code,
                        "source": chunk["metadata"]["source"],
                        "chunk_index": chunk["chunk_index"],
                        "char_count": chunk["char_count"],
                    },
                )
                for chunk, emb in zip(chunks, embeddings)
            ]

            all_points.extend(points)
            total_chunks += len(chunks)
            lang_stats[lang_code] = len(chunks)
            logger.info(
                "Tenant=%s lang=%s chunks=%d", tenant_id, lang_code, len(chunks)
            )

        # Batch upsert with retry resilience
        self._batch_upsert(collection_name, all_points)

        result = {
            "tenant_id": tenant_id,
            "collection": collection_name,
            "total_chunks": total_chunks,
            "total_vectors": len(all_points),
            "per_language": lang_stats,
        }
        logger.info("Ingestion complete: %s", result)
        return result
