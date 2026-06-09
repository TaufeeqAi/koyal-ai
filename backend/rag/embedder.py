from __future__ import annotations

import logging
from typing import List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.config import EMBEDDING_BATCH_SIZE, EMBEDDING_DIMENSION, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

_embedder_instance: MultilingualEmbedder | None = None

class MultilingualEmbedder:
    """
    Thread-safe wrapper around the LaBSE sentence-transformer.

    Instance should be created once and reused — model loading takes ~3s.
    For production (Phase 8), wrap in a singleton or inject via dependency.
    """

    MODEL_NAME: str = EMBEDDING_MODEL
    DIMENSION: int = EMBEDDING_DIMENSION

    def __init__(self) -> None:
        logger.info("Loading LaBSE multilingual model: %s", self.MODEL_NAME)
        self._model = SentenceTransformer(self.MODEL_NAME, device="cpu")
        self.dimension: int = self.DIMENSION
        logger.info(
            "LaBSE loaded. Output dim=%d, device=cpu", self.dimension
        )

    def embed(
        self,
        texts: Sequence[str],
        batch_size: int | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of texts into L2-normalised float32 embeddings.

        Args:
            texts:          List of raw strings (any language, any script).
            batch_size:     Sentences per forward pass. None → uses config default.
            show_progress:  Logs a progress bar during ingestion.

        Returns:
            np.ndarray of shape (len(texts), 768), dtype float32.
            All rows are L2-normalised (unit vectors).
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        effective_batch = batch_size if batch_size is not None else EMBEDDING_BATCH_SIZE

        return self._model.encode(
            list(texts),
            batch_size=effective_batch,
            show_progress_bar=show_progress,
            normalize_embeddings=True,   # cosine = dot product after L2-norm
            convert_to_numpy=True,
        )

    def embed_single(self, text: str) -> List[float]:
        """
        Encode a single query string.

        Returns:
            Python list[float] of length 768 (Qdrant's query_points expects list).

        Raises:
            ValueError: if text is empty or whitespace-only.
        """
        if not text or not text.strip():
            raise ValueError("embed_single received empty string.")

        vector: np.ndarray = self._model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        return vector.tolist()

def get_embedder() -> MultilingualEmbedder:
    """Return the process-wide MultilingualEmbedder, creating it on first call."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = MultilingualEmbedder()
    return _embedder_instance