"""
Multilingual emergency detection: keyword matching + semantic similarity.

Two-layer safety net
────────────────────
Layer 1 — Keyword matching (O(n), deterministic, zero-latency):
    Covers exact / substring matches across Hindi, English, Hinglish,
    and banking-specific emergencies.

Layer 2 — Semantic similarity via LaBSE cosine distance:
    Catches paraphrases that bypass keyword lists, e.g.
    "मेरी साँसें नहीं चल रही" ≈ "can't breathe" in embedding space.

The two-layer approach is deliberate: keyword matching is O(n) and
runs in microseconds — it handles the common case without loading
any model.  Semantic matching only fires when keywords don't match.

Why LaBSE for safety (not a separate safety model):
    - LaBSE is already loaded for retrieval; sharing avoids a second
      471 MB model in memory.
    - Cross-lingual semantic space means "I want to die" and "मैं
      जीना नहीं चाहता" are adjacent — no per-language tuning needed.

Usage example:
    from backend.safety.emergency_keywords import MultilingualEmergencyDetector
    det = MultilingualEmergencyDetector()
    is_emrg, reason = det.is_emergency("मेरा दिल का दौरा पड़ रहा है")
    # (True, "Emergency keyword: 'दिल का दौरा'")
"""

from __future__ import annotations

import logging
from typing import Optional
import threading

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from backend.config import EMERGENCY_SEMANTIC_THRESHOLD
from backend.exceptions import EmergencyDetectionError

logger = logging.getLogger(__name__)

_detector_instance: Optional["MultilingualEmergencyDetector"] = None
_detector_lock = threading.Lock()
# ── Keyword Lists 

EMERGENCY_KEYWORDS_HINDI: list[str] = [
    "दिल का दौरा", "छाती में दर्द", "साँस नहीं आ रही",
    "बेहोश", "खून बह रहा", "आत्महत्या", "मरना चाहता",
    "लकवा", "स्ट्रोक", "नब्ज नहीं", "दम घुट रहा",
    "जीना नहीं चाहता", "मर जाऊँगा", "मर जाना चाहता",
    "बहुत दर्द", "अस्पताल बुलाओ",
]

EMERGENCY_KEYWORDS_ENGLISH: list[str] = [
    "chest pain", "heart attack", "can't breathe", "cannot breathe",
    "unconscious", "stroke", "severe bleeding", "suicidal",
    "want to die", "overdose", "not breathing", "collapsed",
    "call ambulance", "emergency", "dying","kill myself",
    "kill", "end my life", "want to die", "hacked", "account hacked",
    "money stolen", "fraud", "unauthorized transaction",
]

EMERGENCY_KEYWORDS_HINGLISH: list[str] = [
    "dil mein bahut dard", "sans nahi aa raha", "behosh ho gaya",
    "khoon bahut aa raha", "marna chahta", "hospital bulao",
    "ambulance bulao", "mar jaunga", "jina nahi chahta",
]

EMERGENCY_KEYWORDS_BANKING: list[str] = [
    "खाता खाली हो गया", "fraud ho gaya", "account hack",
    "पैसे चोरी हो गए", "unauthorized transaction",
    "OTP किसी को दे दिया", "account hacked", "money stolen",
    "धोखाधड़ी", "पैसे गायब","khata khali ho gaya","paise chori ho gaye",
    "otp kisi ko de diya",
]

ALL_EMERGENCY_KEYWORDS: list[str] = (
    EMERGENCY_KEYWORDS_HINDI
    + EMERGENCY_KEYWORDS_ENGLISH
    + EMERGENCY_KEYWORDS_HINGLISH
    + EMERGENCY_KEYWORDS_BANKING
)

# Reference sentences for semantic similarity (covers all emergency categories)
EMERGENCY_REFERENCE_SENTENCES: list[str] = [
    "I am having a heart attack",
    "मुझे दिल का दौरा पड़ रहा है",
    "Someone needs emergency medical help immediately",
    "मेरा बैंक खाता हैक हो गया है",
    "There has been unauthorized access to my account and money is gone",
    "I want to end my life",
    "मैं जीना नहीं चाहता",
    "I cannot breathe and need help",
    "मुझे साँस नहीं आ रही, help करो",
    "Call an ambulance right now",
    "I have been defrauded, my money is stolen",
]


class MultilingualEmergencyDetector:
    """Two-layer emergency detector: keyword match → semantic similarity.

    Attributes:
        threshold: Cosine similarity threshold for semantic match (0–1).
        embedder: LaBSE model for cross-lingual embeddings.
        ref_embeddings: Pre-computed embeddings for reference sentences.

    Args:
        semantic_threshold: Cosine similarity cutoff for semantic detection.
            Defaults to ``EMERGENCY_SEMANTIC_THRESHOLD`` from config.
        model_name: Sentence-transformer model to use for semantic layer.

    Raises:
        EmergencyDetectionError: If the embedding model fails to load.

    Example:
        >>> det = MultilingualEmergencyDetector()
        >>> det.is_emergency("chest pain")
        (True, "Emergency keyword: 'chest pain'")
        >>> det.is_emergency("मेरी EMI कब कटती है")
        (False, "")
    """

    def __init__(
        self,
        semantic_threshold: float = EMERGENCY_SEMANTIC_THRESHOLD,
        model_name: str = "sentence-transformers/LaBSE",
    ) -> None:
        self.threshold = semantic_threshold
        logger.info(
            "Loading LaBSE for emergency detection "
            "(semantic_threshold=%.2f)...",
            semantic_threshold,
        )
        try:
            self.embedder = SentenceTransformer(model_name)
            self.ref_embeddings: np.ndarray = self.embedder.encode(
                EMERGENCY_REFERENCE_SENTENCES,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmergencyDetectionError(
                f"Failed to load emergency detector model '{model_name}': {exc}"
            ) from exc
        logger.info("Emergency detector ready (%d reference sentences).", len(EMERGENCY_REFERENCE_SENTENCES))

    def is_emergency(self, query: str) -> tuple[bool, str]:
        """Check whether the query represents an emergency situation.

        Args:
            query: Raw caller utterance in any supported language.

        Returns:
            A tuple ``(is_emergency, reason_string)``.
            ``reason_string`` is empty when no emergency is detected.

        Raises:
            EmergencyDetectionError: If semantic embedding fails.

        Example:
            >>> det.is_emergency("मुझे दिल का दौरा आ रहा है")
            (True, "Emergency keyword: 'दिल का दौरा'")
        """
        if not query or not query.strip():
            logger.debug("Empty query passed to emergency detector — returning safe.")
            return False, ""

        query_lower = query.lower()

        # ── Layer 1: Keyword match (deterministic, zero-latency) 
        for keyword in ALL_EMERGENCY_KEYWORDS:
            if keyword.lower() in query_lower:
                reason = f"Emergency keyword: '{keyword}'"
                logger.warning(
                    "EMERGENCY DETECTED (keyword): query=%r reason=%r",
                    query[:80],
                    reason,
                )
                return True, reason

        # ── Layer 2: Semantic similarity (catches paraphrases) 
        try:
            query_emb: np.ndarray = self.embedder.encode(
                [query],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            sims: np.ndarray = cosine_similarity(query_emb, self.ref_embeddings)[0]
            max_score: float = float(sims.max())
            max_idx: int = int(sims.argmax())

            logger.debug(
                "Emergency semantic score=%.3f threshold=%.2f query=%r",
                max_score,
                self.threshold,
                query[:60],
            )

            if max_score >= self.threshold:
                matched_ref = EMERGENCY_REFERENCE_SENTENCES[max_idx]
                reason = (
                    f"Semantic match: '{matched_ref}' "
                    f"(score: {max_score:.3f} ≥ {self.threshold:.2f})"
                )
                logger.warning(
                    "EMERGENCY DETECTED (semantic): query=%r reason=%r",
                    query[:80],
                    reason,
                )
                return True, reason

        except Exception as exc:
            raise EmergencyDetectionError(
                f"Semantic emergency check failed: {exc}",
                query=query[:100],
            ) from exc

        logger.debug("Query cleared safety gate: %r", query[:60])
        return False, ""


def get_default_detector() -> MultilingualEmergencyDetector:
    """Return a module-level singleton emergency detector.

    The singleton is created lazily on first call to avoid loading LaBSE
    at import time (prevents slow test collection).

    Returns:
        The shared ``MultilingualEmergencyDetector`` instance.
    """
    global _detector_instance  
    if _detector_instance is not None:
        return _detector_instance
    with _detector_lock:
        if _detector_instance is None:
            logger.info("First call to get_default_detector — initializing...")
            _detector_instance = MultilingualEmergencyDetector()
        return _detector_instance


