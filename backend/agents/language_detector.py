"""
Two-layer language detection: fast script analysis + Sarvam LID API.

Layer 1 — Script analysis (deterministic, O(1)):
    Scans for Devanagari vs Latin codepoints.
    Outputs: "hi-IN" | "en-IN" | "hi-IN+en-IN" (Hinglish)

Layer 2 — Sarvam LID API (only called for ambiguous / Hinglish):
    Provides confidence + regional variant (mr-IN, ta-IN, etc.).
    Falls back gracefully to Layer 1 result on timeout/error.

Why two layers:
    Script analysis catches 90%+ of cases instantly, without a network
    call.  Sarvam LID is invoked only for mixed-script text where
    distinguishing Hindi from Marathi or Hinglish from Roman-Hindi
    is ambiguous.

Retry policy:
    Sarvam LID calls retry up to SARVAM_MAX_RETRIES times with
    exponential backoff (SARVAM_BACKOFF_BASE ** attempt seconds).

Usage example:
    from backend.agents.language_detector import LanguageDetector
    det = LanguageDetector()
    result = det.detect("मेरी EMI कब कटती है")
    # {"language": "hi-IN", "is_code_mixed": False, "method": "script"}
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

import requests

from backend.config import (
    SARVAM_API_KEY,
    SARVAM_BACKOFF_BASE,
    SARVAM_LID_URL,
    SARVAM_MAX_RETRIES,
    SARVAM_TIMEOUT,
)
from backend.exceptions import SarvamAPIError

logger = logging.getLogger(__name__)

# Unicode ranges for script detection
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_LATIN_RE = re.compile(r"[a-zA-Z]")


class LanguageDetector:
    """Detect the language of an utterance using script analysis + Sarvam LID.

    Args:
        sarvam_enabled: When False, skip the Sarvam LID API call entirely.
            Useful in tests or when SARVAM_API_KEY is absent.

    Example:
        >>> det = LanguageDetector()
        >>> det.detect("Hello, how are you?")
        {'language': 'en-IN', 'is_code_mixed': False, 'method': 'script', ...}
    """

    def __init__(self, sarvam_enabled: bool = True) -> None:
        self._sarvam_enabled = sarvam_enabled and bool(SARVAM_API_KEY)
        if not self._sarvam_enabled:
            logger.warning(
                "Sarvam LID disabled — running script-analysis only. "
                "Set SARVAM_API_KEY to enable API-based detection."
            )

    # ── Public API 

    def detect(self, text: str) -> dict:
        """Detect language, returning a result dict with all detection metadata.

        Args:
            text: Raw utterance from the caller.

        Returns:
            Dict with keys:
                ``language``        — BCP-47 code or "hi-IN+en-IN" for Hinglish.
                ``is_code_mixed``   — True for Hinglish.
                ``method``          — "script" | "sarvam_lid" | "fallback".
                ``confidence``      — Float 0–1 (1.0 for script-only results).

        Example:
            >>> det.detect("मेरा loan EMI कब कटेगा")
            {'language': 'hi-IN+en-IN', 'is_code_mixed': True, 'method': 'script', ...}
        """
        if not text or not text.strip():
            logger.debug("Empty text passed to language detector — defaulting to en-IN.")
            return _make_result("en-IN", False, "fallback", 0.5)

        script_result = self._detect_by_script(text)
        logger.debug("Script detection result: %s", script_result)

        # For code-mixed or short text, confirm with Sarvam LID
        if self._sarvam_enabled and (
            script_result["is_code_mixed"] or len(text) < 20
        ):
            try:
                sarvam_result = self._detect_via_sarvam(text)
                if sarvam_result:
                    merged = {**script_result, **sarvam_result}
                    logger.debug("Sarvam LID merged result: %s", merged)
                    return merged
            except SarvamAPIError as exc:
                logger.warning(
                    "Sarvam LID failed, falling back to script result: %s", exc
                )

        return script_result

    # ── Private helpers 

    def _detect_by_script(self, text: str) -> dict:
        """Fast O(1) script analysis using Unicode codepoint ranges.

        Args:
            text: Any utterance.

        Returns:
            Detection result dict (``method="script"``, ``confidence=1.0``).
        """
        has_devanagari = bool(_DEVANAGARI_RE.search(text))
        has_latin = bool(_LATIN_RE.search(text))

        if has_devanagari and has_latin:
            return _make_result("hi-IN+en-IN", True, "script", 1.0)
        if has_devanagari:
            return _make_result("hi-IN", False, "script", 1.0)
        return _make_result("en-IN", False, "script", 1.0)

    def _detect_via_sarvam(self, text: str) -> Optional[dict]:
        """Call Sarvam LID API with retry + exponential backoff.

        Args:
            text: Utterance to identify.

        Returns:
            Detection result dict (``method="sarvam_lid"``) or ``None``
            if the API returned an unexpected response structure.

        Raises:
            SarvamAPIError: After ``SARVAM_MAX_RETRIES`` failures.
        """
        last_error: Optional[Exception] = None

        for attempt in range(SARVAM_MAX_RETRIES):
            try:
                resp = requests.post(
                    SARVAM_LID_URL,
                    headers={
                        "api-subscription-key": SARVAM_API_KEY,
                        "Content-Type": "application/json",
                    },
                    json={"input": text},
                    timeout=SARVAM_TIMEOUT,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    lang = data.get("language_code", "en-IN")
                    confidence = float(data.get("confidence", 0.85))
                    is_code_mixed = "+" in lang or data.get("is_code_mixed", False)
                    return _make_result(lang, is_code_mixed, "sarvam_lid", confidence)

                if resp.status_code in (429, 503):
                    # Rate-limited or unavailable — respect retry
                    last_error = SarvamAPIError(
                        f"Sarvam LID HTTP {resp.status_code}",
                        status_code=resp.status_code,
                    )
                else:
                    # Non-retryable client error
                    logger.error(
                        "Sarvam LID non-retryable error: status=%d body=%r",
                        resp.status_code,
                        resp.text[:200],
                    )
                    return None

            except requests.exceptions.Timeout:
                last_error = SarvamAPIError(
                    f"Sarvam LID timed out after {SARVAM_TIMEOUT}s",
                    attempt=attempt + 1,
                )
                logger.warning("Sarvam LID timeout (attempt %d/%d)", attempt + 1, SARVAM_MAX_RETRIES)

            except requests.exceptions.RequestException as exc:
                last_error = SarvamAPIError(
                    f"Sarvam LID request error: {exc}", attempt=attempt + 1
                )
                logger.warning(
                    "Sarvam LID request exception (attempt %d/%d): %s",
                    attempt + 1, SARVAM_MAX_RETRIES, exc,
                )

            if attempt < SARVAM_MAX_RETRIES - 1:
                sleep_for = SARVAM_BACKOFF_BASE**attempt
                logger.debug("Sarvam LID backoff: sleeping %.1fs", sleep_for)
                time.sleep(sleep_for)

        raise SarvamAPIError(
            f"Sarvam LID failed after {SARVAM_MAX_RETRIES} attempts",
            last_error=str(last_error),
        )


def _make_result(
    language: str,
    is_code_mixed: bool,
    method: str,
    confidence: float,
) -> dict:
    return {
        "language": language,
        "is_code_mixed": is_code_mixed,
        "method": method,
        "confidence": confidence,
    }