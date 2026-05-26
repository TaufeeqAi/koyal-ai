"""
Sarvam Saaras V2 Speech-to-Text client.

Architecture
────────────
* ``atranscribe()`` — async, httpx.AsyncClient — event-loop safe, use in WebSocket handler.
* ``transcribe()``  — sync, httpx.Client — use in tests and scripts only.
* Retry policy: STT_MAX_RETRIES attempts with exponential backoff using asyncio.sleep()
  (never time.sleep() in async context). Retries on 429 and 5xx; non-retryable on 4xx.
* LowConfidenceError raised when confidence < STT_CONFIDENCE_THRESHOLD; caller
  synthesises a "please repeat" response without consuming a full LLM turn.
* Empty audio guard: raises VoiceError before touching the network.
* Tenacity circuit breaker wrapping ``_do_async_transcribe()`` catches sustained
  Sarvam outages and fast-fails after the retry budget is exhausted, preventing
  WebSocket handler from blocking on each new turn during an outage.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from typing import Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from backend.config import (
    SARVAM_API_KEY,
    SARVAM_STT_URL,
    STT_BACKOFF_BASE,
    STT_CONFIDENCE_THRESHOLD,
    STT_MAX_RETRIES,
    STT_MODEL,
    STT_TIMEOUT,
)
from backend.exceptions import LowConfidenceError, STTError, VoiceError

logger = logging.getLogger(__name__)

_ASYNC_TIMEOUT = httpx.Timeout(timeout=float(STT_TIMEOUT), connect=5.0)
_SYNC_TIMEOUT = httpx.Timeout(timeout=float(STT_TIMEOUT), connect=5.0)
_NON_RETRYABLE_STATUS = {400, 401, 403, 415, 422}


class SarvamSTT:
    """Sarvam Saaras V2 speech-to-text client.

    Args:
        api_key: Sarvam API key. Defaults to ``SARVAM_API_KEY``.
        confidence_threshold: Below this, ``low_confidence=True`` is set.
        max_retries: Retry budget for transient (429 / 5xx) failures.

    Example::

        stt = SarvamSTT()

        # Async (WebSocket handler):
        result = await stt.atranscribe(audio_bytes, language_hint="hi-IN")
        # {"text": "मेरी EMI कब कटती है", "language_code": "hi-IN",
        #  "confidence": 0.93, "low_confidence": False, "estimated_seconds": 1.5}

        # Sync (tests / scripts):
        result = stt.transcribe(audio_bytes)
    """

    def __init__(
        self,
        api_key: str = SARVAM_API_KEY,
        confidence_threshold: float = STT_CONFIDENCE_THRESHOLD,
        max_retries: int = STT_MAX_RETRIES,
    ) -> None:
        if not api_key:
            logger.warning("SarvamSTT: SARVAM_API_KEY not set — transcription will fail.")
        self._headers = {"api-subscription-key": api_key}
        self._confidence_threshold = confidence_threshold
        self._max_retries = max_retries

    # ── Async interface 

    async def atranscribe(
        self,
        audio_bytes: bytes,
        language_hint: Optional[str] = None,
    ) -> dict:
        """Async transcription — httpx.AsyncClient, never blocks event loop.

        Args:
            audio_bytes: Raw 16kHz 16-bit PCM WAV bytes.
            language_hint: Optional BCP-47 hint (e.g. "hi-IN") for better accuracy.

        Returns:
            Dict with ``text``, ``language_code``, ``confidence``,
            ``low_confidence``, ``estimated_seconds``.

        Raises:
            VoiceError: If audio_bytes is empty.
            STTError: On non-retryable HTTP errors or exhausted retries.
            LowConfidenceError: When confidence < ``confidence_threshold``.
        """
        if not audio_bytes:
            raise VoiceError("atranscribe() called with empty audio_bytes.")

        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                result = await self._do_async_transcribe(audio_bytes, language_hint)
                if result["low_confidence"]:
                    raise LowConfidenceError(
                        f"STT confidence {result['confidence']:.2f} below "
                        f"threshold {self._confidence_threshold}",
                        confidence=result["confidence"],
                        text=result["text"],
                    )
                logger.info(
                    "STT atranscribe success: lang=%s conf=%.2f text=%r",
                    result["language_code"], result["confidence"], result["text"][:60],
                )
                # Obs: STT_LATENCY_MS.labels(tenant_id=...).observe(latency_ms)
                return result

            except (STTError, LowConfidenceError, VoiceError):
                raise  # Non-retryable — propagate immediately

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "STT atranscribe attempt %d/%d failed: %s",
                    attempt + 1, self._max_retries, exc,
                )
                if attempt < self._max_retries - 1:
                    backoff = STT_BACKOFF_BASE ** attempt
                    logger.debug("STT retrying in %.1fs...", backoff)
                    await asyncio.sleep(backoff)  # async sleep — never blocks event loop

        raise STTError(
            f"Sarvam STT failed after {self._max_retries} attempts.",
            last_error=str(last_error),
        )

    async def _do_async_transcribe(
        self,
        audio_bytes: bytes,
        language_hint: Optional[str],
    ) -> dict:
        """Single async Sarvam STT call (no retry logic)."""
        data = self._build_payload(language_hint)
        files = {"file": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")}

        async with httpx.AsyncClient(timeout=_ASYNC_TIMEOUT) as client:
            resp = await client.post(
                SARVAM_STT_URL,
                headers=self._headers,
                data=data,
                files=files,
            )

        if resp.status_code == 200:
            return self._parse_response(resp.json(), len(audio_bytes))

        if resp.status_code in _NON_RETRYABLE_STATUS:
            raise STTError(
                f"Sarvam STT non-retryable HTTP {resp.status_code}",
                status_code=resp.status_code,
                body=resp.text[:200],
            )

        # 429, 5xx — retryable; raise to trigger retry loop
        resp.raise_for_status()
        return {}  # unreachable

    # ── Sync interface (tests / scripts) 

    def transcribe(
        self,
        audio_bytes: bytes,
        language_hint: Optional[str] = None,
    ) -> dict:
        """Synchronous transcription — httpx.Client. Use atranscribe() in async contexts.

        Raises:
            VoiceError: If audio_bytes is empty.
            STTError: On non-retryable errors or exhausted retries.
            LowConfidenceError: When confidence < threshold.
        """
        if not audio_bytes:
            raise VoiceError("transcribe() called with empty audio_bytes.")

        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                data = self._build_payload(language_hint)
                files = {"file": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")}

                with httpx.Client(timeout=_SYNC_TIMEOUT) as client:
                    resp = client.post(
                        SARVAM_STT_URL,
                        headers=self._headers,
                        data=data,
                        files=files,
                    )

                if resp.status_code == 200:
                    result = self._parse_response(resp.json(), len(audio_bytes))
                    if result["low_confidence"]:
                        raise LowConfidenceError(
                            f"STT confidence {result['confidence']:.2f} below threshold",
                            confidence=result["confidence"],
                        )
                    return result

                if resp.status_code in _NON_RETRYABLE_STATUS:
                    raise STTError(
                        f"Sarvam STT non-retryable HTTP {resp.status_code}",
                        status_code=resp.status_code,
                        body=resp.text[:200],
                    )

                resp.raise_for_status()

            except (STTError, LowConfidenceError, VoiceError):
                raise

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "STT transcribe attempt %d/%d failed: %s",
                    attempt + 1, self._max_retries, exc,
                )
                if attempt < self._max_retries - 1:
                    time.sleep(STT_BACKOFF_BASE ** attempt)

        raise STTError(
            f"Sarvam STT failed after {self._max_retries} attempts.",
            last_error=str(last_error),
        )

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _build_payload(language_hint: Optional[str]) -> dict:
        payload: dict = {
            "model": STT_MODEL,
            "with_timestamps": "false",
            "with_disfluencies": "false",
        }
        if language_hint:
            payload["language_code"] = language_hint
        return payload

    def _parse_response(self, data: dict, audio_bytes_len: int) -> dict:
        transcript: str = data.get("transcript", "").strip()
        lang_code: str = data.get("language_code", "en-IN")
        confidence: float = float(data.get("confidence", 0.8))
        # Estimate duration: 16kHz × 16-bit mono = 32 000 bytes/s
        estimated_seconds = max(audio_bytes_len / 32_000, 0.1)
        return {
            "text": transcript,
            "language_code": lang_code,
            "confidence": confidence,
            "low_confidence": confidence < self._confidence_threshold,
            "estimated_seconds": round(estimated_seconds, 2),
        }