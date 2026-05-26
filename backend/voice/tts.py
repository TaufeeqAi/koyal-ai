"""
Sarvam Bulbul V1 Text-to-Speech client.

Architecture
────────────
* ``asynthesize()``          — async, chunks in parallel via asyncio.gather()
* ``asynthesize_streaming()``— async generator, yields per-sentence audio chunks
* ``synthesize()``           — sync, for tests and outbound campaign pre-generation

Chunking: Sarvam TTS accepts max 500 chars per request. Longer text is split at
sentence boundaries (danda ।, .!?) then at word boundaries if a sentence exceeds
the limit. For asynthesize(), all chunks are gathered in parallel — latency is
max(chunk_latencies) rather than sum. For streaming, first chunk is yielded
while the rest are pending, minimising time-to-first-audio.

Retry: each chunk uses a synchronous retry loop (sync) or individual async call
(async). Chunk failures are logged and silently skipped; the remaining chunks
continue — partial audio is better than silence.

Voice mapping: 9 Indian languages including Gujarati and Punjabi.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import time
from typing import AsyncGenerator, Optional

import httpx

from backend.config import (
    SARVAM_API_KEY,
    SARVAM_TTS_URL,
    TTS_BACKOFF_BASE,
    TTS_MAX_CHARS_PER_CHUNK,
    TTS_MAX_RETRIES,
    TTS_MODEL,
    TTS_PACE,
    TTS_SAMPLE_RATE,
    TTS_TIMEOUT,
)
from backend.exceptions import TTSError

logger = logging.getLogger(__name__)

_ASYNC_TIMEOUT = httpx.Timeout(timeout=float(TTS_TIMEOUT), connect=5.0)
_SYNC_TIMEOUT = httpx.Timeout(timeout=float(TTS_TIMEOUT), connect=5.0)

# Full language → Sarvam voice mapping (9 Indian languages)
LANGUAGE_VOICE_MAP: dict[str, str] = {
    "hi-IN":       "meera",    # Hindi female — warm, natural
    "en-IN":       "anushka",  # Indian-accented English
    "hi-IN+en-IN": "meera",    # Hinglish — Hindi voice handles mixed script best
    "mr-IN":       "madhura",  # Marathi
    "ta-IN":       "pavithra", # Tamil
    "te-IN":       "hema",     # Telugu
    "kn-IN":       "gagan",    # Kannada
    "bn-IN":       "bani",     # Bengali
    "gu-IN":       "diya",     # Gujarati
    "pa-IN":       "arjun",    # Punjabi
}
DEFAULT_VOICE: str = "meera"

_HINDI_SENTENCE_RE = re.compile(r"[।!?.]")
_ENGLISH_SENTENCE_RE = re.compile(r"[.!?]")


def _get_voice(language_code: str) -> str:
    return LANGUAGE_VOICE_MAP.get(language_code, DEFAULT_VOICE)


def _get_target_lang(language_code: str) -> str:
    """Strip code-mixed suffix: 'hi-IN+en-IN' → 'hi-IN'."""
    return language_code.split("+")[0]


def _chunk_text(text: str, max_chars: int) -> list[str]:
    """Split text at word boundaries, each chunk ≤ max_chars.

    Example:
        >>> _chunk_text("Hello world foo bar", max_chars=10)
        ['Hello', 'world foo', 'bar']
    """
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining)
            break
        split_at = remaining.rfind(" ", 0, max_chars)
        if split_at == -1:
            split_at = max_chars
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip()
    return [c for c in chunks if c]


def _split_sentences(text: str, language_code: str) -> list[str]:
    """Split text at sentence boundaries for streaming synthesis.

    Hindi: splits on danda (।) plus .!?
    Other: splits on .!?

    Example:
        >>> _split_sentences("Hello. How are you?", "en-IN")
        ['Hello', 'How are you?']
    """
    if "hi" in language_code:
        parts = _HINDI_SENTENCE_RE.split(text)
    else:
        parts = _ENGLISH_SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


class SarvamTTS:
    """Sarvam Bulbul V1 text-to-speech client.

    Args:
        api_key: Sarvam API key. Defaults to ``SARVAM_API_KEY``.
        pace: Speech pace multiplier (0.5–2.0). Defaults to ``TTS_PACE``.
        sample_rate: Output audio sample rate. Defaults to ``TTS_SAMPLE_RATE``.

    Example::

        tts = SarvamTTS()

        # Async single (WebSocket handler — all chunks parallel):
        audio = await tts.asynthesize("आपकी EMI 5 तारीख को कटेगी", "hi-IN")

        # Async streaming (first sentence early):
        async for chunk in tts.asynthesize_streaming(response, "hi-IN"):
            await websocket.send_bytes(chunk)

        # Sync (tests / outbound campaign):
        audio = tts.synthesize("Your refund arrives in 24 hours", "en-IN")
    """

    def __init__(
        self,
        api_key: str = SARVAM_API_KEY,
        pace: float = TTS_PACE,
        sample_rate: int = TTS_SAMPLE_RATE,
    ) -> None:
        if not api_key:
            logger.warning("SarvamTTS: SARVAM_API_KEY not set — synthesis will fail.")
        self._headers = {
            "api-subscription-key": api_key,
            "Content-Type": "application/json",
        }
        self._pace = pace
        self._sample_rate = sample_rate
        logger.info(
            "SarvamTTS initialised (model=%s, pace=%.1f, sample_rate=%d)",
            TTS_MODEL, pace, sample_rate,
        )

    # ── Async interface 

    async def asynthesize(
        self,
        text: str,
        language_code: str = "hi-IN",
        pace: Optional[float] = None,
    ) -> bytes:
        """Async full-text synthesis. All chunks gathered in parallel.

        Args:
            text: Text to synthesise (any length; auto-chunked).
            language_code: BCP-47 language code.
            pace: Override pace for this call.

        Returns:
            Concatenated WAV bytes. Empty bytes if text is empty or all chunks fail.
        """
        if not text or not text.strip():
            return b""

        effective_pace = pace if pace is not None else self._pace
        chunks = _chunk_text(text, TTS_MAX_CHARS_PER_CHUNK)
        tasks = [
            self._async_synthesize_chunk(chunk, language_code, effective_pace)
            for chunk in chunks
        ]
        results: list[bytes] = await asyncio.gather(*tasks, return_exceptions=False)
        audio = b"".join(r for r in results if r)
        logger.debug(
            "TTS asynthesize: lang=%s chars=%d chunks=%d audio=%d bytes",
            language_code, len(text), len(chunks), len(audio),
        )
        return audio

    async def asynthesize_streaming(
        self,
        text: str,
        language_code: str = "hi-IN",
        pace: Optional[float] = None,
    ) -> AsyncGenerator[bytes, None]:
        """Async streaming synthesis — yields audio per sentence as soon as ready.

        Enables time-to-first-audio optimisation: first sentence is yielded
        while subsequent sentences are synthesised in the background.

        Args:
            text: Full response text.
            language_code: BCP-47 language code.

        Yields:
            WAV bytes for each sentence (send directly over WebSocket).
        """
        effective_pace = pace if pace is not None else self._pace
        sentences = _split_sentences(text, language_code)
        logger.debug(
            "TTS streaming: %d sentences lang=%s", len(sentences), language_code
        )
        for sentence in sentences:
            if not sentence.strip():
                continue
            try:
                audio = await self._async_synthesize_chunk(
                    sentence, language_code, effective_pace
                )
                if audio:
                    yield audio
            except Exception as exc:
                logger.error(
                    "TTS streaming sentence failed (skipping): %s (text=%r)",
                    exc, sentence[:40],
                )

    async def _async_synthesize_chunk(
        self,
        text: str,
        language_code: str,
        pace: float,
    ) -> bytes:
        """Async synthesis of a single chunk (≤ TTS_MAX_CHARS_PER_CHUNK chars)."""
        payload = self._build_payload(text, language_code, pace)
        try:
            async with httpx.AsyncClient(timeout=_ASYNC_TIMEOUT) as client:
                resp = await client.post(
                    SARVAM_TTS_URL, headers=self._headers, json=payload
                )
                if resp.status_code in {400, 401, 403, 422}:
                    raise TTSError(
                        f"Sarvam TTS non-retryable HTTP {resp.status_code}",
                        body=resp.text[:200],
                    )
                resp.raise_for_status()
                return self._decode_audio(resp.json())
        except TTSError:
            raise
        except httpx.TimeoutException:
            logger.warning("Sarvam TTS chunk timeout: text=%r", text[:40])
            return b""
        except httpx.HTTPStatusError as exc:
            logger.error(
                "Sarvam TTS HTTP %d: %s",
                exc.response.status_code, exc.response.text[:200],
            )
            return b""

    # ── Sync interface 

    def synthesize(
        self,
        text: str,
        language_code: str = "hi-IN",
        pace: Optional[float] = None,
    ) -> bytes:
        """Synchronous synthesis. Use asynthesize() in all async contexts.

        Returns:
            WAV bytes. Empty bytes if text is empty or all chunks fail.
        """
        if not text or not text.strip():
            return b""

        effective_pace = pace if pace is not None else self._pace
        chunks = _chunk_text(text, TTS_MAX_CHARS_PER_CHUNK)
        audio_parts: list[bytes] = []

        for i, chunk in enumerate(chunks):
            part = self._sync_synthesize_chunk(chunk, language_code, effective_pace)
            if part:
                audio_parts.append(part)
            else:
                logger.warning("TTS chunk %d/%d returned empty bytes", i + 1, len(chunks))

        combined = b"".join(audio_parts)
        logger.info(
            "TTS synthesize: lang=%s chars=%d → %d bytes audio",
            language_code, len(text), len(combined),
        )
        return combined

    def _sync_synthesize_chunk(
        self,
        text: str,
        language_code: str,
        pace: float,
    ) -> bytes:
        """Sync synthesis with retry for a single chunk."""
        payload = self._build_payload(text, language_code, pace)
        last_error: Optional[Exception] = None

        for attempt in range(TTS_MAX_RETRIES):
            try:
                with httpx.Client(timeout=_SYNC_TIMEOUT) as client:
                    resp = client.post(
                        SARVAM_TTS_URL, headers=self._headers, json=payload
                    )
                if resp.status_code in {400, 401, 403, 422}:
                    raise TTSError(
                        f"Sarvam TTS non-retryable HTTP {resp.status_code}",
                        body=resp.text[:200],
                    )
                resp.raise_for_status()
                return self._decode_audio(resp.json())
            except TTSError:
                raise
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "TTS chunk attempt %d/%d failed: %s", attempt + 1, TTS_MAX_RETRIES, exc
                )
                if attempt < TTS_MAX_RETRIES - 1:
                    time.sleep(TTS_BACKOFF_BASE ** attempt)

        logger.error("TTS chunk exhausted %d retries: %s", TTS_MAX_RETRIES, last_error)
        return b""

    # ── Private helpers 

    def _build_payload(self, text: str, language_code: str, pace: float) -> dict:
        return {
            "inputs": [text],
            "target_language_code": _get_target_lang(language_code),
            "speaker": _get_voice(language_code),
            "pitch": 0,
            "pace": pace,
            "loudness": 1.5,
            "speech_sample_rate": self._sample_rate,
            "enable_preprocessing": True,
            "model": TTS_MODEL,
        }

    @staticmethod
    def _decode_audio(response_json: dict) -> bytes:
        audios = response_json.get("audios", [])
        if not audios:
            logger.warning("TTS response contained no audio data.")
            return b""
        return base64.b64decode(audios[0])