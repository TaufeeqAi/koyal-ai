"""
PCM ↔ WAV conversion helpers for the LiveKit audio bridge.

Why this module exists:
    LiveKit delivers raw PCM int16 audio frames.
    Sarvam STT expects a multipart WAV file (PCM 16-bit with RIFF header).
    Sarvam TTS returns base64-decoded WAV bytes (PCM 16-bit with RIFF header).
    rtc.AudioSource.capture_frame() expects raw PCM int16 bytes (no header).

These conversions are isolated here so that audio_bridge.py stays focused
on the LiveKit event loop, not audio format bookkeeping. Both audio_bridge.py
and outbound_dialer.py import from this single module (DRY).

Usage example:
    from backend.telephony.audio_utils import pcm_to_wav, wav_to_pcm_frames

    wav_bytes = pcm_to_wav(pcm_bytes, sample_rate=16000)
    for frame_pcm in wav_to_pcm_frames(wav_bytes, frame_duration_ms=20):
        await audio_source.capture_frame(
            rtc.AudioFrame(
                data=frame_pcm,
                sample_rate=16000,
                num_channels=1,
                samples_per_channel=len(frame_pcm) // 2,
            )
        )
"""

from __future__ import annotations

import io
import logging
import wave
from typing import Iterator

logger = logging.getLogger(__name__)


def pcm_to_wav(
    pcm_bytes: bytes,
    sample_rate: int = 16000,
    num_channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """Wrap raw PCM int16 bytes in a standard WAV (RIFF) container.

    Args:
        pcm_bytes: Raw PCM audio — signed 16-bit little-endian.
        sample_rate: Samples per second (Hz). Default: 16000.
        num_channels: 1 for mono, 2 for stereo. Default: 1.
        sample_width: Bytes per sample. 2 for int16. Default: 2.

    Returns:
        WAV-wrapped bytes ready to send as a multipart file upload to Sarvam STT.

    Raises:
        ValueError: If pcm_bytes is empty.

    Example:
        >>> wav = pcm_to_wav(raw_pcm, sample_rate=16000)
        >>> stt.transcribe(wav, language_hint="hi-IN")
    """
    if not pcm_bytes:
        raise ValueError("pcm_to_wav: pcm_bytes must not be empty.")

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    wav_bytes = buf.getvalue()
    logger.debug(
        "pcm_to_wav: %d PCM bytes → %d WAV bytes (rate=%d)",
        len(pcm_bytes), len(wav_bytes), sample_rate,
    )
    return wav_bytes


def wav_to_pcm(wav_bytes: bytes) -> tuple[bytes, int, int]:
    """Extract raw PCM bytes and audio parameters from a WAV container.

    Args:
        wav_bytes: Standard WAV bytes with RIFF header.

    Returns:
        Tuple of ``(pcm_bytes, sample_rate, num_channels)``.

    Raises:
        ValueError: If wav_bytes is not a valid WAV file.

    Example:
        >>> pcm, sr, ch = wav_to_pcm(tts_wav_bytes)
        >>> assert sr == 16000 and ch == 1
    """
    if not wav_bytes:
        raise ValueError("wav_to_pcm: wav_bytes must not be empty.")

    try:
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            sample_rate = wf.getframerate()
            num_channels = wf.getnchannels()
            pcm_bytes = wf.readframes(wf.getnframes())
        logger.debug(
            "wav_to_pcm: %d WAV bytes → %d PCM bytes (rate=%d ch=%d)",
            len(wav_bytes), len(pcm_bytes), sample_rate, num_channels,
        )
        return pcm_bytes, sample_rate, num_channels
    except Exception as exc:
        raise ValueError(f"wav_to_pcm: Invalid WAV data: {exc}") from exc


def wav_to_pcm_frames(
    wav_bytes: bytes,
    frame_duration_ms: int = 20,
    target_sample_rate: int = 16000,
) -> Iterator[bytes]:
    """Yield fixed-duration PCM frames from a WAV file.

    Used for pushing TTS audio to LiveKit in bite-sized 20ms chunks that
    maintain smooth real-time playback (avoids buffer underrun or jitter).
    Incomplete last frames are padded with silence.

    Args:
        wav_bytes: WAV audio from Sarvam TTS.
        frame_duration_ms: Duration of each yielded frame in milliseconds.
            20ms is standard for real-time audio (ITU-T G.711 frame size).
        target_sample_rate: Expected sample rate. Warns if mismatch.

    Yields:
        Raw PCM int16 bytes for each ``frame_duration_ms`` chunk.

    Raises:
        ValueError: If wav_bytes is not valid WAV.

    Example:
        >>> for frame in wav_to_pcm_frames(tts_wav, frame_duration_ms=20):
        ...     audio_frame = rtc.AudioFrame(data=frame, sample_rate=16000,
        ...                                  num_channels=1,
        ...                                  samples_per_channel=len(frame)//2)
        ...     await source.capture_frame(audio_frame)
    """
    pcm_bytes, sample_rate, num_channels = wav_to_pcm(wav_bytes)

    if sample_rate != target_sample_rate:
        logger.warning(
            "wav_to_pcm_frames: WAV sample rate %d != expected %d — "
            "LiveKit will handle resampling.",
            sample_rate, target_sample_rate,
        )

    # bytes per frame = (samples/s × ms / 1000) × channels × 2 bytes/sample
    bytes_per_frame = int(sample_rate * frame_duration_ms / 1000) * num_channels * 2

    offset = 0
    while offset < len(pcm_bytes):
        chunk = pcm_bytes[offset: offset + bytes_per_frame]
        if len(chunk) < bytes_per_frame:
            # Pad last incomplete frame with silence
            chunk = chunk + b"\x00" * (bytes_per_frame - len(chunk))
        yield chunk
        offset += bytes_per_frame


def estimate_audio_duration_ms(pcm_bytes: bytes, sample_rate: int = 16000) -> float:
    """Estimate audio duration in milliseconds from PCM byte count.

    Args:
        pcm_bytes: Raw PCM int16 mono bytes.
        sample_rate: Sample rate in Hz.

    Returns:
        Estimated duration in milliseconds.

    Example:
        >>> ms = estimate_audio_duration_ms(buffer, 16000)
        >>> print(f"Buffer holds ~{ms:.0f}ms of audio")
    """
    num_samples = len(pcm_bytes) // 2  # int16 = 2 bytes/sample
    return (num_samples / sample_rate) * 1000


def silence_pcm(duration_ms: int, sample_rate: int = 16000) -> bytes:
    """Generate silent PCM bytes of the specified duration.

    Args:
        duration_ms: Duration of silence in milliseconds.
        sample_rate: Sample rate in Hz.

    Returns:
        Bytes of silence (all zeros, int16 format).

    Example:
        >>> silence = silence_pcm(200, 16000)  # 200ms of silence
    """
    num_samples = int(sample_rate * duration_ms / 1000)
    return b"\x00\x00" * num_samples