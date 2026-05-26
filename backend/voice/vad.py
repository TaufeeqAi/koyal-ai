"""
WebRTC Voice Activity Detection with frame buffering and speech segmentation.

Architecture (two-class design for independent testability)
──────────────────────────────────────────────────────────
1. ``FrameBuffer``      — Accumulates variable-size incoming bytes into fixed-
                          size VAD frames (e.g. 960 bytes for 30ms at 16kHz).
                          Protocol-level concern only; no VAD logic.

2. ``SpeechSegmenter``  — Stateful state machine (IDLE → IN_SPEECH) using
                          consecutive-frame hysteresis to extract clean utterance
                          buffers for STT submission.
                          Uses FrameBuffer internally.

Energy guard (from B)
─────────────────────
Before passing a frame to webrtcvad, RMS energy is computed. Frames below
VAD_ENERGY_THRESHOLD are forced to SILENCE — this guards against constant-
amplitude noise (HVAC, line buzz) that webrtcvad may classify as speech.

State machine
─────────────
    ┌───────┐  speech_frames ≥ VAD_SPEECH_THRESHOLD  ┌─────────────┐
    │ IDLE  │ ──────────────────────────────────────► │ IN_SPEECH   │
    └───────┘                                         └──────┬──────┘
        ▲                                                    │
        │  silence_frames ≥ VAD_SILENCE_THRESHOLD            │
        └────────────────────────────────────────────────────┘
                          flush accumulated speech → STT

Pre-speech ring buffer
──────────────────────
Captures the last VAD_SPEECH_THRESHOLD frames *before* speech onset.
This recovers the attack transient (first syllable onset) that occurs before
the VAD threshold fires, preventing word-initial clipping at the STT boundary.

VAD flush control message
─────────────────────────
Call ``segmenter.flush()`` to force-submit any buffered speech segment
regardless of silence threshold. The WebSocket handler issues this when
the client sends ``{"type": "flush"}`` — allows callers with low-energy
microphones to bypass the silence gate.
"""

from __future__ import annotations

import collections
import logging
import struct
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import webrtcvad

from backend.config import (
    VAD_AGGRESSIVENESS,
    VAD_ENERGY_THRESHOLD,
    VAD_FRAME_DURATION_MS,
    VAD_MIN_SPEECH_FRAMES,
    VAD_SAMPLE_RATE,
    VAD_SILENCE_THRESHOLD,
    VAD_SPEECH_THRESHOLD,
)
from backend.exceptions import VADError

logger = logging.getLogger(__name__)

# Derived frame size constant: sample_rate × frame_duration_ms/1000 × 2 bytes/sample
FRAME_BYTES: int = int(VAD_SAMPLE_RATE * VAD_FRAME_DURATION_MS / 1000) * 2
# Example: 16000 × 30/1000 × 2 = 960 bytes for 30ms at 16kHz


class VADState(Enum):
    """State machine states for the SpeechSegmenter."""
    IDLE = auto()
    IN_SPEECH = auto()


@dataclass
class VADResult:
    """Result of processing an audio chunk through the segmenter.

    Attributes:
        state: Current VAD state after processing this chunk.
        utterance_complete: True when a complete utterance is ready.
        speech_bytes: Complete PCM utterance audio (non-empty when utterance_complete).
        speech_started: True on the first chunk that triggered speech onset.
        num_speech_frames: Total voiced frames accumulated in the current segment.
    """
    state: VADState
    utterance_complete: bool
    speech_bytes: bytes
    speech_started: bool
    num_speech_frames: int


def _compute_rms(frame_bytes: bytes) -> float:
    """Compute root-mean-square energy of a 16-bit PCM audio frame."""
    n = len(frame_bytes) // 2
    if n == 0:
        return 0.0
    samples = struct.unpack(f"<{n}h", frame_bytes)
    return (sum(s * s for s in samples) / n) ** 0.5


class FrameBuffer:
    """Accumulate variable-length byte streams into fixed-size VAD frames.

    WebRTC VAD requires exactly N bytes per frame (e.g. 960 for 30ms at 16kHz).
    WebSocket audio arrives in variable-size chunks. This class buffers and
    yields complete frames.

    Args:
        frame_bytes: Fixed frame size in bytes. Defaults to ``FRAME_BYTES``.

    Example::

        buf = FrameBuffer()
        frames = buf.push(audio_chunk)  # May be empty if chunk < frame_bytes
        for frame in frames:
            is_speech = vad.is_speech(frame, 16000)
    """

    def __init__(self, frame_bytes: int = FRAME_BYTES) -> None:
        self._frame_bytes = frame_bytes
        self._buffer = bytearray()

    def push(self, data: bytes) -> list[bytes]:
        """Add audio bytes and return all complete frames.

        Args:
            data: Raw PCM audio bytes (any length).

        Returns:
            List of fixed-size frames. Empty list if data < frame size.
        """
        self._buffer.extend(data)
        frames: list[bytes] = []
        while len(self._buffer) >= self._frame_bytes:
            frame = bytes(self._buffer[: self._frame_bytes])
            frames.append(frame)
            del self._buffer[: self._frame_bytes]
        return frames

    def clear(self) -> None:
        """Discard all buffered bytes."""
        self._buffer.clear()

    @property
    def buffered_bytes(self) -> int:
        """Bytes currently held in the internal buffer."""
        return len(self._buffer)


class SpeechSegmenter:
    """Stateful VAD that detects complete utterances from a stream of audio frames.

    Uses hysteresis: requires ``VAD_SPEECH_THRESHOLD`` consecutive voiced frames
    to start speech, and ``VAD_SILENCE_THRESHOLD`` consecutive silent frames to
    end it. Filters micro-bursts shorter than ``VAD_MIN_SPEECH_FRAMES``. Applies
    energy guard (RMS) as a secondary silence detector before webrtcvad.

    Args:
        aggressiveness: WebRTC VAD aggressiveness 0–3.
        sample_rate: PCM sample rate. Must be 8000/16000/32000/48000.
        frame_duration_ms: Frame duration. Must be 10, 20, or 30.

    Raises:
        VADError: On invalid parameters or webrtcvad initialisation failure.

    Example::

        seg = SpeechSegmenter()
        result = seg.process_chunk(audio_chunk_bytes)
        if result.utterance_complete:
            transcript = await stt.atranscribe(result.speech_bytes)

        # Force-flush any buffered speech (VAD flush control message):
        flushed = seg.flush()
        if flushed:
            transcript = await stt.atranscribe(flushed)
    """

    def __init__(
        self,
        aggressiveness: int = VAD_AGGRESSIVENESS,
        sample_rate: int = VAD_SAMPLE_RATE,
        frame_duration_ms: int = VAD_FRAME_DURATION_MS,
    ) -> None:
        if sample_rate not in (8000, 16000, 32000, 48000):
            raise VADError(
                f"Unsupported VAD sample rate: {sample_rate}. "
                f"Must be 8000, 16000, 32000, or 48000 Hz.",
                sample_rate=sample_rate,
            )
        if frame_duration_ms not in (10, 20, 30):
            raise VADError(
                f"Unsupported VAD frame duration: {frame_duration_ms}ms. "
                f"Must be 10, 20, or 30.",
                frame_duration_ms=frame_duration_ms,
            )
        try:
            self._vad = webrtcvad.Vad(aggressiveness)
        except Exception as exc:
            raise VADError(
                f"Failed to initialise WebRTC VAD (aggressiveness={aggressiveness}): {exc}"
            ) from exc

        self._sample_rate = sample_rate
        self._frame_duration_ms = frame_duration_ms
        self._frame_buffer = FrameBuffer(
            frame_bytes=int(sample_rate * frame_duration_ms / 1000) * 2
        )

        # State machine
        self._state = VADState.IDLE
        self._speech_frame_count: int = 0
        self._silence_frame_count: int = 0
        self._speech_buffer: bytearray = bytearray()

        # Pre-speech ring buffer: captures attack transient before VAD threshold
        self._pre_speech_ring: collections.deque[bytes] = collections.deque(
            maxlen=VAD_SPEECH_THRESHOLD
        )

    def process_chunk(self, data: bytes) -> VADResult:
        """Process a variable-length audio chunk through the VAD state machine.

        Args:
            data: Raw PCM bytes from WebSocket (any size).

        Returns:
            VADResult with utterance_complete=True when a full segment is ready.
        """
        frames = self._frame_buffer.push(data)
        result = VADResult(
            state=self._state,
            utterance_complete=False,
            speech_bytes=b"",
            speech_started=False,
            num_speech_frames=self._speech_frame_count,
        )

        for frame in frames:
            frame_result = self._process_frame(frame)
            if frame_result is not None:
                return VADResult(
                    state=self._state,
                    utterance_complete=True,
                    speech_bytes=frame_result,
                    speech_started=False,
                    num_speech_frames=self._speech_frame_count,
                )

        result.state = self._state
        result.num_speech_frames = self._speech_frame_count
        return result

    def flush(self) -> Optional[bytes]:
        """Force-flush any buffered speech regardless of silence threshold.

        Called when client sends ``{"type": "flush"}`` (low-energy mic guard)
        or at end-of-stream. Returns PCM bytes if enough speech is buffered,
        otherwise None.

        Returns:
            Speech bytes if buffered speech ≥ VAD_MIN_SPEECH_FRAMES, else None.
        """
        if (
            self._state == VADState.IN_SPEECH
            and self._speech_frame_count >= VAD_MIN_SPEECH_FRAMES
        ):
            segment = bytes(self._speech_buffer)
            duration_s = len(segment) / (self._sample_rate * 2)
            logger.debug(
                "VAD flush: %.2fs speech segment (%d frames)",
                duration_s, self._speech_frame_count,
            )
            self._reset()
            return segment
        self._reset()
        return None

    def reset(self) -> None:
        """Reset all VAD state. Call between sessions."""
        self._reset()
        self._frame_buffer.clear()

    # ── Private helpers 

    def _process_frame(self, frame: bytes) -> Optional[bytes]:
        """Run one frame through the energy guard + webrtcvad + state machine.

        Returns utterance bytes if the frame completes an utterance, else None.
        """
        # Energy guard: very quiet frames are always silence (HVAC, line buzz)
        rms = _compute_rms(frame)
        if rms < VAD_ENERGY_THRESHOLD:
            is_speech = False
        else:
            try:
                is_speech = self._vad.is_speech(frame, self._sample_rate)
            except Exception:
                is_speech = False

        if self._state == VADState.IDLE:
            self._pre_speech_ring.append(frame)
            if is_speech:
                self._speech_frame_count += 1
                if self._speech_frame_count >= VAD_SPEECH_THRESHOLD:
                    # Transition to IN_SPEECH; prepend pre-speech ring buffer
                    self._state = VADState.IN_SPEECH
                    self._speech_buffer = bytearray()
                    for pre_frame in self._pre_speech_ring:
                        self._speech_buffer.extend(pre_frame)
                    self._silence_frame_count = 0
                    logger.debug("VAD: speech onset detected")
            else:
                self._speech_frame_count = 0
            return None

        # IN_SPEECH state
        self._speech_buffer.extend(frame)

        if is_speech:
            self._speech_frame_count += 1
            self._silence_frame_count = 0
        else:
            self._silence_frame_count += 1

        if self._silence_frame_count >= VAD_SILENCE_THRESHOLD:
            if self._speech_frame_count >= VAD_MIN_SPEECH_FRAMES:
                segment = bytes(self._speech_buffer)
                duration_s = len(segment) / (self._sample_rate * 2)
                logger.debug(
                    "VAD: utterance complete — %.2fs, %d voiced frames",
                    duration_s, self._speech_frame_count,
                )
                self._reset()
                return segment
            else:
                logger.debug(
                    "VAD: discarding short burst (%d frames < min %d)",
                    self._speech_frame_count, VAD_MIN_SPEECH_FRAMES,
                )
                self._reset()
        return None

    def _reset(self) -> None:
        self._state = VADState.IDLE
        self._speech_frame_count = 0
        self._silence_frame_count = 0
        self._speech_buffer = bytearray()
        self._pre_speech_ring.clear()