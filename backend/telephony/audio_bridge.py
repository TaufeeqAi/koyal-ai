from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Optional

from livekit import rtc

from backend.agents.graph import koyal_graph
from backend.exceptions import VoiceError
from backend.agents.state import make_initial_state
from backend.observability.prometheus_metrics import (
    record_call_end,
    record_call_start,
    record_escalation,
    record_language_detection,
    record_llm_latency,
    record_pipeline_latency,
    record_safety_cleared,
    record_stt_latency,
    record_tts_latency,
    record_ttfr,
)
from backend.config import (
    TTS_SAMPLE_RATE,
    VAD_SAMPLE_RATE,
    FALLBACK_BUFFER_THRESHOLD,
    MAX_UTTERANCE_BYTES,
    load_tenant_config,
)
from backend.telephony.audio_utils import pcm_to_wav, wav_to_pcm_frames
from backend.voice.stt import SarvamSTT
from backend.voice.tts import SarvamTTS
from backend.voice.vad import SpeechSegmenter
from backend.config import PIPELINE_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


# ── Pre-canned apology messages
# Spoken when the pipeline fails to produce a response. Ensures the caller
# hears something rather than dead air.
_APOLOGY: dict[str, str] = {
    "hi-IN":       "मुझे खेद है, मैं अभी ठीक से सुन नहीं पाया। कृपया फिर से बोलें।",
    "en-IN":       "I'm sorry, I didn't catch that. Could you please repeat?",
    "hi-IN+en-IN": "Sorry, samajh nahi aaya. Please dobara bolein.",
}
_DEFAULT_APOLOGY_LANG: str = "en-IN"

# ── Deduplication window: ignore same utterance for n seconds
_UTTERANCE_DEDUP_SECONDS: float = 25


class LiveKitAudioBridge:
    """Bridges a single LiveKit Room to the KoyalAI voice pipeline.

    Non-breaking wrapper around STT/TTS and LangGraph.
    Handles full-duplex audio with VAD-based utterance detection.

    Args:
        room: Connected ``rtc.Room`` instance representing this call.
        tenant_id: Tenant identifier for RAG retrieval and cost tracking.
        call_type: ``"inbound"`` or ``"outbound"``.

    Raises:
        VoiceError: If audio publishing cannot be set up during start().

    Example:
        >>> bridge = LiveKitAudioBridge(room, "tenant_hdfc_bank")
        >>> await bridge.start()
    """

    def __init__(
        self,
        room: rtc.Room,
        tenant_id: str,
        call_type: str = "inbound",
    ) -> None:
        self.room = room
        self.tenant_id = tenant_id
        self.call_type = call_type
        self.session_id: str = str(uuid.uuid4())

        # voice pipeline components 
        self.stt = SarvamSTT()
        self.tts = SarvamTTS()
        self._segmenter = SpeechSegmenter()

        # LiveKit audio publishing
        self._audio_source: Optional[rtc.AudioSource] = None
        self._local_track: Optional[rtc.LocalAudioTrack] = None

        # State
        self._detected_language: str = "hi-IN"
        self._is_speaking: bool = False
        self._processing: bool = False
        self._start_time: float = time.monotonic()
        self._stop_event: asyncio.Event = asyncio.Event()
        self._fallback_buffer: bytearray = bytearray()

        # Transcript-based deduplication state
        self._last_transcript: str = ""
        self._last_transcript_time: float = 0.0

        logger.info(
            "LiveKitAudioBridge created: session=%s tenant=%s type=%s room=%s",
            self.session_id, tenant_id, call_type, room.name,
        )

    # ── Public API 

    async def start(self) -> None:
        """Publish the agent's output track and subscribe to caller input.

        Creates an ``rtc.AudioSource`` and publishes a local audio track,
        then registers callbacks for remote audio track subscriptions.

        Raises:
            VoiceError: If the local track cannot be published.
        """
        try:
            self._audio_source = rtc.AudioSource(
                sample_rate=TTS_SAMPLE_RATE,
                num_channels=1,
            )
            self._local_track = rtc.LocalAudioTrack.create_audio_track(
                "koyal-response",
                self._audio_source,
            )
            await self.room.local_participant.publish_track(
                self._local_track,
                rtc.TrackPublishOptions(
                    source=rtc.TrackSource.SOURCE_MICROPHONE,
                ),
            )
            logger.info(
                "[%s] Published local audio track to room '%s'.",
                self.session_id, self.room.name,
            )
        except Exception as exc:
            raise VoiceError(
                f"Failed to publish local audio track: {exc}",
                session_id=self.session_id,
                room=self.room.name,
            ) from exc

        # Subscribe to already-published remote tracks
        for participant in self.room.remote_participants.values():
            for pub in participant.track_publications.values():
                if pub.track and isinstance(pub.track, rtc.RemoteAudioTrack):
                    asyncio.get_event_loop().create_task(
                        self._receive_audio_track(pub.track)
                    )

        # Register for future track subscriptions (livekit 1.x method call, not decorator)
        self.room.on("track_subscribed", self._on_track_subscribed)
        self.room.on("disconnected", self._on_room_disconnected)

        record_call_start(self.tenant_id, self._detected_language, self.call_type)

        # Play tenant greeting (non-blocking, non-fatal)
        asyncio.get_event_loop().create_task(self._play_greeting())
        logger.info("[%s] AudioBridge started.", self.session_id)

    async def stop(self, outcome: str = "completed") -> None:
        """Clean up resources and record call end metrics.

        Args:
            outcome: ``"completed"``, ``"escalated"``, or ``"failed"``.
        """
        if self._stop_event.is_set():
            return
        
        self._stop_event.set()
        duration = time.monotonic() - self._start_time
        self._segmenter.reset()
        record_call_end(
            self.tenant_id,
            self._detected_language,
            duration_seconds=duration,
            call_type=self.call_type,
            outcome=outcome,
        )
        logger.info(
            "[%s] AudioBridge stopped: outcome=%s duration=%.1fs",
            self.session_id, outcome, duration,
        )

    # ── Track subscription callbacks 

    def _on_track_subscribed(
        self,
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        """Called when a remote participant publishes an audio track.

        Uses loop.create_task() — not asyncio.ensure_future() which is
        deprecated since Python 3.10 and removed in 3.14.
        """
        if isinstance(track, rtc.RemoteAudioTrack):
            logger.info(
                "[%s] Remote audio track subscribed from participant '%s'.",
                self.session_id, participant.identity,
            )
            asyncio.get_event_loop().create_task(
                self._receive_audio_track(track)
            )

    def _on_room_disconnected(self) -> None:
        """Handle unexpected room disconnection."""
        logger.warning(
            "[%s] Room '%s' disconnected unexpectedly.",
            self.session_id, self.room.name,
        )
        asyncio.get_event_loop().create_task(self.stop(outcome="failed"))

    # ── Audio receive loop 

    async def _receive_audio_track(self, track: rtc.RemoteAudioTrack) -> None:
        """Stream audio frames from a remote track through the VAD pipeline.

        Requests LiveKit to resample to VAD_SAMPLE_RATE Hz mono so
        SpeechSegmenter receives correctly-sized frames without extra
        resampling code on our side.

        Args:
            track: Remote audio track from a caller or SIP participant.
        """
        logger.info(
            "[%s] Receiving audio from track: %s", self.session_id, track.sid
        )
        stream = rtc.AudioStream.from_track(
            track=track,
            sample_rate=VAD_SAMPLE_RATE,
            num_channels=1,
        )

        async for event in stream:
            if self._stop_event.is_set():
                break

            audio_chunk = bytes(event.frame.data)
            if not audio_chunk:
                continue

            # Interruption suppression: Drop ALL input while processing or speaking
            if self._is_speaking or self._processing:
                # Clear fallback buffer to prevent stale accumulation
                if self._processing and self._fallback_buffer:
                    self._fallback_buffer.clear()
                logger.debug(
                    "[%s] Agent busy (speaking=%s processing=%s) — dropping %d bytes.",
                    self.session_id, self._is_speaking, self._processing, len(audio_chunk)
                )
                continue

            # ── VAD-based utterance detection 
            try:
                vad_result = self._segmenter.process_chunk(audio_chunk)
                if vad_result.utterance_complete and vad_result.speech_bytes:
                    if self._processing:
                        logger.info("[%s] Already processing — dropping utterance.", self.session_id)
                        self._fallback_buffer.clear()
                        continue
                    self._processing = True
                    asyncio.get_event_loop().create_task(
                        self._process_utterance(vad_result.speech_bytes)
                    )
                    self._fallback_buffer.clear()
                    continue
            except Exception as exc:
                logger.warning(
                    "[%s] VAD error, falling back to time-based buffer: %s",
                    self.session_id, exc,
                )

            # ── Fallback: time-based buffer flush 
            self._fallback_buffer.extend(audio_chunk)

            # 3-second threshold flush (normal fallback)
            if len(self._fallback_buffer) >= FALLBACK_BUFFER_THRESHOLD:
                audio_copy = bytes(self._fallback_buffer)
                self._fallback_buffer.clear()
                if not self._processing:
                    self._processing = True
                    asyncio.get_event_loop().create_task(
                        self._process_utterance(audio_copy)
                    )
                continue

            # 30-second safety cap 
            # Handles continuous background noise where VAD never detects silence.
            if len(self._fallback_buffer) >= MAX_UTTERANCE_BYTES:
                logger.warning(
                    "[%s] 30-second safety cap reached — forcing utterance flush.",
                    self.session_id,
                )
                audio_copy = bytes(self._fallback_buffer)
                self._fallback_buffer.clear()
                if not self._processing:
                    self._processing = True
                    asyncio.get_event_loop().create_task(
                        self._process_utterance(audio_copy)
                    )

    # ── Pipeline processing 

    async def _process_utterance(self, speech_pcm: bytes) -> None:
        """Run STT → LangGraph → TTS with hard timeout and processing lock."""
        try:
            await asyncio.wait_for(
                self._run_pipeline(speech_pcm),
                timeout=PIPELINE_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            logger.error(
                "[%s] Pipeline timed out after %.0fs! Speaking apology.",
                self.session_id, PIPELINE_TIMEOUT_SECONDS
            )
            await self._speak_apology(self._detected_language)
        except Exception as exc:
            logger.exception("[%s] Pipeline error: %s", self.session_id, exc)
            await self._speak_apology(self._detected_language)
        finally:
            self._processing = False
            self._fallback_buffer.clear()


    async def _run_pipeline(self, speech_pcm: bytes) -> None:
        """Run STT → LangGraph → TTS for one complete caller utterance.

        All external calls run in executor threads so the asyncio event
        loop is never blocked. If any step fails, _speak_apology() is
        called to ensure the caller hears a message rather than dead air.

        Args:
            speech_pcm: Raw PCM int16 bytes of the complete utterance.
        """
        pipeline_start = time.perf_counter_ns()
        loop = asyncio.get_event_loop()

        # ── STT 
        stt_start = time.perf_counter_ns()
        try:
            wav_bytes = pcm_to_wav(speech_pcm, sample_rate=VAD_SAMPLE_RATE)
            stt_result = await loop.run_in_executor(
                None,
                lambda: self.stt.transcribe(wav_bytes, language_hint=self._detected_language),
            )
        except Exception as exc:
            logger.error("[%s] Unexpected STT error: %s", self.session_id, exc)
            raise

        transcript: str = stt_result.get("text", "").strip()
        lang: str = stt_result.get("language_code", self._detected_language)

        if not transcript:
            logger.debug("[%s] Empty transcript — skipping.", self.session_id)
            return
        
        # ── Transcript deduplication
        now = time.monotonic()
        normalized = transcript.lower().replace("?", "").replace("।", "").replace("!", "").strip()
        if normalized == self._last_transcript and (now - self._last_transcript_time) < _UTTERANCE_DEDUP_SECONDS:
            logger.info("[%s] Duplicate transcript '%s' — skipping.", self.session_id, transcript[:40])
            return
        
        self._last_transcript = normalized
        self._last_transcript_time = now

        stt_latency_ms = (time.perf_counter_ns() - stt_start) / 1_000_000
        record_stt_latency(self.tenant_id, stt_result.get("language_code", "unknown"), stt_latency_ms)

        if stt_result.get("low_confidence"):
            logger.warning(
                "[%s] Low-confidence transcript (%.2f): %r",
                self.session_id, stt_result.get("confidence", 0.0), transcript[:60],
            )

        self._detected_language = lang
        record_language_detection(self.tenant_id, lang, is_code_mixed="+" in lang)
        logger.info(
            "[%s] STT: lang=%s conf=%.2f transcript=%r",
            self.session_id, lang, stt_result.get("confidence", 0.0), transcript[:80],
        )

        # ── LangGraph 
        llm_start = time.perf_counter_ns()
        try:
            initial_state = make_initial_state(
                query=transcript,
                tenant_id=self.tenant_id,
                session_id=self.session_id,
                call_type=self.call_type,
            )
            pipeline_state = await loop.run_in_executor(
                None,
                lambda: koyal_graph.invoke(initial_state),
            )
        except Exception as exc:
            logger.error("[%s] LangGraph pipeline error: %s", self.session_id, exc)
            raise

        llm_latency_ms = (time.perf_counter_ns() - llm_start) / 1_000_000
        record_llm_latency(self.tenant_id, llm_latency_ms)

        final_response: str = pipeline_state.get("final_response") or ""
        response_lang: str = pipeline_state.get("detected_language") or lang
        is_escalation: bool = bool(pipeline_state.get("escalate"))

        if not final_response:
            logger.warning("[%s] Empty final_response from graph.", self.session_id)
            raise ValueError("Empty final_response")

        record_ttfr(self.tenant_id, lang, llm_latency_ms)

        if is_escalation:
            record_escalation(
                self.tenant_id, lang,
                pipeline_state.get("escalation_reason") or "emergency",
            )
            logger.warning(
                "[%s] ESCALATING: reason=%s",
                self.session_id, pipeline_state.get("escalation_reason"),
            )
        else:
            record_safety_cleared(self.tenant_id)

        # ── TTS 
        tts_start = time.perf_counter_ns()
        try:
            tts_wav = await loop.run_in_executor(
                None,
                lambda: self.tts.synthesize(final_response, language_code=response_lang),
            )
        except Exception as exc:
            logger.error("[%s] TTS synthesis failed: %s", self.session_id, exc)
            raise

        tts_latency_ms = (time.perf_counter_ns() - tts_start) / 1_000_000
        record_tts_latency(self.tenant_id, response_lang, tts_latency_ms)

        pipeline_latency_ms = (time.perf_counter_ns() - pipeline_start) / 1_000_000
        record_pipeline_latency(self.tenant_id, lang, pipeline_latency_ms)

        logger.info(
            "[%s] Pipeline done: lang=%s escalate=%s "
            "stt=%.0fms llm=%.0fms tts=%.0fms total=%.0fms",
            self.session_id, lang, is_escalation,
            stt_latency_ms, llm_latency_ms, tts_latency_ms, pipeline_latency_ms,
        )

        if tts_wav:
            await self._publish_audio(tts_wav)

        # End call after escalation — records outcome in Prometheus
        if is_escalation:
            await self.stop(outcome="escalated")

    async def _publish_audio(self, wav_bytes: bytes) -> None:
        """Push WAV audio to the LiveKit room in 20ms PCM frames.

        Args:
            wav_bytes: WAV bytes from Sarvam TTS (PCM 16-bit, 16kHz, mono).
        """
        if not self._audio_source or not wav_bytes:
            logger.debug("[%s] _publish_audio: no audio source or empty WAV.", self.session_id)
            return

        self._is_speaking = True
        logger.debug(
            "[%s] Publishing %d WAV bytes to room.", self.session_id, len(wav_bytes)
        )

        try:
            frame_count = 0
            for frame_pcm in wav_to_pcm_frames(wav_bytes, frame_duration_ms=20):
                if self._stop_event.is_set():
                    logger.debug("[%s] Stop event — aborting audio publish.", self.session_id)
                    break
                samples_per_channel = len(frame_pcm) // 2  # int16 = 2 bytes/sample
                audio_frame = rtc.AudioFrame(
                    data=frame_pcm,
                    sample_rate=TTS_SAMPLE_RATE,
                    num_channels=1,
                    samples_per_channel=samples_per_channel,
                )
                await self._audio_source.capture_frame(audio_frame)
                frame_count += 1
            logger.debug(
                "[%s] Published %d audio frames to room.", self.session_id, frame_count
            )
        except Exception as exc:
            logger.error("[%s] Audio publish error: %s", self.session_id, exc)
        finally:
            self._is_speaking = False

    async def _play_greeting(self) -> None:
        """Synthesise and play the tenant's greeting message at call start.

        Loaded from tenant config — fully customisable per tenant without
        code changes. Falls back to a default greeting if not configured.
        Errors are non-fatal (greeting failure must not abort the call).
        """
        try:
            cfg = load_tenant_config(self.tenant_id)
            lang = cfg.get("primary_language", "hi-IN")
            greeting = cfg.get("greeting_message", "")
            if not greeting:
                company = cfg.get("company_name", "KoyalAI")
                if "hi" in lang:
                    greeting = f"नमस्ते! {company} में आपका स्वागत है। मैं आपकी कैसे मदद कर सकता हूँ?"
                else:
                    greeting = f"Hello! Welcome to {company}. How can I help you today?"
            loop = asyncio.get_event_loop()
            tts_wav = await loop.run_in_executor(
                None,
                lambda: self.tts.synthesize(greeting, language_code=lang),
            )
            if tts_wav:
                await self._publish_audio(tts_wav)
                logger.info("[%s] Greeting played in %s.", self.session_id, lang)
        except Exception as exc:
            logger.warning("[%s] Greeting failed (non-fatal): %s", self.session_id, exc)

    async def _speak_apology(self, language: str) -> None:
        """Synthesise and publish a pre-canned apology in the caller's language.

        Called whenever the pipeline fails to produce a response. Ensures the
        caller hears a language-appropriate message rather than dead air.

        Ported from Implementation A's _speak_apology() pattern.

        Args:
            language: BCP-47 language code (e.g. ``"hi-IN"``).
        """
        apology_text = _APOLOGY.get(language, _APOLOGY[_DEFAULT_APOLOGY_LANG])
        loop = asyncio.get_event_loop()
        try:
            apology_audio = await loop.run_in_executor(
                None, lambda: self.tts.synthesize(apology_text, language_code=language)
            )
            if apology_audio:
                await self._publish_audio(apology_audio)
                logger.info(
                    "[%s] Apology spoken in %s.", self.session_id, language
                )
        except Exception as exc:
            logger.error("[%s] Apology TTS failed: %s", self.session_id, exc)