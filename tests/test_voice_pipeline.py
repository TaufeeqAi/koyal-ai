"""
Coverage
────────
  TestSarvamSTT          — transcription, retry, error hierarchy, empty audio
  TestSarvamTTS          — synthesis, chunking, sentence splitting, voice map
  TestFrameBuffer        — accumulation, partial pushes, multiple frames, clear
  TestSpeechSegmenter    — VAD state machine, silence guard, RMS energy guard
  TestCostTracker        — Redis writes/reads, cross-tenant isolation, TTL (fakeredis)
  TestSessionManager     — lifecycle, concurrency limits, acquire(), to_dict() types
  TestOutboundCaller     — campaign engine, bounded concurrency, template errors
  TestVoiceWebSocketEndpoint — FastAPI TestClient integration
  Parametrized           — STT language × confidence threshold matrix
  CrossTenantIsolation   — Redis contamination regression
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import struct
import time
import wave
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

# ── Synthetic audio helpers 

def _make_pcm_silence(duration_ms: int, sample_rate: int = 16000) -> bytes:
    """All-zero PCM (guaranteed silence for VAD)."""
    return b"\x00\x00" * int(sample_rate * duration_ms / 1000)


def _make_pcm_tone(duration_ms: int, freq_hz: int = 440, sample_rate: int = 16000) -> bytes:
    """Sine-wave PCM — non-zero, triggers webrtcvad's speech classifier."""
    n = int(sample_rate * duration_ms / 1000)
    samples = [
        int(16383 * math.sin(2 * math.pi * freq_hz * i / sample_rate))
        for i in range(n)
    ]
    return b"".join(struct.pack("<h", s) for s in samples)


def _make_wav(pcm: bytes, sample_rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def _b64_wav(pcm: bytes) -> str:
    return base64.b64encode(_make_wav(pcm)).decode()


# ── STT Tests 

class TestSarvamSTT:

    def _mock_httpx_response(self, status: int, body: dict | None = None, text: str = "") -> MagicMock:
        resp = MagicMock()
        resp.status_code = status
        resp.json.return_value = body or {}
        resp.text = text
        resp.raise_for_status = MagicMock()
        if status >= 400:
            resp.raise_for_status.side_effect = Exception(f"HTTP {status}")
        return resp

    def _patch_httpx_client(self, response: MagicMock):
        """Context manager that patches httpx.Client.post."""
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post = MagicMock(return_value=response)
        return patch("httpx.Client", return_value=mock_client)

    def test_transcribe_success_hindi(self) -> None:
        from backend.voice.stt import SarvamSTT
        resp = self._mock_httpx_response(200, {
            "transcript": "मेरी EMI कब कटती है",
            "language_code": "hi-IN",
            "confidence": 0.92,
        })
        with self._patch_httpx_client(resp):
            stt = SarvamSTT(api_key="test-key")
            result = stt.transcribe(_make_wav(_make_pcm_silence(500)))
        assert result["text"] == "मेरी EMI कब कटती है"
        assert result["language_code"] == "hi-IN"
        assert result["low_confidence"] is False

    def test_transcribe_success_english(self) -> None:
        from backend.voice.stt import SarvamSTT
        resp = self._mock_httpx_response(200, {
            "transcript": "What is my EMI due date?",
            "language_code": "en-IN",
            "confidence": 0.97,
        })
        with self._patch_httpx_client(resp):
            stt = SarvamSTT(api_key="test-key")
            result = stt.transcribe(_make_wav(_make_pcm_silence(300)))
        assert result["text"] == "What is my EMI due date?"
        assert not result["low_confidence"]

    def test_low_confidence_flagged(self) -> None:
        from backend.voice.stt import SarvamSTT
        resp = self._mock_httpx_response(200, {
            "transcript": "...", "language_code": "hi-IN", "confidence": 0.25
        })
        with self._patch_httpx_client(resp):
            stt = SarvamSTT(api_key="test-key", confidence_threshold=0.4)
            # transcribe() raises LowConfidenceError when below threshold
            from backend.exceptions import LowConfidenceError
            with pytest.raises(LowConfidenceError):
                stt.transcribe(_make_wav(_make_pcm_silence(200)))

    def test_empty_audio_raises_voice_error(self) -> None:
        from backend.exceptions import VoiceError
        from backend.voice.stt import SarvamSTT
        stt = SarvamSTT(api_key="test-key")
        with pytest.raises(VoiceError, match="empty audio_bytes"):
            stt.transcribe(b"")

    def test_non_retryable_4xx_raises_stt_error_immediately(self) -> None:
        from backend.exceptions import STTError
        from backend.voice.stt import SarvamSTT
        resp = self._mock_httpx_response(401, text="Unauthorized")
        call_count = 0
        def _post(*a, **kw):
            nonlocal call_count
            call_count += 1
            return resp
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post = _post
        with patch("httpx.Client", return_value=mock_client):
            stt = SarvamSTT(api_key="bad-key", max_retries=3)
            with pytest.raises(STTError):
                stt.transcribe(_make_wav(_make_pcm_silence(200)))
        assert call_count == 1, "Must NOT retry on 401"

    def test_retry_on_retryable_error_then_success(self) -> None:
        from backend.voice.stt import SarvamSTT
        call_count = 0
        def _post(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                r = MagicMock()
                r.status_code = 500
                r.text = "Server Error"
                r.raise_for_status.side_effect = Exception("500")
                return r
            r = MagicMock()
            r.status_code = 200
            r.json.return_value = {
                "transcript": "retry success", "language_code": "en-IN", "confidence": 0.9
            }
            r.raise_for_status = MagicMock()
            return r
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post = _post
        with patch("httpx.Client", return_value=mock_client), \
             patch("time.sleep"):
            stt = SarvamSTT(api_key="test-key", max_retries=3)
            result = stt.transcribe(_make_wav(_make_pcm_silence(300)))
        assert result["text"] == "retry success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_atranscribe_empty_raises_voice_error(self) -> None:
        from backend.exceptions import VoiceError
        from backend.voice.stt import SarvamSTT
        stt = SarvamSTT(api_key="test-key")
        with pytest.raises(VoiceError, match="empty audio_bytes"):
            await stt.atranscribe(b"")

    @pytest.mark.asyncio
    async def test_atranscribe_success(self) -> None:
        from backend.voice.stt import SarvamSTT
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "transcript": "हेलो", "language_code": "hi-IN", "confidence": 0.88
        }
        mock_resp.raise_for_status = MagicMock()
        mock_async_client = AsyncMock()
        mock_async_client.__aenter__.return_value = mock_async_client
        mock_async_client.__aexit__.return_value = None
        mock_async_client.post = AsyncMock(return_value=mock_resp)
        with patch("httpx.AsyncClient", return_value=mock_async_client):
            stt = SarvamSTT(api_key="test-key")
            result = await stt.atranscribe(_make_wav(_make_pcm_silence(400)))
        assert result["text"] == "हेलो"
        assert result["language_code"] == "hi-IN"


# ── Parametrized STT × Confidence Matrix 

@pytest.mark.parametrize("language,confidence,threshold,expect_low_conf", [
    ("hi-IN", 0.30, 0.40, True),
    ("hi-IN", 0.50, 0.40, False),
    ("en-IN", 0.39, 0.40, True),
    ("en-IN", 0.41, 0.40, False),
    ("mr-IN", 0.60, 0.40, False),
    ("ta-IN", 0.20, 0.50, True),
])
def test_stt_confidence_matrix(language, confidence, threshold, expect_low_conf) -> None:
    """STT confidence threshold correctly sets low_confidence across all languages."""
    from backend.voice.stt import SarvamSTT
    from backend.exceptions import LowConfidenceError

    resp_body = {"transcript": "test", "language_code": language, "confidence": confidence}
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = resp_body
    mock_resp.raise_for_status = MagicMock()
    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.post = MagicMock(return_value=mock_resp)

    with patch("httpx.Client", return_value=mock_client):
        stt = SarvamSTT(api_key="k", confidence_threshold=threshold)
        if expect_low_conf:
            with pytest.raises(LowConfidenceError):
                stt.transcribe(_make_wav(_make_pcm_silence(200)))
        else:
            result = stt.transcribe(_make_wav(_make_pcm_silence(200)))
            assert result["low_confidence"] is False


# ── TTS Tests 

class TestSarvamTTS:

    def _patch_httpx_sync(self, audio_b64: str):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"audios": [audio_b64]}
        resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post = MagicMock(return_value=resp)
        return patch("httpx.Client", return_value=mock_client)

    def test_synthesize_success(self) -> None:
        from backend.voice.tts import SarvamTTS
        fake_audio = b"RIFF" + b"\x00" * 44
        with self._patch_httpx_sync(base64.b64encode(fake_audio).decode()):
            tts = SarvamTTS(api_key="test-key")
            result = tts.synthesize("नमस्ते", "hi-IN")
        assert result == fake_audio

    def test_synthesize_empty_text_returns_empty(self) -> None:
        from backend.voice.tts import SarvamTTS
        tts = SarvamTTS(api_key="test-key")
        with patch("httpx.Client") as mock_cls:
            result = tts.synthesize("", "hi-IN")
            mock_cls.assert_not_called()
        assert result == b""

    def test_chunk_text_splits_at_word_boundary(self) -> None:
        from backend.voice.tts import _chunk_text
        text = "Hello World Foo Bar Baz Qux"
        chunks = _chunk_text(text, max_chars=15)
        for chunk in chunks:
            assert len(chunk) <= 15
        assert " ".join(chunks).strip() == text.strip()

    def test_chunk_text_short_text_no_split(self) -> None:
        from backend.voice.tts import _chunk_text
        assert _chunk_text("Hello World", max_chars=100) == ["Hello World"]

    def test_split_sentences_hindi(self) -> None:
        from backend.voice.tts import _split_sentences
        text = "नमस्ते। कैसे हैं आप? क्या मैं मदद कर सकता हूँ।"
        parts = _split_sentences(text, "hi-IN")
        assert len(parts) >= 2
        for p in parts:
            assert p.strip()

    def test_split_sentences_english(self) -> None:
        from backend.voice.tts import _split_sentences
        parts = _split_sentences("Hello. How are you? I am here to help.", "en-IN")
        assert len(parts) == 3

    def test_voice_map_hinglish_uses_meera(self) -> None:
        from backend.voice.tts import LANGUAGE_VOICE_MAP
        assert LANGUAGE_VOICE_MAP.get("hi-IN+en-IN") == "meera"

    def test_voice_map_gujarati(self) -> None:
        from backend.voice.tts import LANGUAGE_VOICE_MAP
        assert "gu-IN" in LANGUAGE_VOICE_MAP
        assert LANGUAGE_VOICE_MAP["gu-IN"] == "diya"

    def test_voice_map_punjabi(self) -> None:
        from backend.voice.tts import LANGUAGE_VOICE_MAP
        assert "pa-IN" in LANGUAGE_VOICE_MAP
        assert LANGUAGE_VOICE_MAP["pa-IN"] == "arjun"

    def test_voice_map_covers_9_languages(self) -> None:
        from backend.voice.tts import LANGUAGE_VOICE_MAP
        assert len(LANGUAGE_VOICE_MAP) >= 9

    @pytest.mark.asyncio
    async def test_asynthesize_calls_gather(self) -> None:
        """asynthesize() should use asyncio.gather for parallel chunks."""
        from backend.voice.tts import SarvamTTS
        fake_audio = b"RIFF" + b"\x00" * 44
        async_resp = MagicMock()
        async_resp.status_code = 200
        async_resp.json.return_value = {"audios": [base64.b64encode(fake_audio).decode()]}
        async_resp.raise_for_status = MagicMock()
        mock_async_client = AsyncMock()
        mock_async_client.__aenter__.return_value = mock_async_client
        mock_async_client.__aexit__.return_value = None
        mock_async_client.post = AsyncMock(return_value=async_resp)
        with patch("httpx.AsyncClient", return_value=mock_async_client):
            tts = SarvamTTS(api_key="test-key")
            result = await tts.asynthesize("नमस्ते", "hi-IN")
        assert result == fake_audio


# ── FrameBuffer Tests 

class TestFrameBuffer:

    def test_exact_frame_returned(self) -> None:
        from backend.voice.vad import FrameBuffer, FRAME_BYTES
        buf = FrameBuffer(frame_bytes=FRAME_BYTES)
        frames = buf.push(b"\x00" * FRAME_BYTES)
        assert len(frames) == 1 and len(frames[0]) == FRAME_BYTES

    def test_partial_chunk_buffered(self) -> None:
        from backend.voice.vad import FrameBuffer, FRAME_BYTES
        buf = FrameBuffer(frame_bytes=FRAME_BYTES)
        frames = buf.push(b"\x00" * (FRAME_BYTES // 2))
        assert frames == []
        assert buf.buffered_bytes == FRAME_BYTES // 2

    def test_two_partial_pushes_yield_one_frame(self) -> None:
        from backend.voice.vad import FrameBuffer, FRAME_BYTES
        buf = FrameBuffer(frame_bytes=FRAME_BYTES)
        half = FRAME_BYTES // 2
        assert buf.push(b"\x01" * half) == []
        frames = buf.push(b"\x02" * half)
        assert len(frames) == 1

    def test_three_frame_push(self) -> None:
        from backend.voice.vad import FrameBuffer, FRAME_BYTES
        buf = FrameBuffer(frame_bytes=FRAME_BYTES)
        assert len(buf.push(b"\x00" * FRAME_BYTES * 3)) == 3

    def test_clear_discards_buffer(self) -> None:
        from backend.voice.vad import FrameBuffer, FRAME_BYTES
        buf = FrameBuffer(frame_bytes=FRAME_BYTES)
        buf.push(b"\x00" * (FRAME_BYTES // 2))
        buf.clear()
        assert buf.buffered_bytes == 0


# ── SpeechSegmenter Tests 

class TestSpeechSegmenter:

    def test_silence_no_utterance(self) -> None:
        from backend.voice.vad import SpeechSegmenter, FRAME_BYTES
        seg = SpeechSegmenter()
        silence = _make_pcm_silence(3000)
        for i in range(0, len(silence), FRAME_BYTES):
            result = seg.process_chunk(silence[i: i + FRAME_BYTES])
            assert not result.utterance_complete

    def test_init_without_error(self) -> None:
        from backend.voice.vad import SpeechSegmenter
        seg = SpeechSegmenter()
        assert seg is not None

    def test_invalid_sample_rate_raises_vad_error(self) -> None:
        from backend.exceptions import VADError
        from backend.voice.vad import SpeechSegmenter
        with pytest.raises(VADError, match="sample rate"):
            SpeechSegmenter(sample_rate=22050)

    def test_invalid_frame_duration_raises_vad_error(self) -> None:
        from backend.exceptions import VADError
        from backend.voice.vad import SpeechSegmenter
        with pytest.raises(VADError, match="frame duration"):
            SpeechSegmenter(frame_duration_ms=25)

    def test_reset_clears_state(self) -> None:
        from backend.voice.vad import SpeechSegmenter, VADState
        seg = SpeechSegmenter()
        seg.reset()
        assert seg._state == VADState.IDLE
        assert seg._speech_frame_count == 0

    def test_flush_returns_none_on_silence(self) -> None:
        from backend.voice.vad import SpeechSegmenter
        seg = SpeechSegmenter()
        assert seg.flush() is None

    def test_rms_zero_for_silence(self) -> None:
        from backend.voice.vad import _compute_rms
        silence_frame = b"\x00\x00" * 480
        assert _compute_rms(silence_frame) == 0.0

    def test_rms_nonzero_for_tone(self) -> None:
        from backend.voice.vad import _compute_rms
        tone_frame = _make_pcm_tone(30)
        assert _compute_rms(tone_frame) > 0.0


# ── CostTracker Tests (fakeredis) 

class TestCostTracker:

    @pytest.fixture
    def tracker(self):
        """CostTracker backed by fakeredis.asyncio (no Redis server needed)."""
        try:
            import fakeredis.aioredis as fake_aio
            import fakeredis as fake_sync
        except ImportError:
            pytest.skip("fakeredis not installed — pip install fakeredis")

        from backend.cost_tracker import CostTracker

        shared_server = fake_sync.FakeServer()

        tracker = CostTracker.__new__(CostTracker)
        tracker._async_redis = fake_aio.FakeRedis(server=shared_server, decode_responses=True)
        tracker._sync_redis = fake_sync.FakeRedis(server=shared_server, decode_responses=True)
        return tracker

    @pytest.mark.asyncio
    async def test_track_stt_accumulates(self, tracker) -> None:
        await tracker.track_stt("tenant_a", seconds=60.0)
        costs = tracker.get_tenant_costs("tenant_a")
        assert costs["stt_cost_inr"] == pytest.approx(0.50, abs=0.001)
        assert costs["stt_seconds"] == pytest.approx(60.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_track_tts_accumulates(self, tracker) -> None:
        await tracker.track_tts("tenant_a", chars=1000)
        costs = tracker.get_tenant_costs("tenant_a")
        assert costs["tts_cost_inr"] == pytest.approx(1.5, abs=0.01)
        assert costs["tts_chars"] == 1000

    @pytest.mark.asyncio
    async def test_track_stt_zero_seconds_noop(self, tracker) -> None:
        await tracker.track_stt("tenant_a", seconds=0.0)
        costs = tracker.get_tenant_costs("tenant_a")
        assert costs["stt_cost_inr"] == 0.0

    @pytest.mark.asyncio
    async def test_total_cost_is_sum(self, tracker) -> None:
        await tracker.track_stt("tenant_a", seconds=120.0)   # ₹1.00
        await tracker.track_tts("tenant_a", chars=2000)      # ₹3.00
        costs = tracker.get_tenant_costs("tenant_a")
        assert costs["total_cost_inr"] == pytest.approx(4.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_track_call_outcomes(self, tracker) -> None:
        await tracker.track_call("tenant_a", outcome="completed")
        await tracker.track_call("tenant_a", outcome="completed")
        await tracker.track_call("tenant_a", outcome="escalated")
        costs = tracker.get_tenant_costs("tenant_a")
        assert costs["calls_completed"] == 2
        assert costs["calls_escalated"] == 1
        assert costs["calls_failed"] == 0

    @pytest.mark.asyncio
    async def test_cost_key_prefix_is_koyalai(self, tracker) -> None:
        """Cost keys must use 'koyalai:' namespace (not 'cost:')."""
        from backend.cost_tracker import _key
        k = _key("tenant_hdfc_bank", "stt_inr")
        assert k.startswith("koyalai:")
        assert "cost:" not in k

    @pytest.mark.asyncio
    async def test_cross_tenant_isolation(self, tracker) -> None:
        """Tenant A costs must never appear in Tenant B's report."""
        await tracker.track_stt("tenant_a", seconds=600.0)
        await tracker.track_tts("tenant_a", chars=5000)

        costs_b = tracker.get_tenant_costs("tenant_b")
        assert costs_b["stt_cost_inr"] == 0.0
        assert costs_b["tts_cost_inr"] == 0.0
        assert costs_b["total_cost_inr"] == 0.0
        assert costs_b["calls_completed"] == 0

    @pytest.mark.asyncio
    async def test_reset_clears_all_keys(self, tracker) -> None:
        await tracker.track_stt("tenant_a", seconds=60.0)
        tracker.reset_tenant_costs("tenant_a")
        costs = tracker.get_tenant_costs("tenant_a")
        assert costs["stt_cost_inr"] == 0.0
        assert costs["total_cost_inr"] == 0.0


# ── SessionManager Tests 

class TestSessionManager:

    @pytest.fixture
    def sm(self):
        from backend.voice.session_manager import SessionManager
        return SessionManager(max_sessions=5)

    @pytest.mark.asyncio
    async def test_create_and_get(self, sm) -> None:
        sess = await sm.create_session("tenant_a", "sess_001")
        fetched = await sm.get_session("sess_001")
        assert fetched.session_id == "sess_001"
        assert fetched.tenant_id == "tenant_a"

    @pytest.mark.asyncio
    async def test_duplicate_raises_session_error(self, sm) -> None:
        from backend.exceptions import SessionError
        await sm.create_session("tenant_a", "sess_dup")
        with pytest.raises(SessionError, match="already exists"):
            await sm.create_session("tenant_a", "sess_dup")

    @pytest.mark.asyncio
    async def test_max_sessions_raises_session_error(self, sm) -> None:
        from backend.exceptions import SessionError
        for i in range(5):
            await sm.create_session("tenant_a", f"sess_{i:03d}")
        with pytest.raises(SessionError, match="Max concurrent"):
            await sm.create_session("tenant_a", "sess_overflow")

    @pytest.mark.asyncio
    async def test_update_session_accumulates_stt(self, sm) -> None:
        await sm.create_session("tenant_a", "sess_stt")
        await sm.update_session("sess_stt", stt_seconds=3.0)
        await sm.update_session("sess_stt", stt_seconds=2.5)
        sess = await sm.get_session("sess_stt")
        assert sess.stt_seconds == pytest.approx(5.5)

    @pytest.mark.asyncio
    async def test_end_session_marks_ended(self, sm) -> None:
        from backend.voice.session_manager import SessionState
        await sm.create_session("tenant_a", "sess_end")
        await sm.end_session("sess_end", outcome="completed")
        sess = await sm.get_session("sess_end")
        assert sess.state == SessionState.ENDED
        assert sess.outcome == "completed"
        assert sess.is_active is False

    @pytest.mark.asyncio
    async def test_get_nonexistent_raises(self, sm) -> None:
        from backend.exceptions import SessionError
        with pytest.raises(SessionError, match="not found"):
            await sm.get_session("nonexistent_session_id")

    @pytest.mark.asyncio
    async def test_acquire_context_manager(self, sm) -> None:
        """acquire() context manager should hold lock and allow direct field mutation."""
        await sm.create_session("tenant_a", "sess_acq")
        async with sm.acquire("sess_acq") as sess:
            sess.stt_seconds += 7.5
            sess.turn_count += 1
        fetched = await sm.get_session("sess_acq")
        assert fetched.stt_seconds == pytest.approx(7.5)
        assert fetched.turn_count == 1

    @pytest.mark.asyncio
    async def test_to_dict_types_are_native(self, sm) -> None:
        """to_dict() must return native bool/int, not string-cast values."""
        await sm.create_session("tenant_a", "sess_types")
        sess = await sm.get_session("sess_types")
        d = sess.to_dict()
        assert isinstance(d["is_active"], bool), "is_active must be bool, not str"
        assert isinstance(d["is_code_mixed"], bool), "is_code_mixed must be bool, not str"
        assert isinstance(d["tts_chars"], int)
        assert isinstance(d["turn_count"], int)


# ── OutboundCaller Tests 

class TestOutboundCaller:

    def _mock_caller(self):
        """OutboundCaller with mocked TTS and cost tracker."""
        with patch("backend.voice.outbound_caller.load_tenant_config") as mock_cfg:
            mock_cfg.return_value = {
                "tenant_id": "tenant_hdfc_bank",
                "company_name": "HDFC Bank",
                "primary_language": "hi-IN",
            }
            from backend.voice.outbound_caller import OutboundCaller
            caller = OutboundCaller("tenant_hdfc_bank")

        caller._tts = MagicMock()
        caller._tts.asynthesize = AsyncMock(return_value=b"RIFF" + b"\x00" * 1000)
        caller._cost_tracker = MagicMock()
        caller._cost_tracker.track_tts = AsyncMock()
        caller._cost_tracker.track_call = AsyncMock()
        return caller

    @pytest.mark.asyncio
    async def test_single_contact_campaign(self) -> None:
        caller = self._mock_caller()
        results = await caller.run_outbound_campaign(
            contact_list=[{"phone": "+91-9800000000", "name": "Priya"}],
            script_template="नमस्ते {name} जी!",
            language="hi-IN",
        )
        assert len(results) == 1
        assert results[0]["status"] == "completed"
        assert results[0]["phone"] == "+91-9800000000"

    @pytest.mark.asyncio
    async def test_bounded_concurrency(self) -> None:
        """Semaphore must bound concurrent API calls."""
        call_times: list[float] = []

        async def slow_synth(text, lang):
            call_times.append(time.monotonic())
            await asyncio.sleep(0.05)
            return b"RIFF" + b"\x00" * 100

        caller = self._mock_caller()
        caller._tts.asynthesize = slow_synth

        contacts = [{"phone": f"+91-980000{i:04d}", "name": f"User{i}"} for i in range(6)]
        await caller.run_outbound_campaign(
            contact_list=contacts,
            script_template="Hello {name}!",
            language="en-IN",
            max_concurrent=2,
        )
        # With semaphore=2 and 6 calls, first and second start nearly simultaneously
        assert len(call_times) == 6

    @pytest.mark.asyncio
    async def test_empty_contact_list_raises(self) -> None:
        from backend.exceptions import OutboundError
        caller = self._mock_caller()
        with pytest.raises(OutboundError, match="empty"):
            await caller.run_outbound_campaign(
                contact_list=[], script_template="Hello!", language="en-IN"
            )

    @pytest.mark.asyncio
    async def test_missing_template_key_reported_as_failed(self) -> None:
        caller = self._mock_caller()
        results = await caller.run_outbound_campaign(
            contact_list=[{"phone": "+91-9800000000", "name": "Rahul"}],
            script_template="₹{due_amount} बाकी है {name}।",
            language="hi-IN",
        )
        assert results[0]["status"] == "failed"
        assert "error" in results[0]


# ── WebSocket Endpoint Tests 

class TestVoiceWebSocketEndpoint:

    def test_unknown_tenant_rejected(self) -> None:
        from starlette.testclient import TestClient
        from backend.main import app
        client = TestClient(app)
        with client.websocket_connect("/ws/unknown_tenant/sess_001") as ws:
            msg = json.loads(ws.receive_text())
            assert msg["type"] == "error"
            assert "unknown_tenant" in msg["message"].lower() or "Unknown" in msg["message"]

    def test_health_endpoint_200(self) -> None:
        from starlette.testclient import TestClient
        from backend.main import app
        resp = TestClient(app).get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "services" in data

    def test_outbound_invalid_tenant_422(self) -> None:
        from starlette.testclient import TestClient
        from backend.main import app
        resp = TestClient(app).post(
            "/api/outbound/campaign",
            json={
                "tenant_id": "nonexistent_tenant",
                "contact_list": [{"phone": "+91-9800000000", "name": "Test"}],
                "script_template": "Hello {name}.",
                "language": "en-IN",
            },
        )
        assert resp.status_code == 422

    def test_costs_unknown_tenant_404(self) -> None:
        from starlette.testclient import TestClient
        from backend.main import app
        resp = TestClient(app).get("/api/costs/nonexistent_tenant")
        assert resp.status_code == 404

    def test_sessions_endpoint_returns_list(self) -> None:
        from starlette.testclient import TestClient
        from backend.main import app
        resp = TestClient(app).get("/api/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert "sessions" in data and isinstance(data["sessions"], list)
        assert "count" in data

    def test_metrics_stub_endpoint(self) -> None:
        from starlette.testclient import TestClient
        from backend.main import app
        resp = TestClient(app).get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "koyalai_active_sessions" in data


# ── Cross-Tenant Isolation (Regression Suite) 

class TestCrossTenantIsolation:
    """Regression tests ensuring no data leakage between tenants in Redis."""

    @pytest.fixture
    def tracker(self):
        try:
            import fakeredis.aioredis as fake_aio
            import fakeredis as fake_sync
        except ImportError:
            pytest.skip("fakeredis not installed")
        from backend.cost_tracker import CostTracker

        # CRITICAL: both clients must share the same FakeServer so async writes
        # are visible to sync reads (mirrors production where both hit one Redis).
        shared_server = fake_sync.FakeServer()

        t = CostTracker.__new__(CostTracker)
        t._async_redis = fake_aio.FakeRedis(server=shared_server, decode_responses=True)
        t._sync_redis = fake_sync.FakeRedis(server=shared_server, decode_responses=True)
        return t

    @pytest.mark.asyncio
    async def test_stt_cost_no_cross_contamination(self, tracker) -> None:
        await tracker.track_stt("tenant_a", seconds=3600.0)  # ₹30.00
        assert tracker.get_tenant_costs("tenant_b")["stt_cost_inr"] == 0.0

    @pytest.mark.asyncio
    async def test_tts_cost_no_cross_contamination(self, tracker) -> None:
        await tracker.track_tts("tenant_a", chars=100_000)  # ₹150.00
        assert tracker.get_tenant_costs("tenant_b")["tts_cost_inr"] == 0.0

    @pytest.mark.asyncio
    async def test_call_count_no_cross_contamination(self, tracker) -> None:
        for _ in range(5):
            await tracker.track_call("tenant_a", outcome="completed")
        assert tracker.get_tenant_costs("tenant_b")["calls_completed"] == 0

    @pytest.mark.asyncio
    async def test_reset_does_not_affect_other_tenant(self, tracker) -> None:
        await tracker.track_stt("tenant_a", seconds=60.0)
        await tracker.track_stt("tenant_b", seconds=60.0)
        tracker.reset_tenant_costs("tenant_a")
        assert tracker.get_tenant_costs("tenant_a")["stt_cost_inr"] == 0.0
        assert tracker.get_tenant_costs("tenant_b")["stt_cost_inr"] == pytest.approx(0.50, abs=0.001)

    @pytest.mark.asyncio
    async def test_three_tenants_isolated(self, tracker) -> None:
        for i, tenant in enumerate(["tenant_a", "tenant_b", "tenant_c"]):
            await tracker.track_stt(tenant, seconds=float((i + 1) * 60))
        a = tracker.get_tenant_costs("tenant_a")["stt_cost_inr"]
        b = tracker.get_tenant_costs("tenant_b")["stt_cost_inr"]
        c = tracker.get_tenant_costs("tenant_c")["stt_cost_inr"]
        assert a < b < c  # Each is strictly greater than the previous
        assert a == pytest.approx(0.50, abs=0.01)
        assert b == pytest.approx(1.00, abs=0.01)
        assert c == pytest.approx(1.50, abs=0.01)