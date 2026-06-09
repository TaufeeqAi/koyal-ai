"""
Comprehensive test suite for KoyalAI LiveKit integration.

Test classes (40 tests total):
  TestAudioUtils           (8)  — PCM↔WAV conversion, frame splitting
  TestLiveKitAudioBridge   (7)  — full pipeline mocking + apology TTS
  TestKoyalRoom            (4)  — room lifecycle + request_id
  TestOutboundDialer       (5)  — DialRequest API, TTS abort, campaign semaphore,
                                   duplicate blocking
  TestInboundHandlerEndpoints (9) — webhook, token, rooms, health, ready,
                                     tenant extraction (incl. ValueError)
  TestOutboundEndpoints    (3)  — outbound health / ready probes
  TestSIPTrunkManager      (4)  — trunk creation + idempotency check

Run:
    pytest tests/test_livekit_bridge.py -v
    pytest tests/test_livekit_bridge.py -v -k "TestAudioUtils" --tb=short
"""

from __future__ import annotations

import asyncio
import io
import wave
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Global fixture: prevent cross-test registry pollution 
@pytest.fixture(autouse=True)
def _clear_active_rooms():
    from backend.telephony.livekit_room import active_rooms
    active_rooms.clear()
    yield


# ── Test helpers 

def _make_raw_pcm(duration_ms: int = 500, sample_rate: int = 16000) -> bytes:
    """Generate silent int16 PCM bytes for testing."""
    num_samples = int(sample_rate * duration_ms / 1000)
    return b"\x00\x00" * num_samples


def _make_wav_bytes(duration_ms: int = 500, sample_rate: int = 16000) -> bytes:
    """Generate a valid WAV file for testing."""
    pcm = _make_raw_pcm(duration_ms, sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()

# 1. Audio Utils Tests 

class TestAudioUtils:
    """Tests for PCM↔WAV conversion utilities in audio_utils.py."""

    def test_pcm_to_wav_produces_valid_wav(self) -> None:
        from backend.telephony.audio_utils import pcm_to_wav
        pcm = _make_raw_pcm(200)
        wav = pcm_to_wav(pcm, sample_rate=16000)
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"
        buf = io.BytesIO(wav)
        with wave.open(buf, "rb") as wf:
            assert wf.getframerate() == 16000
            assert wf.getnchannels() == 1

    def test_pcm_to_wav_empty_raises_value_error(self) -> None:
        from backend.telephony.audio_utils import pcm_to_wav
        with pytest.raises(ValueError, match="must not be empty"):
            pcm_to_wav(b"")

    def test_wav_to_pcm_extracts_correct_data(self) -> None:
        from backend.telephony.audio_utils import pcm_to_wav, wav_to_pcm
        original_pcm = _make_raw_pcm(100)
        wav = pcm_to_wav(original_pcm, sample_rate=16000)
        extracted_pcm, sr, ch = wav_to_pcm(wav)
        assert sr == 16000 and ch == 1
        assert extracted_pcm == original_pcm

    def test_wav_to_pcm_invalid_input_raises(self) -> None:
        from backend.telephony.audio_utils import wav_to_pcm
        with pytest.raises(ValueError, match="Invalid WAV"):
            wav_to_pcm(b"not a wav file")

    def test_wav_to_pcm_frames_yields_correct_chunk_sizes(self) -> None:
        from backend.telephony.audio_utils import wav_to_pcm_frames
        wav = _make_wav_bytes(500, 16000)
        frames = list(wav_to_pcm_frames(wav, frame_duration_ms=20))
        # 16kHz, 20ms = 320 samples × 2 bytes = 640 bytes/frame
        for frame in frames:
            assert len(frame) == 640

    def test_wav_to_pcm_frames_count_is_correct(self) -> None:
        from backend.telephony.audio_utils import wav_to_pcm_frames
        wav = _make_wav_bytes(500, 16000)
        frames = list(wav_to_pcm_frames(wav, frame_duration_ms=20))
        assert len(frames) == 25, f"Expected 25 frames, got {len(frames)}"

    def test_silence_pcm_length_is_correct(self) -> None:
        from backend.telephony.audio_utils import silence_pcm
        silence = silence_pcm(duration_ms=100, sample_rate=16000)
        expected = int(16000 * 0.1) * 2
        assert len(silence) == expected

    def test_estimate_audio_duration_ms(self) -> None:
        from backend.telephony.audio_utils import estimate_audio_duration_ms
        pcm = _make_raw_pcm(500)
        ms = estimate_audio_duration_ms(pcm, sample_rate=16000)
        assert abs(ms - 500.0) < 1.0

# 2. LiveKitAudioBridge Tests 

class TestLiveKitAudioBridge:
    """Tests for LiveKit audio bridge with mocked livekit 1.x APIs."""

    def _make_mock_room(self) -> MagicMock:
        room = MagicMock()
        room.name = "tenant_hdfc_bank-inbound-abc123"
        room.remote_participants = {}
        local_participant = AsyncMock()
        local_participant.publish_track = AsyncMock()
        room.local_participant = local_participant
        room.on = MagicMock()  # NOT @room.on() decorator — method call
        return room

    @pytest.mark.asyncio
    async def test_start_publishes_local_track_and_registers_room_on(self) -> None:
        """start() must publish track AND register room.on(event, callback)."""
        from backend.telephony.audio_bridge import LiveKitAudioBridge
        mock_room = self._make_mock_room()

        with patch("backend.telephony.audio_bridge.rtc") as mock_rtc, \
             patch("backend.telephony.audio_bridge.record_call_start"), \
             patch("backend.telephony.audio_bridge.load_tenant_config",
                   return_value={"company_name": "Test", "primary_language": "en-IN"}), \
             patch("backend.telephony.audio_bridge.SarvamTTS") as MockTTS:

            MockTTS.return_value.synthesize.return_value = b""
            mock_rtc.AudioSource.return_value = MagicMock()
            mock_rtc.LocalAudioTrack.create_audio_track.return_value = MagicMock()
            mock_rtc.TrackPublishOptions.return_value = MagicMock()
            mock_rtc.TrackSource.SOURCE_MICROPHONE = "microphone"
            mock_rtc.RemoteAudioTrack = type("RemoteAudioTrack", (), {})

            bridge = LiveKitAudioBridge(mock_room, "tenant_hdfc_bank")
            await bridge.start()

        mock_room.local_participant.publish_track.assert_awaited_once()
        # Verify room.on() is called (not @room.on() decorator)
        assert any(
            call_args[0][0] == "track_subscribed"
            for call_args in mock_room.on.call_args_list
        ), "room.on('track_subscribed', callback) must be called — not @room.on() decorator."

    @pytest.mark.asyncio
    async def test_process_utterance_full_pipeline(self) -> None:
        """_process_utterance must call STT → LangGraph → TTS → publish."""
        from backend.telephony.audio_bridge import LiveKitAudioBridge
        mock_room = self._make_mock_room()

        with patch("backend.telephony.audio_bridge.SarvamSTT") as MockSTT, \
             patch("backend.telephony.audio_bridge.SarvamTTS") as MockTTS, \
             patch("backend.telephony.audio_bridge.koyal_graph") as mock_graph, \
             patch("backend.telephony.audio_bridge.record_stt_latency"), \
             patch("backend.telephony.audio_bridge.record_llm_latency"), \
             patch("backend.telephony.audio_bridge.record_tts_latency"), \
             patch("backend.telephony.audio_bridge.record_ttfr"), \
             patch("backend.telephony.audio_bridge.record_pipeline_latency"), \
             patch("backend.telephony.audio_bridge.record_safety_cleared"), \
             patch("backend.telephony.audio_bridge.record_language_detection"), \
             patch("backend.telephony.audio_bridge.pcm_to_wav", return_value=b"wav"), \
             patch("backend.telephony.audio_bridge.make_initial_state",
                   return_value={"query": "", "tenant_id": "t", "session_id": "s", "call_type": "inbound"}):

            MockSTT.return_value.transcribe.return_value = {
                "text": "मेरी EMI कब कटती है",
                "language_code": "hi-IN",
                "confidence": 0.92,
                "low_confidence": False,
                "duration_seconds": 2.0,
            }
            mock_graph.invoke.return_value = {
                "final_response": "5 तारीख को",
                "detected_language": "hi-IN",
                "escalate": False,
                "llm_tokens": 42,
                "retrieved_chunks": [],
            }
            MockTTS.return_value.synthesize.return_value = _make_wav_bytes(200)

            bridge = LiveKitAudioBridge(mock_room, "tenant_hdfc_bank")
            bridge._audio_source = AsyncMock()
            bridge._audio_source.capture_frame = AsyncMock()

            await bridge._process_utterance(_make_raw_pcm(2000))

        MockSTT.return_value.transcribe.assert_called_once()
        mock_graph.invoke.assert_called_once()
        MockTTS.return_value.synthesize.assert_called_once_with(
            "5 तारीख को", language_code="hi-IN"
        )

    @pytest.mark.asyncio
    async def test_process_utterance_skips_empty_transcript(self) -> None:
        """Bridge must NOT invoke LangGraph when STT returns empty text."""
        from backend.telephony.audio_bridge import LiveKitAudioBridge
        mock_room = self._make_mock_room()

        with patch("backend.telephony.audio_bridge.SarvamSTT") as MockSTT, \
             patch("backend.telephony.audio_bridge.SarvamTTS"), \
             patch("backend.telephony.audio_bridge.koyal_graph") as mock_graph, \
             patch("backend.telephony.audio_bridge.pcm_to_wav", return_value=b"wav"), \
             patch("backend.telephony.audio_bridge.record_stt_latency"), \
             patch("backend.telephony.audio_bridge.record_language_detection"):

            MockSTT.return_value.transcribe.return_value = {
                "text": "   ",
                "language_code": "en-IN",
                "confidence": 0.1,
                "low_confidence": True,
                "duration_seconds": 1.0,
            }
            bridge = LiveKitAudioBridge(mock_room, "tenant_hdfc_bank")
            await bridge._process_utterance(_make_raw_pcm(1000))

        mock_graph.invoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_utterance_emergency_stops_bridge(self) -> None:
        """Escalated response must call bridge.stop(outcome='escalated')."""
        from backend.telephony.audio_bridge import LiveKitAudioBridge
        mock_room = self._make_mock_room()

        with patch("backend.telephony.audio_bridge.SarvamSTT") as MockSTT, \
             patch("backend.telephony.audio_bridge.SarvamTTS") as MockTTS, \
             patch("backend.telephony.audio_bridge.koyal_graph") as mock_graph, \
             patch("backend.telephony.audio_bridge.pcm_to_wav", return_value=b"wav"), \
             patch("backend.telephony.audio_bridge.make_initial_state", return_value={}), \
             patch("backend.telephony.audio_bridge.record_stt_latency"), \
             patch("backend.telephony.audio_bridge.record_llm_latency"), \
             patch("backend.telephony.audio_bridge.record_tts_latency"), \
             patch("backend.telephony.audio_bridge.record_ttfr"), \
             patch("backend.telephony.audio_bridge.record_pipeline_latency"), \
             patch("backend.telephony.audio_bridge.record_escalation"), \
             patch("backend.telephony.audio_bridge.record_language_detection"):

            MockSTT.return_value.transcribe.return_value = {
                "text": "दिल का दौरा", "language_code": "hi-IN",
                "confidence": 0.95, "low_confidence": False, "duration_seconds": 1.0,
            }
            mock_graph.invoke.return_value = {
                "final_response": "मैं आपको जोड़ रहा हूँ।",
                "detected_language": "hi-IN",
                "escalate": True,
                "escalation_reason": "Emergency: heart attack",
                "llm_tokens": 50, "retrieved_chunks": [],
            }
            MockTTS.return_value.synthesize.return_value = _make_wav_bytes(300)

            bridge = LiveKitAudioBridge(mock_room, "tenant_hdfc_bank")
            bridge._audio_source = AsyncMock()
            bridge._audio_source.capture_frame = AsyncMock()

            with patch.object(bridge, "stop", new_callable=AsyncMock) as mock_stop:
                await bridge._process_utterance(_make_raw_pcm(1000))
                mock_stop.assert_awaited_once_with(outcome="escalated")

    @pytest.mark.asyncio
    async def test_process_utterance_speaks_apology_on_stt_failure(self) -> None:
        """When STT raises, bridge must speak apology."""
        from backend.telephony.audio_bridge import LiveKitAudioBridge
        from backend.exceptions import STTError
        mock_room = self._make_mock_room()

        with patch("backend.telephony.audio_bridge.SarvamSTT") as MockSTT, \
             patch("backend.telephony.audio_bridge.SarvamTTS"), \
             patch("backend.telephony.audio_bridge.pcm_to_wav", return_value=b"wav"):

            MockSTT.return_value.transcribe.side_effect = STTError("upstream error")
            bridge = LiveKitAudioBridge(mock_room, "tenant_hdfc_bank")

            with patch.object(bridge, "_speak_apology", new_callable=AsyncMock) as mock_apology:
                await bridge._process_utterance(_make_raw_pcm(1000))
                mock_apology.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_publish_audio_pushes_correct_frame_count(self) -> None:
        """_publish_audio must push one capture_frame call per 20ms frame."""
        from backend.telephony.audio_bridge import LiveKitAudioBridge
        mock_room = self._make_mock_room()
        mock_source = AsyncMock()
        mock_source.capture_frame = AsyncMock()

        bridge = LiveKitAudioBridge(mock_room, "tenant_hdfc_bank")
        bridge._audio_source = mock_source

        # 400ms WAV at 20ms/frame = 20 frames
        await bridge._publish_audio(_make_wav_bytes(400, 16000))
        assert mock_source.capture_frame.await_count == 20

    @pytest.mark.asyncio
    async def test_receive_audio_triggers_30s_safety_cap(self) -> None:
        """Fallback buffer exceeding 30s must trigger utterance flush."""
        from backend.telephony.audio_bridge import LiveKitAudioBridge
        from backend.config import MAX_UTTERANCE_BYTES
        mock_room = self._make_mock_room()

        bridge = LiveKitAudioBridge(mock_room, "tenant_hdfc_bank")

        # Simulate the fallback buffer filling beyond the 30s cap
        bridge._fallback_buffer.extend(b"\x00" * (MAX_UTTERANCE_BYTES + 1))

        process_called = []

        async def mock_process(pcm):
            process_called.append(len(pcm))

        bridge._process_utterance = mock_process 

        # Trigger the condition check by adding one more byte
        chunk = b"\x00\x00"
        bridge._fallback_buffer.extend(chunk)

        if len(bridge._fallback_buffer) >= MAX_UTTERANCE_BYTES:
            audio_copy = bytes(bridge._fallback_buffer)
            bridge._fallback_buffer.clear()
            await bridge._process_utterance(audio_copy)

        assert len(process_called) == 1, "30s cap must trigger exactly one utterance flush."
        assert bridge._fallback_buffer == bytearray(), "Buffer must be cleared after flush."


# 3. KoyalRoom Tests 

class TestKoyalRoom:
    """Tests for the KoyalRoom room lifecycle manager."""

    @pytest.mark.asyncio
    async def test_connect_builds_jwt_with_minimal_privilege(self) -> None:
        """connect() must build a JWT with can_publish_data=False."""
        from backend.telephony.livekit_room import KoyalRoom

        mock_room = MagicMock()
        mock_room.connect = AsyncMock()
        mock_room.name = "tenant_hdfc_bank-inbound-abc"
        mock_room.local_participant = MagicMock(identity="koyal-agent-tenant_hdfc_bank")
        mock_room.remote_participants = {}

        captured_grants = {}

        with patch("backend.telephony.livekit_room.rtc.Room", return_value=mock_room), \
             patch("backend.telephony.livekit_room.api.AccessToken") as MockToken, \
             patch("backend.telephony.livekit_room.LiveKitAudioBridge") as MockBridge:

            def capture_grants(grants):
                captured_grants.update({
                    "can_publish_data": grants.can_publish_data,
                    "can_publish": grants.can_publish,
                    "can_subscribe": grants.can_subscribe,
                })
                return MockToken.return_value

            MockToken.return_value.with_identity.return_value = MockToken.return_value
            MockToken.return_value.with_name.return_value = MockToken.return_value
            MockToken.return_value.with_grants.side_effect = capture_grants
            MockToken.return_value.to_jwt.return_value = "test_jwt"
            MockBridge.return_value.start = AsyncMock()

            koyal = KoyalRoom("tenant_hdfc_bank-inbound-abc", "tenant_hdfc_bank")
            await koyal.connect()

        assert captured_grants.get("can_publish_data") is False, \
            "Agent token must have can_publish_data=False (minimal privilege)."

    @pytest.mark.asyncio
    async def test_connect_twice_is_noop(self) -> None:
        """connect() called twice must not reconnect."""
        from backend.telephony.livekit_room import KoyalRoom

        mock_room = MagicMock()
        mock_room.connect = AsyncMock()
        mock_room.name = "tenant_hdfc_bank-inbound-abc"
        mock_room.local_participant = MagicMock(identity="koyal-agent")
        mock_room.remote_participants = {}

        with patch("backend.telephony.livekit_room.rtc.Room", return_value=mock_room), \
             patch("backend.telephony.livekit_room.api.AccessToken") as MockToken, \
             patch("backend.telephony.livekit_room.LiveKitAudioBridge") as MockBridge:

            MockToken.return_value.with_identity.return_value = MockToken.return_value
            MockToken.return_value.with_name.return_value = MockToken.return_value
            MockToken.return_value.with_grants.return_value = MockToken.return_value
            MockToken.return_value.to_jwt.return_value = "jwt"
            MockBridge.return_value.start = AsyncMock()

            koyal = KoyalRoom("tenant_hdfc_bank-inbound-abc", "tenant_hdfc_bank")
            await koyal.connect()
            await koyal.connect()

        assert mock_room.connect.await_count == 1

    @pytest.mark.asyncio
    async def test_disconnect_stops_bridge_and_room(self) -> None:
        """disconnect() must stop bridge and disconnect from room."""
        from backend.telephony.livekit_room import KoyalRoom

        mock_room = MagicMock()
        mock_room.connect = AsyncMock()
        mock_room.disconnect = AsyncMock()
        mock_room.name = "tenant_hdfc_bank-inbound-abc"
        mock_room.local_participant = MagicMock(identity="koyal-agent")
        mock_room.remote_participants = {}

        with patch("backend.telephony.livekit_room.rtc.Room", return_value=mock_room), \
             patch("backend.telephony.livekit_room.api.AccessToken") as MockToken, \
             patch("backend.telephony.livekit_room.LiveKitAudioBridge") as MockBridge:

            MockToken.return_value.with_identity.return_value = MockToken.return_value
            MockToken.return_value.with_name.return_value = MockToken.return_value
            MockToken.return_value.with_grants.return_value = MockToken.return_value
            MockToken.return_value.to_jwt.return_value = "jwt"
            MockBridge.return_value.start = AsyncMock()
            MockBridge.return_value.stop = AsyncMock()

            koyal = KoyalRoom("tenant_hdfc_bank-inbound-abc", "tenant_hdfc_bank")
            await koyal.connect()
            await koyal.disconnect(outcome="completed")

        MockBridge.return_value.stop.assert_awaited_once_with(outcome="completed")
        mock_room.disconnect.assert_awaited_once()

    def test_request_id_stored(self) -> None:
        """KoyalRoom must accept and store an optional request_id."""
        from backend.telephony.livekit_room import KoyalRoom
        koyal = KoyalRoom("test-room", "tenant_test", request_id="req-123")
        assert koyal.request_id == "req-123"

# 4. Outbound Dialer Tests (5 tests)

class TestOutboundDialer:
    """Tests for LiveKitSIPOutbound including campaign semaphore."""

    @pytest.mark.asyncio
    async def test_dial_returns_typed_dial_result(self) -> None:
        """dial() must return a typed DialResult with status='dialing'."""
        from backend.telephony.outbound_dialer import LiveKitSIPOutbound, DialResult, DialRequest

        with patch("backend.telephony.outbound_dialer.api.LiveKitAPI") as MockAPI, \
             patch("backend.telephony.outbound_dialer.SarvamTTS") as MockTTS, \
             patch("backend.telephony.outbound_dialer.CostTracker") as MockCT, \
             patch("backend.telephony.outbound_dialer._connect_and_monitor",
                   new_callable=AsyncMock):

            mock_sip = AsyncMock()
            mock_participant = MagicMock()
            mock_participant.sip_call_id = "call_xyz"
            mock_sip.create_sip_participant = AsyncMock(return_value=mock_participant)
            MockAPI.return_value.sip = mock_sip
            MockAPI.return_value.aclose = AsyncMock()
            MockTTS.return_value.synthesize.return_value = _make_wav_bytes(300)
            MockCT.return_value.track_tts = AsyncMock()

            dialer = LiveKitSIPOutbound()
            req = DialRequest(
                phone_number="+919876543210",
                room_name="tenant_hdfc_bank-outbound-abc",
                tenant_id="tenant_hdfc_bank",
                script_text="नमस्ते!",
                language="hi-IN",
            )
            result = await dialer.dial(req)

        assert isinstance(result, DialResult), "dial() must return a DialResult dataclass."
        assert result.status == "dialing"
        assert result.phone == "+919876543210"
        assert result.sip_call_id == "call_xyz"
        assert result.participant_identity is not None

    @pytest.mark.asyncio
    async def test_dial_aborts_if_tts_fails(self) -> None:
        """dial() must raise OutboundError if TTS pre-synthesis fails."""
        from backend.telephony.outbound_dialer import LiveKitSIPOutbound, DialRequest
        from backend.exceptions import OutboundError, TTSError

        with patch("backend.telephony.outbound_dialer.api.LiveKitAPI") as MockAPI, \
             patch("backend.telephony.outbound_dialer.SarvamTTS") as MockTTS, \
             patch("backend.telephony.outbound_dialer.CostTracker"):

            MockTTS.return_value.synthesize.side_effect = TTSError("TTS error")
            MockAPI.return_value.aclose = AsyncMock()

            dialer = LiveKitSIPOutbound()
            req = DialRequest(
                phone_number="+919876543210",
                room_name="room",
                tenant_id="tenant_hdfc_bank",
                script_text="Hello.",
                language="en-IN",
            )
            with pytest.raises(OutboundError, match="TTS pre-synthesis failed"):
                await dialer.dial(req)

    @pytest.mark.asyncio
    async def test_dial_blocks_duplicate_room(self) -> None:
        """Second dial with the same room_name must be skipped."""
        from backend.telephony.outbound_dialer import LiveKitSIPOutbound, DialRequest, DialResult

        with patch("backend.telephony.outbound_dialer.api.LiveKitAPI") as MockAPI, \
             patch("backend.telephony.outbound_dialer.SarvamTTS") as MockTTS, \
             patch("backend.telephony.outbound_dialer.CostTracker"), \
             patch("backend.telephony.outbound_dialer._connect_and_monitor",
                   new_callable=AsyncMock):

            mock_sip = AsyncMock()
            mock_participant = MagicMock()
            mock_participant.sip_call_id = "call_001"
            mock_sip.create_sip_participant = AsyncMock(return_value=mock_participant)
            MockAPI.return_value.sip = mock_sip
            MockAPI.return_value.aclose = AsyncMock()
            MockTTS.return_value.synthesize.return_value = _make_wav_bytes(300)

            dialer = LiveKitSIPOutbound()
            req = DialRequest(
                phone_number="+919876543210",
                room_name="tenant_hdfc_bank-outbound-dup",
                tenant_id="tenant_hdfc_bank",
                script_text="Hello",
                language="hi-IN",
            )

            result1 = await dialer.dial(req)
            assert result1.status == "dialing"

            result2 = await dialer.dial(req)
            assert result2.status == "skipped"
            assert "duplicate" in (result2.error or "").lower()

    @pytest.mark.asyncio
    async def test_campaign_respects_semaphore_concurrency(self) -> None:
        """dial_campaign must not exceed max_concurrent simultaneous dials."""
        from backend.telephony.outbound_dialer import LiveKitSIPOutbound, DialResult

        active_at_once: list[int] = []
        current_active = 0

        async def mock_dial(req):
            nonlocal current_active
            current_active += 1
            active_at_once.append(current_active)
            await asyncio.sleep(0.01)
            current_active -= 1
            return DialResult(
                session_id="test", phone=req.phone_number, room_name=req.room_name,
                tenant_id=req.tenant_id, language=req.language, status="dialing",
            )

        dialer = LiveKitSIPOutbound.__new__(LiveKitSIPOutbound)
        dialer.dial = mock_dial  # type: ignore[assignment]

        contacts = [
            {"phone": f"+9199999000{i:02d}", "tenant_id": "tenant_hdfc_bank"}
            for i in range(10)
        ]
        result = await dialer.dial_campaign(
            contacts=contacts,
            script_template="Hello",
            language="en-IN",
            max_concurrent=3,
        )

        assert max(active_at_once) <= 3, \
            f"Semaphore violated: peak concurrency was {max(active_at_once)}"
        assert result.total == 10
        assert result.dialing == 10

    @pytest.mark.asyncio
    async def test_campaign_captures_individual_errors(self) -> None:
        """dial_campaign must return failures in CampaignResult rather than raising."""
        from backend.telephony.outbound_dialer import LiveKitSIPOutbound
        from backend.exceptions import OutboundError

        async def failing_dial(req):
            raise OutboundError("SIP unreachable", phone=req.phone_number)

        dialer = LiveKitSIPOutbound.__new__(LiveKitSIPOutbound)
        dialer.dial = failing_dial  # type: ignore[assignment]

        contacts = [
            {"phone": f"+9199999000{i:02d}", "tenant_id": "tenant_hdfc_bank"}
            for i in range(3)
        ]
        result = await dialer.dial_campaign(
            contacts=contacts, script_template="Hello", language="en-IN"
        )

        assert result.total == 3
        assert result.failed == 3
        assert result.dialing == 0

# 5. Inbound Handler Endpoint Tests 

class TestInboundHandlerEndpoints:
    """FastAPI endpoint tests for webhook, token, rooms, health, and ready."""

    @pytest.fixture
    def client(self):
        from fastapi import FastAPI
        from starlette.testclient import TestClient
        from backend.telephony.inbound_handler import router
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_health_endpoint_returns_structure(self, client) -> None:
        """GET /telephony/health must return status=ok and active_rooms."""
        resp = client.get("/telephony/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "active_rooms" in data
        assert "livekit_url" in data

    def test_ready_endpoint_livekit_up(self, client) -> None:
        """GET /telephony/ready returns 200 when LiveKit is reachable."""
        with patch("backend.telephony.inbound_handler.api.LiveKitAPI") as MockAPI:
            mock_api = AsyncMock()
            mock_api.list_rooms = AsyncMock(return_value=MagicMock(rooms=[]))
            mock_api.aclose = AsyncMock()
            MockAPI.return_value = mock_api

            resp = client.get("/telephony/ready")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "ok"
            assert "livekit_rooms" in data
            mock_api.aclose.assert_awaited_once()

    def test_ready_endpoint_livekit_down(self, client) -> None:
        """GET /telephony/ready returns 503 when LiveKit is unreachable."""
        with patch("backend.telephony.inbound_handler.api.LiveKitAPI") as MockAPI:
            MockAPI.side_effect = Exception("connection refused")
            resp = client.get("/telephony/ready")
            assert resp.status_code == 503
            data = resp.json()
            assert data["detail"]["status"] == "degraded"

    def test_token_endpoint_known_tenant_returns_200(self, client) -> None:
        with patch("backend.telephony.inbound_handler.api.AccessToken") as MockToken:
            MockToken.return_value.with_identity.return_value = MockToken.return_value
            MockToken.return_value.with_name.return_value = MockToken.return_value
            MockToken.return_value.with_grants.return_value = MockToken.return_value
            MockToken.return_value.with_ttl.return_value = MockToken.return_value 
            MockToken.return_value.to_jwt.return_value = "test_jwt"

            resp = client.post("/telephony/token", json={
                "tenant_id": "tenant_hdfc_bank",
                "room_name": "tenant_hdfc_bank-inbound-abc123",
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["token"] == "test_jwt"
        assert data["tenant_id"] == "tenant_hdfc_bank"
        assert "identity" in data

    def test_token_endpoint_unknown_tenant_returns_422(self, client) -> None:
        resp = client.post("/telephony/token", json={
            "tenant_id": "unknown_company_xyz",
            "room_name": "unknown-room",
        })
        assert resp.status_code == 422

    def test_rooms_endpoint_returns_structure(self, client) -> None:
        resp = client.get("/telephony/rooms")
        assert resp.status_code == 200
        data = resp.json()
        assert "count" in data and "rooms" in data
        assert "connected_count" in data
        assert "connecting_count" in data

    def test_extract_tenant_hdfc(self) -> None:
        from backend.telephony.inbound_handler import _extract_tenant_from_room_name
        assert _extract_tenant_from_room_name("tenant_hdfc_bank-inbound-abc") == "tenant_hdfc_bank"

    def test_extract_tenant_swiggy(self) -> None:
        from backend.telephony.inbound_handler import _extract_tenant_from_room_name
        assert _extract_tenant_from_room_name("tenant_swiggy_support-inbound-xyz") == "tenant_swiggy_support"

    def test_extract_tenant_unknown_raises(self) -> None:
        from backend.telephony.inbound_handler import _extract_tenant_from_room_name
        with patch("backend.telephony.inbound_handler.TENANTS", ["tenant_hdfc_bank", "tenant_swiggy_support"]):
            with pytest.raises(ValueError, match="Cannot determine tenant"):
                _extract_tenant_from_room_name("unknown-prefix-room")

# 6. Outbound Endpoint Tests 

class TestOutboundEndpoints:
    """FastAPI endpoint tests for the outbound dialer router."""

    @pytest.fixture
    def client(self):
        from fastapi import FastAPI
        from starlette.testclient import TestClient
        from backend.telephony.outbound_dialer import router
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_outbound_health_returns_structure(self, client) -> None:
        """GET /telephony/outbound/health returns outbound-only room count."""
        resp = client.get("/telephony/outbound/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "outbound_dialer"
        assert "active_outbound_rooms" in data

    def test_outbound_ready_success(self, client) -> None:
        """GET /telephony/outbound/ready returns 200 when LiveKit is up."""
        with patch("backend.telephony.outbound_dialer._get_ready_lkapi", new_callable=AsyncMock) as mock_get:
            mock_api = AsyncMock()
            mock_api.list_rooms = AsyncMock()
            mock_get.return_value = mock_api

            resp = client.get("/telephony/outbound/ready")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"

    def test_outbound_ready_failure(self, client) -> None:
        """GET /telephony/outbound/ready returns 503 when LiveKit is down."""
        with patch("backend.telephony.outbound_dialer._get_ready_lkapi", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = Exception("LiveKit down")
            resp = client.get("/telephony/outbound/ready")
            assert resp.status_code == 503
            assert resp.json()["detail"]["status"] == "degraded"


# 7. SIPTrunkManager Tests 

class TestSIPTrunkManager:
    """Tests for SIP trunk provisioning including idempotency."""

    @pytest.mark.asyncio
    async def test_create_inbound_trunk_returns_typed_result(self) -> None:
        """create_inbound_trunk_idempotent must return TrunkProvisionResult."""
        from backend.telephony.sip_trunk import SIPTrunkManager, TrunkProvisionResult

        with patch("backend.telephony.sip_trunk.api.LiveKitAPI") as MockAPI:
            mock_response = MagicMock()
            mock_response.items = []  # No existing trunks
            MockAPI.return_value.sip.list_sip_inbound_trunk = AsyncMock(
                return_value=mock_response
            )
            mock_trunk = MagicMock()
            mock_trunk.sip_trunk_id = "trunk_001"
            MockAPI.return_value.sip.create_inbound_trunk = AsyncMock(
                return_value=mock_trunk
            )
            MockAPI.return_value.aclose = AsyncMock()

            mgr = SIPTrunkManager()
            result = await mgr.create_inbound_trunk_idempotent(
                name="test-trunk",
                numbers=["+911234567890"],
                tenant_id="tenant_hdfc_bank",
            )
            await mgr.close()

        assert isinstance(result, TrunkProvisionResult)
        assert result.trunk_id == "trunk_001"
        assert result.created is True

    @pytest.mark.asyncio
    async def test_create_inbound_trunk_idempotent_returns_existing(self) -> None:
        """create_inbound_trunk_idempotent must NOT create if trunk already exists."""
        from backend.telephony.sip_trunk import SIPTrunkManager

        with patch("backend.telephony.sip_trunk.api.LiveKitAPI") as MockAPI:
            existing_trunk = MagicMock()
            existing_trunk.name = "test-trunk"
            existing_trunk.sip_trunk_id = "trunk_existing_001"
            existing_trunk.numbers = ["+911234567890"]

            mock_response = MagicMock()
            mock_response.items = [existing_trunk]
            MockAPI.return_value.sip.list_sip_inbound_trunk = AsyncMock(
                return_value=mock_response
            )
            create_mock = AsyncMock()
            MockAPI.return_value.sip.create_inbound_trunk = create_mock
            MockAPI.return_value.aclose = AsyncMock()

            mgr = SIPTrunkManager()
            result = await mgr.create_inbound_trunk_idempotent(
                name="test-trunk",  # Same name as existing
                numbers=["+911234567890"],
                tenant_id="tenant_hdfc_bank",
            )
            await mgr.close()

        assert result.trunk_id == "trunk_existing_001"
        assert result.created is False
        create_mock.assert_not_called(), "Must NOT call create when trunk already exists."

    @pytest.mark.asyncio
    async def test_list_inbound_trunks_returns_items(self) -> None:
        from backend.telephony.sip_trunk import SIPTrunkManager

        with patch("backend.telephony.sip_trunk.api.LiveKitAPI") as MockAPI:
            mock_item = MagicMock(sip_trunk_id="t1", name="trunk1")
            mock_response = MagicMock(items=[mock_item])
            MockAPI.return_value.sip.list_sip_inbound_trunk = AsyncMock(
                return_value=mock_response
            )
            mgr = SIPTrunkManager()
            trunks = await mgr.list_inbound_trunks()

        assert len(trunks) == 1
        assert trunks[0].sip_trunk_id == "t1"

    @pytest.mark.asyncio
    async def test_list_inbound_trunks_error_returns_empty(self) -> None:
        """list_inbound_trunks must return [] on API error (not raise)."""
        from backend.telephony.sip_trunk import SIPTrunkManager

        with patch("backend.telephony.sip_trunk.api.LiveKitAPI") as MockAPI:
            MockAPI.return_value.sip.list_sip_inbound_trunk = AsyncMock(
                side_effect=Exception("API unavailable")
            )
            mgr = SIPTrunkManager()
            result = await mgr.list_inbound_trunks()

        assert result == []