"""
Public surface:
    SarvamSTT               — Sarvam Saaras V2 speech-to-text
    SarvamTTS               — Sarvam Bulbul V1 text-to-speech
    SpeechSegmenter         — WebRTC VAD state machine
    FrameBuffer             — Fixed-frame accumulator for variable-size audio
    SessionManager          — Per-call session lifecycle
    OutboundCaller          — Concurrent outbound call engine
    WebSocketVoiceHandler   — Full-duplex voice loop
"""

from backend.voice.stt import SarvamSTT
from backend.voice.tts import SarvamTTS
from backend.voice.vad import FrameBuffer, SpeechSegmenter
from backend.voice.session_manager import SessionManager, get_session_manager
from backend.voice.outbound_caller import OutboundCaller
from backend.voice.websocket_handler import WebSocketVoiceHandler

__all__ = [
    "SarvamSTT",
    "SarvamTTS",
    "FrameBuffer",
    "SpeechSegmenter",
    "SessionManager",
    "get_session_manager",
    "OutboundCaller",
    "WebSocketVoiceHandler",
]