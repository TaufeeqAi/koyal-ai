"""
Full-duplex async voice loop: PCM → VAD → STT → LangGraph → TTS → audio.

Per-turn pipeline
─────────────────
1. Receive binary PCM frames via WebSocket (client → server)
2. SpeechSegmenter accumulates frames → speech segment when utterance ends
3. SarvamSTT.atranscribe()           → transcript + detected language
4. AgentState built from transcript + session context + harmful_attempt_count
5. koyal_graph.ainvoke(state)        → final_response in caller's language
6. Guardrails post-processing:
     - Persist updated strike count to Redis
     - Check end_session → terminate WebSocket (3rd strike)
     - Check wait_for_next_input → speak warning, loop back (1st/2nd strike)
7. SarvamTTS.asynthesize_streaming() → yield audio chunks per sentence
8. websocket.send_bytes(audio_chunk) per sentence (streaming)
9. Update session: state, language, usage counters
10. CostTracker.track_*()            → Redis atomic writes

Control messages (JSON text frames)
────────────────────────────────────
  {"type": "config", "language": "hi-IN"}   — override detected language
  {"type": "flush"}                          — force-flush VAD buffer → STT
  {"type": "end"}                            — graceful call termination

Interruption detection
──────────────────────
When binary audio arrives while TTS is streaming (_is_speaking=True),
_cancel_speaking() is called and TTS stream is abandoned. This enables
natural barge-in / interruption behaviour without waiting for the full
TTS response to complete.

Guardrails 3-strike integration
─────────────────────────────
* harmful_attempt_count is loaded from Redis per-turn (cross-worker safety)
* Passed to AgentState for graph routing decisions
* Updated count is persisted back to Redis after graph invocation
* 1st/2nd strike: warning spoken, session continues (wait_for_next_input)
* 3rd strike: termination message spoken, WebSocket closed (end_session)
* Emergency escalation bypasses strike system entirely (safety_gate authority)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect

from backend.agents.graph import koyal_graph
from backend.agents.state import AgentState
from backend.config import DEFAULT_LANGUAGE, WS_GREETING_ENABLED, load_tenant_config
from backend.cost_tracker import CostTracker
from backend.exceptions import LowConfidenceError, SessionError, STTError, TTSError
from backend.voice.session_manager import SessionState, get_session_manager
from backend.voice.stt import SarvamSTT
from backend.voice.tts import SarvamTTS
from backend.voice.vad import SpeechSegmenter

logger = logging.getLogger(__name__)

_FALLBACK_TEXT: dict[str, str] = {
    "hi-IN":       "क्षमा करें, मुझे समझने में परेशानी हो रही है। कृपया फिर से बोलें।",
    "en-IN":       "I'm sorry, I had trouble understanding you. Please try again.",
    "hi-IN+en-IN": "Sorry, mujhe samajhne mein problem ho rahi hai. Please repeat.",
}
_DEFAULT_FALLBACK = _FALLBACK_TEXT["en-IN"]


class WebSocketVoiceHandler:
    """Manages the full lifecycle of one WebSocket voice call with guardrails
    3-strike progressive discipline integration.

    Args:
        websocket: FastAPI WebSocket connection.
        tenant_id: Validated tenant identifier.
        session_id: Unique session ID.
        call_type: ``"inbound"`` or ``"outbound"``.

    Example (from FastAPI WebSocket endpoint)::

        handler = WebSocketVoiceHandler(
            websocket=ws,
            tenant_id="tenant_hdfc_bank",
            session_id="sess_001",
        )
        await handler.run()
    """

    def __init__(
        self,
        websocket: WebSocket,
        tenant_id: str,
        session_id: str,
        call_type: str = "inbound",
    ) -> None:
        self._ws = websocket
        self._tenant_id = tenant_id
        self._session_id = session_id
        self._call_type = call_type

        self._stt = SarvamSTT()
        self._tts = SarvamTTS()
        self._segmenter = SpeechSegmenter()
        self._cost_tracker = CostTracker()
        self._session_manager = get_session_manager()

        self._detected_language: str = DEFAULT_LANGUAGE
        self._is_speaking: bool = False
        self._interrupted: bool = False
        self._speaking_task: Optional[asyncio.Task] = None

    async def run(self) -> None:
        """Accept WebSocket, create session, run voice loop, clean up on exit."""
        await self._ws.accept()

        try:
            session = await self._session_manager.create_session(
                self._tenant_id, self._session_id, self._call_type
            )
        except SessionError as exc:
            logger.error("Session creation failed: %s", exc)
            await self._send_error(str(exc))
            await self._ws.close(code=4009)
            return

        if WS_GREETING_ENABLED:
            await self._send_greeting()

        await self._send_status("listening")
        logger.info(
            "Voice loop started: tenant=%s session=%s strikes=%d",
            self._tenant_id, self._session_id,
            (await self._session_manager.get_session(self._session_id)).harmful_attempt_count,
        )

        try:
            await self._voice_loop()
        except WebSocketDisconnect:
            logger.info("Client disconnected: session=%s", self._session_id)
        except Exception as exc:
            logger.exception("Voice loop error: session=%s error=%s", self._session_id, exc)
        finally:
            await self._cleanup()

    async def _voice_loop(self) -> None:
        """Main audio receive loop."""
        while True:
            try:
                message = await self._ws.receive()
            except WebSocketDisconnect:
                raise

            if "bytes" in message and message["bytes"]:
                audio_data: bytes = message["bytes"]

                if self._is_speaking:
                    await self._cancel_speaking()

                await self._session_manager.update_session(
                    self._session_id, state=SessionState.LISTENING
                )
                result = self._segmenter.process_chunk(audio_data)

                if result.utterance_complete and result.speech_bytes:
                    await self._process_speech_turn(result.speech_bytes)

            elif "text" in message and message["text"]:
                await self._handle_control_message(message["text"])

    async def _process_speech_turn(self, speech_bytes: bytes) -> None:
        """Run one complete STT → LangGraph → TTS turn with guardrails
        3-strike post-processing.
        """
        turn_start = time.monotonic()
        await self._session_manager.update_session(
            self._session_id, state=SessionState.PROCESSING
        )
        await self._send_status("processing")

        # ── STT 
        try:
            stt_start = time.monotonic()
            stt_result = await self._stt.atranscribe(
                speech_bytes, language_hint=self._detected_language
            )
            stt_latency_ms = (time.monotonic() - stt_start) * 1000
            # obs: STT_LATENCY_MS.labels(tenant_id=self._tenant_id).observe(stt_latency_ms)
        except LowConfidenceError as exc:
            logger.warning("[%s] Low STT confidence: %.2f", self._session_id, exc.context.get("confidence", 0))
            await self._speak_response(
                _FALLBACK_TEXT.get(self._detected_language, _DEFAULT_FALLBACK),
                self._detected_language,
            )
            return
        except STTError as exc:
            logger.error("[%s] STT failed: %s", self._session_id, exc)
            await self._send_error("Speech recognition failed. Please try again.")
            await self._send_status("listening")
            # obs: STT_ERRORS_TOTAL.labels(tenant_id=self._tenant_id, reason="http_error").inc()
            return
        except Exception as exc:
            logger.error("[%s] STT unexpected error: %s", self._session_id, exc)
            await self._send_status("listening")
            return

        transcript = stt_result["text"]
        detected_lang = stt_result["language_code"]
        stt_seconds = stt_result["estimated_seconds"]

        if not transcript:
            logger.debug("[%s] Empty STT transcript — skipping turn.", self._session_id)
            await self._send_status("listening")
            return

        if detected_lang and detected_lang != self._detected_language:
            self._detected_language = detected_lang
            logger.info("[%s] Language updated: %s", self._session_id, detected_lang)

        await self._cost_tracker.track_stt(self._tenant_id, stt_seconds)
        async with self._session_manager.acquire(self._session_id) as sess:
            sess.language = detected_lang
            sess.stt_seconds += stt_seconds

        logger.info(
            "[%s] STT: lang=%s conf=%.2f latency=%.0fms text=%r",
            self._session_id, detected_lang, stt_result["confidence"],
            stt_latency_ms, transcript[:60],
        )

        # ── Guardrails: Load strike count from Redis (per-turn, cross-worker) ─
        current_strikes = await self._session_manager.load_session_strikes(self._session_id)
        async with self._session_manager.acquire(self._session_id) as sess:
            # Sync in-memory state with Redis (handles worker migration / reconnect)
            if sess.harmful_attempt_count != current_strikes:
                logger.info(
                    "[%s] Strike count synced: memory=%d → redis=%d",
                    self._session_id, sess.harmful_attempt_count, current_strikes,
                )
                sess.harmful_attempt_count = current_strikes

        # ── LangGraph 
        initial_state = AgentState(
            query=transcript,
            tenant_id=self._tenant_id,
            detected_language=detected_lang,
            session_id=self._session_id,
            harmful_attempt_count=current_strikes,
        )

        try:
            agent_result: dict = await koyal_graph.ainvoke(initial_state)
        except Exception as exc:
            logger.error("[%s] LangGraph failed: %s", self._session_id, exc)
            await self._speak_response(
                _FALLBACK_TEXT.get(self._detected_language, _DEFAULT_FALLBACK),
                self._detected_language,
            )
            return

        # ── Guardrails post-processing 
        updated_strikes = agent_result.get("harmful_attempt_count", current_strikes)
        guardrail_blocked = agent_result.get("guardrail_input_blocked", False)
        end_session = agent_result.get("end_session", False)
        wait_for_next_input = agent_result.get("wait_for_next_input", False)

        # Update in-memory session and persist to Redis
        async with self._session_manager.acquire(self._session_id) as sess:
            sess.harmful_attempt_count = updated_strikes
        await self._session_manager.save_session_strikes(self._session_id, updated_strikes)

        logger.info(
            "[%s] Guardrails: strikes=%d→%d blocked=%s end_session=%s wait=%s",
            self._session_id, current_strikes, updated_strikes,
            guardrail_blocked, end_session, wait_for_next_input,
        )

        # ── Handle 3rd strike: termination 
        if end_session:
            final_response = agent_result.get("final_response") or _DEFAULT_FALLBACK
            response_lang = agent_result.get("detected_language") or self._detected_language

            await self._send_text_frame({
                "type": "termination",
                "text": final_response,
                "language": response_lang,
                "reason": "three_strike_policy",
            })
            await self._speak_response(final_response, response_lang)

            await self._session_manager.update_session(
                self._session_id, state=SessionState.ENDED, outcome="terminated"
            )
            await self._cost_tracker.track_call(self._tenant_id, outcome="terminated")
            # obs: CALLS_TOTAL.labels(..., outcome="terminated").inc()

            logger.warning(
                "[%s] Session terminated by 3-strike policy. Strikes=%d",
                self._session_id, updated_strikes,
            )
            raise WebSocketDisconnect()

        # ── Handle 1st/2nd strike: warning, skip normal pipeline 
        if guardrail_blocked and wait_for_next_input:
            warning_response = agent_result.get("final_response") or _DEFAULT_FALLBACK
            warning_lang = agent_result.get("detected_language") or self._detected_language

            await self._send_text_frame({
                "type": "warning",
                "text": warning_response,
                "language": warning_lang,
                "strike_number": updated_strikes,
            })
            await self._speak_response(warning_response, warning_lang)

            await self._session_manager.update_session(
                self._session_id, state=SessionState.LISTENING
            )
            await self._send_status("listening")

            logger.info(
                "[%s] Strike %d warning issued. Waiting for next input.",
                self._session_id, updated_strikes,
            )
            return  # Skip normal TTS, LLM cost tracking, turn counter increment

        # ── Normal pipeline: extract response fields 
        llm_tokens = agent_result.get("llm_tokens", 0) or 0
        if llm_tokens:
            await self._cost_tracker.track_llm(self._tenant_id, llm_tokens)
            async with self._session_manager.acquire(self._session_id) as sess:
                sess.llm_tokens += llm_tokens

        final_response = agent_result.get("final_response") or _DEFAULT_FALLBACK
        is_escalation = agent_result.get("escalate", False)
        response_lang = agent_result.get("detected_language") or self._detected_language

        await self._send_text_frame({
            "type": "escalation" if is_escalation else "response",
            "text": final_response,
            "verified": agent_result.get("verified", False),
            "language": response_lang,
            "reason": agent_result.get("escalation_reason") if is_escalation else None,
        })

        if is_escalation:
            await self._session_manager.update_session(
                self._session_id, state=SessionState.ESCALATED, outcome="escalated"
            )
            await self._cost_tracker.track_call(self._tenant_id, outcome="escalated")

        # ── TTS 
        await self._session_manager.update_session(
            self._session_id, state=SessionState.SPEAKING
        )
        await self._send_status("speaking")
        await self._speak_response(final_response, response_lang)

        tts_chars = len(final_response)
        await self._cost_tracker.track_tts(self._tenant_id, tts_chars)
        async with self._session_manager.acquire(self._session_id) as sess:
            sess.tts_chars += tts_chars
            sess.turn_count += 1

        turn_latency_ms = (time.monotonic() - turn_start) * 1000
        logger.info(
            "[%s] Turn complete: lang=%s latency=%.0fms escalated=%s strikes=%d",
            self._session_id, response_lang, turn_latency_ms, is_escalation, updated_strikes,
        )
        # obs: TURN_LATENCY_MS.labels(tenant_id=self._tenant_id).observe(turn_latency_ms)

        if is_escalation:
            await self._session_manager.end_session(self._session_id, outcome="escalated")
            raise WebSocketDisconnect()

        await self._session_manager.update_session(
            self._session_id, state=SessionState.LISTENING
        )
        await self._send_status("listening")

    async def _speak_response(self, text: str, language_code: str) -> None:
        """Stream TTS audio to the client, sentence by sentence."""
        self._is_speaking = True
        self._interrupted = False

        async def _stream() -> None:
            try:
                async for audio_chunk in self._tts.asynthesize_streaming(text, language_code):
                    if self._interrupted:
                        logger.debug("[%s] TTS interrupted.", self._session_id)
                        break
                    try:
                        await self._ws.send_bytes(audio_chunk)
                    except WebSocketDisconnect:
                        raise
                    except Exception as exc:
                        logger.debug("[%s] send_bytes error: %s", self._session_id, exc)
            except TTSError as exc:
                logger.error("[%s] TTS error: %s", self._session_id, exc)
                # obs: TTS_ERRORS_TOTAL.labels(...).inc()
            finally:
                self._is_speaking = False

        self._speaking_task = asyncio.ensure_future(_stream())
        await self._speaking_task

    async def _cancel_speaking(self) -> None:
        """Signal barge-in interruption and cancel the TTS stream."""
        self._interrupted = True
        if self._speaking_task and not self._speaking_task.done():
            self._speaking_task.cancel()
            try:
                await self._speaking_task
            except asyncio.CancelledError:
                pass
        self._is_speaking = False
        logger.debug("[%s] TTS cancelled (barge-in).", self._session_id)

    async def _send_greeting(self) -> None:
        """Synthesise and send a tenant-specific greeting in the tenant's language."""
        try:
            cfg = load_tenant_config(self._tenant_id)
            primary_lang = cfg.get("primary_language", DEFAULT_LANGUAGE)
            greeting = cfg.get("greeting_message", "")
            if not greeting:
                company = cfg.get("company_name", "")
                if "hi" in primary_lang:
                    greeting = f"नमस्ते! {company} में आपका स्वागत है। मैं आपकी कैसे मदद कर सकता हूँ?"
                else:
                    greeting = f"Hello! Welcome to {company}. How can I help you today?"
            await self._speak_response(greeting, primary_lang)
        except Exception as exc:
            logger.warning("[%s] Greeting failed (continuing): %s", self._session_id, exc)

    async def _handle_control_message(self, text: str) -> None:
        """Handle incoming JSON control messages.

        Supported types:
            ``{"type": "config", "language": "hi-IN"}`` — language override
            ``{"type": "flush"}``                        — force VAD flush → STT
            ``{"type": "end"}``                          — graceful call end
        """
        try:
            msg = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("[%s] Non-JSON text frame: %r", self._session_id, text[:80])
            return

        msg_type = msg.get("type")

        if msg_type == "config":
            lang = msg.get("language")
            if lang:
                self._detected_language = lang
                logger.debug("[%s] Language override: %s", self._session_id, lang)

        elif msg_type == "flush":
            logger.debug("[%s] VAD flush requested by client.", self._session_id)
            flushed = self._segmenter.flush()
            if flushed:
                await self._process_speech_turn(flushed)
            else:
                logger.debug("[%s] VAD flush: no speech buffered.", self._session_id)

        elif msg_type == "end":
            logger.info("[%s] Client requested call end.", self._session_id)
            await self._session_manager.end_session(self._session_id, outcome="completed")
            raise WebSocketDisconnect()

        elif msg_type == "ping":
            await self._send_text_frame({"type": "pong"})

        else:
            logger.debug("[%s] Unknown control message type: %r", self._session_id, msg_type)

    async def _send_status(self, state: str) -> None:
        await self._send_text_frame({"type": "status", "state": state})

    async def _send_error(self, message: str) -> None:
        await self._send_text_frame({"type": "error", "message": message})

    async def _send_text_frame(self, payload: dict) -> None:
        """Send a JSON text frame, swallowing disconnect errors."""
        try:
            await self._ws.send_text(json.dumps(payload, ensure_ascii=False))
        except (WebSocketDisconnect, RuntimeError):
            pass
        except Exception as exc:
            logger.debug("[%s] send_text_frame ignored: %s", self._session_id, exc)

    async def _cleanup(self) -> None:
        """End session and flush costs on any exit path."""
        try:
            session = await self._session_manager.get_session(self._session_id)
            if session.is_active:
                # Determine outcome: terminated takes precedence
                outcome = session.outcome or "completed"
                await self._session_manager.end_session(
                    self._session_id, outcome=outcome
                )
                await self._cost_tracker.track_call(self._tenant_id, outcome=outcome)
                # obs: CALLS_TOTAL.labels(tenant_id=self._tenant_id, outcome=outcome).inc()
        except SessionError:
            pass
        except Exception as exc:
            logger.error("[%s] Cleanup error: %s", self._session_id, exc)

        self._segmenter.reset()
        logger.info("[%s] WebSocketVoiceHandler cleanup complete.", self._session_id)