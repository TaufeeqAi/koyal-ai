from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

from backend.agents.graph import koyal_graph
from backend.observability.langfuse_client import make_callback_handler
from backend.config import STT_COST_PER_SECOND, TTS_COST_PER_CHAR
from backend.agents.state import AgentState, make_initial_state
from backend.observability.prometheus_metrics import (
    record_asr_confidence,
    record_call_end,
    record_cost_inr,
    record_escalation,
    record_language_detection,
    record_llm_latency,
    record_pipeline_latency,
    record_retrieval_score,
    record_safety_cleared,
    record_stt_latency,
    record_tts_latency,
    record_ttfr,
    record_guardrail_input_block,
    record_guardrail_output_block,
    record_three_strike_termination,
    update_harmful_attempts,
    remove_harmful_attempts_gauge,
)

logger = logging.getLogger(__name__)


def _invoke_graph_with_callbacks(
    initial_state: AgentState,
    callbacks: Optional[list[Any]] = None,
) -> AgentState:
    """Invoke the compiled LangGraph pipeline with optional LangChain callbacks.

    SYNCHRONOUS — always called via asyncio.to_thread(). Do not call directly
    from an async context without to_thread() wrapping.

    Args:
        initial_state: Fully-initialised AgentState (from _make_initial_state).
        callbacks:     List of LangChain callback handlers. Langfuse
                       CallbackHandler auto-traces every node as a child span.

    Returns:
        Final AgentState after all nodes have executed.
    """
    config: dict = {}
    if callbacks:
        config["callbacks"] = callbacks
        for cb in callbacks:
            if hasattr(cb, "_koyal_metadata"):
                config.setdefault("metadata", {}).update(cb._koyal_metadata)

    t0 = time.perf_counter()
    final_state: AgentState = (
        koyal_graph.invoke(initial_state, config=config)
        if config
        else koyal_graph.invoke(initial_state)
    )
    latency_ms = (time.perf_counter() - t0) * 1000
    final_state["latency_ms"] = round(latency_ms, 2)
    return final_state


async def observed_invoke_graph(
    query: str,
    tenant_id: str,
    session_id: str,
    call_type: str = "inbound",
    stt_latency_ms: float = 0.0,
    stt_confidence: float = 0.85,
    stt_duration_seconds: float = 0.0,
    tts_latency_ms: float = 0.0,
    tts_chars: int = 0,
    harmful_attempt_count: int = 0,
    call_start_time: Optional[float] = None,
) -> AgentState:
    """Fully-instrumented async wrapper for the LangGraph pipeline.

    This is the primary entry point for WebSocketHandler and
    main.py. Replaces direct koyal_graph.invoke() calls.

    Responsibilities:
        1. Builds AgentState with trace_id
        2. Creates Langfuse CallbackHandler (real or NoOp)
        3. Runs synchronous graph in asyncio thread pool (non-blocking)
        4. Records all Prometheus metrics from the completed state
        5. Returns bilingual fallback response on pipeline failure

    Args:
        query:                Caller transcript text from STT.
        tenant_id:            Tenant identifier (e.g. "tenant_hdfc_bank").
        session_id:           Call session UUID from SessionManager.
        call_type:            "inbound" or "outbound".
        stt_latency_ms:       STT API round-trip latency in ms.
        stt_confidence:       ASR confidence score (0.0–1.0).
        stt_duration_seconds: Audio duration in seconds (for STT cost calculation).
        tts_latency_ms:       TTS synthesis latency in ms (TTS call).
        tts_chars:            TTS output character count (for TTS cost calculation).
        call_start_time:      time.perf_counter() at utterance start (for TTFR).

    Returns:
        Final AgentState with all fields populated. On pipeline failure,
        returns a minimal state with 'error' set and a bilingual fallback
        in 'final_response'. Never raises.
    """
    initial_state = make_initial_state(
        query=query, 
        tenant_id=tenant_id, 
        session_id=session_id, 
        call_type=call_type,
        harmful_attempt_count=harmful_attempt_count,
    )

    # Langfuse CallbackHandler — injected into LangGraph via RunnableConfig
    handler = make_callback_handler(
        session_id=session_id,
        tenant_id=tenant_id,
        trace_id=initial_state["trace_id"],
        call_type=call_type,
  
    )

    # Run synchronous LangGraph in thread pool (non-blocking)
    try:
        final_state: AgentState = await asyncio.to_thread(
            _invoke_graph_with_callbacks,
            initial_state,
            [handler],
        )
    except Exception as exc:
        logger.error(
            "observed_invoke_graph: pipeline failed — tenant=%s session=%s error=%s",
            tenant_id, session_id, exc,
            exc_info=True,
        )
        initial_state["error"] = str(exc)
        # Bilingual fallback — contextually appropriate for BFSI voice AI
        initial_state["final_response"] = (
            "मुझे खेद है, अभी तकनीकी समस्या है। कृपया हमारे हेल्पलाइन पर कॉल करें।"
            if "hi" in (initial_state.get("detected_language") or "en")
            else "I'm sorry, there's a technical issue. Please call our helpline."
        )
        return initial_state

    # ── Prometheus metric recording 

    detected_lang = final_state.get("detected_language") or "en-in"
    escalated = final_state.get("escalate", False)
    llm_tokens = final_state.get("llm_tokens") or 0
    latency_ms = final_state.get("latency_ms") or 0.0
    retrieved_chunks = final_state.get("retrieved_chunks") or []
    is_code_mixed = final_state.get("is_code_mixed", False)

    # STT metrics
    if stt_latency_ms > 0 or stt_confidence > 0:
        record_stt_latency(tenant_id, detected_lang, stt_latency_ms)
        record_asr_confidence(tenant_id, detected_lang, stt_confidence)
        record_language_detection(tenant_id, detected_lang, is_code_mixed)
        if stt_duration_seconds > 0:
            record_cost_inr(tenant_id, "stt", stt_duration_seconds * STT_COST_PER_SECOND)

    # LLM metrics
    record_llm_latency(tenant_id, latency_ms)
    # Groq free tier: ₹0 LLM cost — counter still incremented for visibility
    if llm_tokens > 0:
        record_cost_inr(tenant_id, "llm", 0.0)

    # TTS metrics
    if tts_latency_ms > 0:
        record_tts_latency(tenant_id, detected_lang, tts_latency_ms)
    if tts_chars > 0:
        record_cost_inr(tenant_id, "tts", tts_chars * TTS_COST_PER_CHAR)

    # Retrieval scores
    for chunk in retrieved_chunks:
        score = chunk.get("rerank_score") or chunk.get("score")
        if score:
            record_retrieval_score(tenant_id, float(score))

    # Safety gate
    if escalated:
        record_escalation(
            tenant_id=tenant_id,
            language=detected_lang,
            reason=final_state.get("escalation_reason") or "unknown",
        )
    else:
        record_safety_cleared(tenant_id)

    # TTFR
    if call_start_time is not None:
        ttfr_ms = (time.perf_counter() - call_start_time) * 1000
        record_ttfr(tenant_id, detected_lang, ttfr_ms)

    # Pipeline latency (STT + LLM; TTS is recorded separately above)
    pipeline_ms = stt_latency_ms + latency_ms
    record_pipeline_latency(tenant_id, detected_lang, pipeline_ms)

    # ── Guardrail metrics (NeMo + 3‑strike) 
    if final_state.get("guardrail_input_blocked"):
        reason = final_state.get("guardrail_input_reason") or "unknown"
        record_guardrail_input_block(tenant_id, reason)

    if final_state.get("guardrail_output_blocked"):
        reason = final_state.get("guardrail_output_reason") or "unknown"
        record_guardrail_output_block(tenant_id, reason)

    # 3‑strike termination – only if session ended due to strikes
    if final_state.get("end_session") and final_state.get("harmful_attempt_count", 0) >= 3:
        record_three_strike_termination(tenant_id)

    # Update harmful attempts gauge for this session (per turn)
    update_harmful_attempts(
        tenant_id,
        session_id,
        final_state.get("harmful_attempt_count", 0)
    )

    logger.info(
        "observed_invoke_graph: completed — tenant=%s lang=%s escalated=%s "
        "tokens=%d llm_latency=%.0fms pipeline=%.0fms",
        tenant_id, detected_lang, escalated, llm_tokens, latency_ms, pipeline_ms,
    )

    return final_state


async def observed_call_lifecycle(
    tenant_id: str,
    session_id: str,  
    language: str,
    call_type: str,
    duration_seconds: float,
    outcome: str = "completed",
) -> None:
    """Record call-level lifecycle metrics at session close.

    Called by WebSocketHandler._cleanup() to record the full call duration.

    Args:
        tenant_id:         Tenant identifier.
        language:          Final detected language for this call.
        call_type:         "inbound" or "outbound".
        duration_seconds:  Total wall-clock duration of the call.
        outcome:           "completed", "escalated", "dropped", or "error".
    """
    record_call_end(
        tenant_id=tenant_id,
        language=language,
        call_type=call_type,
        duration_seconds=duration_seconds,
        outcome=outcome,
    )
    remove_harmful_attempts_gauge(tenant_id, session_id)
