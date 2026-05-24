from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """Shared mutable state threaded through every LangGraph node.

    Fields with ``total=False`` are optional and start as ``None``; the
    four required fields (query, tenant_id, session_id, call_type) must
    be supplied in the initial invocation dict.
    """

    # ── Required input fields 
    query: str               # Raw caller utterance (Hindi / English / Hinglish)
    tenant_id: str           # e.g. "tenant_hdfc_bank"
    session_id: str          # Unique per call session
    call_type: str           # "inbound" | "outbound"

    # ── Trace / correlation 
    trace_id: str            # UUID for this single pipeline invocation
    timestamp: str           # ISO-8601 UTC start time

    # ── Language detection 
    detected_language: Optional[str]   # "hi-IN" | "en-IN" | "hi-IN+en-IN"
    language_confidence: Optional[float]
    is_code_mixed: bool                # True when Hinglish detected
    detection_method: Optional[str]    # "script" | "sarvam_lid" | "fallback"

    # ── Safety gate 
    safety_cleared: bool
    escalate: bool
    escalation_reason: Optional[str]

    # ── Guardrails (NeMo) 
    guardrail_input_blocked: bool
    guardrail_input_reason: Optional[str]
    guardrail_output_blocked: bool
    guardrail_output_reason: Optional[str]
    guardrail_pii_masked: Optional[list[str]]
    guardrail_hallucination_score: Optional[float]
    guardrail_pii_leaked: Optional[list[str]]

    # ── 3-Strike Policy (NeMo Input Rails)
    harmful_attempt_count: int
    end_session: bool
    wait_for_next_input: bool

    # ── Language bridge 
    query_english: Optional[str]       # Translated query for LLM reasoning

    # ── Retrieval 
    retrieved_chunks: Optional[list[dict[str, Any]]]
    retrieval_context: Optional[str]   # Formatted string passed to LLM

    # ── Response generation 
    raw_response: Optional[str]        # English response from LLM
    final_response: Optional[str]      # Response translated back to caller lang

    # ── Verification (Chain-of-Verification) 
    verified: bool
    verification_score: Optional[float]
    verification_notes: Optional[str]

    # ── Cost tracking 
    stt_seconds: Optional[float]
    tts_chars: Optional[int]
    llm_tokens: Optional[int]          # Total tokens (prompt + completion)

    # ── Metadata 
    latency_ms: Optional[float]
    error: Optional[str]               # Non-None signals pipeline error


def make_initial_state(
    query: str,
    tenant_id: str,
    session_id: str,
    call_type: str = "inbound",
    harmful_attempt_count: int = 0,
) -> AgentState:
    """Build a fully-initialised AgentState for graph invocation.

    All optional fields are set to safe defaults so that downstream nodes
    never encounter KeyError.

    Args:
        query: Raw caller utterance.
        tenant_id: Tenant directory identifier.
        session_id: Unique call-session identifier.
        call_type: ``"inbound"`` or ``"outbound"``.
        harmful_attempt_count: Strikes from previous turns (loaded from Redis).

    Returns:
        An AgentState dict ready to pass to ``koyal_graph.invoke()``.

    Raises:
        ValueError: If query or tenant_id is empty.

    Example:
        >>> s = make_initial_state("Hello", "tenant_hdfc_bank", "s1")
        >>> s["call_type"]
        'inbound'
    """
    if not query or not query.strip():
        raise ValueError("query must be a non-empty string.")
    if not tenant_id or not tenant_id.strip():
        raise ValueError("tenant_id must be a non-empty string.")

    return AgentState(
        query=query.strip(),
        tenant_id=tenant_id.strip(),
        session_id=session_id,
        call_type=call_type,
        trace_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        detected_language=None,
        language_confidence=None,
        is_code_mixed=False,
        detection_method=None,
        safety_cleared=False,
        escalate=False,
        escalation_reason=None,
        guardrail_input_blocked=False,
        guardrail_input_reason=None,
        guardrail_output_blocked=False,
        guardrail_output_reason=None,
        guardrail_pii_masked=None,
        guardrail_hallucination_score=None,
        guardrail_pii_leaked=None,
        harmful_attempt_count=harmful_attempt_count,
        end_session=False,
        wait_for_next_input=False,
        query_english=None,
        retrieved_chunks=None,
        retrieval_context=None,
        raw_response=None,
        final_response=None,
        verified=False,
        verification_score=None,
        verification_notes=None,
        stt_seconds=None,
        tts_chars=None,
        llm_tokens=None,
        latency_ms=None,
        error=None,
    )