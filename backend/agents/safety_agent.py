"""
LangGraph node: multilingual safety gate.

Responsibility
──────────────
Run the MultilingualEmergencyDetector against the raw caller utterance
and update state.  If an emergency is detected:
    • ``state["escalate"]`` → True
    • ``state["escalation_reason"]`` → human-readable reason
    • ``state["safety_cleared"]`` → False

Non-emergency path:
    • ``state["safety_cleared"]`` → True
    • ``state["escalate"]`` → False

The node is intentionally stateless — the detector singleton is created
once at module level to avoid reloading LaBSE on every call.

Usage example (direct node call, e.g. in tests):
    from backend.agents.safety_agent import safety_gate_agent
    from backend.agents.state import make_initial_state
    s = make_initial_state("chest pain", "tenant_hdfc_bank", "s1")
    result = safety_gate_agent(s)
    # {"safety_cleared": False, "escalate": True, "escalation_reason": "..."}
"""

from __future__ import annotations

import logging

from backend.agents.state import AgentState
from backend.exceptions import EmergencyDetectionError
from backend.safety.emergency_keywords import get_default_detector

logger = logging.getLogger(__name__)


def safety_gate_agent(state: AgentState) -> dict:
    """LangGraph node: check query against multilingual emergency detector.

    Args:
        state: Current pipeline state.  Reads ``state["query"]``,
               ``state["tenant_id"]``, ``state["trace_id"]``.

    Returns:
        Partial state dict updating safety fields:
            ``safety_cleared``, ``escalate``, ``escalation_reason``.

    Example:
        >>> safety_gate_agent({"query": "heart attack", ...})
        {"safety_cleared": False, "escalate": True,
         "escalation_reason": "Emergency keyword: 'heart attack'"}
    """
    query: str = state.get("query", "")
    tenant_id: str = state.get("tenant_id", "unknown")
    trace_id: str = state.get("trace_id", "?")

    logger.info(
        "[%s] Safety gate running for tenant=%s query=%r",
        trace_id, tenant_id, query[:80],
    )

    try:
        detector = get_default_detector()
        is_emergency, reason = detector.is_emergency(query)
    except EmergencyDetectionError as exc:
        # Fail-safe: treat detection failure as escalation — better to
        # over-escalate than miss a real emergency.
        logger.error(
            "[%s] Emergency detection error — escalating as precaution: %s",
            trace_id, exc,
        )
        return {
            "safety_cleared": False,
            "escalate": True,
            "escalation_reason": f"Detection error (safe escalation): {exc}",
        }

    if is_emergency:
        logger.warning(
            "[%s] ESCALATING tenant=%s query=%r reason=%r",
            trace_id, tenant_id, query[:80], reason,
        )
        return {
            "safety_cleared": False,
            "escalate": True,
            "escalation_reason": reason,
        }

    logger.info("[%s] Safety cleared for tenant=%s", trace_id, tenant_id)
    return {
        "safety_cleared": True,
        "escalate": False,
        "escalation_reason": None,
    }