r"""
NeMo Guardrails + 3-Strike Policy Integration:
    • input_guardrails: 3-strike progressive discipline for non-emergency
      violations (jailbreak, off-topic, profanity). NEVER escalates.
    • safety_gate: SOLE escalation authority for emergencies.
    • output_guardrails: sanitizes LLM response before TTS.

3-Strike Policy:
    • Strike 1-2: Warning message, wait_for_next_input=True
    • Strike 3+: Termination, end_session=True
    • Safe after warnings: Reset counter to 0
    • Emergency: Bypasses strike system entirely

Graph topology (CLEAN — single escalation authority):

    ┌─────────────────────────────────────────────────────────────────────┐
    │                              START                                  │
    │                                │                                    │
    │                                ▼                                    │
    │                      [language_detect]                              │
    │                                │                                    │
    │                                ▼                                    │
    │                      [input_guardrails]                             │
    │                         /            \                              │
    │                   blocked?          safe?                           │
    │                      │                │                             │
    │                      ▼                ▼                             │
    │                    END          [safety_gate]                       │
    │                (polite refusal)      │                              │
    │                                 /         \                         │
    │                          escalate?      cleared?                    │
    │                              │            │                         │
    │                              ▼            ▼                         │
    │                        [escalation]  [language_bridge]              │
    │                              │            │                         │
    │                              │            ▼                         │
    │                              │       [retrieval]                    │
    │                              │            │                         │
    │                              │            ▼                         │
    │                              │        [response]                    │
    │                              │            │                         │
    │                              │            ▼                         │
    │                              │      [verification]                  │
    │                              │            │                         │
    │                              │            ▼                         │
    │                              │   [translate_response]               │
    │                              │            │                         │
    │                              └──────► [output_guardrails]           │
    │                                           │                         │
    │                                           ▼                         │
    │                                         END                         │
    └─────────────────────────────────────────────────────────────────────┘

Escalation authority: ONLY safety_gate handles emergencies.
  • input_guardrails blocks (jailbreak/PII/off-topic) → END with polite refusal
  • input_guardrails safe → safety_gate → emergency? → escalation (human)
  • No duplicate emergency detection — safety_gate owns the detector singleton

LangGraph 1.x API notes:
    • Nodes return partial state dicts (only updated fields).
    • Entry point uses ``graph.add_edge(START, "node")``.
    • ``END`` is imported from ``langgraph.graph``.
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph

from backend.config import GUARDRAILS_ENABLED
from backend.agents.escalation_handler import escalation_handler
from backend.agents.language_bridge import LanguageBridge
from backend.agents.language_detector import LanguageDetector
from backend.agents.response_agent import response_agent
from backend.agents.retrieval_agent import retrieval_agent
from backend.agents.safety_agent import safety_gate_agent
from backend.agents.state import AgentState
from backend.agents.verification_agent import verification_agent

logger = logging.getLogger(__name__)

# Module-level singletons — LangGraph node functions close over these
_detector = LanguageDetector()
_bridge = LanguageBridge()

_guardrails = None


def _get_guardrails():
    """Lazy initializer for the GuardrailsHandler singleton."""
    global _guardrails
    if _guardrails is None:
        from backend.safety.guardrails_handler import get_guardrails_handler

        _guardrails = get_guardrails_handler()
    return _guardrails


# ── Node Functions 


def language_detection_node(state: AgentState) -> dict:
    """Detect language of the caller's utterance."""
    query: str = state.get("query", "")
    trace_id: str = state.get("trace_id", "?")
    logger.info("[%s] language_detection_node: query=%r", trace_id, query[:80])

    result = _detector.detect(query)
    logger.info(
        "[%s] Detected language=%s is_code_mixed=%s method=%s confidence=%.2f",
        trace_id,
        result.get("language"),
        result.get("is_code_mixed"),
        result.get("method"),
        result.get("confidence", 1.0),
    )
    return {
        "detected_language": result.get("language", "en-IN"),
        "is_code_mixed": result.get("is_code_mixed", False),
        "detection_method": result.get("method", "script"),
        "language_confidence": result.get("confidence", 1.0),
    }


def input_guardrails_node(state: AgentState) -> dict:
    """Run input guardrails (jailbreak, PII mask, off-topic, moderation) with 3-strike policy.

    NEVER escalates — only blocks non-emergency violations or passes through.
    Emergency detection is delegated to safety_gate (single authority).

    Returns:
        Partial state dict. If blocked, sets ``final_response`` to polite
        refusal/termination and ``guardrail_input_blocked=True``. If safe,
        returns (potentially PII-masked) query for safety_gate processing.
    """
    if not GUARDRAILS_ENABLED:
        logger.info("[%s] Guardrails disabled, passing through", state.get("trace_id", "?"))
        return {}
    return _get_guardrails().input_rail_node(state)


def output_guardrails_node(state: AgentState) -> dict:
    """Run output guardrails (hallucination, PII leak, language consistency).

    Returns partial state. If blocked, rewrites ``final_response`` to a
    safe fallback before TTS. Does NOT escalate — only sanitizes output.
    """
    if not GUARDRAILS_ENABLED:
        logger.info("[%s] Guardrails disabled, passing through", state.get("trace_id", "?"))
        return {}
    
    return _get_guardrails().output_rail_node(state)


def language_bridge_node(state: AgentState) -> dict:
    """Translate query to English for LLM reasoning."""
    query: str = state.get("query", "")
    lang: str = state.get("detected_language") or "en-IN"
    trace_id: str = state.get("trace_id", "?")
    logger.info("[%s] language_bridge_node: translating lang=%s -> en-IN", trace_id, lang)
    query_english = _bridge.translate_to_english(query, lang)
    logger.debug("[%s] Translated: %r -> %r", trace_id, query[:60], query_english[:60])
    return {"query_english": query_english}


def translate_response_node(state: AgentState) -> dict:
    """Translate English LLM response back to caller's language."""
    raw_response: str = state.get("raw_response") or ""
    lang: str = state.get("detected_language") or "en-IN"
    trace_id: str = state.get("trace_id", "?")
    logger.info("[%s] translate_response_node: translating response -> %s", trace_id, lang)

    if lang == "en-IN":
        final_response = raw_response
        logger.debug("[%s] Target is English — no translation needed.", trace_id)
    else:
        final_response = _bridge.translate_to_language(raw_response, lang)
        logger.debug(
            "[%s] Translated response: %r -> %r",
            trace_id,
            raw_response[:60],
            final_response[:60],
        )

    return {"final_response": final_response}


# ── Routers 


def _route_after_input_guardrails(state: AgentState) -> Literal["safety_gate", "end"]:
    """Two-way router after input guardrails.

    input_guardrails NEVER escalates. It either:
        • Blocks (non-emergency) → END with warning/termination message
        • Passes (safe or emergency) → safety_gate

    Emergency detection is the sole responsibility of safety_gate.
    """
    trace_id: str = state.get("trace_id", "?")

    # Already terminated? End immediately
    if state.get("end_session"):
        logger.warning("[%s] Session terminated — routing to END", trace_id)
        return "end"

    # Blocked (warning or termination)? End with message in final_response
    if state.get("guardrail_input_blocked"):
        logger.info(
            "[%s] Input guardrails blocked — routing to END (warning/termination)",
            trace_id,
        )
        return "end"

    # Safe or emergency (passed through) → safety_gate
    logger.info("[%s] Input guardrails passed — routing to SAFETY_GATE", trace_id)
    return "safety_gate"


def _route_after_safety(state: AgentState) -> Literal["escalation", "language_bridge"]:
    """Conditional edge: SOLE escalation authority.

    safety_gate is the ONLY node that can trigger escalation to human agents.
    All emergency detection flows through here.
    """
    if state.get("escalate"):
        reason: str = state.get("escalation_reason") or ""
        logger.warning(
            "[%s] SAFETY_GATE escalating (reason=%r)",
            state.get("trace_id", "?"),
            reason[:80],
        )
        return "escalation"
    return "language_bridge"


# ── Graph Builder 


def build_koyal_graph():
    """Build and compile the KoyalAI LangGraph pipeline."""
    graph = StateGraph(AgentState)

    # ── Core nodes
    graph.add_node("language_detect", language_detection_node)
    graph.add_node("safety_gate", safety_gate_agent)
    graph.add_node("language_bridge", language_bridge_node)
    graph.add_node("retrieval", retrieval_agent)
    graph.add_node("response", response_agent)
    graph.add_node("verification", verification_agent)
    graph.add_node("translate_response", translate_response_node)
    graph.add_node("escalation", escalation_handler)

    # Entry point
    graph.add_edge(START, "language_detect")

    # ── Guardrails path: bypass entirely when disabled
    if GUARDRAILS_ENABLED:
        graph.add_node("input_guardrails", input_guardrails_node)
        graph.add_node("output_guardrails", output_guardrails_node)

        graph.add_edge("language_detect", "input_guardrails")
        graph.add_conditional_edges(
            "input_guardrails",
            _route_after_input_guardrails,
            {"safety_gate": "safety_gate", "end": END},
        )
        graph.add_edge("translate_response", "output_guardrails")
        graph.add_edge("output_guardrails", END)

    else:
        graph.add_edge("language_detect", "safety_gate")
        graph.add_edge("translate_response", END)


    # Safety gate: SOLE escalation authority
    graph.add_conditional_edges(
        "safety_gate",
        _route_after_safety,
        {
            "escalation": "escalation",
            "language_bridge": "language_bridge",
        },
    )

    # Happy-path continuation
    graph.add_edge("language_bridge", "retrieval")
    graph.add_edge("retrieval", "response")
    graph.add_edge("response", "verification")
    graph.add_edge("verification", "translate_response")

    # Escalation ends
    graph.add_edge("escalation", END)

    compiled = graph.compile()
    logger.info(
        "KoyalAI LangGraph compiled (guardrails=%s).",
        "enabled" if GUARDRAILS_ENABLED else "DISABLED"
    )
    return compiled


# Module-level compiled graph
koyal_graph = build_koyal_graph()