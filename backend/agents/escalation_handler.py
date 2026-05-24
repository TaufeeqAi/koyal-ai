"""
LangGraph node: language-matched emergency escalation handler.

When the safety gate routes to escalation, this node:
  1. Looks up the tenant's escalation messages (Hindi + English) from
     the tenant config.json.
  2. Selects the message matching the caller's detected language.
  3. Writes it to ``state["final_response"]`` — this is what the TTS
     engine will speak back to the caller.

Fallback chain:
    Tenant Hindi message → Tenant English message → Built-in defaults.

The handler never raises — escalation must always produce a response.

Usage example:
    from backend.agents.escalation_handler import escalation_handler
    result = escalation_handler({
        "tenant_id": "tenant_hdfc_bank",
        "detected_language": "hi-IN",
        "escalation_reason": "Emergency keyword: 'आत्महत्या'",
        "trace_id": "abc",
    })
    # {"final_response": "मैं आपको हमारे वरिष्ठ अधिकारी से..."}
"""

from __future__ import annotations

import logging

from backend.agents.state import AgentState
from backend.config import load_tenant_config

logger = logging.getLogger(__name__)

# Built-in fallback messages when tenant config is absent or incomplete
_DEFAULT_HINDI_ESCALATION = (
    "मैं आपको हमारे वरिष्ठ अधिकारी से जोड़ रहा हूँ। "
    "कृपया लाइन पर रहें। आपकी मदद जल्द होगी।"
)
_DEFAULT_ENGLISH_ESCALATION = (
    "I am connecting you to our senior officer immediately. "
    "Please hold the line. Help is on the way."
)
_DEFAULT_BANKING_ESCALATION_HINDI = (
    "आपके खाते की सुरक्षा के लिए हम तत्काल कार्रवाई कर रहे हैं। "
    "कृपया 1800-202-6161 पर कॉल करें या लाइन पर रहें।"
)
_DEFAULT_BANKING_ESCALATION_ENGLISH = (
    "We are taking immediate action to secure your account. "
    "Please call 1800-202-6161 or stay on the line."
)


def escalation_handler(state: AgentState) -> dict:
    """LangGraph node: produce a language-appropriate escalation response.

    Args:
        state: Pipeline state.  Reads ``tenant_id``, ``detected_language``,
               ``escalation_reason``, ``trace_id``.

    Returns:
        Partial state dict:
            ``final_response`` — escalation message in caller's language.
            ``escalate``       — True (confirmed).
            ``safety_cleared`` — False (confirmed).

    Example:
        >>> escalation_handler({"detected_language": "hi-IN", ...})
        {"final_response": "मैं आपको हमारे वरिष्ठ अधिकारी से...", ...}
    """
    tenant_id: str = state.get("tenant_id", "")
    detected_language: str = state.get("detected_language") or "en-IN"
    escalation_reason: str = state.get("escalation_reason") or "Emergency"
    trace_id: str = state.get("trace_id", "?")

    logger.warning(
        "[%s] ESCALATION for tenant=%s lang=%s reason=%r",
        trace_id, tenant_id, detected_language, escalation_reason[:100],
    )

    # Determine whether this is a banking fraud emergency
    is_banking_emergency = _is_banking_emergency(escalation_reason)

    # Load tenant-specific escalation messages
    msg_hindi, msg_english = _load_tenant_escalation(
        tenant_id, is_banking=is_banking_emergency
    )

    # Select message matching detected language
    is_hindi = "hi" in detected_language
    final_response = msg_hindi if is_hindi else msg_english

    logger.info(
        "[%s] Escalation response selected (lang=%s, banking=%s): %r",
        trace_id, detected_language, is_banking_emergency, final_response[:80],
    )

    return {
        "final_response": final_response,
        "escalate": True,
        "safety_cleared": False,
    }


def _load_tenant_escalation(tenant_id: str, is_banking: bool) -> tuple[str, str]:
    """Load escalation messages from tenant config, with fallback to defaults.

    Banking emergencies use dedicated banking keys if present:
        - escalation_message_banking_hindi
        - escalation_message_banking_english

    For banking emergencies, generic escalation keys are intentionally
    ignored so that a tenant's default "senior officer" message does not
    override the banking-specific "account security" message.

    Args:
        tenant_id: Tenant identifier.
        is_banking: Whether to prefer banking-specific escalation messages.

    Returns:
        Tuple ``(hindi_message, english_message)``.
    """
    try:
        cfg = load_tenant_config(tenant_id)

        if is_banking:
            # Banking-specific keys take priority for banking emergencies
            hindi_msg = cfg.get("escalation_message_banking_hindi")
            english_msg = cfg.get("escalation_message_banking_english")
            if hindi_msg and english_msg:
                return hindi_msg, english_msg
            # If tenant lacks banking-specific keys, use banking defaults
            return _DEFAULT_BANKING_ESCALATION_HINDI, _DEFAULT_BANKING_ESCALATION_ENGLISH

        # Non-banking: use generic keys with generic defaults
        hindi_msg = cfg.get(
            "escalation_message_hindi", _DEFAULT_HINDI_ESCALATION
        )
        english_msg = cfg.get(
            "escalation_message_english", _DEFAULT_ENGLISH_ESCALATION
        )
        return hindi_msg, english_msg

    except Exception as exc:
        logger.warning(
            "Could not load tenant escalation config for '%s': %s — using defaults.",
            tenant_id, exc,
        )
        if is_banking:
            return _DEFAULT_BANKING_ESCALATION_HINDI, _DEFAULT_BANKING_ESCALATION_ENGLISH
        return _DEFAULT_HINDI_ESCALATION, _DEFAULT_ENGLISH_ESCALATION


def _is_banking_emergency(escalation_reason: str) -> bool:
    """Check if the escalation is triggered by a banking/fraud keyword."""
    banking_signals = [
        "fraud", "hack", "OTP", "unauthorized", "खाता खाली",
        "पैसे चोरी", "account hack", "धोखाधड़ी", "account hacked",
        "money stolen", "unauthorized transaction",
    ]
    reason_lower = escalation_reason.lower()
    return any(sig.lower() in reason_lower for sig in banking_signals)