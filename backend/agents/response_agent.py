"""
LangGraph node: generate English response via Groq Llama 3.3-70B.

Design decisions
────────────────
* Temperature = 0 — deterministic, auditable responses for BFSI and
  e-commerce where factual precision is non-negotiable.
* Context-first prompt — retrieved chunks are injected before the query
  so the model has the knowledge base in full view.
* Tenant persona — each tenant's ``system_prompt_extras`` is appended to
  the base system prompt (loaded from config.json at runtime).
* Token tracking — ``usage_metadata`` from the response is stored in
  state for cost accounting.

Retry policy:
    Groq calls retry up to AGENT_MAX_RETRIES times with exponential
    backoff.  ResponseGenerationError is raised after exhaustion.

Usage example:
    from backend.agents.response_agent import response_agent
    result = response_agent({
        "query_english": "When is my EMI deducted?",
        "retrieval_context": "[Source 1 — English]\\nEMI is on the 5th...",
        "tenant_id": "tenant_hdfc_bank",
        "detected_language": "hi-IN",
    })
    # {"raw_response": "Your EMI is automatically...", "llm_tokens": 243}
"""

from __future__ import annotations
import backend.groq_patch
import logging
import time
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from backend.agents.state import AgentState
from backend.config import (
    AGENT_BACKOFF_BASE,
    AGENT_MAX_RETRIES,
    GROQ_API_KEY,
    GROQ_MAX_TOKENS,
    GROQ_MODEL_NAME,
    GROQ_TEMPERATURE,
    GROQ_TIMEOUT,
    load_tenant_config,
)
from backend.exceptions import GroqAPIError, ResponseGenerationError

logger = logging.getLogger(__name__)

_BASE_SYSTEM_PROMPT = """\
You are KoyalAI, an intelligent customer service agent.
Answer the caller's question accurately and concisely using ONLY the provided \
context sections below.

Rules:
1. If the answer is not in the context, say: "I don't have that information right now. \
Please call our helpline for assistance."
2. Be factual — do not invent numbers, dates, or policies.
3. Keep your response under 100 words unless a longer answer is essential.
4. Always respond in English — translation to the caller's language happens separately.
5. Use a warm, professional, respectful tone.
"""


def _build_llm() -> ChatGroq:
    """Construct a ChatGroq instance from centralised config.

    Returns:
        Configured ChatGroq client.

    Raises:
        GroqAPIError: If GROQ_API_KEY is not set.
    """
    if not GROQ_API_KEY:
        raise GroqAPIError("GROQ_API_KEY is not set — cannot initialise ChatGroq.")
    return ChatGroq(
        model=GROQ_MODEL_NAME,
        api_key=GROQ_API_KEY,
        temperature=GROQ_TEMPERATURE,
        max_tokens=GROQ_MAX_TOKENS,
        timeout=GROQ_TIMEOUT,
    )


# Module-level singleton — avoids reconstructing on every call
_llm: Optional[ChatGroq] = None


def _get_llm() -> ChatGroq:
    global _llm  # noqa: PLW0603
    if _llm is None:
        _llm = _build_llm()
    return _llm


def response_agent(state: AgentState) -> dict:
    """LangGraph node: generate a grounded English response via Groq.

    Args:
        state: Pipeline state.  Reads ``query_english``,
               ``retrieval_context``, ``tenant_id``, ``trace_id``.

    Returns:
        Partial state dict:
            ``raw_response`` — English answer from the LLM.
            ``llm_tokens``   — Total tokens consumed (prompt + completion).

    Raises:
        ResponseGenerationError: After all retries are exhausted.

    Example:
        >>> response_agent({"query_english": "EMI date?", ...})
        {"raw_response": "Your EMI is deducted on the 5th ...", "llm_tokens": 197}
    """
    query_english: str = state.get("query_english") or state.get("query", "")
    context: str = state.get("retrieval_context") or "No context available."
    tenant_id: str = state.get("tenant_id", "")
    trace_id: str = state.get("trace_id", "?")
    detected_language: str = state.get("detected_language") or "en-IN"

    if not query_english.strip():
        logger.warning("[%s] response_agent: empty query — returning fallback.", trace_id)
        return {
            "raw_response": "I'm sorry, I didn't catch your question. Could you please repeat?",
            "llm_tokens": 0,
        }

    # Build tenant-aware system prompt
    system_prompt = _build_system_prompt(tenant_id, detected_language)

    # Build user message with context + query
    user_message = (
        f"KNOWLEDGE BASE:\n{context}\n\n"
        f"CALLER'S QUESTION:\n{query_english}"
    )

    logger.info(
        "[%s] Calling Groq %s for tenant=%s (context=%d chars)",
        trace_id, GROQ_MODEL_NAME, tenant_id, len(context),
    )

    last_error: Optional[Exception] = None
    for attempt in range(AGENT_MAX_RETRIES):
        try:
            llm = _get_llm()
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ])

            raw_response: str = response.content.strip()
            token_usage: int = 0
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                token_usage = response.usage_metadata.get("total_tokens", 0)

            logger.info(
                "[%s] Groq response: %d chars, %d tokens",
                trace_id, len(raw_response), token_usage,
            )
            logger.debug("[%s] Raw LLM output: %r", trace_id, raw_response[:200])

            return {
                "raw_response": raw_response,
                "llm_tokens": token_usage,
            }

        except GroqAPIError:
            raise  

        except Exception as exc:
            last_error = exc
            logger.warning(
                "[%s] Groq call failed (attempt %d/%d): %s",
                trace_id, attempt + 1, AGENT_MAX_RETRIES, exc,
            )
            if attempt < AGENT_MAX_RETRIES - 1:
                sleep_for = AGENT_BACKOFF_BASE**attempt
                logger.debug("[%s] Retrying Groq in %.1fs...", trace_id, sleep_for)
                time.sleep(sleep_for)

    raise ResponseGenerationError(
        f"Groq response generation failed after {AGENT_MAX_RETRIES} attempts.",
        tenant_id=tenant_id,
        last_error=str(last_error),
    )


def _build_system_prompt(tenant_id: str, detected_language: str) -> str:
    """Build a tenant-personalised system prompt.

    Appends the tenant's ``system_prompt_extras`` from config.json.
    Falls back to the base prompt if the config cannot be loaded.

    Args:
        tenant_id: Tenant identifier.
        detected_language: Detected caller language (informational only).

    Returns:
        Full system prompt string.
    """
    extras = ""
    try:
        cfg = load_tenant_config(tenant_id)
        extras = cfg.get("system_prompt_extras", "")
        company = cfg.get("company_name", tenant_id)
    except Exception as exc:
        logger.warning(
            "Could not load tenant config for '%s' — using base prompt: %s",
            tenant_id, exc,
        )
        company = tenant_id

    prompt_parts = [
        _BASE_SYSTEM_PROMPT,
        f"You are speaking on behalf of {company}.",
    ]
    if extras:
        prompt_parts.append(extras)

    return "\n\n".join(prompt_parts)