"""
LangGraph node: Chain-of-Verification (CoVe) faithfulness check.

What it does
────────────
After the response agent generates an English answer, this node asks
the LLM to verify that the answer is:
  1. Grounded in the retrieved context (no hallucinations).
  2. Responsive to the caller's question.
  3. Free of fabricated numbers, dates, or policies.

Output format from the LLM (JSON):
    {
      "verdict": "PASS" | "FAIL",
      "score": 0.0-1.0,
      "reason": "brief explanation"
    }

Behaviour on FAIL or error:
    - ``state["verified"]`` → False
    - ``state["verification_notes"]`` → reason string
    - The pipeline still continues — verification is advisory in Phase 2.
      A FAIL flag can be used in Phase 5 RAGAS evaluation to detect
      systematic issues.

Usage example:
    from backend.agents.verification_agent import verification_agent
    result = verification_agent({
        "query_english": "When is my EMI?",
        "retrieval_context": "EMI is on the 5th...",
        "raw_response": "Your EMI is on the 5th of every month.",
        "trace_id": "abc",
    })
    # {"verified": True, "verification_score": 0.95, "verification_notes": "..."}
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from backend.agents.state import AgentState
from backend.config import (
    AGENT_BACKOFF_BASE,
    AGENT_MAX_RETRIES,
    GROQ_API_KEY,
    GROQ_MODEL_NAME,
    GROQ_TIMEOUT,
    VERIFICATION_SCORE_THRESHOLD,
)
from backend.exceptions import VerificationError

logger = logging.getLogger(__name__)

_VERIFICATION_SYSTEM_PROMPT = """\
You are a strict faithfulness verifier for a customer service AI.
Given:
  - CONTEXT: retrieved knowledge base excerpts
  - QUESTION: the caller's question (in English)
  - ANSWER: the AI's response to verify

Evaluate whether the ANSWER is:
  1. Factually grounded in CONTEXT — no invented numbers or policies.
  2. Responsive — it actually answers QUESTION.
  3. Complete enough — it doesn't omit critical information present in CONTEXT.

Respond ONLY with valid JSON (no markdown, no explanation outside the JSON):
{
  "verdict": "PASS" or "FAIL",
  "score": <float 0.0 to 1.0>,
  "reason": "<one sentence>"
}

PASS = answer is faithful and responsive. FAIL = hallucination or non-answer.
"""

_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def verification_agent(state: AgentState) -> dict:
    """LangGraph node: verify LLM response faithfulness via Groq CoVe.

    Args:
        state: Pipeline state.  Reads ``query_english``,
               ``retrieval_context``, ``raw_response``, ``trace_id``.

    Returns:
        Partial state dict:
            ``verified``            — True if verdict == "PASS".
            ``verification_score``  — Float 0–1 faithfulness score.
            ``verification_notes``  — LLM's reasoning.

    Example:
        >>> verification_agent({"raw_response": "EMI is on the 5th.", ...})
        {"verified": True, "verification_score": 0.95, "verification_notes": "..."}
    """
    query_english: str = state.get("query_english") or state.get("query", "")
    context: str = state.get("retrieval_context") or "No context."
    raw_response: str = state.get("raw_response") or ""
    trace_id: str = state.get("trace_id", "?")

    if not raw_response.strip():
        logger.warning("[%s] verification_agent: empty response — marking unverified.", trace_id)
        return {
            "verified": False,
            "verification_score": 0.0,
            "verification_notes": "Empty response — nothing to verify.",
        }

    user_content = (
        f"CONTEXT:\n{context[:2000]}\n\n"   # truncate to avoid token overflow
        f"QUESTION:\n{query_english}\n\n"
        f"ANSWER:\n{raw_response}"
    )

    logger.info("[%s] Running Chain-of-Verification...", trace_id)

    last_error: Optional[Exception] = None
    for attempt in range(AGENT_MAX_RETRIES):
        try:
            llm = _get_verification_llm()
            response = llm.invoke([
                SystemMessage(content=_VERIFICATION_SYSTEM_PROMPT),
                HumanMessage(content=user_content),
            ])
            raw_text: str = response.content.strip()

            # Parse JSON verdict
            verdict, score, reason = _parse_verdict(raw_text, trace_id)
            verified = verdict == "PASS" and score >= VERIFICATION_SCORE_THRESHOLD

            logger.info(
                "[%s] Verification: verdict=%s score=%.2f verified=%s",
                trace_id, verdict, score, verified,
            )

            return {
                "verified": verified,
                "verification_score": score,
                "verification_notes": reason,
            }

        except VerificationError:
            raise
        except Exception as exc:
            last_error = exc
            logger.warning(
                "[%s] Verification attempt %d/%d failed: %s",
                trace_id, attempt + 1, AGENT_MAX_RETRIES, exc,
            )
            if attempt < AGENT_MAX_RETRIES - 1:
                time.sleep(AGENT_BACKOFF_BASE**attempt)

    # Graceful degradation — treat as unverified but don't block pipeline
    logger.error(
        "[%s] Verification exhausted retries: %s. Marking unverified.",
        trace_id, last_error,
    )
    return {
        "verified": False,
        "verification_score": 0.0,
        "verification_notes": f"Verification failed after retries: {last_error}",
    }


# ── Helpers 

_verification_llm: Optional[ChatGroq] = None


def _get_verification_llm() -> ChatGroq:
    global _verification_llm  # noqa: PLW0603
    if _verification_llm is None:
        if not GROQ_API_KEY:
            raise VerificationError("GROQ_API_KEY not set — cannot verify responses.")
        _verification_llm = ChatGroq(
            model=GROQ_MODEL_NAME,
            api_key=GROQ_API_KEY,
            temperature=0,          # Deterministic — verification must be consistent
            max_tokens=256,         # Verdict JSON is tiny
            timeout=GROQ_TIMEOUT,
        )
    return _verification_llm


def _parse_verdict(raw_text: str, trace_id: str) -> tuple[str, float, str]:
    """Extract verdict, score, and reason from the LLM's JSON output.

    Args:
        raw_text: Raw LLM output (expected to contain a JSON object).
        trace_id: Correlation ID for logging.

    Returns:
        Tuple ``(verdict, score, reason)``.

    Raises:
        VerificationError: If JSON is missing or malformed after all retries.
    """
    match = _JSON_RE.search(raw_text)
    if not match:
        logger.warning(
            "[%s] Verification: LLM returned no JSON — raw=%r", trace_id, raw_text[:200]
        )
        raise VerificationError(
            "Verification LLM returned no JSON block.",
            raw_output=raw_text[:200],
        )

    try:
        data = json.loads(match.group())
        verdict: str = data.get("verdict", "FAIL").upper()
        score: float = float(data.get("score", 0.0))
        reason: str = data.get("reason", "No reason provided.")
        return verdict, score, reason
    except (json.JSONDecodeError, ValueError) as exc:
        raise VerificationError(
            f"Failed to parse verification JSON: {exc}",
            raw_output=raw_text[:200],
        ) from exc