"""
LangGraph node: multilingual Qdrant retrieval with NeMo Guardrails validation.

Reads ``state["query_english"]`` (translated) for retrieval — using the
English query against the cross-lingual LaBSE embedding space gives the
best recall for both Hindi and English document chunks.

Falls back to the original ``state["query"]`` if no English translation
was produced (e.g. because the query was already in English or the bridge
was disabled).

Guardrails Integration:
    After raw retrieval, ``guardrails.process_retrieval()`` validates chunks:
        • Tenant isolation check (drop cross-tenant contamination)
        • Relevance threshold filter (score >= 0.20)
        • PII redaction in chunk text (deterministic regex)

    PII redaction metadata is tracked in ``guardrail_pii_masked`` so
    output rails can cross-check for leakage against what was already masked.

Usage example:
    from backend.agents.retrieval_agent import retrieval_agent
    result = retrieval_agent({
        "query": "EMI payment date",
        "query_english": "When is my EMI deducted?",
        "tenant_id": "tenant_hdfc_bank",
        "trace_id": "abc",
    })
    # {
    #   "retrieved_chunks": [...],
    #   "retrieval_context": "[Source 1 — English | score=0.912]\n...",
    #   "guardrail_pii_masked": ["PAN: ABCDE1234F"],
    # }
"""

from __future__ import annotations

import logging

from backend.agents.state import AgentState
from backend.exceptions import RetrievalError
from backend.rag.retriever import MultilingualRetriever
from backend.safety.guardrails_handler import get_guardrails_handler

logger = logging.getLogger(__name__)

# Module-level singletons — shared across all invocations in the same process
_retriever: MultilingualRetriever | None = None
_guardrails = None  # Lazy-initialized below


def _get_retriever() -> MultilingualRetriever:
    """Lazily initialise and return the shared retriever singleton."""
    global _retriever  # noqa: PLW0603
    if _retriever is None:
        logger.info("Initialising MultilingualRetriever singleton...")
        _retriever = MultilingualRetriever()
    return _retriever


def _get_guardrails():
    """Lazily initialise and return the shared guardrails singleton."""
    global _guardrails  # noqa: PLW0603
    if _guardrails is None:
        _guardrails = get_guardrails_handler()
    return _guardrails


def retrieval_agent(state: AgentState) -> dict:
    """LangGraph node: retrieve relevant chunks from the tenant Qdrant collection.

    Priority for the search query:
        1. ``state["query_english"]`` — translated English query (best recall).
        2. ``state["query"]`` — original query as fallback.

    After retrieval, runs guardrails validation:
        • Tenant isolation (drop contaminated chunks)
        • Relevance filtering (score >= 0.20)
        • PII redaction in chunk text
        • Tracks redacted entities in ``guardrail_pii_masked``

    Args:
        state: Pipeline state.  Reads ``query``, ``query_english``,
               ``tenant_id``, ``detected_language``, ``trace_id``.

    Returns:
        Partial state dict:
            ``retrieved_chunks`` — list of validated chunk dicts.
            ``retrieval_context`` — formatted string ready for LLM prompt.
            ``guardrail_pii_masked`` — list of PII entities redacted from chunks.

    Raises:
        RetrievalError: If Qdrant is unreachable or the collection is absent.

    Example:
        >>> retrieval_agent({"query_english": "EMI due date", ...})
        {
            "retrieved_chunks": [...],
            "retrieval_context": "[Source 1 — English | score=0.912]\n...",
            "guardrail_pii_masked": ["PAN: ABCDE1234F"],
        }
    """
    query_english: str = state.get("query_english") or state.get("query", "")
    tenant_id: str = state.get("tenant_id", "")
    detected_language: str = state.get("detected_language") or "en-IN"
    trace_id: str = state.get("trace_id", "?")

    if not query_english.strip():
        logger.warning(
            "[%s] retrieval_agent received empty query — returning empty context.",
            trace_id,
        )
        return {
            "retrieved_chunks": [],
            "retrieval_context": "No relevant information found.",
            "guardrail_pii_masked": [],
        }

    logger.info(
        "[%s] Retrieving for tenant=%s lang=%s query=%r",
        trace_id, tenant_id, detected_language, query_english[:80],
    )

    # ── Step 1: Raw retrieval from Qdrant 
    try:
        retriever = _get_retriever()
        raw_chunks = retriever.retrieve(
            query=query_english,
            tenant_id=tenant_id,
            preferred_language=detected_language,
        )
    except Exception as exc:
        raise RetrievalError(
            f"Qdrant retrieval failed for tenant '{tenant_id}': {exc}",
            tenant_id=tenant_id,
            query=query_english[:100],
        ) from exc

    if not raw_chunks:
        logger.warning(
            "[%s] No chunks retrieved for tenant=%s query=%r",
            trace_id, tenant_id, query_english[:80],
        )
        return {
            "retrieved_chunks": [],
            "retrieval_context": "No relevant information found in the knowledge base.",
            "guardrail_pii_masked": [],
        }

    # ── Step 2: Guardrails validation on retrieved chunks 
    guardrails = _get_guardrails()
    safe_chunks, pii_redacted = guardrails.process_retrieval(
        raw_chunks, tenant_id, trace_id
    )

    logger.info(
        "[%s] Retrieval guardrails: %d raw chunks -> %d safe chunks (%d PII redacted)",
        trace_id, len(raw_chunks), len(safe_chunks), len(pii_redacted),
    )

    # ── Step 3: Format context for LLM prompt 
    context = _format_context(safe_chunks)
    logger.debug(
        "[%s] Final context: %d chunks, %d chars.",
        trace_id, len(safe_chunks), len(context),
    )

    return {
        "retrieved_chunks": safe_chunks,
        "retrieval_context": context,
        "guardrail_pii_masked": pii_redacted,
    }


def _format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context string for the LLM.

    Args:
        chunks: List of validated chunk dicts with ``text`` and ``language`` keys.

    Returns:
        Formatted multi-source context string, or a default "not found" string.
    """
    if not chunks:
        return "No relevant information found in the knowledge base."

    parts: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        lang_label = "Hindi" if "hi" in chunk.get("language", "") else "English"
        score = chunk.get("rerank_score", chunk.get("score", 0.0))
        parts.append(
            f"[Source {i} — {lang_label} | score={score:.3f}]\n{chunk['text']}"
        )
    return "\n\n".join(parts)