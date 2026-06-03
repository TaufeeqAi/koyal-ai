from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Env var configuration 
_LANGFUSE_BASE_URL: str = (
    os.getenv("LANGFUSE_BASE_URL")
    or os.getenv("LANGFUSE_HOST")
    or "http://localhost:3001"
)
_LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY") or ""
_LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY") or ""

if not os.getenv("LANGFUSE_BASE_URL") and os.getenv("LANGFUSE_HOST"):
    os.environ["LANGFUSE_BASE_URL"] = _LANGFUSE_BASE_URL

_langfuse_client: Optional[Any] = None
_langfuse_available: bool = False


# ── NoOp implementations 

class _NoOpCallbackHandler:
    """NoOp LangChain/LangGraph CallbackHandler.

    Implements the same interface as langfuse.langchain.CallbackHandler
    but discards all calls. Returned when Langfuse is disabled or unreachable.
    """
    ignore_chain: bool = False
    ignore_llm: bool = False
    ignore_agent: bool = False
    ignore_retriever: bool = False
    ignore_chat_model: bool = False
    raise_error: bool = False
    run_inline: bool = False

    def on_llm_start(self, *args: Any, **kwargs: Any) -> None: pass
    def on_llm_end(self, *args: Any, **kwargs: Any) -> None: pass
    def on_llm_error(self, *args: Any, **kwargs: Any) -> None: pass
    def on_llm_new_token(self, *args: Any, **kwargs: Any) -> None: pass

    def on_chain_start(self, *args: Any, **kwargs: Any) -> None: pass
    def on_chain_end(self, *args: Any, **kwargs: Any) -> None: pass
    def on_chain_error(self, *args: Any, **kwargs: Any) -> None: pass

    def on_tool_start(self, *args: Any, **kwargs: Any) -> None: pass
    def on_tool_end(self, *args: Any, **kwargs: Any) -> None: pass
    def on_tool_error(self, *args: Any, **kwargs: Any) -> None: pass

    def on_retriever_start(self, *args: Any, **kwargs: Any) -> None: pass
    def on_retriever_end(self, *args: Any, **kwargs: Any) -> None: pass
    def on_retriever_error(self, *args: Any, **kwargs: Any) -> None: pass

    def on_agent_action(self, *args: Any, **kwargs: Any) -> None: pass
    def on_agent_finish(self, *args: Any, **kwargs: Any) -> None: pass

    def on_chat_model_start(self, *args: Any, **kwargs: Any) -> None: pass
    def on_chat_model_end(self, *args: Any, **kwargs: Any) -> None: pass
    def on_chat_model_error(self, *args: Any, **kwargs: Any) -> None: pass


class _NoOpLangfuseClient:
    """NoOp Langfuse client.

    Implements the subset of Langfuse() interface used by KoyalAI.
    Returned by get_langfuse_client() when Langfuse is not available.
    """

    enabled: bool = False

    def flush(self) -> None: pass
    def shutdown(self) -> None: pass

    def start_as_current_observation(self, *args: Any, **kwargs: Any):
        """Context manager NoOp — yields self."""
        from contextlib import contextmanager

        @contextmanager
        def _noop():
            yield self

        return _noop()

    def score_trace(self, *args: Any, **kwargs: Any) -> None: pass


# ── Smart span filter 

def _should_export_span(span: Any) -> bool:
    """OTel span filter. Controls which spans reach Langfuse server.

    Strategy: export LLM/AI spans and KoyalAI custom spans. Block infrastructure
    noise (httpx, redis, asyncio, starlette internals) that would flood Langfuse
    with uninformative span data.

    Returns True for:
        - Any span passing the Langfuse default filter (langfuse-sdk, gen_ai.*)
        - Any span with "koyal" in its instrumentation scope name

    Returns False for:
        - httpx transport spans
        - Redis connection spans
        - asyncio task spans
        - Starlette ASGI middleware spans
    """
    try:
        from langfuse.span_filter import is_default_export_span  # type: ignore[import-untyped]
        if is_default_export_span(span):
            return True
        scope = getattr(span, "instrumentation_scope", None)
        if scope and scope.name and "koyal" in scope.name.lower():
            return True
        return False
    except (ImportError, AttributeError):
        # If filter module unavailable, export all (safe fallback)
        return True


# ── Client initialisation 

def init_langfuse() -> Any:
    """Initialise and return the process-level Langfuse singleton.

    Gracefully degrades to _NoOpLangfuseClient when:
        - LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set
        - langfuse package is not installed
        - Server is unreachable at startup

    Call this once in FastAPI lifespan before accepting requests.
    Subsequent calls return the cached singleton (idempotent).

    Returns:
        Real Langfuse client if available, _NoOpLangfuseClient otherwise.
        Never raises.
    """
    global _langfuse_client, _langfuse_available

    if _langfuse_client is not None:
        return _langfuse_client

    if not (_LANGFUSE_PUBLIC_KEY and _LANGFUSE_SECRET_KEY):
        logger.warning(
            "Langfuse DISABLED — LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY not set. "
            "Traces will not be recorded. Set keys in .env to enable."
        )
        _langfuse_client = _NoOpLangfuseClient()
        _langfuse_available = False
        return _langfuse_client

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=_LANGFUSE_PUBLIC_KEY,
            secret_key=_LANGFUSE_SECRET_KEY,
            host=_LANGFUSE_BASE_URL,
            flush_at=int(os.getenv("LANGFUSE_FLUSH_AT", "20")),
            flush_interval=float(os.getenv("LANGFUSE_FLUSH_INTERVAL", "500")) / 1000,
            timeout=20,
            debug=False,
            should_export_span=_should_export_span,
        )
        _langfuse_available = True
        logger.info(
            "Langfuse client initialised (host=%s, key=%s...)",
            _LANGFUSE_BASE_URL,
            _LANGFUSE_PUBLIC_KEY[:12],
        )
    except Exception as exc:
        logger.warning(
            "Langfuse initialisation failed (non-fatal): %s. "
            "Traces will not be recorded. Voice pipeline unaffected.",
            exc,
        )
        _langfuse_client = _NoOpLangfuseClient()
        _langfuse_available = False

    return _langfuse_client


def get_langfuse_client() -> Any:
    """Return the Langfuse singleton. Initialises on first call.

    Always returns a valid object (real client or _NoOpLangfuseClient).
    Callers never need to null-check the return value.
    """
    global _langfuse_client
    if _langfuse_client is None:
        return init_langfuse()
    return _langfuse_client


def is_langfuse_available() -> bool:
    """Return True if the real Langfuse client is active and traces will be recorded."""
    return _langfuse_available


def make_callback_handler(
    session_id: str,
    tenant_id: str,
    trace_id: str,
    call_type: str = "inbound",
    language: str | None = None,
) -> Any:
    """Create a Langfuse LangChain/LangGraph CallbackHandler for one pipeline turn.

     The CallbackHandler constructor ONLY accepts
    `public_key` and `update_trace` as direct kwargs. All other attributes (session_id,
    trace_id, tags, tenant_id, etc.) MUST be passed via the LangChain `config` metadata
    dict when invoking the graph, NOT via the CallbackHandler constructor.

    This function now returns a plain CallbackHandler() and builds the metadata dict
    that the caller must pass in config={"metadata": {...}}.

    Usage (in instrumented_graph.py):
        handler = make_callback_handler(session_id, tenant_id, trace_id)
        result = koyal_graph.invoke(
            state,
            config={
                "callbacks": [handler],
                "metadata": handler._koyal_metadata,  # ← auto-attached by this function
            }
        )

    Returns:
        CallbackHandler with `_koyal_metadata` attribute attached, or _NoOpCallbackHandler.
    """
    if not _langfuse_available:
        return _NoOpCallbackHandler()

    try:
        from langfuse.langchain import CallbackHandler

        handler = CallbackHandler()

        # Attach metadata dict for the caller to use in config
        handler._koyal_metadata = {
            "langfuse_session_id": session_id,
            "langfuse_trace_id": trace_id,
            "langfuse_user_id": tenant_id,
            "langfuse_tags": [tenant_id, f"type:{call_type}"] + ([f"lang:{language}"] if language else []),
            "tenant_id": tenant_id,
            "call_type": call_type,
            "language": language or "unknown",
            "session_id": session_id,
        }

        return handler

    except Exception as exc:
        logger.warning("make_callback_handler: failed to create handler: %s", exc)
        return _NoOpCallbackHandler()


async def flush() -> None:
    """Flush all pending spans to Langfuse server. Call on FastAPI shutdown.

    Prevents losing the last few turns of in-flight calls when the process exits.
    Non-fatal: errors are logged and suppressed.
    """
    client = get_langfuse_client()
    try:
        if hasattr(client, "flush") and callable(client.flush):
            client.flush()
            logger.info("Langfuse: flushed pending spans on shutdown.")
    except Exception as exc:
        logger.warning("Langfuse flush error (non-fatal): %s", exc)


def score_turn(tenant_id: str, language: str, score: float) -> None:
    """Record a RAGAS score on the current Langfuse trace as a span score.

    No-op if Langfuse is not available. Non-fatal on all errors.
    """
    if not _langfuse_available:
        return
    try:
        client = get_langfuse_client()
        if hasattr(client, "score"):
            client.score(
                name="ragas_faithfulness",
                value=score,
                comment=f"language={language}, tenant={tenant_id}",
            )
    except Exception as exc:
        logger.debug("Langfuse score_turn error (non-fatal): %s", exc)