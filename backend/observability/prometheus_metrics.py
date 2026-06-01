from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


# ── Safe metric creation guards 


def _counter(name: str, doc: str, labels: list[str]) -> Counter:
    """Create Counter with DuplicateMetric protection."""
    try:
        return Counter(name, doc, labels)
    except ValueError:
        from prometheus_client import REGISTRY
        return REGISTRY._names_to_collectors.get(name + "_total") or Counter(name, doc, labels)


def _gauge(name: str, doc: str, labels: list[str]) -> Gauge:
    """Create Gauge with DuplicateMetric protection."""
    try:
        return Gauge(name, doc, labels)
    except ValueError:
        from prometheus_client import REGISTRY
        return REGISTRY._names_to_collectors.get(name) or Gauge(name, doc, labels)


def _histogram(name: str, doc: str, labels: list[str], buckets: list) -> Histogram:
    """Create Histogram with DuplicateMetric protection."""
    try:
        return Histogram(name, doc, labels, buckets=buckets)
    except ValueError:
        from prometheus_client import REGISTRY
        return REGISTRY._names_to_collectors.get(name) or Histogram(name, doc, labels, buckets=buckets)


# ── Label normalisation (A) ──────────────────────────────────────────────────

def _norm(label: str | None) -> str:
    """Normalise label value: lowercase, strip whitespace, default 'unknown'.

    Prevents phantom time series from case mismatches like "Hi-IN" vs "hi-in"
    or "  Tenant_HDFC_Bank  " vs "tenant_hdfc_bank".

    Args:
        label: Raw label value (may be None for unset optional fields).

    Returns:
        Normalised lowercase stripped string. Never empty — returns "unknown"
        for None, empty string, or whitespace-only input.
    """
    return (label or "unknown").lower().strip() or "unknown"


# ── Call Lifecycle Metrics 

_CALLS_TOTAL: Counter = _counter(
    "koyal_calls_total",
    "Total calls processed by the KoyalAI pipeline",
    ["tenant_id", "language", "call_type", "outcome"],
)

_ACTIVE_CALLS: Gauge = _gauge(
    "koyal_active_calls",
    "Number of currently active voice call sessions",
    ["tenant_id"],
)

_CALL_DURATION_SECONDS: Histogram = _histogram(
    "koyal_call_duration_seconds",
    "End-to-end call duration in seconds",
    ["tenant_id", "language"],
    buckets=[10, 30, 60, 120, 300, 600],
)

# ── Latency Metrics

_STT_LATENCY_MS: Histogram = _histogram(
    "koyal_stt_latency_ms",
    "Sarvam Saaras V3 STT request latency in milliseconds",
    ["tenant_id", "language"],      # ← tenant_id REQUIRED for per-tenant SLA dashboards
    buckets=[50, 100, 200, 500, 1000, 2000, 5000],
)

_LLM_LATENCY_MS: Histogram = _histogram(
    "koyal_llm_latency_ms",
    "Groq Llama LLM inference latency in milliseconds",
    ["tenant_id"],                   # ← tenant_id REQUIRED for per-tenant SLA dashboards
    buckets=[100, 200, 500, 1000, 2000, 5000, 10000],
)

_TTS_LATENCY_MS: Histogram = _histogram(
    "koyal_tts_latency_ms",
    "Sarvam Bulbul V3 TTS synthesis latency in milliseconds",
    ["tenant_id", "language"],      # ← tenant_id REQUIRED for per-tenant SLA dashboards
    buckets=[50, 100, 200, 500, 1000, 2000],
)

_TTFR_MS: Histogram = _histogram(
    "koyal_ttfr_ms",
    "Time to first response (utterance end → TTS start) in milliseconds",
    ["tenant_id", "language"],
    buckets=[200, 400, 600, 800, 1000, 1500, 2000, 3000, 5000],
)

_PIPELINE_LATENCY_MS: Histogram = _histogram(
    "koyal_pipeline_latency_ms",
    "Full STT→LangGraph→TTS pipeline latency in milliseconds",
    ["tenant_id", "language"],
    buckets=[500, 1000, 1500, 2000, 3000, 5000, 10000],
)

# ── Safety Metrics 

_EMERGENCY_ESCALATIONS_TOTAL: Counter = _counter(
    "koyal_emergency_escalations_total",
    "Emergency escalations triggered by the safety gate",
    ["tenant_id", "language", "reason_category"],   
)

_SAFETY_GATE_DECISIONS_TOTAL: Counter = _counter(
    "koyal_safety_gate_decisions_total",
    "Safety gate pass/fail decisions",
    ["tenant_id", "decision"],
)


# ── Guardrail Metrics (NeMo + 3-Strike Policy) 

_GUARDRAIL_INPUT_BLOCKS_TOTAL: Counter = _counter(
    "koyal_guardrail_input_blocks_total",
    "Input guardrail blocks (jailbreak, PII, off-topic, profanity, etc.)",
    ["tenant_id", "block_reason_category"],
)

_GUARDRAIL_OUTPUT_BLOCKS_TOTAL: Counter= _counter(
    "koyal_guardrail_output_blocks_total",
    "Output guardrail blocks (hallucination, PII leak, language mismatch)",
    ["tenant_id", "block_reason_category"],
)

_THREE_STRIKE_TERMINATIONS_TOTAL: Counter = _counter(
    "koyal_three_strike_terminations_total",
    "Sessions terminated by 3-strike policy after repeated non‑emergency violations",
    ["tenant_id"],
)

# Gauge for current harmful‑attempt count (includes session_id label)
_HARMFUL_ATTEMPTS_GAUGE: Gauge= _gauge(
    "koyal_harmful_attempts_current",
    "Current harmful-attempt counter for the active session (resets after each safe turn)",
    ["tenant_id", "session_id"],
)


# ── Language Metrics 

_LANGUAGE_DETECTIONS_TOTAL: Counter = _counter(
    "koyal_language_detections_total",
    "Detected languages per utterance",
    ["tenant_id", "language", "is_code_mixed"],     # ← tenant_id REQUIRED
)

# ── Cost Metrics (₹ INR) 

_COST_INR_TOTAL: Counter = _counter(
    "koyal_cost_inr_total",
    "Cumulative cost in Indian Rupees by tenant and service type",
    ["tenant_id", "cost_type"],
)

# ── Quality Metrics 

_RAGAS_FAITHFULNESS: Gauge = _gauge(
    "koyal_ragas_faithfulness",
    "Latest RAGAS faithfulness score (0.0–1.0). Written by Phase 5 evaluation runs.",
    ["tenant_id", "language"],
)

_RETRIEVAL_RELEVANCE_SCORE: Histogram = _histogram(
    "koyal_retrieval_relevance_score",
    "Qdrant retrieval chunk rerank score distribution",
    ["tenant_id"],
    buckets=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0],
)

_VERIFICATION_SCORE: Histogram = _histogram(
    "koyal_verification_score",
    "Chain-of-Verification faithfulness score distribution (0.0–1.0)",
    ["tenant_id"],
    buckets=[0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

_ASR_CONFIDENCE: Gauge = _gauge(
    "koyal_asr_confidence",
    "Sarvam STT confidence score for the most recent utterance (0.0–1.0)",
    ["tenant_id", "language"],
)

# ── HTTP Metrics 

_HTTP_REQUESTS_TOTAL: Counter = _counter(
    "koyal_http_requests_total",
    "Total HTTP requests served",
    ["method", "endpoint", "status_code"],
)

_HTTP_REQUEST_DURATION_SECONDS: Histogram = _histogram(
    "koyal_http_request_duration_seconds",
    "HTTP request processing duration in seconds (Prometheus base unit)",
    ["method", "endpoint"],
    buckets=[0.005, 0.010, 0.025, 0.050, 0.100, 0.250, 0.500, 1.0, 2.5, 5.0],
)

# ── WebSocket Metrics 

_WS_CONNECTIONS_TOTAL: Counter = _counter(
    "koyal_ws_connections_total",
    "Total WebSocket voice connections established (session churn rate)",
    ["tenant_id"],
)

_WS_ACTIVE_CONNECTIONS: Gauge = _gauge(
    "koyal_ws_active_connections",
    "Currently active WebSocket voice connections",
    ["tenant_id"],
)

# ── Current Cost Gauge (synced from Redis) 
_COST_INR_CURRENT: Gauge = _gauge(
    "koyal_cost_inr_current",
    "Current accumulated cost in Indian Rupees per tenant (synced every 30s)",
    ["tenant_id", "cost_type"],
)

# ── Typed metrics dataclass 

@dataclass(frozen=True)
class _KoyalMetrics:
    """
    Typed container for all KoyalAI Prometheus metric instruments.

    All callers access metrics through the module-level METRICS singleton.
    The frozen dataclass ensures no metric can be accidentally replaced.

    Type safety advantage over flat module constants:
        METRICS.sst_latency    → mypy error at import (typo caught at parse time)
        STT_LATENCY_MS_TYPO    → AttributeError at runtime on first scrape

    Usage:
        from backend.observability.prometheus_metrics import METRICS
        METRICS.stt_latency.labels(tenant_id="tenant_hdfc_bank", language="hi-in").observe(250.0)
        METRICS.calls_total.labels(...).inc()
    """
    calls_total: Counter
    call_duration: Histogram
    active_calls: Gauge
    stt_latency: Histogram
    llm_latency: Histogram
    tts_latency: Histogram
    ttfr: Histogram
    pipeline_latency: Histogram
    asr_confidence: Gauge
    ragas_faithfulness: Gauge
    retrieval_relevance: Histogram
    verification_score: Histogram
    cost_inr: Counter
    cost_inr_current: Gauge  
    emergency_escalations: Counter
    safety_gate_decisions: Counter
    language_detections: Counter
    http_requests_total: Counter
    http_request_duration: Histogram
    ws_connections_total: Counter
    ws_active_connections: Gauge
    guardrail_input_blocks: Counter
    guardrail_output_blocks: Counter
    three_strike_terminations: Counter
    harmful_attempts_gauge: Gauge


# ── Module-level singleton 

METRICS = _KoyalMetrics(
    calls_total=_CALLS_TOTAL,
    call_duration=_CALL_DURATION_SECONDS,
    active_calls=_ACTIVE_CALLS,
    stt_latency=_STT_LATENCY_MS,
    llm_latency=_LLM_LATENCY_MS,
    tts_latency=_TTS_LATENCY_MS,
    ttfr=_TTFR_MS,
    pipeline_latency=_PIPELINE_LATENCY_MS,
    asr_confidence=_ASR_CONFIDENCE,
    ragas_faithfulness=_RAGAS_FAITHFULNESS,
    retrieval_relevance=_RETRIEVAL_RELEVANCE_SCORE,
    verification_score=_VERIFICATION_SCORE,
    cost_inr=_COST_INR_TOTAL,
    cost_inr_current=_COST_INR_CURRENT,
    emergency_escalations=_EMERGENCY_ESCALATIONS_TOTAL,
    safety_gate_decisions=_SAFETY_GATE_DECISIONS_TOTAL,
    language_detections=_LANGUAGE_DETECTIONS_TOTAL,
    http_requests_total=_HTTP_REQUESTS_TOTAL,
    http_request_duration=_HTTP_REQUEST_DURATION_SECONDS,
    ws_connections_total=_WS_CONNECTIONS_TOTAL,
    ws_active_connections=_WS_ACTIVE_CONNECTIONS,
    guardrail_input_blocks=_GUARDRAIL_INPUT_BLOCKS_TOTAL,
    guardrail_output_blocks=_GUARDRAIL_OUTPUT_BLOCKS_TOTAL,
    three_strike_terminations=_THREE_STRIKE_TERMINATIONS_TOTAL,
    harmful_attempts_gauge=_HARMFUL_ATTEMPTS_GAUGE,
)


# ── Scrape endpoint helper 

def get_metrics_response() -> tuple[bytes, str]:
    """Generate the Prometheus text-format scrape response.

    Typical response time: <1ms (generate_latest is in-process, no network).
    Used by backend/main.py PrometheusMiddleware exclusion and test assertions.
    """
    return generate_latest(), CONTENT_TYPE_LATEST


# ── Escalation reason categorisation 

def _categorise_escalation_reason(reason: str) -> str:
    """Map a raw escalation reason string to a closed-enum category label.

    CRITICAL: Raw reason strings must NEVER reach Prometheus label values.
    Even with [:50] truncation, free-text labels cause TSDB cardinality
    explosion at production scale (10K+ unique reasons = 10K+ time series
    per escalation metric).

    This function maps all reasons to exactly 6 values:
        "keyword_medical"   — medical emergency keywords
        "keyword_banking"   — financial fraud/security keywords
        "keyword_hinglish"  — Hinglish emergency cross-language keywords
        "semantic"          — semantic similarity threshold match
        "error"             — detection error / malfunction
        "other"             — none of the above

    Args:
        reason: Raw reason string from the safety gate (any length).

    Returns:
        One of the 6 category strings above. Never raises.
    """
    reason_lower = reason.lower()
    if any(kw in reason_lower for kw in [
        "medical", "दिल", "heart", "chest", "breath", "stroke",
        "suicide", "suicidal", "overdose", "behosh", "ambulance",
        "आत्महत्या", "jaan", "marna", "khoon",
    ]):
        return "keyword_medical"
    if any(kw in reason_lower for kw in [
        "fraud", "hack", "otp", "unauthorized", "dhokha",
        "chori", "bank", "account", "upi", "password",
    ]):
        return "keyword_banking"
    if any(kw in reason_lower for kw in ["hospital", "sans", "dard"]):
        return "keyword_hinglish"
    if "semantic" in reason_lower:
        return "semantic"
    if "error" in reason_lower or "detection" in reason_lower:
        return "error"
    return "other"


# ── Guardrail reason categorisation 

def _categorise_guardrail_reason(reason: str) -> str:
    """Map raw guardrail reason to a closed category (max 7 values)."""
    r = reason.lower()
    if "jailbreak" in r or "prompt injection" in r:
        return "jailbreak"
    if "pii" in r or "personal info" in r or "aadhaar" in r or "phone" in r or "email" in r:
        return "pii_detected"
    if "off-topic" in r or "off_topic" in r:
        return "off_topic"
    if "profanity" in r or "toxic" in r:
        return "profanity"
    if "hallucination" in r or "factual" in r:
        return "hallucination"
    if "language" in r or "script" in r:
        return "language_mismatch"
    return "other"

# ── Recording Helpers 

def record_call_start(tenant_id: str, language: str, call_type: str = "inbound") -> None:
    """Increment active_calls gauge and WebSocket connection counters."""
    t = _norm(tenant_id)
    METRICS.active_calls.labels(tenant_id=t).inc()
    METRICS.ws_connections_total.labels(tenant_id=t).inc()
    METRICS.ws_active_connections.labels(tenant_id=t).inc()
    logger.debug("metrics: call_start tenant=%s lang=%s type=%s", t, _norm(language), call_type)


def record_call_end(
    tenant_id: str,
    language: str,
    duration_seconds: float,
    call_type: str = "inbound",
    outcome: str = "completed",
) -> None:
    """Record call completion. Decrements active gauges."""
    t, l = _norm(tenant_id), _norm(language)
    METRICS.calls_total.labels(
        tenant_id=t, language=l, call_type=call_type, outcome=outcome
    ).inc()
    METRICS.call_duration.labels(tenant_id=t, language=l).observe(duration_seconds)
    METRICS.active_calls.labels(tenant_id=t).dec()
    METRICS.ws_active_connections.labels(tenant_id=t).dec()
    logger.debug(
        "metrics: call_end tenant=%s lang=%s duration=%.1fs outcome=%s",
        t, l, duration_seconds, outcome,
    )


def record_stt_latency(tenant_id: str, language: str, latency_ms: float) -> None:
    METRICS.stt_latency.labels(tenant_id=_norm(tenant_id), language=_norm(language)).observe(latency_ms)


def record_llm_latency(tenant_id: str, latency_ms: float) -> None:
    METRICS.llm_latency.labels(tenant_id=_norm(tenant_id)).observe(latency_ms)


def record_tts_latency(tenant_id: str, language: str, latency_ms: float) -> None:
    METRICS.tts_latency.labels(tenant_id=_norm(tenant_id), language=_norm(language)).observe(latency_ms)


def record_ttfr(tenant_id: str, language: str, latency_ms: float) -> None:
    METRICS.ttfr.labels(tenant_id=_norm(tenant_id), language=_norm(language)).observe(latency_ms)


def record_pipeline_latency(tenant_id: str, language: str, latency_ms: float) -> None:
    METRICS.pipeline_latency.labels(
        tenant_id=_norm(tenant_id), language=_norm(language)
    ).observe(latency_ms)


def record_escalation(tenant_id: str, language: str, reason: str) -> None:
    """Record escalation with closed-enum reason_category label."""
    category = _categorise_escalation_reason(reason)
    METRICS.emergency_escalations.labels(
        tenant_id=_norm(tenant_id),
        language=_norm(language),
        reason_category=category,
    ).inc()
    METRICS.safety_gate_decisions.labels(
        tenant_id=_norm(tenant_id), decision="escalated"
    ).inc()
    logger.debug("metrics: escalation tenant=%s lang=%s category=%s", tenant_id, language, category)


def record_safety_cleared(tenant_id: str) -> None:
    METRICS.safety_gate_decisions.labels(tenant_id=_norm(tenant_id), decision="cleared").inc()


def record_language_detection(tenant_id: str, language: str, is_code_mixed: bool) -> None:
    METRICS.language_detections.labels(
        tenant_id=_norm(tenant_id),
        language=_norm(language),
        is_code_mixed=str(is_code_mixed).lower(),
    ).inc()


def record_cost_inr(tenant_id: str, cost_type: str, amount_inr: float) -> None:
    METRICS.cost_inr.labels(tenant_id=_norm(tenant_id), cost_type=cost_type).inc(amount_inr)


def update_ragas_score(tenant_id: str, language: str, faithfulness_score: float) -> None:
    """Update RAGAS faithfulness gauge. Called by Phase 5 evaluation runs."""
    METRICS.ragas_faithfulness.labels(
        tenant_id=_norm(tenant_id), language=_norm(language)
    ).set(faithfulness_score)


def record_retrieval_score(tenant_id: str, score: float) -> None:
    METRICS.retrieval_relevance.labels(tenant_id=_norm(tenant_id)).observe(score)


def record_verification_score(tenant_id: str, score: float) -> None:
    METRICS.verification_score.labels(tenant_id=_norm(tenant_id)).observe(score)


def record_asr_confidence(tenant_id: str, language: str, confidence: float) -> None:
    METRICS.asr_confidence.labels(
        tenant_id=_norm(tenant_id), language=_norm(language)
    ).set(confidence)

# ── Guardrail Recording Helpers 

def record_guardrail_input_block(tenant_id: str, reason: str) -> None:
    """Increment input guardrail block counter with categorised reason."""
    category = _categorise_guardrail_reason(reason)
    METRICS.guardrail_input_blocks.labels(
        tenant_id=_norm(tenant_id), block_reason_category=category
    ).inc()
    logger.debug("guardrail_input_block: tenant=%s category=%s", tenant_id, category)


def record_guardrail_output_block(tenant_id: str, reason: str) -> None:
    """Increment output guardrail block counter."""
    category = _categorise_guardrail_reason(reason)
    METRICS.guardrail_output_blocks.labels(
        tenant_id=_norm(tenant_id), block_reason_category=category
    ).inc()
    logger.debug("guardrail_output_block: tenant=%s category=%s", tenant_id, category)


def record_three_strike_termination(tenant_id: str) -> None:
    """Increment 3‑strike termination counter."""
    METRICS.three_strike_terminations.labels(tenant_id=_norm(tenant_id)).inc()
    logger.warning("3‑strike termination: tenant=%s", tenant_id)


def update_harmful_attempts(tenant_id: str, session_id: str, count: int) -> None:
    """Set the current harmful-attempt count for an active session."""
    METRICS.harmful_attempts_gauge.labels(
        tenant_id=_norm(tenant_id), session_id=session_id
    ).set(count)
    logger.debug("harmful_attempts: tenant=%s session=%s count=%d", tenant_id, session_id, count)


def remove_harmful_attempts_gauge(tenant_id: str, session_id: str) -> None:
    """Remove the gauge time series for a finished session (prevents leak)."""
    from prometheus_client import REGISTRY
    try:
        sample = METRICS.harmful_attempts_gauge.labels(
            tenant_id=_norm(tenant_id), session_id=session_id
        )
        # The gauge collector is named "koyal_harmful_attempts_current"
        collector = REGISTRY._names_to_collectors.get("koyal_harmful_attempts_current")
        if collector:
            # Remove the specific label set
            collector.remove(*sample._metrics.keys())
    except Exception as e:
        logger.debug("Failed to remove harmful_attempts gauge: %s", e)

def update_current_cost(tenant_id: str, cost_type: str, amount_inr: float) -> None:
    """Update the current cost gauge (used by background sync)."""
    METRICS.cost_inr_current.labels(
        tenant_id=_norm(tenant_id), cost_type=cost_type
    ).set(amount_inr)

# ── ASGI Middleware 

class PrometheusMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that records Prometheus HTTP metrics for every request.

    Records:
        koyal_http_requests_total{method, endpoint, status_code}
        koyal_http_request_duration_seconds{method, endpoint}

    Excludes internal/operational paths from metric recording to prevent
    cardinality growth from health checks and Prometheus scrapes themselves.

    Path normalisation:
        /ws/tenant_hdfc_bank/sess_001  →  /ws/{tenant_id}/{session_id}
        /api/costs/tenant_hdfc_bank   →  /api/costs/{tenant_id}
    """

    _EXCLUDED_PATHS: frozenset[str] = frozenset({
        "/metrics", "/health", "/ready", "/docs", "/openapi.json", "/redoc",
    })

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        path = request.url.path

        if path in self._EXCLUDED_PATHS:
            return await call_next(request)

        method = request.method
        normalised_path = self._normalise_path(path)
        t0 = time.perf_counter()
        status_code = "500"

        try:
            response = await call_next(request)
            status_code = str(response.status_code)
            return response
        except Exception:
            status_code = "500"
            raise
        finally:
            duration = time.perf_counter() - t0
            METRICS.http_requests_total.labels(
                method=method,
                endpoint=normalised_path,
                status_code=status_code,
            ).inc()
            METRICS.http_request_duration.labels(
                method=method,
                endpoint=normalised_path,
            ).observe(duration)

    @staticmethod
    def _normalise_path(path: str) -> str:
        """Collapse dynamic path segments to template variables.

        Prevents per-session and per-tenant path cardinality explosion.
        Without normalisation, /ws/{unique_session_id} creates one time series
        per session — millions at production scale.
        """
        parts = path.strip("/").split("/")
        if not parts or parts == [""]:
            return "/"

        normalised: list[str] = [parts[0]]
        for part in parts[1:]:
            if part.startswith("tenant_"):
                normalised.append("{tenant_id}")
            elif len(part) == 36 and part.count("-") == 4:
                # UUID v4 session IDs
                normalised.append("{session_id}")
            else:
                normalised.append(part)
        return "/" + "/".join(normalised)


# ── Module-level aliases 
CALLS_TOTAL    = _CALLS_TOTAL
COST_INR_TOTAL = _COST_INR_TOTAL
COST_INR_GAUGE = _COST_INR_CURRENT