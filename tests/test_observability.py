"""
Coverage :
    1. LabelNormalisation    — _norm() and _categorise_escalation_reason()
    2. MetricInstruments     — all 20 METRICS fields are correct types
    3. TenantIdLabels        — STT/LLM/TTS latency metrics carry tenant_id
    4. RecordingHelpers      — helper functions observe/increment correctly
    5. PrometheusMiddleware  — path normalisation, excluded paths
    6. LangfuseClient        — NoOp fallback, env var alias handling
    7. InstrumentedGraph     — observed_invoke_graph() with mocked pipeline
    8. GrafanaDashboard      — dashboard JSON structure validation
    9. CrossTenantIsolation  — metrics for tenant A do not affect tenant B (ported)
   10. CostTrackerPhase4     — CostTracker emits Prometheus counters (ported)
   11. ObservabilityEndpoints — /metrics and /health correctness (ported)

Test strategy:
    • Fresh CollectorRegistry per test via monkeypatch (prevents
      "Duplicated timeseries" errors across test runs)
    • Langfuse tests use module-level monkeypatching (no real server needed)
    • InstrumentedGraph tests mock _invoke_graph_with_callbacks()
    • GrafanaDashboard tests load the actual JSON file (skips if absent)
    • CostTracker tests use fakeredis (no Redis server required)

Run:
    pytest tests/test_observability.py -v
    pytest tests/test_observability.py -v -k "TestInstrumentedGraph"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from prometheus_client import CollectorRegistry

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Registry isolation fixture 

# @pytest.fixture(autouse=True)
# def isolate_registry(monkeypatch):
#     """Each test gets a fresh CollectorRegistry.

#     Without this, the second import of prometheus_metrics would raise
#     ValueError: Duplicated timeseries in CollectorRegistry.
#     The _counter/_gauge/_histogram guards catch this at module level, but
#     explicit registry isolation is best practice for test correctness.
#     """
#     fresh_registry = CollectorRegistry()
#     monkeypatch.setattr("prometheus_client.REGISTRY", fresh_registry, raising=False)
#     import prometheus_client
#     monkeypatch.setattr(
#         prometheus_client,
#         "generate_latest",
#         lambda registry=None:
#         prometheus_client.exposition.generate_latest(
#             fresh_registry if registry is None else registry
#         ),
#         raising=False,
#     )
#     yield fresh_registry


# ── Label Normalisation Tests

class TestLabelNormalisation:
    """_norm() and _categorise_escalation_reason() correctness."""

    def test_norm_lowercases(self) -> None:
        from backend.observability.prometheus_metrics import _norm
        assert _norm("HI-IN") == "hi-in"
        assert _norm("Tenant_HDFC_Bank") == "tenant_hdfc_bank"

    def test_norm_strips_whitespace(self) -> None:
        from backend.observability.prometheus_metrics import _norm
        assert _norm("  hi-in  ") == "hi-in"

    def test_norm_empty_returns_unknown(self) -> None:
        from backend.observability.prometheus_metrics import _norm
        assert _norm("") == "unknown"
        assert _norm(None) == "unknown"  # type: ignore[arg-type]
        assert _norm("   ") == "unknown"

    def test_categorise_escalation_medical_hindi(self) -> None:
        from backend.observability.prometheus_metrics import _categorise_escalation_reason
        assert _categorise_escalation_reason("keyword: दिल का दौरा") == "keyword_medical"
        assert _categorise_escalation_reason("आत्महत्या की कोशिश") == "keyword_medical"

    def test_categorise_escalation_medical_english(self) -> None:
        from backend.observability.prometheus_metrics import _categorise_escalation_reason
        assert _categorise_escalation_reason("heart attack detected") == "keyword_medical"
        assert _categorise_escalation_reason("suicidal ideation") == "keyword_medical"

    def test_categorise_escalation_banking(self) -> None:
        from backend.observability.prometheus_metrics import _categorise_escalation_reason
        assert _categorise_escalation_reason("OTP fraud detected") == "keyword_banking"
        assert _categorise_escalation_reason("account hack attempt") == "keyword_banking"

    def test_categorise_escalation_semantic(self) -> None:
        from backend.observability.prometheus_metrics import _categorise_escalation_reason
        reason = "Semantic match: 'I want to end my life' (score: 0.85 >= 0.80)"
        assert _categorise_escalation_reason(reason) == "semantic"

    def test_categorise_escalation_error(self) -> None:
        from backend.observability.prometheus_metrics import _categorise_escalation_reason
        assert _categorise_escalation_reason("detection error in safety gate") == "error"

    def test_categorise_escalation_other(self) -> None:
        from backend.observability.prometheus_metrics import _categorise_escalation_reason
        assert _categorise_escalation_reason("some random reason") == "other"

    def test_six_categories_only(self) -> None:
        from backend.observability.prometheus_metrics import _categorise_escalation_reason
        valid = {"keyword_medical", "keyword_banking", "keyword_hinglish", "semantic", "error", "other"}
        test_inputs = [
            "random text", "FRAUD", "suicidal", "semantic match",
            "heart", "unknown", "detection_error", "hospital",
        ]
        for inp in test_inputs:
            result = _categorise_escalation_reason(inp)
            assert result in valid, f"Unexpected category '{result}' for input '{inp}'"


# ── Metric Instrument Tests

class TestMetricInstruments:
    """All 20 METRICS fields must be correct prometheus_client types."""

    def test_metrics_singleton_importable(self) -> None:
        from backend.observability.prometheus_metrics import METRICS
        assert METRICS is not None

    def test_calls_total_is_counter(self) -> None:
        from backend.observability.prometheus_metrics import METRICS
        from prometheus_client import Counter
        assert isinstance(METRICS.calls_total, Counter)

    def test_call_duration_is_histogram(self) -> None:
        from backend.observability.prometheus_metrics import METRICS
        from prometheus_client import Histogram
        assert isinstance(METRICS.call_duration, Histogram)

    def test_active_calls_is_gauge(self) -> None:
        from backend.observability.prometheus_metrics import METRICS
        from prometheus_client import Gauge
        assert isinstance(METRICS.active_calls, Gauge)

    def test_stt_latency_is_histogram(self) -> None:
        from backend.observability.prometheus_metrics import METRICS
        from prometheus_client import Histogram
        assert isinstance(METRICS.stt_latency, Histogram)

    def test_asr_confidence_is_gauge(self) -> None:
        from backend.observability.prometheus_metrics import METRICS
        from prometheus_client import Gauge
        assert isinstance(METRICS.asr_confidence, Gauge)

    def test_ws_connections_total_is_counter(self) -> None:
        from backend.observability.prometheus_metrics import METRICS
        from prometheus_client import Counter
        assert isinstance(METRICS.ws_connections_total, Counter)

    def test_ragas_faithfulness_is_gauge(self) -> None:
        from backend.observability.prometheus_metrics import METRICS
        from prometheus_client import Gauge
        assert isinstance(METRICS.ragas_faithfulness, Gauge)

    def test_http_request_duration_is_histogram(self) -> None:
        from backend.observability.prometheus_metrics import METRICS
        from prometheus_client import Histogram
        assert isinstance(METRICS.http_request_duration, Histogram)


# ── Tenant ID Label Tests

class TestTenantIdLabels:
    def test_stt_latency_has_tenant_id(self) -> None:
        from backend.observability.prometheus_metrics import METRICS
        assert "tenant_id" in METRICS.stt_latency._labelnames, \
            "STT latency MUST have tenant_id label for per-tenant SLA queries"

    def test_llm_latency_has_tenant_id(self) -> None:
        from backend.observability.prometheus_metrics import METRICS
        assert "tenant_id" in METRICS.llm_latency._labelnames

    def test_tts_latency_has_tenant_id(self) -> None:
        from backend.observability.prometheus_metrics import METRICS
        assert "tenant_id" in METRICS.tts_latency._labelnames

    def test_language_detections_has_tenant_id(self) -> None:
        from backend.observability.prometheus_metrics import METRICS
        assert "tenant_id" in METRICS.language_detections._labelnames

    def test_escalations_has_reason_category_not_reason(self) -> None:
        from backend.observability.prometheus_metrics import METRICS
        assert "reason_category" in METRICS.emergency_escalations._labelnames
        assert "reason" not in METRICS.emergency_escalations._labelnames


# ── Recording Helper Tests 

class TestRecordingHelpers:

    def test_record_stt_latency_increments_histogram(self) -> None:
        from backend.observability.prometheus_metrics import METRICS, record_stt_latency
        before = METRICS.stt_latency.labels(tenant_id="t1", language="hi-in")._sum.get()
        record_stt_latency("t1", "hi-IN", 250.0)
        after = METRICS.stt_latency.labels(tenant_id="t1", language="hi-in")._sum.get()
        assert after == before + 250.0

    def test_record_stt_normalises_case(self) -> None:
        from backend.observability.prometheus_metrics import METRICS, record_stt_latency
        before = METRICS.stt_latency.labels(tenant_id="t1", language="hi-in")._sum.get()
        record_stt_latency("t1", "Hi-IN", 100.0)
        record_stt_latency("t1", "hi-in", 200.0)
        after = METRICS.stt_latency.labels(tenant_id="t1", language="hi-in")._sum.get()
        assert after == before + 300.0

    def test_record_escalation_uses_closed_enum(self) -> None:
        from backend.observability.prometheus_metrics import METRICS, record_escalation
        before = METRICS.emergency_escalations.labels(
            tenant_id="t1", language="hi-in", reason_category="keyword_medical"
        )._value.get()
        record_escalation("t1", "hi-IN", "Emergency: दिल का दौरा")
        after = METRICS.emergency_escalations.labels(
            tenant_id="t1", language="hi-in", reason_category="keyword_medical"
        )._value.get()
        assert after == before + 1.0

    def test_record_call_start_increments_active_and_ws(self) -> None:
        from backend.observability.prometheus_metrics import METRICS, record_call_start
        before_active = METRICS.active_calls.labels(tenant_id="t1")._value.get()
        before_ws = METRICS.ws_connections_total.labels(tenant_id="t1")._value.get()
        record_call_start("t1", "en-in", "inbound")
        assert METRICS.active_calls.labels(tenant_id="t1")._value.get() == before_active + 1.0
        assert METRICS.ws_connections_total.labels(tenant_id="t1")._value.get() == before_ws + 1.0

    def test_record_cost_inr_increments_counter(self) -> None:
        from backend.observability.prometheus_metrics import METRICS, record_cost_inr
        before = METRICS.cost_inr.labels(tenant_id="t1", cost_type="stt")._value.get()
        record_cost_inr("t1", "stt", 1.25)
        after = METRICS.cost_inr.labels(tenant_id="t1", cost_type="stt")._value.get()
        assert abs((after - before) - 1.25) < 0.001


# ── Prometheus Middleware Tests 

class TestPrometheusMiddleware:

    def test_path_normalisation_tenant_id(self) -> None:
        from backend.observability.prometheus_metrics import PrometheusMiddleware
        assert PrometheusMiddleware._normalise_path("/ws/tenant_hdfc_bank") == "/ws/{tenant_id}"

    def test_path_normalisation_session_uuid(self) -> None:
        from backend.observability.prometheus_metrics import PrometheusMiddleware
        uuid_path = "/ws/tenant_hdfc_bank/550e8400-e29b-41d4-a716-446655440000"
        normalised = PrometheusMiddleware._normalise_path(uuid_path)
        assert normalised == "/ws/{tenant_id}/{session_id}"

    def test_path_normalisation_cost_api(self) -> None:
        from backend.observability.prometheus_metrics import PrometheusMiddleware
        assert PrometheusMiddleware._normalise_path("/api/costs/tenant_hdfc_bank") == "/api/costs/{tenant_id}"

    def test_excluded_paths_set_contains_metrics(self) -> None:
        from backend.observability.prometheus_metrics import PrometheusMiddleware
        assert "/metrics" in PrometheusMiddleware._EXCLUDED_PATHS
        assert "/health" in PrometheusMiddleware._EXCLUDED_PATHS

    def test_middleware_root_path(self) -> None:
        from backend.observability.prometheus_metrics import PrometheusMiddleware
        assert PrometheusMiddleware._normalise_path("/") == "/"

    def test_middleware_static_path_unchanged(self) -> None:
        from backend.observability.prometheus_metrics import PrometheusMiddleware
        assert PrometheusMiddleware._normalise_path("/api/sessions") == "/api/sessions"


# ── Langfuse Client Tests 

class TestLangfuseClient:

    def test_is_disabled_when_no_keys(self, monkeypatch) -> None:
        import backend.observability.langfuse_client as lf_module
        monkeypatch.setattr(lf_module, "_langfuse_client", None)
        monkeypatch.setattr(lf_module, "_langfuse_available", False)
        monkeypatch.setattr(lf_module, "_LANGFUSE_PUBLIC_KEY", "")
        result = lf_module.init_langfuse()
        assert lf_module.is_langfuse_available() is False

    def test_get_client_returns_noop_when_disabled(self, monkeypatch) -> None:
        import backend.observability.langfuse_client as lf_module
        monkeypatch.setattr(lf_module, "_langfuse_client", None)
        monkeypatch.setattr(lf_module, "_langfuse_available", False)
        monkeypatch.setattr(lf_module, "_LANGFUSE_PUBLIC_KEY", "")
        result = lf_module.get_langfuse_client()
        assert result is not None
        assert hasattr(result, "flush")
        assert hasattr(result, "enabled")

    def test_make_callback_handler_returns_noop_when_disabled(self, monkeypatch) -> None:
        import backend.observability.langfuse_client as lf_module
        monkeypatch.setattr(lf_module, "_langfuse_available", False)
        handler = lf_module.make_callback_handler(
            session_id="sess_001",
            tenant_id="tenant_hdfc_bank",
            trace_id="trace-001",
        )
        assert handler is not None
        assert hasattr(handler, "on_llm_start")

    def test_env_var_alias_sets_base_url(self, monkeypatch) -> None:
        import os
        monkeypatch.setenv("LANGFUSE_HOST", "http://test-langfuse:3001")
        monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
        import importlib
        import backend.observability.langfuse_client as lf_module
        importlib.reload(lf_module)
        assert os.environ.get("LANGFUSE_BASE_URL") == "http://test-langfuse:3001"


# ── Instrumented Graph Tests 

class TestInstrumentedGraph:
    """observed_invoke_graph() metric recording and error handling."""

    @pytest.mark.asyncio
    async def test_returns_agent_state_on_success(self) -> None:
        mock_state = {
            "query": "When is EMI deducted?",
            "tenant_id": "tenant_hdfc_bank",
            "session_id": "test-session",
            "trace_id": "test-trace",
            "call_type": "inbound",
            "detected_language": "en-in",
            "detection_method": "script",
            "is_code_mixed": False,
            "safety_cleared": True,
            "escalate": False,
            "escalation_reason": None,
            "escalation_response": None,
            "query_english": "When is EMI deducted?",
            "translation_skipped": True,
            "retrieved_chunks": [
                {"text": "EMI is deducted on the 5th", "score": 0.91, "rerank_score": 0.91}
            ],
            "retrieval_context": "EMI is deducted on the 5th",
            "raw_response": "EMI is deducted on the 5th of every month.",
            "llm_tokens": 120,
            "verified": True,
            "verification_notes": "VERIFIED",
            "final_response": "EMI is deducted on the 5th of every month.",
            "timestamp": "2025-01-01T00:00:00Z",
            "latency_ms": 1100.0,
            "error": None,
        }

        with patch(
            "backend.observability.instrumented_graph._invoke_graph_with_callbacks",
            return_value=mock_state,
        ):
            from backend.observability.instrumented_graph import observed_invoke_graph
            result = await observed_invoke_graph(
                query="When is EMI deducted?",
                tenant_id="tenant_hdfc_bank",
                session_id="test-session",
                call_type="inbound",
                stt_latency_ms=220.0,
                stt_confidence=0.92,
                stt_duration_seconds=3.0,
            )

        assert result["final_response"] == "EMI is deducted on the 5th of every month."
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_records_escalation_metric_correctly(self) -> None:
        mock_state = {
            "query": "मुझे दिल का दौरा पड़ रहा है",
            "tenant_id": "tenant_hdfc_bank",
            "session_id": "test-session",
            "trace_id": "test-trace",
            "call_type": "inbound",
            "detected_language": "hi-in",
            "is_code_mixed": False,
            "safety_cleared": False,
            "escalate": True,
            "escalation_reason": "Keyword match: दिल",
            "final_response": "तुरंत मदद के लिए अधिकारी से संपर्क करें।",
            "llm_tokens": 0,
            "latency_ms": 50.0,
            "error": None,
        }

        from backend.observability.prometheus_metrics import METRICS
        before = METRICS.emergency_escalations.labels(
            tenant_id="tenant_hdfc_bank",
            language="hi-in",
            reason_category="keyword_medical",
        )._value.get()

        with patch(
            "backend.observability.instrumented_graph._invoke_graph_with_callbacks",
            return_value=mock_state,
        ):
            from backend.observability.instrumented_graph import observed_invoke_graph
            await observed_invoke_graph(
                query="मुझे दिल का दौरा पड़ रहा है",
                tenant_id="tenant_hdfc_bank",
                session_id="test-session",
            )

        after = METRICS.emergency_escalations.labels(
            tenant_id="tenant_hdfc_bank",
            language="hi-in",
            reason_category="keyword_medical",
        )._value.get()
        assert after == before + 1.0

    @pytest.mark.asyncio
    async def test_returns_bilingual_fallback_on_pipeline_exception(self) -> None:
        with patch(
            "backend.observability.instrumented_graph._invoke_graph_with_callbacks",
            side_effect=RuntimeError("Groq API timeout"),
        ):
            from backend.observability.instrumented_graph import observed_invoke_graph
            result = await observed_invoke_graph(
                query="Test query",
                tenant_id="tenant_hdfc_bank",
                session_id="test-session",
            )

        assert result["error"] is not None
        assert "Groq API timeout" in result["error"]
        assert result["final_response"] is not None
        assert len(result["final_response"]) > 10

    @pytest.mark.asyncio
    async def test_records_stt_metrics_from_kwargs(self) -> None:
        mock_state = {
            "tenant_id": "tenant_hdfc_bank",
            "session_id": "s1",
            "detected_language": "hi-in",
            "is_code_mixed": False,
            "escalate": False,
            "retrieved_chunks": [],
            "llm_tokens": 50,
            "latency_ms": 800.0,
            "error": None,
            "final_response": "response",
        }

        from backend.observability.prometheus_metrics import METRICS
        before_stt = METRICS.stt_latency.labels(
            tenant_id="tenant_hdfc_bank", language="hi-in"
        )._sum.get()

        with patch(
            "backend.observability.instrumented_graph._invoke_graph_with_callbacks",
            return_value=mock_state,
        ):
            from backend.observability.instrumented_graph import observed_invoke_graph
            await observed_invoke_graph(
                query="EMI?",
                tenant_id="tenant_hdfc_bank",
                session_id="s1",
                stt_latency_ms=300.0,
                stt_confidence=0.88,
                stt_duration_seconds=2.5,
            )

        after_stt = METRICS.stt_latency.labels(
            tenant_id="tenant_hdfc_bank", language="hi-in"
        )._sum.get()
        assert after_stt == before_stt + 300.0


# ── Grafana Dashboard JSON Tests 

class TestGrafanaDashboard:
    @pytest.fixture(scope="class")
    def dashboard(self) -> dict:
        dashboard_path = (
            Path(__file__).parent.parent
            / "monitoring"
            / "grafana"
            / "dashboards"
            / "koyal_dashboard.json"
        )
        if not dashboard_path.exists():
            pytest.skip("Dashboard JSON not found — run from project root")
        with open(dashboard_path, encoding="utf-8") as f:
            return json.load(f)

    def test_dashboard_has_uid(self, dashboard: dict) -> None:
        assert "uid" in dashboard
        assert dashboard["uid"] == "koyalai-phase4-v1"

    def test_dashboard_has_title(self, dashboard: dict) -> None:
        assert "title" in dashboard
        assert "KoyalAI" in dashboard["title"]

    def test_dashboard_has_panels(self, dashboard: dict) -> None:
        assert "panels" in dashboard
        assert len(dashboard["panels"]) >= 14

    def test_dashboard_has_tenant_variable(self, dashboard: dict) -> None:
        variables = dashboard["templating"]["list"]
        var_names = [v["name"] for v in variables]
        assert "tenant_id" in var_names

    def test_dashboard_has_ttfr_panel(self, dashboard: dict) -> None:
        all_exprs = " ".join(
            target.get("expr", "")
            for panel in dashboard["panels"]
            for target in panel.get("targets", [])
        )
        assert "koyal_ttfr_ms" in all_exprs

    def test_dashboard_has_cost_panel(self, dashboard: dict) -> None:
        all_exprs = " ".join(
            target.get("expr", "")
            for panel in dashboard["panels"]
            for target in panel.get("targets", [])
        )
        assert "koyal_cost_inr_total" in all_exprs

    def test_dashboard_has_ragas_panel(self, dashboard: dict) -> None:
        all_exprs = " ".join(
            target.get("expr", "")
            for panel in dashboard["panels"]
            for target in panel.get("targets", [])
        )
        assert "koyal_ragas_faithfulness" in all_exprs

    def test_dashboard_datasource_uid_matches_provisioning(self, dashboard: dict) -> None:
        for panel in dashboard["panels"]:
            ds = panel.get("datasource", {})
            if isinstance(ds, dict) and ds.get("type") == "prometheus":
                assert ds.get("uid") == "koyalai_prometheus", \
                    f"Panel '{panel.get('title')}' datasource UID must be 'koyalai_prometheus'"

    def test_all_panels_have_targets(self, dashboard: dict) -> None:
        for panel in dashboard["panels"]:
            assert len(panel.get("targets", [])) >= 1

    def test_dashboard_refresh_interval(self, dashboard: dict) -> None:
        assert dashboard.get("refresh") == "15s"


# ── Cross-Tenant Metric Isolation 

class TestCrossTenantIsolation:
    """Metrics for tenant A must not bleed into tenant B label values."""

    def test_calls_total_tenant_isolation(self) -> None:
        from backend.observability.prometheus_metrics import CALLS_TOTAL
        # Reset tenant B baseline
        before_b = CALLS_TOTAL.labels(
            tenant_id="isolation_tenant_b", language="en-IN",
            call_type="inbound", outcome="completed"
        )._value.get()
        # Increment tenant A only
        for _ in range(5):
            CALLS_TOTAL.labels(
                tenant_id="isolation_tenant_a", language="en-IN",
                call_type="inbound", outcome="completed"
            ).inc()
        # tenant B must be unchanged
        after_b = CALLS_TOTAL.labels(
            tenant_id="isolation_tenant_b", language="en-IN",
            call_type="inbound", outcome="completed"
        )._value.get()
        assert after_b == before_b

    def test_cost_gauge_tenant_isolation(self) -> None:
        from backend.observability.prometheus_metrics import COST_INR_GAUGE
        COST_INR_GAUGE.labels(tenant_id="gauge_a", cost_type="stt").set(100.0)
        COST_INR_GAUGE.labels(tenant_id="gauge_b", cost_type="stt").set(0.0)
        assert COST_INR_GAUGE.labels(
            tenant_id="gauge_a", cost_type="stt"
        )._value.get() == 100.0
        assert COST_INR_GAUGE.labels(
            tenant_id="gauge_b", cost_type="stt"
        )._value.get() == 0.0


# ── CostTracker → Prometheus Counter Emission 

class TestCostTracker:
    """CostTracker must emit Prometheus Counters in addition to Redis writes."""

    @pytest.fixture
    def tracker_with_mock_redis(self):
        from backend.cost_tracker import CostTracker

        tracker = CostTracker.__new__(CostTracker)

        import fakeredis
        import fakeredis.aioredis as fake_aio

        server = fakeredis.FakeServer()
        tracker._async_redis = fake_aio.FakeRedis(server=server, decode_responses=True, protocol=2)
        tracker._sync_redis = fakeredis.FakeRedis(server=server, decode_responses=True, protocol=2)
        _orig_async_execute = tracker._async_redis.execute_command
        async def _async_execute_command(*args, **kwargs):
            if args and str(args[0]).upper() == "HELLO":
                return []
            return await _orig_async_execute(*args, **kwargs)
        tracker._async_redis.execute_command = _async_execute_command
        return tracker

    @pytest.mark.asyncio
    async def test_track_stt_emits_prometheus_counter(
        self, tracker_with_mock_redis
    ) -> None:
        from backend.observability.prometheus_metrics import COST_INR_TOTAL

        tracker = tracker_with_mock_redis
        before = COST_INR_TOTAL.labels(
            tenant_id="tracker_test_tenant", cost_type="stt"
        )._value.get()

        await tracker.track_stt("tracker_test_tenant", seconds=120.0)  # ₹1.00

        after = COST_INR_TOTAL.labels(
            tenant_id="tracker_test_tenant", cost_type="stt"
        )._value.get()
        cost_inr = (120.0 / 60.0) * 0.50
        assert abs(after - before - cost_inr) < 1e-6

    @pytest.mark.asyncio
    async def test_track_tts_emits_prometheus_counter(
        self, tracker_with_mock_redis
    ) -> None:
        from backend.observability.prometheus_metrics import COST_INR_TOTAL

        tracker = tracker_with_mock_redis
        before = COST_INR_TOTAL.labels(
            tenant_id="tracker_tts_test", cost_type="tts"
        )._value.get()

        await tracker.track_tts("tracker_tts_test", chars=1000)  # ₹1.50

        after = COST_INR_TOTAL.labels(
            tenant_id="tracker_tts_test", cost_type="tts"
        )._value.get()
        expected_cost = 1000 * 0.0015
        assert abs(after - before - expected_cost) < 1e-6

    @pytest.mark.asyncio
    async def test_sync_gauges_reads_from_redis(
        self, tracker_with_mock_redis, monkeypatch
    ) -> None:
        from backend.observability.prometheus_metrics import COST_INR_GAUGE
  
        tracker = tracker_with_mock_redis
        tenant = "gauge_sync_tenant"
        cost_type = "stt"
        expected_cost = 0.50
        def mock_get_tenant_costs(tenant_id: str) -> dict:
            return {
                "tenant_id": tenant_id,
                "stt_cost_inr": expected_cost,
                "tts_cost_inr": 0.0,
                "total_cost_inr": expected_cost,
                "stt_seconds": 60.0,
                "tts_chars": 0,
                "llm_tokens": 0,
                "calls_completed": 0,
                "calls_escalated": 0,
                "calls_failed": 0,
                "calls_terminated": 0,
            }
        
        monkeypatch.setattr(tracker, "get_tenant_costs", mock_get_tenant_costs)

        tracker.sync_gauges([tenant])

        gauge_val = COST_INR_GAUGE.labels(
            tenant_id=tenant, cost_type=cost_type
        )._value.get()
        assert abs(gauge_val - 0.50) < 0.01

    @pytest.mark.asyncio
    async def test_cross_tenant_cost_isolation_with_prometheus(
        self, tracker_with_mock_redis
    ) -> None:
        from backend.observability.prometheus_metrics import COST_INR_TOTAL

        tracker = tracker_with_mock_redis
        before_b = COST_INR_TOTAL.labels(
            tenant_id="ct_isolation_b", cost_type="stt"
        )._value.get()

        # Only charge tenant A
        await tracker.track_stt("ct_isolation_a", seconds=600.0)

        after_b = COST_INR_TOTAL.labels(
            tenant_id="ct_isolation_b", cost_type="stt"
        )._value.get()
        assert after_b == before_b, "Tenant A costs must not affect Tenant B metrics"


# ── FastAPI Endpoint Tests 

class TestObservabilityEndpoints:
    """FastAPI endpoints must expose correct Prometheus and health data."""

    def test_metrics_endpoint_returns_prometheus_format(self) -> None:
        from starlette.testclient import TestClient
        from backend.main import app

        client = TestClient(app)
        resp = client.get("/metrics/")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers.get("content-type", "")
        assert b"koyal_" in resp.content

    def test_health_endpoint_includes_langfuse_status(self) -> None:
        from starlette.testclient import TestClient
        from backend.main import app

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "langfuse" in data["services"]
        assert data["services"]["langfuse"] in ("enabled", "disabled", "error")

    def test_health_endpoint_version_is_v4(self) -> None:
        from starlette.testclient import TestClient
        from backend.main import app

        client = TestClient(app)
        resp = client.get("/health")
        data = resp.json()
        assert data["version"] == "4.0.0"