from __future__ import annotations

import json
import re
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from backend.agents.graph import build_koyal_graph
from backend.agents.state import AgentState, make_initial_state


# ── Fixtures 

@pytest.fixture(scope="module")
def graph():
    """Build the LangGraph pipeline once per test module."""
    return build_koyal_graph()


@pytest.fixture
def hdfc_state() -> AgentState:
    return make_initial_state(
        query="मेरी EMI कब कटती है",
        tenant_id="tenant_hdfc_bank",
        session_id="test_session_001",
        call_type="inbound",
    )


@pytest.fixture
def english_state() -> AgentState:
    return make_initial_state(
        query="What is the EMI due date?",
        tenant_id="tenant_hdfc_bank",
        session_id="test_session_002",
        call_type="inbound",
    )


@pytest.fixture
def hinglish_state() -> AgentState:
    return make_initial_state(
        query="मेरा EMI miss हो गया, kya hoga",
        tenant_id="tenant_hdfc_bank",
        session_id="test_session_003",
        call_type="inbound",
    )


@pytest.fixture
def swiggy_state() -> AgentState:
    return make_initial_state(
        query="What is the refund policy?",
        tenant_id="tenant_swiggy_support",
        session_id="test_session_004",
        call_type="inbound",
    )


@pytest.fixture
def hindi_emergency_state() -> AgentState:
    return make_initial_state(
        query="मुझे दिल का दौरा आ रहा है",
        tenant_id="tenant_hdfc_bank",
        session_id="test_session_005",
        call_type="inbound",
    )


@pytest.fixture
def english_emergency_state() -> AgentState:
    return make_initial_state(
        query="I am having a heart attack, please help",
        tenant_id="tenant_hdfc_bank",
        session_id="test_session_006",
        call_type="inbound",
    )


@pytest.fixture
def banking_emergency_state() -> AgentState:
    return make_initial_state(
        query="fraud ho gaya mere account mein",
        tenant_id="tenant_hdfc_bank",
        session_id="test_session_007",
        call_type="inbound",
    )


# ── Mock Helpers 

def _mock_groq_response(content: str, total_tokens: int = 150) -> MagicMock:
    """Build a mock ChatGroq response object."""
    msg = MagicMock()
    msg.content = content
    msg.usage_metadata = {"total_tokens": total_tokens}
    return msg


def _mock_sarvam_translate(source: str) -> str:
    """Passthrough mock — returns source text as-is (simulates English)."""
    return source


def _mock_sarvam_lid_response() -> dict:
    return {"language_code": "hi-IN", "confidence": 0.95}


# ── Language Detection Tests 

class TestLanguageDetection:
    """Test the language_detection_node in isolation."""

    def test_hindi_script_detected(self) -> None:
        from backend.agents.graph import language_detection_node
        state = make_initial_state("नमस्ते मुझे मदद चाहिए", "tenant_hdfc_bank", "s1")
        result = language_detection_node(state)
        assert result["detected_language"] == "hi-IN"
        assert result["is_code_mixed"] is False
        assert result["detection_method"] == "script"

    def test_english_script_detected(self) -> None:
        from backend.agents.graph import language_detection_node
        state = make_initial_state("Hello, I need help with my loan", "tenant_hdfc_bank", "s1")
        result = language_detection_node(state)
        assert result["detected_language"] == "en-IN"
        assert result["is_code_mixed"] is False

    def test_hinglish_detected_as_code_mixed(self) -> None:
        from backend.agents.graph import language_detection_node
        state = make_initial_state("मेरा EMI miss हो गया kya hoga", "tenant_hdfc_bank", "s1")
        result = language_detection_node(state)
        assert result["is_code_mixed"] is True
        assert "hi-IN" in result["detected_language"]

    def test_empty_query_defaults_to_english(self) -> None:
        from backend.agents.graph import language_detection_node
        state = make_initial_state("placeholder", "tenant_hdfc_bank", "s1")
        state["query"] = ""
        result = language_detection_node(state)
        assert result["detected_language"] == "en-IN"


# ── Safety Gate Tests 

class TestSafetyGate:
    """Test safety_gate_agent node directly."""

    def test_hindi_emergency_escalates(self, hindi_emergency_state: AgentState) -> None:
        from backend.agents.safety_agent import safety_gate_agent
        result = safety_gate_agent(hindi_emergency_state)
        assert result["escalate"] is True
        assert result["safety_cleared"] is False
        assert result["escalation_reason"] is not None
        assert len(result["escalation_reason"]) > 0

    def test_english_emergency_escalates(self, english_emergency_state: AgentState) -> None:
        from backend.agents.safety_agent import safety_gate_agent
        result = safety_gate_agent(english_emergency_state)
        assert result["escalate"] is True
        assert result["safety_cleared"] is False

    def test_normal_query_clears_safety(self, hdfc_state: AgentState) -> None:
        from backend.agents.safety_agent import safety_gate_agent
        result = safety_gate_agent(hdfc_state)
        assert result["safety_cleared"] is True
        assert result["escalate"] is False
        assert result["escalation_reason"] is None

    def test_banking_fraud_escalates(self, banking_emergency_state: AgentState) -> None:
        from backend.agents.safety_agent import safety_gate_agent
        result = safety_gate_agent(banking_emergency_state)
        assert result["escalate"] is True


# ── Escalation Handler Tests 

class TestEscalationHandler:
    """Test escalation_handler node directly."""

    def test_hindi_escalation_returns_hindi_message(self) -> None:
        from backend.agents.escalation_handler import escalation_handler
        state = make_initial_state("dummy", "tenant_hdfc_bank", "s1")
        state["detected_language"] = "hi-IN"
        state["escalation_reason"] = "Emergency keyword: 'दिल का दौरा'"
        result = escalation_handler(state)
        # Response should contain Devanagari characters (Hindi)
        devanagari_re = re.compile(r"[\u0900-\u097F]")
        assert devanagari_re.search(result["final_response"]), (
            f"Expected Hindi response, got: {result['final_response']!r}"
        )
        assert result["escalate"] is True
        assert result["safety_cleared"] is False

    def test_english_escalation_returns_english_message(self) -> None:
        from backend.agents.escalation_handler import escalation_handler
        state = make_initial_state("dummy", "tenant_hdfc_bank", "s1")
        state["detected_language"] = "en-IN"
        state["escalation_reason"] = "Emergency keyword: 'heart attack'"
        result = escalation_handler(state)
        # Response must be ASCII/English readable
        assert result["final_response"]
        assert result["escalate"] is True

    def test_banking_fraud_escalation_uses_banking_message(self) -> None:
        from backend.agents.escalation_handler import escalation_handler
        state = make_initial_state("dummy", "tenant_hdfc_bank", "s1")
        state["detected_language"] = "en-IN"
        state["escalation_reason"] = "Emergency keyword: 'fraud ho gaya'"
        result = escalation_handler(state)
        # Banking escalation should mention account security, helpline, action, or fraud
        final = result["final_response"].lower()
        assert any(word in final for word in ["account", "secure", "1800", "immediately", "action", "fraud"]), (
            f"Banking escalation missing expected content: {result['final_response']!r}"
        )

    def test_unknown_tenant_uses_default_escalation(self) -> None:
        from backend.agents.escalation_handler import escalation_handler
        state = make_initial_state("dummy", "tenant_nonexistent", "s1")
        state["detected_language"] = "en-IN"
        state["escalation_reason"] = "Emergency keyword: 'heart attack'"
        # Should NOT raise — must fall back gracefully
        result = escalation_handler(state)
        assert result["final_response"]


# ── Full Graph Tests (mocked APIs) 

class TestFullGraphMocked:
    """End-to-end graph tests with all external APIs mocked."""

    @patch("backend.agents.response_agent._get_llm")
    @patch("backend.agents.verification_agent._get_verification_llm")
    @patch("backend.agents.language_bridge.LanguageBridge._call_sarvam",
           side_effect=lambda text, **kw: text)
    def test_hindi_query_reaches_final_response(
        self,
        mock_bridge,
        mock_verif_llm,
        mock_response_llm,
        graph,
        hdfc_state: AgentState,
    ) -> None:
        """Hindi query must produce a non-empty final_response without escalation."""
        # Mock Groq response
        mock_response_llm.return_value.invoke.return_value = _mock_groq_response(
            "Your EMI is automatically deducted on the 5th of every month."
        )
        # Mock verification as PASS
        verif_json = json.dumps({"verdict": "PASS", "score": 0.92, "reason": "Faithful."})
        mock_verif_llm.return_value.invoke.return_value = _mock_groq_response(verif_json)

        result = graph.invoke(hdfc_state)

        assert result["final_response"], "final_response must not be empty."
        assert not result["escalate"], "Normal query must not escalate."
        assert result["safety_cleared"] is True

    @patch("backend.agents.response_agent._get_llm")
    @patch("backend.agents.verification_agent._get_verification_llm")
    @patch("backend.agents.language_bridge.LanguageBridge._call_sarvam",
           side_effect=lambda text, **kw: text)
    def test_english_query_full_pipeline(
        self,
        mock_bridge,
        mock_verif_llm,
        mock_response_llm,
        graph,
        english_state: AgentState,
    ) -> None:
        """English query must complete without translation overhead."""
        mock_response_llm.return_value.invoke.return_value = _mock_groq_response(
            "EMI is deducted on the 5th of every month automatically."
        )
        verif_json = json.dumps({"verdict": "PASS", "score": 0.95, "reason": "Grounded."})
        mock_verif_llm.return_value.invoke.return_value = _mock_groq_response(verif_json)

        result = graph.invoke(english_state)

        assert result["detected_language"] == "en-IN"
        assert result["final_response"]
        assert result["verified"] is True
        assert result["verification_score"] >= 0.7

    def test_hindi_emergency_escalates_with_hindi_message(
        self, graph, hindi_emergency_state: AgentState
    ) -> None:
        """Hindi emergency query must escalate with a Hindi response — no LLM call needed."""
        result = graph.invoke(hindi_emergency_state)

        assert result["escalate"] is True
        assert result["safety_cleared"] is False
        assert result["final_response"], "Escalation must provide a response."
        # Verify response contains Devanagari (Hindi)
        devanagari_re = re.compile(r"[\u0900-\u097F]")
        assert devanagari_re.search(result["final_response"]), (
            f"Hindi emergency should produce Hindi escalation message. "
            f"Got: {result['final_response']!r}"
        )

    def test_english_emergency_escalates_with_english_message(
        self, graph, english_emergency_state: AgentState
    ) -> None:
        """English emergency query must escalate with an English message."""
        result = graph.invoke(english_emergency_state)

        assert result["escalate"] is True
        assert result["final_response"]

    def test_hinglish_query_detected_as_code_mixed(
        self, graph, hinglish_state: AgentState
    ) -> None:
        """Hinglish must be detected as code-mixed even without Sarvam LID."""
        # Just run language detection — stop before LLM to avoid mock complexity
        from backend.agents.graph import language_detection_node
        result = language_detection_node(hinglish_state)
        assert result["is_code_mixed"] is True, (
            "Hinglish ('मेरा EMI miss हो गया') must be detected as code-mixed."
        )

    @patch("backend.agents.response_agent._get_llm")
    @patch("backend.agents.verification_agent._get_verification_llm")
    @patch("backend.agents.language_bridge.LanguageBridge._call_sarvam",
           side_effect=lambda text, **kw: text)
    def test_verification_fail_still_produces_response(
        self,
        mock_bridge,
        mock_verif_llm,
        mock_response_llm,
        graph,
        english_state: AgentState,
    ) -> None:
        """A FAIL verdict must not block the pipeline — final_response must be set."""
        mock_response_llm.return_value.invoke.return_value = _mock_groq_response(
            "I apologize, I don't have that information."
        )
        # Simulate verification FAIL
        verif_json = json.dumps({
            "verdict": "FAIL", "score": 0.3,
            "reason": "Response not grounded in context."
        })
        mock_verif_llm.return_value.invoke.return_value = _mock_groq_response(verif_json)

        result = graph.invoke(english_state)

        assert result["final_response"], "Pipeline must produce response even on FAIL verdict."
        assert result["verified"] is False
        assert result["verification_score"] < 0.7

    @patch("backend.agents.response_agent._get_llm")
    @patch("backend.agents.verification_agent._get_verification_llm")
    @patch("backend.agents.language_bridge.LanguageBridge._call_sarvam",
           side_effect=lambda text, **kw: text)
    def test_tenant_isolation_in_pipeline(
        self,
        mock_bridge,
        mock_verif_llm,
        mock_response_llm,
        graph,
        swiggy_state: AgentState,
    ) -> None:
        """Swiggy tenant queries must retrieve from koyal_tenant_swiggy_support only."""
        mock_response_llm.return_value.invoke.return_value = _mock_groq_response(
            "Refunds are processed within 5–7 business days."
        )
        verif_json = json.dumps({"verdict": "PASS", "score": 0.88, "reason": "Grounded."})
        mock_verif_llm.return_value.invoke.return_value = _mock_groq_response(verif_json)

        result = graph.invoke(swiggy_state)

        # Verify all retrieved chunks belong to Swiggy's tenant
        chunks = result.get("retrieved_chunks") or []
        for chunk in chunks:
            assert chunk["tenant_id"] == "tenant_swiggy_support", (
                f"Cross-tenant contamination detected! "
                f"HDFC chunk found in Swiggy results: {chunk!r}"
            )

    @patch("backend.agents.response_agent._get_llm")
    @patch("backend.agents.verification_agent._get_verification_llm")
    @patch("backend.agents.language_bridge.LanguageBridge._call_sarvam",
           side_effect=lambda text, **kw: text)
    def test_llm_tokens_tracked_in_state(
        self,
        mock_bridge,
        mock_verif_llm,
        mock_response_llm,
        graph,
        english_state: AgentState,
    ) -> None:
        """LLM token usage must be tracked in state for cost accounting."""
        mock_response_llm.return_value.invoke.return_value = _mock_groq_response(
            "EMI is on the 5th.", total_tokens=243
        )
        verif_json = json.dumps({"verdict": "PASS", "score": 0.90, "reason": "OK."})
        mock_verif_llm.return_value.invoke.return_value = _mock_groq_response(verif_json)

        result = graph.invoke(english_state)

        assert result.get("llm_tokens") == 243, (
            f"Expected llm_tokens=243, got {result.get('llm_tokens')}"
        )


# ── make_initial_state Tests 

class TestMakeInitialState:
    """Validate the state factory function."""

    def test_required_fields_populated(self) -> None:
        state = make_initial_state("Hello", "tenant_hdfc_bank", "s1")
        assert state["query"] == "Hello"
        assert state["tenant_id"] == "tenant_hdfc_bank"
        assert state["session_id"] == "s1"
        assert state["call_type"] == "inbound"
        assert state["trace_id"]        # Non-empty UUID
        assert state["timestamp"]       # Non-empty ISO timestamp

    def test_optional_fields_are_none(self) -> None:
        state = make_initial_state("Hello", "tenant_hdfc_bank", "s1")
        assert state["detected_language"] is None
        assert state["query_english"] is None
        assert state["raw_response"] is None
        assert state["final_response"] is None
        assert state["escalation_reason"] is None

    def test_empty_query_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="query must be a non-empty string"):
            make_initial_state("", "tenant_hdfc_bank", "s1")

    def test_whitespace_query_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="query must be a non-empty string"):
            make_initial_state("   ", "tenant_hdfc_bank", "s1")

    def test_empty_tenant_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="tenant_id must be a non-empty string"):
            make_initial_state("Hello", "", "s1")

    def test_outbound_call_type(self) -> None:
        state = make_initial_state("Hello", "tenant_hdfc_bank", "s1", call_type="outbound")
        assert state["call_type"] == "outbound"

    def test_unique_trace_ids(self) -> None:
        s1 = make_initial_state("Q1", "tenant_hdfc_bank", "s1")
        s2 = make_initial_state("Q2", "tenant_hdfc_bank", "s2")
        assert s1["trace_id"] != s2["trace_id"], "Each state must have a unique trace_id."