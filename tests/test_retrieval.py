"""

Tests:
    1. Hindi queries → HDFC tenant → correct Hindi/English chunks returned
    2. English queries → HDFC tenant → correct English chunks returned
    3. Hinglish (code-mixed) queries → HDFC tenant → handles code-switching
    4. Swiggy refund query → Swiggy tenant → correct refund info returned
    5. Tenant isolation → HDFC-specific data must NOT appear in Swiggy results
    6. Metadata integrity → tenant_id and language fields present in payload
    7. Nonexistent tenant → raises exception, not silent empty list
    8. Context formatting → labelled, non-empty output

Prerequisites:
    docker-compose up -d qdrant redis
    python scripts/ingest_all.py
    pytest tests/test_retrieval.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.rag.retriever import MultilingualRetriever
from backend.rag.ingestor import MultilingualIngestor

# ─── Module-level singleton 
# Shared across all tests in this module to avoid reloading LaBSE per test.
retriever = MultilingualRetriever()


# ─── HDFC — Hindi Query Tests 
HINDI_TESTS = [
    ("मेरी EMI कब कटती है", "5"),
    ("EMI लेट होने पर क्या होता है", "500"),
    ("लोन का पूर्व भुगतान कैसे करें", "prepayment"),
    ("कस्टमर केयर नंबर क्या है", "1800"),
    ("फोरक्लोजर के लिए क्या चार्ज है", ["2%", "2", "शुल्क", "charge", "fixed"]),
    ("KYC के लिए क्या चाहिए", "PAN"),
]

# ─── HDFC — English Query Tests 
ENGLISH_TESTS = [
    ("when is the EMI deducted", "5th"),
    ("late payment penalty amount", "500"),
    ("prepayment charges on floating rate", "floating"),
    ("customer care toll free number", "1800"),
    ("foreclosure charge percentage", ["2%", "2", "शुल्क", "charge", "fixed"]),
    ("minimum balance savings account", "10,000"),
]

# ─── HDFC — Hinglish (code-mixed) Query Tests 
HINGLISH_TESTS = [
    ("मेरा EMI miss हो गया, kya hoga", "500"),
    ("loan prepay karna hai, charges kya hain", "floating"),
    ("mere account se EMI kata, 5th ko hoga kya", "5"),
    ("card kho gaya, kya karun", ["1800", "block", "खो", "तुरंत", "call"]),
]

# ─── Swiggy — English Refund Tests 
SWIGGY_ENGLISH_TESTS = [
    ("what is the refund policy", "refund"),
    ("how long does refund take", "24"),
    ("order not received what to do", ["report", "रिपोर्ट"]),
    ("food poisoning contact", "1800"),
    ("how to cancel order", ["full refund", "पूरा रिफंड", "cancel", "रद्द", "2 मिनट"]),
    ("Swiggy One membership benefits", "delivery"),
]

# ─── Swiggy — Hindi Tests 
SWIGGY_HINDI_TESTS = [
    ("रिफंड में कितने दिन लगते हैं", "24"),
    ("ऑर्डर नहीं मिला तो क्या करें", "रिपोर्ट"),
    ("खराब खाना मिला", "रिपोर्ट"),
    ("ऑर्डर ट्रैक न हो", "ऐप"),
]


# ─── Helpers 

def _combined_text(chunks: list[dict]) -> str:
    return " ".join(c["text"] for c in chunks).lower()


def _assert_non_empty(chunks: list[dict], query: str) -> None:
    assert chunks, (
        f"No results returned for query: {query!r}. "
        "Ensure ingestion ran successfully before tests."
    )


def _assert_substrings(combined: str, expected: str | list[str], query: str) -> None:
    """Assert that at least one expected substring appears in combined text."""
    if isinstance(expected, str):
        expected = [expected]
    combined_lower = combined.lower()
    found = any(exp.lower() in combined_lower for exp in expected)
    assert found, (
        f"Expected any of {expected!r} in results for query {query!r}.\n"
        f"Got:\n{combined[:400]}"
    )


# ─── HDFC — Hindi 

@pytest.mark.parametrize("query,expected_substring", HINDI_TESTS)
def test_hdfc_hindi_retrieval(query: str, expected_substring: str | list[str]) -> None:
    """Hindi queries must return relevant chunks containing expected keyword."""
    chunks = retriever.retrieve(query, "tenant_hdfc_bank", "hi-IN")
    _assert_non_empty(chunks, query)
    combined = _combined_text(chunks)
    _assert_substrings(combined, expected_substring, query)


# ─── HDFC — English 

@pytest.mark.parametrize("query,expected_substring", ENGLISH_TESTS)
def test_hdfc_english_retrieval(query: str, expected_substring: str | list[str]) -> None:
    """English queries must return relevant chunks containing expected keyword."""
    chunks = retriever.retrieve(query, "tenant_hdfc_bank", "en-IN")
    _assert_non_empty(chunks, query)
    combined = _combined_text(chunks)
    _assert_substrings(combined, expected_substring, query)


# ─── HDFC — Hinglish 

@pytest.mark.parametrize("query,expected_substring", HINGLISH_TESTS)
def test_hdfc_hinglish_retrieval(query: str, expected_substring: str | list[str]) -> None:
    """Hinglish (code-mixed) queries must be handled without pre-translation."""
    chunks = retriever.retrieve(query, "tenant_hdfc_bank")
    _assert_non_empty(chunks, query)
    combined = _combined_text(chunks)
    _assert_substrings(combined, expected_substring, query)


# ─── Swiggy — English 

@pytest.mark.parametrize("query,expected_substring", SWIGGY_ENGLISH_TESTS)
def test_swiggy_english_retrieval(query: str, expected_substring: str | list[str]) -> None:
    """Swiggy English queries must return relevant Swiggy support chunks."""
    chunks = retriever.retrieve(query, "tenant_swiggy_support", "en-IN")
    _assert_non_empty(chunks, query)
    combined = _combined_text(chunks)
    _assert_substrings(combined, expected_substring, query)


# ─── Swiggy — Hindi 

@pytest.mark.parametrize("query,expected_substring", SWIGGY_HINDI_TESTS)
def test_swiggy_hindi_retrieval(query: str, expected_substring: str | list[str]) -> None:
    """Swiggy Hindi queries must return relevant Swiggy support chunks."""
    chunks = retriever.retrieve(query, "tenant_swiggy_support", "hi-IN")
    _assert_non_empty(chunks, query)
    combined = _combined_text(chunks)
    _assert_substrings(combined, expected_substring, query)


# ─── Tenant Isolation — CRITICAL SAFETY TESTS 

def test_tenant_isolation_hdfc_not_in_swiggy() -> None:
    """
    CRITICAL: HDFC-specific content must NOT appear in Swiggy's collection.
    Query: "EMI payment" is HDFC-specific vocabulary.
    """
    hdfc_chunks = retriever.retrieve("EMI payment", "tenant_hdfc_bank")
    swiggy_chunks = retriever.retrieve("EMI payment", "tenant_swiggy_support")

    hdfc_text = " ".join(c["text"] for c in hdfc_chunks)
    swiggy_text = " ".join(c["text"] for c in swiggy_chunks)

    assert hdfc_text != swiggy_text, (
        "TENANT ISOLATION VIOLATED: HDFC and Swiggy returned identical results. "
        "Check that separate collections are used per tenant."
    )

    hdfc_specific_terms = ["hdfcbank.com", "1800-202-6161", "foreclosure"]
    for term in hdfc_specific_terms:
        assert term.lower() not in swiggy_text.lower(), (
            f"TENANT ISOLATION VIOLATED: '{term}' (HDFC-specific) found in "
            f"Swiggy retrieval results.\nSwiggy text:\n{swiggy_text[:400]}"
        )


def test_tenant_isolation_swiggy_not_in_hdfc() -> None:
    """
    CRITICAL: Swiggy-specific content must NOT appear in HDFC's collection.
    Query: "refund policy" is Swiggy-specific vocabulary.
    """
    swiggy_chunks = retriever.retrieve("refund policy", "tenant_swiggy_support")
    hdfc_chunks = retriever.retrieve("refund policy", "tenant_hdfc_bank")

    swiggy_text = " ".join(c["text"] for c in swiggy_chunks)
    hdfc_text = " ".join(c["text"] for c in hdfc_chunks)

    assert swiggy_text != hdfc_text, (
        "TENANT ISOLATION VIOLATED: Both tenants returned identical 'refund' results."
    )

    swiggy_specific_terms = ["swiggy.in", "1800-208-9999", "food delivery"]
    for term in swiggy_specific_terms:
        assert term.lower() not in hdfc_text.lower(), (
            f"TENANT ISOLATION VIOLATED: '{term}' (Swiggy-specific) found in "
            f"HDFC retrieval results.\nHDFC text:\n{hdfc_text[:400]}"
        )


def test_tenant_isolation_collection_names_are_separate() -> None:
    """Verify that the two tenants use different Qdrant collections."""
    hdfc_coll = MultilingualIngestor.collection_name("tenant_hdfc_bank")
    swiggy_coll = MultilingualIngestor.collection_name("tenant_swiggy_support")
    assert hdfc_coll != swiggy_coll, "Tenants must use different collections."
    assert hdfc_coll == "koyalai_tenant_hdfc_bank"
    assert swiggy_coll == "koyalai_tenant_swiggy_support"


# ─── Metadata Integrity Tests 

def test_retrieval_returns_tenant_metadata() -> None:
    """All returned chunks must carry correct tenant_id in metadata."""
    chunks = retriever.retrieve("customer care number", "tenant_hdfc_bank")
    _assert_non_empty(chunks, "customer care number")
    for chunk in chunks:
        assert chunk["tenant_id"] == "tenant_hdfc_bank", (
            f"Chunk tenant_id mismatch: expected 'tenant_hdfc_bank', "
            f"got {chunk['tenant_id']!r}"
        )


def test_retrieval_returns_language_metadata() -> None:
    """All returned chunks must carry a language code in metadata."""
    chunks = retriever.retrieve("what is refund timeline", "tenant_swiggy_support")
    _assert_non_empty(chunks, "refund timeline")
    for chunk in chunks:
        assert chunk["language"] in {"hi-IN", "en-IN", "hi-IN+en-IN", "und"}, (
            f"Unknown language code in chunk: {chunk['language']!r}"
        )


def test_retrieval_returns_char_count_metadata() -> None:
    """All returned chunks must carry char_count in payload."""
    chunks = retriever.retrieve("EMI due date", "tenant_hdfc_bank")
    _assert_non_empty(chunks, "EMI due date")
    for chunk in chunks:
        assert "char_count" in chunk, "Missing 'char_count' in chunk payload"
        assert isinstance(chunk["char_count"], int), "char_count must be an int"
        assert chunk["char_count"] > 0, "char_count must be positive"


# ─── Failure Mode Tests 

def test_empty_collection_after_nonexistent_tenant() -> None:
    """Querying a nonexistent tenant must raise, not silently return empty."""
    with pytest.raises(Exception):
        retriever.retrieve("test", "tenant_does_not_exist")


# ─── Context Formatting Tests 

def test_format_context_non_empty() -> None:
    """format_context must return a non-empty labelled string with score info."""
    chunks = retriever.retrieve(
        "What is the refund policy", "tenant_swiggy_support"
    )
    _assert_non_empty(chunks, "refund policy")
    context = retriever.format_context(chunks)
    assert "[Source" in context
    assert "score=" in context
    assert len(context) > 50


def test_format_context_empty_input() -> None:
    """format_context on empty list must return the no-result message."""
    context = retriever.format_context([])
    assert "No relevant information" in context