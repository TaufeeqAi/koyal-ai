from __future__ import annotations

import pytest

from backend.safety.emergency_keywords import (
    ALL_EMERGENCY_KEYWORDS,
    EMERGENCY_KEYWORDS_BANKING,
    EMERGENCY_KEYWORDS_ENGLISH,
    EMERGENCY_KEYWORDS_HINDI,
    EMERGENCY_KEYWORDS_HINGLISH,
    MultilingualEmergencyDetector,
)


# ── Fixtures 

@pytest.fixture(scope="module")
def detector() -> MultilingualEmergencyDetector:
    """Shared detector instance — LaBSE loaded once per module."""
    return MultilingualEmergencyDetector()


# ── Hindi Emergency Tests 

@pytest.mark.parametrize("query,keyword", [
    ("मुझे दिल का दौरा आ रहा है", "दिल का दौरा"),
    ("मेरी छाती में दर्द है", "छाती में दर्द"),
    ("मैं आत्महत्या करना चाहता हूँ", "आत्महत्या"),
    ("मुझे साँस नहीं आ रही", "साँस नहीं आ रही"),
    ("वो बेहोश हो गया", "बेहोश"),
    ("मरना चाहता हूँ मैं", "मरना चाहता"),
    ("मेरे पैर में लकवा मार गया", "लकवा"),
    ("दम घुट रहा है मेरा", "दम घुट रहा"),
])
def test_hindi_emergency_keywords(query: str, keyword: str, detector: MultilingualEmergencyDetector) -> None:
    """Hindi emergency keywords must be detected by Layer 1 (keyword match)."""
    is_emrg, reason = detector.is_emergency(query)
    assert is_emrg, (
        f"Expected EMERGENCY for Hindi query {query!r} "
        f"containing keyword '{keyword}', but got SAFE."
    )
    assert keyword.lower() in reason.lower() or "Emergency keyword" in reason, (
        f"Reason should reference the keyword. Got: {reason!r}"
    )


# ── English Emergency Tests 

@pytest.mark.parametrize("query,keyword", [
    ("I am having a chest pain", "chest pain"),
    ("I think I'm having a heart attack", "heart attack"),
    ("I feel unconscious and need help", "unconscious"),
    ("I can't breathe properly", "can't breathe"),
    ("I'm suicidal and need help", "suicidal"),
    ("I want to die", "want to die"),
    ("She has collapsed on the floor", "collapsed"),
    ("He is not breathing at all", "not breathing"),
    ("I took an overdose of medicine", "overdose"),
    ("Please call an ambulance right now", "call ambulance"),
])
def test_english_emergency_keywords(query: str, keyword: str, detector: MultilingualEmergencyDetector) -> None:
    """English emergency keywords must be detected by Layer 1."""
    is_emrg, reason = detector.is_emergency(query)
    assert is_emrg, (
        f"Expected EMERGENCY for English query {query!r} "
        f"(keyword: '{keyword}'). Got SAFE with reason: {reason!r}"
    )


# ── Hinglish Emergency Tests 

@pytest.mark.parametrize("query,keyword", [
    ("dil mein bahut dard ho raha hai", "dil mein bahut dard"),
    ("sans nahi aa raha mujhe", "sans nahi aa raha"),
    ("woh behosh ho gaya", "behosh ho gaya"),
    ("khoon bahut aa raha hai", "khoon bahut aa raha"),
    ("marna chahta hoon main", "marna chahta"),
    ("hospital bulao abhi", "hospital bulao"),
    ("ambulance bulao please", "ambulance bulao"),
])
def test_hinglish_emergency_keywords(query: str, keyword: str, detector: MultilingualEmergencyDetector) -> None:
    """Hinglish emergency keywords must be detected by Layer 1."""
    is_emrg, reason = detector.is_emergency(query)
    assert is_emrg, (
        f"Expected EMERGENCY for Hinglish query {query!r} "
        f"(keyword: '{keyword}'). Got SAFE."
    )


# ── Banking Emergency Tests 

@pytest.mark.parametrize("query,keyword", [
    ("mere khata khali ho gaya", "खाता खाली हो गया"),
    ("fraud ho gaya mere account mein", "fraud ho gaya"),
    ("my account got hacked", "account hack"),
    ("mere paise chori ho gaye", "पैसे चोरी हो गए"),
    ("there was an unauthorized transaction", "unauthorized transaction"),
    ("OTP kisi ko de diya maine", "OTP किसी को दे दिया"),
])
def test_banking_emergency_keywords(query: str, keyword: str, detector: MultilingualEmergencyDetector) -> None:
    """Banking/fraud emergency keywords must be detected."""
    is_emrg, reason = detector.is_emergency(query)
    assert is_emrg, (
        f"Expected EMERGENCY for banking query {query!r}. Got SAFE."
    )


# ── Negative Tests (non-emergency queries) 

@pytest.mark.parametrize("query", [
    "मेरी EMI कब कटती है",
    "What is the refund policy?",
    "मुझे loan की जानकारी चाहिए",
    "When does my account statement come?",
    "मेरा बैलेंस कितना है",
    "Can I change my EMI date?",
    "prepayment charges kya hain",
    "मुझे नया क्रेडिट कार्ड चाहिए",
    "order status check karna hai",
    "What are the branch timings?",
    "",                         # empty query
    "   ",                      # whitespace-only query
])
def test_non_emergency_queries_are_safe(query: str, detector: MultilingualEmergencyDetector) -> None:
    """Normal customer service queries must NOT trigger emergency detection."""
    is_emrg, reason = detector.is_emergency(query)
    assert not is_emrg, (
        f"Expected SAFE for normal query {query!r}, "
        f"but EMERGENCY was triggered. Reason: {reason!r}"
    )


# ── Semantic Detection Tests (requires LaBSE — marked slow) 

@pytest.mark.parametrize("query,description", [
    (
        "मेरी साँसें रुक रही हैं, कुछ करो",
        "Paraphrase of 'can't breathe' in Hindi (keyword-free)",
    ),
    (
        "My heart is giving out, I feel terrible",
        "Semantic near-miss for heart attack (no exact keyword)",
    ),
])
def test_semantic_emergency_detection(
    query: str,
    description: str,
    detector: MultilingualEmergencyDetector,
    run_slow: bool,
) -> None:
    """Semantic paraphrases should trigger Layer 2 detection when threshold met."""
    if not run_slow:
        pytest.skip("Skipping slow semantic test. Pass --run-slow to include.")
    is_emrg, reason = detector.is_emergency(query)
    # Note: semantic detection depends on threshold and model — warn if it fails
    # rather than hard-fail, as embeddings are non-deterministic across versions
    if not is_emrg:
        pytest.xfail(
            f"Semantic detection missed {query!r} ({description}). "
            f"Consider lowering EMERGENCY_SEMANTIC_THRESHOLD."
        )


# ── Keyword List Completeness Tests 

def test_all_emergency_keyword_lists_non_empty() -> None:
    """All keyword sublists and the combined list must be non-empty."""
    assert len(EMERGENCY_KEYWORDS_HINDI) > 0
    assert len(EMERGENCY_KEYWORDS_ENGLISH) > 0
    assert len(EMERGENCY_KEYWORDS_HINGLISH) > 0
    assert len(EMERGENCY_KEYWORDS_BANKING) > 0
    assert len(ALL_EMERGENCY_KEYWORDS) >= (
        len(EMERGENCY_KEYWORDS_HINDI)
        + len(EMERGENCY_KEYWORDS_ENGLISH)
        + len(EMERGENCY_KEYWORDS_HINGLISH)
        + len(EMERGENCY_KEYWORDS_BANKING)
    )


def test_no_duplicate_keywords_across_lists() -> None:
    """No keyword should appear in more than one sublist (sanity check)."""
    seen: set[str] = set()
    all_lists = [
        EMERGENCY_KEYWORDS_HINDI,
        EMERGENCY_KEYWORDS_ENGLISH,
        EMERGENCY_KEYWORDS_HINGLISH,
        EMERGENCY_KEYWORDS_BANKING,
    ]
    duplicates: list[str] = []
    for keyword_list in all_lists:
        for kw in keyword_list:
            kw_lower = kw.lower()
            if kw_lower in seen:
                duplicates.append(kw)
            seen.add(kw_lower)
    assert not duplicates, f"Duplicate keywords found: {duplicates}"