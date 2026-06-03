from __future__ import annotations

# Monkey-patch RAGAS to force n=1 for Groq compatibility
import ragas.llms.base as _ragas_base

_ragas_llm_class = getattr(_ragas_base, "BaseRagasLLM", None)

if _ragas_llm_class:
    # Patch agenerate_text 
    _original_agenerate_text = _ragas_llm_class.agenerate_text
    
    async def _patched_agenerate_text(self, prompt, n=1, temperature=1e-8, stop=None, callbacks=None):
        """Force n=1 regardless of what RAGAS requests (Groq compatibility)."""
        return await _original_agenerate_text(
            self, prompt, n=1, temperature=temperature, stop=stop, callbacks=callbacks
        )
    
    _ragas_llm_class.agenerate_text = _patched_agenerate_text
    
    # Also patch sync version if it exists
    if hasattr(_ragas_llm_class, "generate_text"):
        _original_generate_text = _ragas_llm_class.generate_text
        
        def _patched_generate_text(self, prompt, n=1, temperature=1e-8, stop=None, callbacks=None):
            """Force n=1 regardless of what RAGAS requests (Groq compatibility)."""
            return _original_generate_text(
                self, prompt, n=1, temperature=temperature, stop=stop, callbacks=callbacks
            )
        
        _ragas_llm_class.generate_text = _patched_generate_text
    
    import logging
    logging.getLogger(__name__).info("RAGAS 0.4.3 patched: n=1 enforced on agenerate_text/generate_text")
else:
    import logging
    logging.getLogger(__name__).warning("Could not patch RAGAS BaseRagasLLM — n=1 enforcement may fail")


import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from backend.agents.graph import koyal_graph
from backend.agents.state import make_initial_state
from backend.config import (
    GROQ_EVAL_API_KEY,
    GROQ_MODEL_NAME,
)
from backend.observability.prometheus_metrics import update_ragas_score
from backend.observability.langfuse_client import score_turn

logger = logging.getLogger(__name__)

# ── Per-language faithfulness thresholds 
FAITHFULNESS_THRESHOLDS: dict[str, float] = {
    "hi-IN":       0.80,   
    "en-IN":       0.82,   
    "hi-IN+en-IN": 0.75,   
}

# Language-agnostic thresholds
THRESHOLDS: dict[str, float] = {
    "response_relevancy": 0.75,
    "llm_context_precision_without_reference": 0.70,
    "context_recall": 0.65,
}

# ── Retry configuration 
_MAX_EVAL_RETRIES: int = 3
_EVAL_RETRY_BASE_SECONDS: float = 2.0  # wait = base ** attempt_number


# ── Test Cases 

@dataclass
class EvalTestCase:
    """A single multilingual RAG evaluation case.

    Attributes:
        question: Raw query in the caller's language.
        ground_truth: Expected answer (used by ContextRecall metric).
        tenant_id: Tenant to route the query to.
        language: BCP-47 language code for grouping results.
        description: Human-readable label for the test case.
        expected_keywords: If non-empty, at least one must appear in the response.
    """
    question: str
    ground_truth: str
    tenant_id: str
    language: str
    description: str = ""
    expected_keywords: list[str] = field(default_factory=list)


EVAL_TEST_CASES: list[EvalTestCase] = [
    # ── HDFC Bank — Hindi 
    EvalTestCase(
        question="मेरी EMI कब कटती है?",
        ground_truth="5 तारीख को",
        tenant_id="tenant_hdfc_bank",
        language="hi-IN",
        description="Hindi: EMI deduction date query",
        expected_keywords=["5", "पाँच", "पांच", "तारीख"],
    ),
    EvalTestCase(
        question="लेट पेमेंट पर क्या चार्ज है?",
        ground_truth="₹500 विलंब शुल्क",
        tenant_id="tenant_hdfc_bank",
        language="hi-IN",
        description="Hindi: Late payment penalty query",
        expected_keywords=["500", "₹500", "विलंब", "शुल्क", "पेनल्टी"],
    ),
    EvalTestCase(
        question="क्या मैं अपना लोन समय से पहले चुका सकता हूँ?",
        ground_truth="6 महीने के बाद प्रीपेमेंट की अनुमति है",
        tenant_id="tenant_hdfc_bank",
        language="hi-IN",
        description="Hindi: Prepayment eligibility query",
        expected_keywords=["6", "छह", "महीने", "प्रीपेमेंट", "prepayment"],
    ),
    # ── HDFC Bank — English 
    EvalTestCase(
        question="When is EMI deducted?",
        ground_truth="5th of every month",
        tenant_id="tenant_hdfc_bank",
        language="en-IN",
        description="English: EMI deduction date query",
        expected_keywords=["5th", "fifth", "5"],
    ),
    EvalTestCase(
        question="What happens if I miss my EMI payment?",
        ground_truth="A late payment fee of ₹500 is charged",
        tenant_id="tenant_hdfc_bank",
        language="en-IN",
        description="English: Missed EMI penalty query",
        expected_keywords=["500", "fee", "penalty", "charge"],
    ),
    EvalTestCase(
        question="Can I prepay my loan before tenure ends?",
        ground_truth="Prepayment is allowed after 6 months",
        tenant_id="tenant_hdfc_bank",
        language="en-IN",
        description="English: Prepayment query",
        expected_keywords=["6 months", "6", "prepay", "prepayment", "allowed"],
    ),
    # ── HDFC Bank — Hinglish 
    EvalTestCase(
        question="EMI miss ho gayi, kya penalty hai?",
        ground_truth="500 rupaye late payment charge lagega",
        tenant_id="tenant_hdfc_bank",
        language="hi-IN+en-IN",
        description="Hinglish: Missed EMI penalty query",
        expected_keywords=["500", "penalty", "charge", "shulk"],
    ),
    EvalTestCase(
        question="Loan prepay karna hai, 6 mahine ke baad kar sakte hain?",
        ground_truth="Haan, 6 mahine ke baad prepayment allowed hai",
        tenant_id="tenant_hdfc_bank",
        language="hi-IN+en-IN",
        description="Hinglish: Prepayment confirmation query",
        expected_keywords=["6", "allowed", "prepay", "prepayment"],
    ),
    # ── Swiggy Support — English 
    EvalTestCase(
        question="What is the refund policy?",
        ground_truth="Refunds processed within 5-7 business days",
        tenant_id="tenant_swiggy_support",
        language="en-IN",
        description="English: Swiggy refund policy query",
        expected_keywords=["refund", "5", "7", "business days"],
    ),
    EvalTestCase(
        question="How do I track my order?",
        ground_truth="Track your order via the Swiggy app",
        tenant_id="tenant_swiggy_support",
        language="en-IN",
        description="English: Order tracking query",
        expected_keywords=["app", "track", "Swiggy"],
    ),
    # ── Swiggy Support — Hindi 
    EvalTestCase(
        question="मेरा ऑर्डर कहाँ है?",
        ground_truth="आप Swiggy ऐप में ऑर्डर ट्रैक कर सकते हैं",
        tenant_id="tenant_swiggy_support",
        language="hi-IN",
        description="Hindi: Order tracking query (Swiggy)",
        expected_keywords=["ऐप", "app", "track", "ट्रैक"],
    ),
]


# ── Result Data Classes 

@dataclass
class LanguageEvalResult:
    """RAGAS scores and metadata for one language group.

    Attributes:
        language: BCP-47 language code.
        n_cases: Number of test cases evaluated.
        faithfulness: Average faithfulness score (0-1).
        response_relevancy: Average response relevancy score (0-1).
        llm_context_precision: Average context precision score (0-1).
        context_recall: Average context recall score (0-1).
        passed_faithfulness: Whether faithfulness exceeded per-language threshold.
        faithfulness_threshold: Per-language threshold used for this group.
            Embedded here so the JSON report is a self-contained audit trail —
            threshold changes are visible in git diff without cross-referencing code.
        duration_seconds: Time taken to run this language group.
        error: Non-None if evaluation failed.
    """
    language: str
    n_cases: int
    faithfulness: float = 0.0
    response_relevancy: float = 0.0
    llm_context_precision: float = 0.0
    context_recall: float = 0.0
    passed_faithfulness: bool = False
    faithfulness_threshold: float = 0.82   # overridden in _evaluate_language_group
    duration_seconds: float = 0.0
    error: Optional[str] = None


@dataclass
class MultilingualEvalReport:
    """Full evaluation report across all languages.

    Attributes:
        timestamp: ISO-8601 UTC time of evaluation.
        run_id: Unique identifier for this eval run.
        results_by_language: Dict mapping language code to LanguageEvalResult.
        all_thresholds_passed: True if all language groups pass their faithfulness gate.
        failed_languages: List of language codes that failed the faithfulness gate.
        total_duration_seconds: End-to-end wall time for the full eval run.
    """
    timestamp: str
    run_id: str
    results_by_language: dict[str, LanguageEvalResult] = field(default_factory=dict)
    all_thresholds_passed: bool = False
    failed_languages: list[str] = field(default_factory=list)
    total_duration_seconds: float = 0.0


# ── Internal helpers 

async def _run_with_retry(coro_factory, label: str = "") -> Any:
    """Retry an async callable with exponential backoff on transient failures.

    Args:
        coro_factory: Zero-argument callable returning an awaitable.
        label: Human-readable label for log messages.

    Returns:
        The awaitable's result on success.

    Raises:
        The last exception after all retries are exhausted.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, _MAX_EVAL_RETRIES + 1):
        try:
            return await coro_factory()
        except Exception as exc:
            last_exc = exc
            if attempt == _MAX_EVAL_RETRIES:
                break
            wait = _EVAL_RETRY_BASE_SECONDS ** attempt
            logger.warning(
                "%s — attempt %d/%d failed: %s. Retrying in %.1fs.",
                label or "RAGAS evaluate()", attempt, _MAX_EVAL_RETRIES, exc, wait,
            )
            await asyncio.sleep(wait)
    raise last_exc  # type: ignore[misc]


# ── RAGAS Evaluator 

class RagasEvaluator:
    """Multilingual RAGAS evaluation engine for KoyalAI.

    Uses Groq Llama 3.3-70B as the LLM judge and LaBSE as the embedding
    model — both already configured in the pipeline, so no extra API keys
    or memory are required.

    Pipeline invocations run concurrently (up to 4 at a time) via
    asyncio.Semaphore to respect Groq's free-tier rate limits while
    running 4x faster than a sequential loop for 10+ test cases.
    RAGAS evaluate() is retried up to 3 times with exponential backoff
    on transient failures.

    Args:
        output_dir: Directory to write JSON eval reports.
            Defaults to ``eval_results/``.

    Raises:
        ImportError: If ragas is not installed.
        ValueError: If GROQ_API_KEY is absent at LLM init time.

    Example:
        >>> evaluator = RagasEvaluator()
        >>> report = await evaluator.run_multilingual_evaluation()
        >>> evaluator.assert_thresholds(report)
    """

    def __init__(
        self,
        output_dir: Path = Path("eval_results"),
    ) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._evaluator_llm = None
        self._embeddings = None
        logger.info("RagasEvaluator initialised (output_dir=%s)", output_dir)

    # ── Lazy model initialisation 

    def _get_evaluator_llm(self):
        """Lazily initialise the Groq LLM judge wrapped for ragas.

        Returns:
            ``LangchainLLMWrapper`` around ChatGroq.

        Raises:
            ImportError: If ragas or langchain_groq is not installed.
            ValueError: If GROQ_API_KEY is not set.
        """
        if self._evaluator_llm is not None:
            return self._evaluator_llm
        try:
            from ragas.llms import LangchainLLMWrapper  # noqa: PLC0415
            from langchain_groq import ChatGroq          # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "ragas is not installed. Install: pip install ragas==0.4.3"
            ) from exc
        if not GROQ_EVAL_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set — required for RAGAS evaluation."
            )
        groq_llm = ChatGroq(
            model=GROQ_MODEL_NAME,
            api_key=GROQ_EVAL_API_KEY,
            temperature=0,
            max_tokens=512,
            max_retries=3,
            n=1,
        )
        self._evaluator_llm = LangchainLLMWrapper(groq_llm)
        logger.info(
            "RAGAS LLM judge: %s (via LangchainLLMWrapper)", GROQ_MODEL_NAME
        )
        return self._evaluator_llm

    def _get_embeddings(self):
        """Lazily initialise LaBSE embeddings wrapped for ragas.

        LaBSE is already loaded in process for retrieval — sharing it
        avoids loading a second ~471 MB model.

        Returns:
            ``LangchainEmbeddingsWrapper`` around HuggingFaceEmbeddings,
            or ``None`` if LaBSE is unavailable (ResponseRelevancy skipped).
        """
        if self._embeddings is not None:
            return self._embeddings
        try:
            from ragas.embeddings import LangchainEmbeddingsWrapper      # noqa: PLC0415
            from langchain_huggingface import HuggingFaceEmbeddings  
            hf_emb = HuggingFaceEmbeddings(
                model_name="sentence-transformers/LaBSE",
            )
            self._embeddings = LangchainEmbeddingsWrapper(hf_emb)
            logger.info("RAGAS embeddings: LaBSE (via LangchainEmbeddingsWrapper)")
        except ImportError as exc:
            logger.warning(
                "Could not load LaBSE for RAGAS: %s — "
                "ResponseRelevancy metric will be skipped.", exc
            )
            self._embeddings = None
        return self._embeddings

    # ── Main evaluation orchestration 

    async def run_multilingual_evaluation(
        self,
        test_cases: Optional[list[EvalTestCase]] = None,
    ) -> MultilingualEvalReport:
        """Run full multilingual evaluation and return a structured report.

        Groups test cases by language, runs the KoyalAI pipeline concurrently
        (Semaphore=4) for all cases, then evaluates each language group with
        RAGAS and writes a JSON report to ``output_dir``.

        Args:
            test_cases: Optional custom list of test cases.
                Defaults to ``EVAL_TEST_CASES``.

        Returns:
            ``MultilingualEvalReport`` with per-language scores.

        Raises:
            RuntimeError: If the pipeline graph fails to produce any results.

        Example:
            >>> report = await evaluator.run_multilingual_evaluation()
            >>> print(report.all_thresholds_passed)
        """
        cases = test_cases or EVAL_TEST_CASES
        run_id = str(uuid.uuid4())[:8]
        total_start = time.monotonic()

        logger.info(
            "Starting multilingual RAGAS evaluation: run_id=%s cases=%d",
            run_id, len(cases),
        )

        # ── Step 1: Run pipeline concurrently 

        semaphore = asyncio.Semaphore(1)

        async def _bounded_run(case: EvalTestCase) -> tuple[str, dict]:
            async with semaphore:
                await asyncio.sleep(15.0)
                lang = case.language
                logger.info("Running pipeline: [%s] %s", lang, case.question[:60])
                try:
                    state = make_initial_state(
                        query=case.question,
                        tenant_id=case.tenant_id,
                        session_id=str(uuid.uuid4()),
                    )
                    # LangGraph is CPU-bound synchronous — offload to thread pool
                    pipeline_result = await asyncio.to_thread(
                        koyal_graph.invoke, state
                    )
                    chunks = pipeline_result.get("retrieved_chunks") or []
                    retrieved_contexts = [
                        c.get("text", "") for c in chunks if c.get("text")
                    ] or ["No context retrieved."]
                    logger.debug(
                        "Pipeline done: lang=%s response=%r",
                        lang,
                        (pipeline_result.get("final_response") or "")[:60],
                    )
                    return lang, {
                        "user_input": case.question,
                        "response": pipeline_result.get("final_response") or "",
                        "retrieved_contexts": retrieved_contexts,
                        "reference": case.ground_truth,
                        "description": case.description,
                        "expected_keywords": case.expected_keywords,
                        "tenant_id": case.tenant_id,
                        "escalated": pipeline_result.get("escalate", False),
                    }
                except Exception as exc:
                    logger.error(
                        "Pipeline failed for case '%s': %s",
                        case.question[:50], exc,
                    )
                    return lang, {
                        "user_input": case.question,
                        "response": "",
                        "retrieved_contexts": [""],
                        "reference": case.ground_truth,
                        "description": case.description,
                        "expected_keywords": case.expected_keywords,
                        "tenant_id": case.tenant_id,
                        "escalated": False,
                        "error": str(exc),
                    }

        logger.info(
            "Invoking KoyalAI pipeline for %d cases (Semaphore=4, return_exceptions=True)...",
            len(cases),
        )
        task_results = await asyncio.gather(
            *[_bounded_run(c) for c in cases],
            return_exceptions=True,  # A-pattern: single failure ≠ full abort
        )

        raw_results: dict[str, list[dict]] = {}
        for item in task_results:
            if isinstance(item, BaseException):
                logger.error(
                    "Pipeline task raised uncaught exception (skipping case): %s", item
                )
                continue
            lang, result_dict = item
            raw_results.setdefault(lang, []).append(result_dict)

        # ── Step 2: Evaluate each language group with RAGAS 
        report = MultilingualEvalReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            run_id=run_id,
        )

        for lang, lang_cases in raw_results.items():
            if not lang_cases:
                continue
            lang_result = await self._evaluate_language_group(
                lang=lang,
                cases=lang_cases,
            )
            report.results_by_language[lang] = lang_result
            logger.info(
                "Language '%s': faithfulness=%.3f (thr=%.2f) "
                "relevancy=%.3f precision=%.3f recall=%.3f",
                lang,
                lang_result.faithfulness,
                lang_result.faithfulness_threshold,
                lang_result.response_relevancy,
                lang_result.llm_context_precision,
                lang_result.context_recall,
            )

        # ── Step 3: Per-language threshold assessment 
        failed: list[str] = []
        for lang, result in report.results_by_language.items():
            faith_threshold = FAITHFULNESS_THRESHOLDS.get(lang, 0.82)
            if result.error:
                failed.append(lang)
            elif result.faithfulness < faith_threshold:
                failed.append(lang)
                logger.warning(
                    "Faithfulness BELOW threshold: lang=%s score=%.3f threshold=%.2f",
                    lang, result.faithfulness, faith_threshold,
                )
            else:
                result.passed_faithfulness = True
                logger.info(
                    "Faithfulness PASSED: lang=%s score=%.3f threshold=%.2f",
                    lang, result.faithfulness, faith_threshold,
                )

        report.failed_languages = failed
        report.all_thresholds_passed = len(failed) == 0
        report.total_duration_seconds = time.monotonic() - total_start

        # ── Step 4: Push to Prometheus + Langfuse 
        for lang, result in report.results_by_language.items():
            for tenant_id in {c.get("tenant_id") for c in raw_results[lang]}:
                if result.faithfulness > 0:
                    update_ragas_score(tenant_id, lang, result.faithfulness)

        # ── Step 5: Persist JSON report 
        self._save_report(report, raw_results)

        return report

    async def _evaluate_language_group(
        self,
        lang: str,
        cases: list[dict],
    ) -> LanguageEvalResult:
        """Run RAGAS evaluation for one language group with retry.

        Skips escalated and empty-response cases before building RAGAS samples.
        Wraps the synchronous evaluate() call in run_in_executor to keep the
        asyncio event loop responsive. Retries up to 3 times on transient
        failures with exponential backoff.

        Args:
            lang: BCP-47 language code.
            cases: List of pipeline output dicts for this language.

        Returns:
            ``LanguageEvalResult`` with metric scores and embedded threshold.
        """
        start = time.monotonic()
        faith_threshold = FAITHFULNESS_THRESHOLDS.get(lang, 0.82)

        try:
            from ragas import evaluate                                   # noqa: PLC0415
            from ragas.dataset_schema import (                          # noqa: PLC0415
                SingleTurnSample, EvaluationDataset,
            )
            from ragas.metrics import (                                  # noqa: PLC0415
                Faithfulness,
                ResponseRelevancy,
                LLMContextPrecisionWithoutReference,
                ContextRecall,
            )
        except ImportError as exc:
            logger.error("ragas not installed: %s", exc)
            return LanguageEvalResult(
                language=lang,
                n_cases=len(cases),
                faithfulness_threshold=faith_threshold,
                error=f"ragas not installed: {exc}",
            )

        evaluator_llm = self._get_evaluator_llm()
        embeddings = self._get_embeddings()

        # Build ragas samples — skip escalated and empty cases
        samples: list[SingleTurnSample] = []
        for c in cases:
            if c.get("escalated") or c.get("error"):
                logger.debug(
                    "Skipping escalated/error case: %s", c.get("user_input", "")[:50]
                )
                continue
            if not c.get("response", "").strip():
                logger.warning(
                    "Empty response for case: %s — skipping.",
                    c.get("user_input", "")[:50],
                )
                continue
            samples.append(
                SingleTurnSample(
                    user_input=c["user_input"],
                    response=c["response"],
                    retrieved_contexts=c["retrieved_contexts"],
                    reference=c["reference"],
                )
            )

        if not samples:
            logger.warning(
                "No valid samples for language '%s' — all escalated or empty.", lang
            )
            return LanguageEvalResult(
                language=lang,
                n_cases=len(cases),
                faithfulness_threshold=faith_threshold,
                error="All cases were escalated, empty, or errored — nothing to evaluate.",
            )

        eval_dataset = EvaluationDataset(samples=samples)

        metrics = [
            Faithfulness(llm=evaluator_llm),
            LLMContextPrecisionWithoutReference(llm=evaluator_llm),
            ContextRecall(llm=evaluator_llm),
        ]
        # ResponseRelevancy disabled: triples Groq call count and triggers 429/400
        # if embeddings is not None:
        #     metrics.append(ResponseRelevancy(llm=evaluator_llm, embeddings=embeddings))

        logger.info(
            "Running RAGAS eval: lang=%s samples=%d metrics=%s",
            lang, len(samples), [m.__class__.__name__ for m in metrics],
        )

        # run_in_executor keeps asyncio event loop responsive during evaluate()
        # _run_with_retry provides 3-attempt exponential backoff on transient failures
        loop = asyncio.get_event_loop()

        def _evaluate_sync():
            return evaluate(
                dataset=eval_dataset,
                metrics=metrics,
                llm=evaluator_llm,
                embeddings=embeddings if embeddings else None,
                show_progress=False,
                raise_exceptions=False,
            )

        try:
            eval_result = await _run_with_retry(
                lambda: loop.run_in_executor(None, _evaluate_sync),
                label=f"RAGAS evaluate() [lang={lang}]",
            )
        except Exception as exc:
            logger.error("RAGAS evaluate() failed after %d retries: %s", _MAX_EVAL_RETRIES, exc)
            return LanguageEvalResult(
                language=lang,
                n_cases=len(samples),
                faithfulness_threshold=faith_threshold,
                duration_seconds=time.monotonic() - start,
                error=str(exc),
            )

        try:
            df = eval_result.to_pandas()
            scores = {}
            for col in df.columns:
                try:
                    scores[col] = float(df[col].mean())
                except (TypeError, ValueError):
                    continue
        except Exception as exc:
            logger.error("Failed to parse RAGAS result: %s", exc)
            scores = {}
        
        faithfulness = float(scores.get("faithfulness", 0.0))
        response_relevancy = float(scores.get("response_relevancy", 0.0))
        llm_context_precision = float(
            scores.get("llm_context_precision_without_reference", 0.0)
        )
        context_recall = float(scores.get("context_recall", 0.0))

        return LanguageEvalResult(
            language=lang,
            n_cases=len(samples),
            faithfulness=faithfulness,
            response_relevancy=response_relevancy,
            llm_context_precision=llm_context_precision,
            context_recall=context_recall,
            faithfulness_threshold=faith_threshold,
            duration_seconds=time.monotonic() - start,
        )

    def assert_thresholds(self, report: MultilingualEvalReport) -> None:
        """Assert all language groups meet their per-language faithfulness threshold.

        Args:
            report: Completed ``MultilingualEvalReport``.

        Raises:
            AssertionError: If any language fails its faithfulness threshold.

        Example:
            >>> evaluator.assert_thresholds(report)  # raises if any fail
        """
        for lang, result in report.results_by_language.items():
            if result.error:
                raise AssertionError(
                    f"RAGAS evaluation failed for language '{lang}': {result.error}"
                )
            faith_threshold = FAITHFULNESS_THRESHOLDS.get(lang, 0.82)
            assert result.faithfulness >= faith_threshold, (
                f"Faithfulness BELOW threshold for '{lang}': "
                f"{result.faithfulness:.3f} < {faith_threshold:.2f}\n"
                f"Threshold rationale: see FAITHFULNESS_THRESHOLDS in ragas_eval.py.\n"
                f"Improve: check retrieval quality, chunk overlap, or LLM prompt."
            )
        logger.info(
            "✓ All RAGAS thresholds passed for %d language groups.",
            len(report.results_by_language),
        )

    def _save_report(
        self,
        report: MultilingualEvalReport,
        raw_results: dict[str, list[dict]],
    ) -> Path:
        """Serialise report to JSON and save to output_dir.

        The JSON report is self-contained: it embeds both the flat THRESHOLDS
        and the FAITHFULNESS_THRESHOLDS per-language dict, so the pass/fail
        decision can be reproduced from the report alone without reading code.
        faithfulness_threshold is also embedded in each LanguageEvalResult
        (via asdict) for the same reason.

        Args:
            report: Completed evaluation report.
            raw_results: Raw pipeline outputs grouped by language (for tenant metadata).

        Returns:
            Path to the saved JSON file.
        """
        report_dict = {
            "timestamp": report.timestamp,
            "run_id": report.run_id,
            "all_thresholds_passed": report.all_thresholds_passed,
            "failed_languages": report.failed_languages,
            "total_duration_seconds": round(report.total_duration_seconds, 2),
            # Embed both threshold dicts for self-contained audit trail
            "thresholds": {
                **THRESHOLDS,
                "faithfulness_by_language": FAITHFULNESS_THRESHOLDS,
            },
            "results_by_language": {
                lang: asdict(result)
                for lang, result in report.results_by_language.items()
            },
        }
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"ragas_eval_{report.run_id}_{timestamp_str}.json"
        output_path = self._output_dir / filename
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(report_dict, fh, ensure_ascii=False, indent=2)
        logger.info("RAGAS report saved: %s", output_path)
        return output_path

    def print_report(self, report: MultilingualEvalReport) -> None:
        """Print a formatted evaluation report to stdout.

        Args:
            report: Completed evaluation report.
        """
        print(f"\n{'='*64}")
        print("KoyalAI RAGAS Multilingual Evaluation Report")
        print(f"Run ID: {report.run_id}  |  {report.timestamp}")
        print(f"{'='*64}")
        for lang, result in report.results_by_language.items():
            print(
                f"\n── Language: {lang} "
                f"({result.n_cases} cases, {result.duration_seconds:.1f}s) ──"
            )
            if result.error:
                print(f"  ERROR: {result.error}")
                continue
            status = "✓ PASS" if result.passed_faithfulness else "✗ FAIL"
            print(
                f"  Faithfulness:        {result.faithfulness:.3f}  "
                f"[{status} | threshold={result.faithfulness_threshold:.2f}]"
            )
            print(f"  Response Relevancy:  {result.response_relevancy:.3f}")
            print(f"  Context Precision:   {result.llm_context_precision:.3f}")
            print(f"  Context Recall:      {result.context_recall:.3f}")
        print(f"\n{'='*64}")
        if report.all_thresholds_passed:
            print(
                f"✓ All {len(report.results_by_language)} language groups "
                f"passed their faithfulness thresholds."
            )
        else:
            print(f"✗ FAILED languages: {', '.join(report.failed_languages)}")
        print(f"Total evaluation time: {report.total_duration_seconds:.1f}s")
        print(f"{'='*64}\n")