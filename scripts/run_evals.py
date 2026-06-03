"""
CLI runner for KoyalAI evaluation suite.

Usage:
    # Run all evaluations (GROQ_API_KEY required for RAGAS):
    python scripts/run_evals.py

    # Safety evaluation only (no API key required):
    python scripts/run_evals.py --safety-only

    # RAGAS evaluation only:
    python scripts/run_evals.py --ragas-only

    # Generate HTML report after RAGAS eval:
    python scripts/run_evals.py --ragas-only --html-report

    # Custom output directory:
    python scripts/run_evals.py --output-dir /tmp/eval_results

    # Fail fast on first threshold violation:
    python scripts/run_evals.py --fail-fast

Exit codes:
    0   All thresholds passed
    1   One or more thresholds failed
    2   Configuration error (missing API key, Qdrant unavailable)
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Force-load .env before any backend imports 
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

import argparse
import asyncio
import json
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KoyalAI Multilingual Evaluation Suite — Phase 5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--safety-only", action="store_true")
    parser.add_argument("--ragas-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("eval_results"))
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--no-assert", action="store_true")
    parser.add_argument(
        "--html-report",
        action="store_true",
        help="Generate HTML report after RAGAS evaluation (requires generate_eval_report.py).",
    )
    return parser.parse_args()


async def run_ragas_evaluation(
    output_dir: Path, fail_fast: bool, no_assert: bool, html_report: bool
) -> bool:
    import os
    if not os.getenv("GROQ_API_KEY"):
        logger.error(
            "GROQ_API_KEY is not set — RAGAS evaluation requires it.\n"
            "  export GROQ_API_KEY=gsk-..."
        )
        return False

    from backend.observability.ragas_eval import RagasEvaluator
    evaluator = RagasEvaluator(output_dir=output_dir)

    logger.info("Starting RAGAS multilingual evaluation...")
    report = await evaluator.run_multilingual_evaluation()
    evaluator.print_report(report)

    passed = True
    if not no_assert:
        try:
            evaluator.assert_thresholds(report)
            logger.info("✓ RAGAS: All thresholds passed.")
        except AssertionError as exc:
            logger.error("✗ RAGAS: Threshold failure:\n%s", exc)
            if fail_fast:
                sys.exit(1)
            passed = False
    else:
        passed = report.all_thresholds_passed

    if html_report:
        _generate_html_report(output_dir)

    return passed


def _generate_html_report(output_dir: Path) -> None:
    """Invoke generate_eval_report.py to produce an HTML report."""
    report_script = Path("scripts") / "generate_eval_report.py"
    if not report_script.exists():
        logger.warning("generate_eval_report.py not found — skipping HTML report.")
        return
    result = subprocess.run(
        [sys.executable, str(report_script), str(output_dir)],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        logger.info("HTML report: %s", result.stdout.strip())
    else:
        logger.warning("HTML report generation failed: %s", result.stderr.strip())


def run_safety_evaluation(output_dir: Path, fail_fast: bool, no_assert: bool) -> bool:
    from backend.observability.deepeval_suite import KoyalSafetyEvaluator

    evaluator = KoyalSafetyEvaluator()
    logger.info("Starting safety gate evaluation...")
    report = evaluator.run_safety_evaluation()
    evaluator.print_report(report)

    output_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime, timezone
    report_path = output_dir / f"safety_eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    report_data = {
        "total": report.total, "passed": report.passed, "failed": report.failed,
        "pass_rate": report.pass_rate,
        "escalation_accuracy": report.escalation_accuracy,
        "non_escalation_accuracy": report.non_escalation_accuracy,
        "failed_cases": [
            {
                "description": r.description, "query": r.query,
                "expected_escalate": r.expected_escalate,
                "actual_escalate": r.actual_escalate,
                "category": r.category, "error": r.error,
            }
            for r in report.failed_cases
        ],
    }
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report_data, fh, ensure_ascii=False, indent=2)
    logger.info("Safety report saved: %s", report_path)

    passed = True
    if not no_assert:
        try:
            evaluator.assert_all_passed(report)
            logger.info("✓ Safety: All %d tests passed.", report.total)
        except AssertionError as exc:
            logger.error("✗ Safety: %s", exc)
            if fail_fast:
                sys.exit(1)
            passed = False
    else:
        passed = report.pass_rate >= 1.0

    return passed


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_passed = True

    if not args.ragas_only:
        all_passed &= run_safety_evaluation(args.output_dir, args.fail_fast, args.no_assert)

    if not args.safety_only:
        all_passed &= asyncio.run(
            run_ragas_evaluation(
                args.output_dir, args.fail_fast, args.no_assert, args.html_report
            )
        )

    if all_passed:
        logger.info("\n" + "="*60)
        logger.info("✓ ALL EVALUATIONS PASSED — Phase 5 checkpoint met.")
        logger.info("="*60)
        sys.exit(0)
    else:
        logger.error("\n" + "="*60)
        logger.error("✗ ONE OR MORE EVALUATIONS FAILED — see reports in %s", args.output_dir)
        logger.error("="*60)
        sys.exit(1)


if __name__ == "__main__":
    main()