"""
Reads ragas_eval_*.json reports produced by RagasEvaluator._save_report()
and generates a colour-coded HTML report for non-engineer stakeholders.

Consumes JSON schema:
  - results_by_language (dict, not list)
  - passed_faithfulness (not passed)
  - all_thresholds_passed (not overall_passed)
  - thresholds.faithfulness_by_language (per-language dict)
  - total_duration_seconds (not duration_seconds at top level)

Usage:
    python scripts/generate_eval_report.py eval_results/
    python scripts/generate_eval_report.py eval_results/ --out eval_results/report.html
    python scripts/generate_eval_report.py eval_results/ --latest 5
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


# ── Rendering helpers 

def _score_cell(score: float, threshold: float) -> str:
    """Render a score with colour-coded pass/fail and threshold annotation."""
    color = "#276749" if score >= threshold else "#c53030"
    return (
        f'<span style="color:{color};font-weight:700">{score:.3f}</span>'
        f'<br><span style="color:#a0aec0;font-size:.82rem">≥{threshold:.2f}</span>'
    )


def _badge(passed: bool) -> str:
    bg = "#c6f6d5;color:#22543d" if passed else "#fed7d7;color:#742a2a"
    text = "PASS" if passed else "FAIL"
    return (
        f'<span style="background:{bg};padding:2px 10px;border-radius:12px;'
        f'font-size:.82rem;font-weight:600">{text}</span>'
    )


# ── Report builder 

def build_report(report_paths: list[Path]) -> str:
    """Build a complete HTML report from one or more ragas_eval_*.json files."""
    runs: list[dict] = []
    for p in report_paths:
        try:
            runs.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception as exc:
            print(f"Warning: {p}: {exc}", file=sys.stderr)

    if not runs:
        return "<html><body><p>No reports found.</p></body></html>"

    all_results = [
        r
        for run in runs
        for r in run.get("results_by_language", {}).values()
    ]
    n_languages   = len({r["language"] for r in all_results})
    total_cases   = sum(r.get("n_cases", 0) for r in all_results)
    overall_passed = all(run.get("all_thresholds_passed", False) for run in runs)

    # ── Summary cards 
    summary_badge_color = "#276749" if overall_passed else "#c53030"
    summary_badge_text  = "ALL PASS" if overall_passed else "FAILURES"

    summary_cards = f"""
    <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:24px">
      <div style="background:#fff;border-radius:8px;padding:16px 24px;box-shadow:0 1px 4px rgba(0,0,0,.08)">
        <div style="font-size:.78rem;color:#718096;text-transform:uppercase;margin-bottom:4px">Overall</div>
        <div style="font-size:1.8rem;font-weight:700;color:{summary_badge_color}">{summary_badge_text}</div>
      </div>
      <div style="background:#fff;border-radius:8px;padding:16px 24px;box-shadow:0 1px 4px rgba(0,0,0,.08)">
        <div style="font-size:.78rem;color:#718096;text-transform:uppercase;margin-bottom:4px">Eval Runs</div>
        <div style="font-size:1.8rem;font-weight:700">{len(runs)}</div>
      </div>
      <div style="background:#fff;border-radius:8px;padding:16px 24px;box-shadow:0 1px 4px rgba(0,0,0,.08)">
        <div style="font-size:.78rem;color:#718096;text-transform:uppercase;margin-bottom:4px">Languages</div>
        <div style="font-size:1.8rem;font-weight:700">{n_languages}</div>
      </div>
      <div style="background:#fff;border-radius:8px;padding:16px 24px;box-shadow:0 1px 4px rgba(0,0,0,.08)">
        <div style="font-size:.78rem;color:#718096;text-transform:uppercase;margin-bottom:4px">Total Cases</div>
        <div style="font-size:1.8rem;font-weight:700">{total_cases}</div>
      </div>
    </div>"""

    # ── Per-run tables 
    run_tables = ""
    for run in runs:
        thresholds      = run.get("thresholds", {})
        faith_by_lang   = thresholds.get("faithfulness_by_language", {})
        relevancy_thr   = thresholds.get("response_relevancy", 0.75)
        precision_thr   = thresholds.get("llm_context_precision_without_reference", 0.70)
        recall_thr      = thresholds.get("context_recall", 0.65)
        run_passed      = run.get("all_thresholds_passed", False)
        run_duration    = run.get("total_duration_seconds", 0)

        rows = ""
        for lang, r in run.get("results_by_language", {}).items():
            faith_thr = r.get("faithfulness_threshold") or faith_by_lang.get(lang, 0.82)
            if r.get("error"):
                rows += (
                    f"<tr><td><code>{lang}</code></td>"
                    f'<td colspan="5" style="color:#c53030">{r["error"]}</td>'
                    f"<td>{_badge(False)}</td></tr>"
                )
            else:
                rows += (
                    f"<tr>"
                    f"<td><code>{lang}</code></td>"
                    f"<td style='text-align:center'>{r.get('n_cases', 0)}</td>"
                    f"<td>{_score_cell(r.get('faithfulness', 0.0), faith_thr)}</td>"
                    f"<td>{_score_cell(r.get('response_relevancy', 0.0), relevancy_thr)}</td>"
                    f"<td>{_score_cell(r.get('llm_context_precision', 0.0), precision_thr)}</td>"
                    f"<td>{_score_cell(r.get('context_recall', 0.0), recall_thr)}</td>"
                    f"<td>{_badge(r.get('passed_faithfulness', False))}</td>"
                    f"</tr>"
                )

        run_badge = _badge(run_passed)
        run_tables += f"""
        <div style="background:#fff;border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,.08);
                    padding:20px;margin-bottom:20px">
          <h2 style="margin:0 0 14px;font-size:1.05rem;border-bottom:2px solid #e2e8f0;padding-bottom:8px">
            Run: <code style="background:#edf2f7;padding:2px 6px;border-radius:4px">{run.get('run_id','?')}</code>
            &nbsp;{run_badge}&nbsp;
            <span style="font-weight:400;font-size:.85rem;color:#718096">
              {run.get('timestamp','')[:19]}Z &nbsp;·&nbsp; {run_duration:.1f}s
            </span>
          </h2>
          <table style="width:100%;border-collapse:collapse;font-size:.88rem">
            <thead style="background:#edf2f7">
              <tr>
                <th style="padding:8px 12px;text-align:left">Language</th>
                <th style="padding:8px 12px;text-align:center">Cases</th>
                <th style="padding:8px 12px;text-align:left">Faithfulness</th>
                <th style="padding:8px 12px;text-align:left">Relevancy</th>
                <th style="padding:8px 12px;text-align:left">Context Precision</th>
                <th style="padding:8px 12px;text-align:left">Context Recall</th>
                <th style="padding:8px 12px;text-align:left">Status</th>
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
        </div>"""

    # ── Threshold legend 
    legend = """
    <div style="background:#fff;border-radius:8px;padding:14px 20px;
                box-shadow:0 1px 4px rgba(0,0,0,.08);margin-bottom:20px;font-size:.85rem">
      <strong>Faithfulness thresholds (per-language):</strong>
      &nbsp; hi-IN ≥ 0.80 &nbsp;·&nbsp; en-IN ≥ 0.82 &nbsp;·&nbsp; hi-IN+en-IN ≥ 0.75
      <br>
      <span style="color:#718096">
        Lower Hinglish threshold is intentional: LLM judges reasoning in English
        underperform on code-mixed Hindi+English content.
      </span>
    </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>KoyalAI Evaluation Report</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    margin: 0; padding: 24px; background: #f5f7fa; color: #1a1a2e;
  }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #e2e8f0; vertical-align: top; }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover {{ background: #f7fafc; }}
</style>
</head>
<body>
<h1 style="color:#2d3748;margin-bottom:4px">🦜 KoyalAI Multilingual Evaluation Report</h1>
<div style="color:#718096;font-size:.9rem;margin-bottom:24px">
  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} &nbsp;·&nbsp;
  ragas 0.4.3 &nbsp;·&nbsp; deepeval ≥4.0
</div>
{summary_cards}
{legend}
{run_tables}
</body>
</html>"""


# ── CLI 

def main() -> int:
    parser = argparse.ArgumentParser(description="Generate HTML RAGAS eval report")
    parser.add_argument("reports_dir", type=Path, help="Directory containing ragas_eval_*.json files")
    parser.add_argument("--out", type=Path, default=None, help="Output HTML path (default: reports_dir/report.html)")
    parser.add_argument("--latest", type=int, default=None, metavar="N", help="Process only the N most recent reports")
    args = parser.parse_args()

    if not args.reports_dir.exists():
        print(f"Error: {args.reports_dir} not found", file=sys.stderr)
        return 2

    files = sorted(args.reports_dir.glob("ragas_eval_*.json"))
    if not files:
        print(f"No ragas_eval_*.json files in {args.reports_dir}", file=sys.stderr)
        return 2

    if args.latest:
        files = files[-args.latest:]

    out = args.out or (args.reports_dir / "report.html")
    out.write_text(build_report(files), encoding="utf-8")
    print(f"✅ Report: {out}  ({len(files)} run(s) included)")
    return 0


if __name__ == "__main__":
    sys.exit(main())