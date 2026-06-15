/**
 * src/app/evals/page.tsx — RAGAS Evaluation Viewer
 * ──────────────────────────────────────────────────
 * Shows multilingual RAGAS eval reports for each tenant.
 * Server Component shell — MetricsPanel is a Client Component.
 */

import type { Metadata } from 'next'
import { TENANT_IDS, TENANT_LABELS, RAGAS_FAITHFULNESS_THRESHOLDS, RAGAS_SHARED_THRESHOLDS } from '@/lib/constants'
import { MetricsPanel } from '@/components/MetricsPanel'
import { Card, CardHeader, CardTitle } from '@/components/ui/Card'

export const metadata: Metadata = {
  title: 'Evaluations',
}

export default function EvalsPage() {
  // Safely access shared thresholds with fallbacks
  const relevancyThreshold = RAGAS_SHARED_THRESHOLDS?.response_relevancy ?? 0.75
  const precisionThreshold = RAGAS_SHARED_THRESHOLDS?.llm_context_precision_without_reference ?? 0.70
  const recallThreshold = RAGAS_SHARED_THRESHOLDS?.context_recall ?? 0.65

  return (
    <div className="space-y-6">
      {/* Page header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-100">RAGAS Evaluation</h1>
        <p className="mt-1 text-sm text-slate-500">
          Multilingual faithfulness · relevancy · precision · recall — per-language thresholds
        </p>
      </div>

      {/* Threshold reference table */}
      <Card>
        <CardHeader>
          <CardTitle>
            Threshold Reference
            <span className="ml-2 text-xs font-normal text-slate-500">Phase 5 FAITHFULNESS_THRESHOLDS</span>
          </CardTitle>
        </CardHeader>
        <div className="px-5 pb-5 overflow-x-auto">
          <table className="w-full text-sm" aria-label="RAGAS thresholds by language">
            <thead>
              <tr className="border-b border-navy-600">
                <th scope="col" className="text-left py-2 px-3 text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Language
                </th>
                <th scope="col" className="text-left py-2 px-3 text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Faithfulness
                </th>
                <th scope="col" className="text-left py-2 px-3 text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Relevancy
                </th>
                <th scope="col" className="text-left py-2 px-3 text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Precision
                </th>
                <th scope="col" className="text-left py-2 px-3 text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Recall
                </th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(RAGAS_FAITHFULNESS_THRESHOLDS ?? {}).map(([lang, thr]) => (
                <tr key={lang} className="border-b border-navy-700 last:border-0">
                  <td className="py-2 px-3 text-slate-300 font-mono-data">{lang}</td>
                  <td className="py-2 px-3 text-emerald-400 font-mono-data">≥ {Math.round(thr * 100)}%</td>
                  <td className="py-2 px-3 text-slate-400 font-mono-data">≥ {Math.round(relevancyThreshold * 100)}%</td>
                  <td className="py-2 px-3 text-slate-400 font-mono-data">≥ {Math.round(precisionThreshold * 100)}%</td>
                  <td className="py-2 px-3 text-slate-400 font-mono-data">≥ {Math.round(recallThreshold * 100)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="px-5 pb-5">
          <p className="text-xs text-slate-500 bg-navy-800/50 rounded-lg px-3 py-2 border border-navy-600">
            Hinglish threshold (0.75) is intentionally lower due to LLM judge bias on code-mixed text — see Phase 5 implementation.
          </p>
        </div>
      </Card>

      {/* Per-tenant eval panels */}
      <div className="space-y-4">
        {(TENANT_IDS ?? []).map((tid) => (
          <MetricsPanel key={tid} tenantId={tid} />
        ))}
      </div>

      {/* Runbook hint */}
      <Card>
        <CardHeader>
          <CardTitle>Run Evaluation</CardTitle>
        </CardHeader>
        <div className="px-5 pb-5 space-y-3">
          <p className="text-sm text-slate-400">
            Trigger a fresh RAGAS evaluation run from the terminal:
          </p>
          <pre className="rounded-lg bg-navy-900 border border-navy-700 p-4 overflow-x-auto">
            <code className="text-xs font-mono-data text-slate-300 leading-relaxed">
{`# Run full multilingual RAGAS + DeepEval suite
python scripts/run_evals.py

# Run RAGAS only (faster)
python scripts/run_evals.py --ragas-only

# Run safety eval only (deterministic, no API key)
python scripts/run_evals.py --safety-only`}
            </code>
          </pre>
          <p className="text-xs text-slate-500">
            Results refresh automatically every 30 seconds in the UI.
          </p>
        </div>
      </Card>
    </div>
  )
}