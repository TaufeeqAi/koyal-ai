/**
 * src/components/MetricsPanel.tsx
 * ────────────────────────────────
 * RAGAS evaluation scores panel with per-language bar charts.
 * Uses recharts BarChart — must be a Client Component.
 * Polls /evals/ragas every 30s (POLL_RAGAS_SCORES).
 */

'use client'

import { BarChart, Bar, Cell, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import { useRagasReport } from '@/lib/api'
import { LANGUAGE_LABELS, RAGAS_FAITHFULNESS_THRESHOLDS } from '@/lib/constants'
import { formatScore, scoreToColor, cn } from '@/lib/utils'
import { Card, CardHeader, CardTitle } from '@/components/ui/Card'
import { PageLoader } from '@/components/ui/LoadingSpinner'
import { ErrorBanner } from '@/components/ui/ErrorBanner'
import { Badge } from '@/components/ui/Badge'

interface MetricsPanelProps {
  tenantId: string
}

// Full labels for tooltips (mapping short chart names → display names)
const METRIC_DISPLAY_NAMES: Record<string, string> = {
  Faith:   'Faithfulness',
  Relev:   'Response Relevancy',
  Prec:    'LLM Context Precision',
  Recall:  'Context Recall',
}

// Default thresholds (fallback if API doesn't provide per-metric thresholds)
const DEFAULT_THRESHOLDS: Record<string, number> = {
  Faith:   0.82,  // Overridden by result.faithfulness_threshold if available
  Relev:   0.75,
  Prec:    0.70,
  Recall:  0.65,
}

const COLOR_MAP: Record<string, string> = {
  emerald: '#34d399',
  amber:   '#fbbf24',
  rose:    '#f87171',
}

export function MetricsPanel({ tenantId }: MetricsPanelProps) {
  const { data, error, isLoading } = useRagasReport(tenantId)

  return (
    <Card>
      <CardHeader>
        <CardTitle>
          RAGAS Scores
          {data && (
            <Badge
              variant={data.all_thresholds_passed ? 'success' : 'error'}
              className="ml-2"
            >
              {data.all_thresholds_passed ? 'All Pass' : 'Failures'}
            </Badge>
          )}
        </CardTitle>
      </CardHeader>

      {isLoading && <PageLoader label="Loading RAGAS scores..." />}
      {error && <ErrorBanner message={error.message} />}

      {!isLoading && !data && !error && (
        <div className="px-5 pb-5 text-center py-8">
          <p className="text-slate-400 text-sm">No evaluation data yet.</p>
          <p className="text-slate-600 text-xs mt-1">
            Run <code className="bg-navy-700 px-1 py-0.5 rounded text-slate-300">python scripts/run_evals.py</code>
          </p>
        </div>
      )}

      {data && (
        <div className="px-5 pb-5 space-y-6">
          {Object.entries(data.results_by_language).map(([lang, result]) => {
            if (result.error) {
              return (
                <div key={lang} className="rounded-lg border border-rose-500/20 bg-rose-500/5 p-3">
                  <p className="text-sm text-rose-400 font-medium">
                    {LANGUAGE_LABELS[lang as keyof typeof LANGUAGE_LABELS] ?? lang}
                  </p>
                  <p className="text-xs text-rose-300/70 mt-1">{result.error}</p>
                </div>
              )
            }

            // Build chart data with per-metric thresholds (API-provided or default)
            const chartData = [
              { 
                name: 'Faith', 
                score: result.faithfulness, 
                threshold: result.faithfulness_threshold ?? DEFAULT_THRESHOLDS.Faith 
              },
              { 
                name: 'Relev', 
                score: result.response_relevancy, 
                threshold: DEFAULT_THRESHOLDS.Relev 
              },
              { 
                name: 'Prec', 
                score: result.llm_context_precision, 
                threshold: DEFAULT_THRESHOLDS.Prec 
              },
              { 
                name: 'Recall', 
                score: result.context_recall, 
                threshold: DEFAULT_THRESHOLDS.Recall 
              },
            ]

            // Get the actual faithfulness threshold used for this language (from result or constant)
            const faithThresholdUsed = result.faithfulness_threshold ?? 
              (RAGAS_FAITHFULNESS_THRESHOLDS[lang as keyof typeof RAGAS_FAITHFULNESS_THRESHOLDS] ?? 0.82)

            return (
              <div key={lang} className="space-y-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-semibold text-slate-200">
                      {LANGUAGE_LABELS[lang as keyof typeof LANGUAGE_LABELS] ?? lang}
                    </span>
                    <span className="text-xs text-slate-500">
                      ({result.n_cases} cases)
                    </span>
                  </div>
                  <Badge variant={result.passed_faithfulness ? 'success' : 'error'}>
                    {result.passed_faithfulness ? 'PASS' : 'FAIL'}
                  </Badge>
                </div>

                <div className="h-40">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={chartData} barSize={32}>
                      <XAxis
                        dataKey="name"
                        tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'JetBrains Mono' }}
                        axisLine={false}
                        tickLine={false}
                      />
                      <YAxis
                        domain={[0, 1]}
                        tick={{ fill: '#64748b', fontSize: 9, fontFamily: 'JetBrains Mono' }}
                        axisLine={false}
                        tickLine={false}
                        tickFormatter={(v: number) => `${Math.round(v * 100)}`}
                      />
                      <Tooltip
                        content={({ active, payload }) => {
                          if (!active || !payload?.length) return null
                          const d = payload[0].payload as { name: string; score: number; threshold: number }
                          const displayName = METRIC_DISPLAY_NAMES[d.name] ?? d.name
                          const colorKey = scoreToColor(d.score, d.threshold)
                          return (
                            <div className="rounded-lg border border-navy-600 bg-navy-800 px-3 py-2 shadow-card">
                              <p className="text-xs font-semibold text-slate-200">
                                {displayName}
                              </p>
                              <p className="text-xs text-slate-400 mt-0.5">
                                Score: <span className="font-mono-data" style={{ color: COLOR_MAP[colorKey] }}>
                                  {formatScore(d.score)}
                                </span>
                              </p>
                              <p className="text-xs text-slate-500">
                                threshold: {formatScore(d.threshold)}
                              </p>
                            </div>
                          )
                        }}
                        cursor={{ fill: 'rgba(30,45,82,0.5)' }}
                      />
                      <Bar dataKey="score" radius={[4, 4, 0, 0]}>
                        {chartData.map((entry, index) => {
                          const colorKey = scoreToColor(entry.score, entry.threshold)
                          return <Cell key={index} fill={COLOR_MAP[colorKey]} />
                        })}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                <p className="text-xs text-slate-600">
                  Faithfulness threshold: ≥{Math.round(faithThresholdUsed * 100)}%
                </p>
              </div>
            )
          })}

          <p className="text-xs text-slate-600 text-center pt-2 border-t border-navy-700">
            Run {new Date(data.timestamp).toLocaleString('en-IN')} · {data.run_id}
          </p>
        </div>
      )}
    </Card>
  )
}