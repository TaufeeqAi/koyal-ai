/**
 * src/components/CostPanel.tsx
 * ─────────────────────────────
 * Per-tenant cost tracker panel. Polls Phase 3 CostTracker via SWR at 5s.
 * Shows STT / TTS / LLM costs in INR.
 */

'use client'

import { useTenantCosts } from '@/lib/api'
import { formatINR } from '@/lib/utils'
import { Card, CardHeader, CardTitle } from '@/components/ui/Card'
import { PageLoader } from '@/components/ui/LoadingSpinner'
import { ErrorBanner } from '@/components/ui/ErrorBanner'

interface CostPanelProps {
  tenantId: string
}

export function CostPanel({ tenantId }: CostPanelProps) {
  const { data, error, isLoading } = useTenantCosts(tenantId)

  return (
    <Card>
      <CardHeader>
        <CardTitle>
          Cost Tracker
          <span className="ml-2 text-xs font-normal text-slate-500">₹ / session</span>
        </CardTitle>
      </CardHeader>

      {isLoading && <PageLoader label="Loading costs..." />}
      {error && <ErrorBanner message={error.message} />}

      {data && (
        <div className="space-y-3 px-5 pb-5">
          {[
            { label: 'Speech-to-Text', value: data.stt_cost_inr,  icon: '🎙' },
            { label: 'Text-to-Speech', value: data.tts_cost_inr,  icon: '🔊' },
            { label: 'LLM Tokens',     value: data.llm_tokens,    icon: '🧠', raw: `${data.llm_tokens.toLocaleString('en-IN')} tokens` },
          ].map(({ label, value, icon, raw }) => (
            <div key={label} className="flex items-center justify-between py-2 border-b border-navy-700 last:border-0">
              <div className="flex items-center gap-2 text-sm text-slate-400">
                <span>{icon}</span>
                <span>{label}</span>
              </div>
              <div className="text-sm font-mono-data text-slate-200">
                {raw ?? formatINR(value)}
              </div>
            </div>
          ))}

          <div className="flex items-center justify-between pt-2 border-t border-navy-600">
            <span className="text-sm font-semibold text-slate-300">Total</span>
            <span className="text-lg font-bold font-mono-data text-koyal">
              {formatINR(data.total_cost_inr)}
            </span>
          </div>
        </div>
      )}
    </Card>
  )
}