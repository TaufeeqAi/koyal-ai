/**
 * src/components/AgentTrace.tsx
 * ──────────────────────────────
 * Visual representation of the LangGraph pipeline stages.
 * Phase 7 renders this from the agent's pipeline trace embedded in
 * transcript WebSocket messages (or as a static template if not yet available).
 *
 * Shows each phase 2 node in sequence with pass/skip/fail status.
 */

import { PIPELINE_STAGE_LABELS } from '@/lib/constants'
import { cn } from '@/lib/utils'
import type { PipelineStageResult, PipelineStage } from '@/types'

const ALL_STAGES: PipelineStage[] = [
  'language_detect',
  'safety_gate',
  'language_bridge',
  'retrieval',
  'response',
  'verification',
  'translate_response',
]

const ESCALATION_STAGE: PipelineStage = 'escalation'

interface AgentTraceProps {
  /** Actual stage results from pipeline. Falls back to static display if undefined. */
  stages?: PipelineStageResult[]
  /** If true, shows escalation path instead of normal path */
  escalated?: boolean
  className?: string
}

const STATUS_CLASSES: Record<string, string> = {
  pass:    'bg-emerald-500/20 border-emerald-500/40 text-emerald-400',
  fail:    'bg-rose-500/20 border-rose-500/40 text-rose-400',
  skip:    'bg-slate-700/50 border-slate-600 text-slate-500',
  pending: 'bg-navy-700 border-navy-600 text-slate-500',
}

export function AgentTrace({ stages, escalated, className }: AgentTraceProps) {
  const stageMap = new Map(stages?.map((s) => [s.stage, s]))
  const displayStages = escalated
    ? ['language_detect', 'safety_gate', ESCALATION_STAGE] as PipelineStage[]
    : ALL_STAGES

  return (
    <div className={cn('space-y-1', className)} aria-label="Agent pipeline trace">
      {displayStages.map((stageName, idx) => {
        const result = stageMap.get(stageName)
        const status = result?.status ?? 'pending'
        const label  = PIPELINE_STAGE_LABELS[stageName] ?? stageName

        return (
          <div key={stageName} className="flex items-center gap-3">
            {/* Connector line (not for last item) */}
            <div className="flex flex-col items-center">
              <div className={cn(
                'w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold border',
                STATUS_CLASSES[status],
              )}>
                {status === 'pass' && '✓'}
                {status === 'fail' && '✗'}
                {status === 'skip' && '–'}
                {status === 'pending' && String(idx + 1)}
              </div>
              {idx < displayStages.length - 1 && (
                <div className="w-px h-4 bg-navy-600 my-1" />
              )}
            </div>

            {/* Stage label and timing */}
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className={cn(
                  'text-xs',
                  status === 'pass' ? 'text-slate-300' : status === 'fail' ? 'text-rose-400' : 'text-slate-500',
                )}>
                  {label}
                </span>
                {result?.duration_ms !== undefined && (
                  <span className="text-xs text-slate-600 font-mono-data">
                    {Math.round(result.duration_ms)}ms
                  </span>
                )}
              </div>
              {result?.label && (
                <p className="text-xs text-slate-500 mt-0.5">{result.label}</p>
              )}
            </div>
          </div>
        )
      })}
    </div>
  )
}