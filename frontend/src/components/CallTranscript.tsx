/**
 * src/components/CallTranscript.tsx
 * ───────────────────────────────────
 * Bilingual real-time transcript viewer.
 *
 * Renders turns in a chat-style layout:
 *   Caller turns → left-aligned (darker bubble)
 *   Agent turns  → right-aligned (koyal blue tint)
 *   Escalation   → full-width rose alert
 *
 * Auto-scrolls to the latest turn on each new message.
 * Shows the per-turn language badge from Phase 6 STT output.
 */

'use client'

import { useEffect, useRef } from 'react'
import { LanguageBadge } from './LanguageBadge'
import { cn } from '@/lib/utils'
import { formatTimestamp } from '@/lib/utils'
import type { TranscriptTurn } from '@/types'

interface CallTranscriptProps {
  turns: TranscriptTurn[]
  className?: string
}

export function CallTranscript({ turns, className }: CallTranscriptProps) {
  const bottomRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to latest turn
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [turns])

  if (turns.length === 0) {
    return (
      <div className={cn(
        'flex h-72 items-center justify-center rounded-xl border border-navy-600 bg-navy-800/50',
        className,
      )}>
        <div className="text-center">
          <div className="text-4xl mb-3">📞</div>
          <p className="text-slate-400 text-sm">Waiting for call…</p>
          <p className="text-slate-600 text-xs mt-1">
            Transcript appears here when a call is active
          </p>
        </div>
      </div>
    )
  }

  return (
    <div
      role="log"
      aria-label="Call transcript"
      aria-live="polite"
      className={cn(
        'h-72 overflow-y-auto space-y-3 rounded-xl border border-navy-600 bg-navy-800/50 p-4',
        className,
      )}
    >
      {turns.map((turn, i) => {
        if (turn.is_escalation) {
          return <EscalationTurn key={i} turn={turn} />
        }
        return turn.speaker === 'caller'
          ? <CallerTurn key={i} turn={turn} />
          : <AgentTurn key={i} turn={turn} />
      })}
      <div ref={bottomRef} />
    </div>
  )
}

function CallerTurn({ turn }: { turn: TranscriptTurn }) {
  return (
    <div className="flex justify-start">
      <div className="max-w-[80%]">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-xs font-medium text-slate-500">Caller</span>
          <LanguageBadge language={turn.language} size="xs" />
          <span className="text-xs text-slate-600 font-mono-data">
            {formatTimestamp(turn.timestamp)}
          </span>
        </div>
        <div className="rounded-xl rounded-tl-sm bg-navy-700 px-4 py-2.5 text-sm text-slate-200">
          {turn.text}
          {turn.confidence !== undefined && turn.confidence < 0.6 && (
            <p className="text-xs text-amber-400 mt-1">
              Low confidence ({Math.round(turn.confidence * 100)}%)
            </p>
          )}
        </div>
      </div>
    </div>
  )
}

function AgentTurn({ turn }: { turn: TranscriptTurn }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[80%]">
        <div className="flex items-center gap-2 mb-1 justify-end">
          <span className="text-xs text-slate-600 font-mono-data">
            {formatTimestamp(turn.timestamp)}
          </span>
          <LanguageBadge language={turn.language} size="xs" />
          <span className="text-xs font-medium text-koyal">KoyalAI</span>
        </div>
        <div className="rounded-xl rounded-tr-sm bg-koyal/10 border border-koyal/20 px-4 py-2.5 text-sm text-slate-200">
          {turn.text}
        </div>
      </div>
    </div>
  )
}

function EscalationTurn({ turn }: { turn: TranscriptTurn }) {
  return (
    <div className="flex justify-center">
      <div className="w-full rounded-lg border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-center">
        <p className="text-sm font-semibold text-rose-400">
          ⚠ Escalation Triggered
        </p>
        <p className="text-xs text-rose-300 mt-1">{turn.text}</p>
        <span className="text-xs text-rose-500/60 font-mono-data mt-1 block">
          {formatTimestamp(turn.timestamp)}
        </span>
      </div>
    </div>
  )
}