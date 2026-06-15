/**
 * src/app/page.tsx — Live Call Monitor
 * ──────────────────────────────────────
 * Displays all active LiveKit call rooms from Phase 6, allows agents
 * to join as listeners via WebRTC, and shows live bilingual transcripts.
 *
 * Architecture (Server + Client):
 *   page.tsx            → Server Component shell (static layout, Suspense)
 *      → Client Component (SWR polls /telephony/rooms @ 2s)
 *      → Client Component (WebRTC + WebSocket transcript)
 *            → Client Component (SWR @ 5s)
 *         → Client Component (SWR @ 30s)
 */

import type { Metadata } from 'next'
import { Suspense } from 'react'
import { PageLoader } from '@/components/ui/LoadingSpinner'
import { ActiveCallsMonitor } from './_components/ActiveCallsMonitor'

export const metadata: Metadata = {
  title: 'Live Monitor',
}

export default function LiveMonitorPage() {
  return (
    <div className="space-y-6">
      {/* Page header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-100">Live Call Monitor</h1>
        <p className="mt-1 text-sm text-slate-500">
          Real-time bilingual transcript · Cost tracking · RAGAS scores
        </p>
      </div>

      <Suspense fallback={<PageLoader label="Loading monitor..." />}>
        <ActiveCallsMonitor />
      </Suspense>
    </div>
  )
}