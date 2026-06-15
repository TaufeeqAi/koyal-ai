/**
 * src/app/outbound/page.tsx — Outbound Campaign Manager
 * ───────────────────────────────────────────────────────
 * Launch SIP outbound campaigns via Phase 6 LiveKitSIPOutbound.dial_campaign().
 *
 * The page is a Server Component shell that renders a Client Component
 * campaign form. The campaign result is displayed inline after launch.
 */

import type { Metadata } from 'next'
import { CampaignManager } from './_components/CampaignManager'

export const metadata: Metadata = {
  title: 'Outbound Campaigns',
}

export default function OutboundPage() {
  return (
    <div className="space-y-6">
      {/* Page header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-100">Outbound Campaign Manager</h1>
        <p className="mt-1 text-sm text-slate-500">
          Launch Hindi · Hinglish · English SIP dialing campaigns via LiveKit
        </p>
      </div>

      <CampaignManager />
    </div>
  )
}