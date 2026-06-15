/**
 * src/app/_components/ActiveCallsMonitor.tsx
 * ────────────────────────────────────────────
 * Client component for the live call monitor page.
 * Polls active rooms and renders the selected room's transcript + panels.
 *
 * Colocated underscore directory (_components) = not a route (Next.js 16 convention).
 */

'use client'

import { useState } from 'react'
import { useActiveRooms } from '@/lib/api'
import { LiveKitCallRoom } from '@/components/LiveKitCallRoom'
import { CostPanel } from '@/components/CostPanel'
import { MetricsPanel } from '@/components/MetricsPanel'
import { AgentTrace } from '@/components/AgentTrace'
import { Card, CardHeader, CardTitle } from '@/components/ui/Card'
import { Badge } from '@/components/ui/Badge'
import { ErrorBanner } from '@/components/ui/ErrorBanner'
import { PageLoader } from '@/components/ui/LoadingSpinner'
import { TENANT_LABELS } from '@/lib/constants'
import { tenantFromRoomName } from '@/lib/utils'

export function ActiveCallsMonitor() {
  const { data: rooms, error, isLoading } = useActiveRooms()
  const [selectedRoom, setSelectedRoom] = useState<string | null>(null)

  const activeRooms = rooms?.rooms ?? []
  const selectedTenantId = selectedRoom ? tenantFromRoomName(selectedRoom) : null

  if (isLoading) return <PageLoader label="Loading active calls..." />
  if (error)     return <ErrorBanner message={error.message} />

  const noRooms =  activeRooms.length === 0

  const getTenantLabel = (tenantId: string) =>
    TENANT_LABELS[tenantId as keyof typeof TENANT_LABELS] ?? tenantId


  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* ── Left column: room list + call room ──────────────────────────── */}
      <div className="lg:col-span-2 space-y-4">
        {/* Active rooms list */}
        <Card>
          <CardHeader>
            <CardTitle>
              Active Calls
              <Badge variant="info" className="ml-2">
                {rooms?.count ?? 0} room{(rooms?.count ?? 0) !== 1 ? 's' : ''}
              </Badge>
            </CardTitle>
          </CardHeader>

          {noRooms ? (
            <div className="px-5 pb-5 text-center py-8">
              <p className="text-slate-400 text-sm">No active calls.</p>
              <p className="text-slate-600 text-xs mt-1">
                Inbound SIP calls appear here when callers dial in.
              </p>
            </div>
          ) : (
            <div className="divide-y divide-navy-600">
              {activeRooms.map((room)  => {
                const tId    = tenantFromRoomName(room.room_name)
                const isSelected = selectedRoom === room.room_name
                return (
                  <div key={room.room_name}>
                    <button
                      onClick={() => setSelectedRoom(isSelected ? null : room.room_name)}
                      aria-pressed={isSelected}
                      className={[
                        'w-full flex items-center justify-between px-5 py-3',
                        'text-left text-sm transition-colors',
                        isSelected
                          ? 'bg-koyal/10 border-l-2 border-koyal'
                          : 'hover:bg-navy-700',
                      ].join(' ')}
                    >
                      <div>
                        <div className="font-mono-data text-slate-200">
                          {room.room_name}
                        </div>
                        <div className="text-xs text-slate-500 mt-0.5">
                          {getTenantLabel(tId)}
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Badge variant={room.connected ? 'success' : 'warning'}>
                          {room.connected ? 'Agent active' : 'Connecting'}
                        </Badge>
                        {isSelected && (
                          <span className="text-koyal text-xs">▲</span>
                        )}
                      </div>
                    </button>

                    {/* Expanded: live call room */}
                    {isSelected && (
                      <div className="px-5 pb-4">
                        <LiveKitCallRoom
                          roomName={room.room_name}
                          onDisconnect={() => setSelectedRoom(null)}
                        />
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          )}
        </Card>

        {/* Agent pipeline trace */}
        {selectedTenantId && (
          <Card>
            <CardHeader>
              <CardTitle>Agent Pipeline</CardTitle>
            </CardHeader>
            <div className="px-5 pb-5">
              <AgentTrace />
              <p className="text-xs text-slate-500 mt-2">LangGraph Phases 2–3</p>
            </div>
          </Card>
        )}
      </div>

      {/* ── Right column: cost + metrics ─────────────────────────────────── */}
      <div className="space-y-4">
        {selectedTenantId && (
          <>
            <CostPanel tenantId={selectedTenantId} />
            <MetricsPanel tenantId={selectedTenantId} />
          </>
        )}
      </div>
    </div>
  )
}