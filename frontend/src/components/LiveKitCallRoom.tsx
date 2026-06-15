/**
 * src/components/LiveKitCallRoom.tsx
 * ────────────────────────────────────
 * Browser-based voice calling component using @livekit/components-react v2.9.
 *
 * Wraps LiveKitRoom with voice-only configuration (no video).
 * Shows connection state, audio controls, and a disconnect button.
 * Integrates with the live transcript hook via useLiveTranscript.
 *
 * Flow:
 *   1. User selects an active room from Phase 6 /telephony/rooms
 *   2. Dashboard requests a caller JWT from /telephony/token
 *   3. This component connects to the LiveKit room via WebRTC
 *   4. The Phase 6 KoyalRoom agent is already in the room waiting
 *   5. Audio is published/subscribed; transcript streams via WS
 */

'use client'

import { useState, useCallback, useEffect, useRef } from 'react'
import {
  LiveKitRoom,
  RoomAudioRenderer,
  useConnectionState,
  useParticipants,
  useVoiceAssistant,
} from '@livekit/components-react'
// import '@livekit/components-react/dist/index.css'
import { ConnectionState } from 'livekit-client'
import { requestCallerToken } from '@/lib/api'
import { useLiveTranscript } from '@/lib/websocket'
import { CallTranscript } from './CallTranscript'
import { Button } from './ui/Button'
import { Badge } from './ui/Badge'
import { ErrorBanner } from './ui/ErrorBanner'
import { cn, tenantFromRoomName } from '@/lib/utils'
import type { TokenResponse } from '@/types'

interface LiveKitCallRoomProps {
  /** LiveKit room name to join (from /telephony/rooms) */
  roomName: string
  onDisconnect?: () => void
  className?: string
}

export function LiveKitCallRoom({ roomName, onDisconnect, className }: LiveKitCallRoomProps) {
  const [tokenResp, setTokenResp] = useState<TokenResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [liveKitConnected, setLiveKitConnected] = useState(false)

  const tenantId = tenantFromRoomName(roomName)
  const { turns, connectionState: wsState, connect: wsConnect, disconnect: wsDisconnect } = useLiveTranscript()

  // Robust session ID extraction
  const sessionIdRef = useRef<string | null>(null)

  const getSessionId = useCallback((): string | null => {
    // Assuming roomName format: "tenant-sessionId" or "tenant-timestamp-sessionId"
    // Last segment after hyphen is the session ID
    const parts = roomName.split('-')
    if (parts.length < 2) {
      console.warn(`[LiveKitCallRoom] Could not extract session ID from roomName: ${roomName}`)
      return null
    }
    return parts[parts.length - 1]
  }, [roomName])

  // Validate tenant before making requests
  const validateTenant = useCallback((): boolean => {
    if (!tenantId || typeof tenantId !== 'string') {
      setError(`Invalid tenant ID extracted from room: ${roomName}`)
      return false
    }
    return true
  }, [tenantId, roomName])

  // Request JWT and prepare connection
  const connect = useCallback(async () => {
    setError(null)
    setLoading(true)

    if (!validateTenant()) {
      setLoading(false)
      return
    }

    const sessionId = getSessionId()
    if (!sessionId) {
      setError(`Invalid room name format: ${roomName} (cannot extract session ID)`)
      setLoading(false)
      return
    }
    sessionIdRef.current = sessionId

    try {
      const resp = await requestCallerToken({ tenant_id: tenantId, room_name: roomName })
      setTokenResp(resp)
      // LiveKitRoom will call onConnected when WebRTC is ready
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to join call.')
      setTokenResp(null)
    } finally {
      setLoading(false)
    }
  }, [roomName, tenantId, validateTenant, getSessionId])

  // Disconnect everything
  const disconnect = useCallback(() => {
    wsDisconnect()
    setLiveKitConnected(false)
    setTokenResp(null)
    sessionIdRef.current = null
    onDisconnect?.()
  }, [wsDisconnect, onDisconnect])

  // Called when LiveKitRoom successfully connects
  const handleLiveKitConnected = useCallback(() => {
    setLiveKitConnected(true)
    // Start transcript WebSocket only after LiveKit is actually connected
    if (sessionIdRef.current) {
      wsConnect(sessionIdRef.current)
    } else {
      console.warn('[LiveKitCallRoom] No session ID available for transcript WebSocket')
    }
  }, [wsConnect])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (sessionIdRef.current) {
        wsDisconnect()
      }
    }
  }, [wsDisconnect])

  // Retry handler
  const handleRetry = useCallback(() => {
    connect()
  }, [connect])

  // ── Not connected yet ────────────────────────────────────────────────────
  if (!tokenResp) {
    return (
      <div className={cn('space-y-3', className)}>
        <div className="rounded-xl border border-navy-600 bg-navy-800/50 p-4">
          <h3 className="text-sm font-semibold text-slate-200">Join call as listener</h3>
          <p className="text-xs text-slate-500 font-mono-data mt-1">{roomName}</p>
        </div>
        {error && (
          <div className="space-y-2">
            <ErrorBanner message={error} />
            <Button variant="secondary" size="sm" onClick={handleRetry} className="w-full">
              Retry
            </Button>
          </div>
        )}
        <Button onClick={connect} loading={loading} className="w-full">
          Join Room →
        </Button>
      </div>
    )
  }

  // ── Connected ────────────────────────────────────────────────────────────
  return (
    <div className={cn('space-y-3', className)}>
      <LiveKitRoom
        token={tokenResp.token}
        serverUrl={tokenResp.ws_url}
        connectOptions={{ autoSubscribe: true }}
        audio={true}
        video={false}
        onConnected={handleLiveKitConnected}
        onDisconnected={disconnect}
        className="rounded-xl border border-navy-600 bg-navy-800/30 overflow-hidden"
      >
        {/* Renders remote audio automatically */}
        <RoomAudioRenderer />

        {/* Inner component (uses hooks that require LiveKitRoom context) */}
        <RoomControls
          roomName={roomName}
          wsState={wsState}
          onDisconnect={disconnect}
        />
      </LiveKitRoom>

      {/* Live transcript - only show if WebSocket connected or at least attempted */}
      {(wsState === 'connected' || wsState === 'connecting') && (
        <CallTranscript turns={turns} />
      )}
    </div>
  )
}

// ── Inner components (inside LiveKitRoom context) ─────────────────────────────

function RoomControls({
  roomName,
  wsState,
  onDisconnect,
}: {
  roomName: string
  wsState: string
  onDisconnect: () => void
}) {
  const connectionState = useConnectionState()
  const participants = useParticipants()
  const assistant = useVoiceAssistant()   // ← real-time agent state

  const isConnected = connectionState === ConnectionState.Connected
  const agentCount = participants.filter((p) => p.identity?.startsWith('koyal-agent')).length

  // Map assistant state to colour + optional animation
  const agentStateConfig = {
    speaking: { color: 'bg-emerald-400', animate: 'animate-pulse', label: 'Speaking' },
    listening: { color: 'bg-blue-400', animate: '', label: 'Listening' },
    thinking: { color: 'bg-amber-400', animate: 'animate-pulse', label: 'Thinking' },
    connecting: { color: 'bg-slate-500', animate: '', label: 'Connecting' },
    idle: { color: 'bg-slate-600', animate: '', label: 'Idle' },
  }
  const stateKey = assistant?.state as keyof typeof agentStateConfig
  const { color, animate, label } = agentStateConfig[stateKey] ?? agentStateConfig.idle

  return (
    <div className="p-4 space-y-3">
      {/* Connection status bar */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Badge variant={isConnected ? 'success' : 'warning'} dot>
            {connectionState === ConnectionState.Connecting && 'Connecting…'}
            {connectionState === ConnectionState.Connected && 'Live'}
            {connectionState === ConnectionState.Disconnected && 'Disconnected'}
            {connectionState === ConnectionState.Reconnecting && 'Reconnecting…'}
          </Badge>
          {agentCount > 0 && (
            <Badge variant="info" className="flex items-center gap-1.5">
              <span
                className={cn('w-1.5 h-1.5 rounded-full', color, animate)}
                title={label}
              />
              KoyalAI Agent
            </Badge>
          )}
        </div>
        <Badge variant={wsState === 'connected' ? 'success' : 'neutral'}>
          WS: {wsState}
        </Badge>
      </div>

      {/* Room info */}
      <div className="flex items-center justify-between text-xs text-slate-500">
        <span className="font-mono-data">{roomName}</span>
        <span>
          {participants.length} participant{participants.length !== 1 ? 's' : ''}
        </span>
      </div>

      {/* Real audio visualizer using CSS only (no random flicker) */}
      <AudioVisualiser isActive={assistant?.state === 'speaking'} />

      {/* Controls */}
      <Button variant="danger" size="sm" onClick={onDisconnect} className="w-full">
        Leave Room
      </Button>
    </div>
  )
}

interface AudioVisualiserProps {
  /** Whether agent is currently speaking (triggers animation) */
  isActive: boolean
}

function AudioVisualiser({ isActive }: AudioVisualiserProps) {
  // Fixed heights with smooth transition when active
  const barHeights = [32, 48, 64, 80, 64, 48, 32, 40]

  return (
    <div className="flex items-end justify-center gap-1 h-10">
      {barHeights.map((height, i) => (
        <div
          key={i}
          className={cn(
            'w-1.5 bg-koyal/60 rounded-full transition-all duration-150',
            isActive && 'animate-pulse'
          )}
          style={{
            height: `${isActive ? height : height * 0.4}%`,
            transitionDelay: `${i * 30}ms`,
          }}
        />
      ))}
    </div>
  )
}