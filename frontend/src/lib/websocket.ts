/**
 * src/lib/websocket.ts
 * ─────────────────────
 * Custom React hook for real-time transcript streaming via WebSocket.
 *
 * Connects to /ws/transcript/{roomName}.
 * Implements:
 *   - Auto-reconnect with exponential backoff (up to 30s)
 *   - Connection state tracking (connecting / connected / disconnected / error)
 *   - Clean disconnect on component unmount
 *   - JSON message parsing with type safety
 *   - Max 500 turns in memory (prevents memory leak on long calls)
 *   - Heartbeat ping (every 30s) to prevent idle timeouts from proxies/load balancers
 */

'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { WS_BASE } from '@/lib/constants'
import type { TranscriptTurn } from '@/types'

export type ConnectionState =
  | 'idle'
  | 'connecting'
  | 'connected'
  | 'disconnected'
  | 'error'

interface UseLiveTranscriptOptions {
  /** Max turns to keep in memory. Older turns are dropped. Default: 500 */
  maxTurns?: number
}

interface UseLiveTranscriptResult {
  turns: TranscriptTurn[]
  connectionState: ConnectionState
  error: string | null
  connect: (roomName: string) => void
  disconnect: () => void
  clearTurns: () => void
}

const MAX_TURNS_DEFAULT = 500
const RECONNECT_BASE_MS = 1_000
const RECONNECT_MAX_MS = 30_000
const RECONNECT_MAX_TRIES = 10

// Heartbeat interval (30 seconds) – prevents idle timeout on proxies/load balancers
const HEARTBEAT_INTERVAL_MS = 30_000

function getTranscriptWsUrl(roomName: string): string {
  return `${WS_BASE}/ws/transcript/${encodeURIComponent(roomName)}`
}

function parseTranscriptTurn(raw: unknown): TranscriptTurn | null {
  if (!raw || typeof raw !== 'object') return null

  const obj = raw as Record<string, unknown>

  const speaker = obj.speaker
  if (speaker !== 'caller' && speaker !== 'agent') return null

  const text = typeof obj.text === 'string' ? obj.text.trim() : ''
  if (!text) return null

  const language = typeof obj.language === 'string' ? obj.language : 'en-IN'

  const timestamp =
    typeof obj.timestamp === 'string' && obj.timestamp.trim().length > 0
      ? obj.timestamp
      : new Date().toISOString()

  const confidence =
    typeof obj.confidence === 'number' && Number.isFinite(obj.confidence)
      ? obj.confidence
      : undefined

  const isEscalation =
    typeof obj.is_escalation === 'boolean' ? obj.is_escalation : false

  return {
    speaker,
    text,
    language,
    timestamp,
    confidence,
    is_escalation: isEscalation,
  }
}

export function useLiveTranscript(
  options: UseLiveTranscriptOptions = {},
): UseLiveTranscriptResult {
  const maxTurns = options.maxTurns ?? MAX_TURNS_DEFAULT

  const [turns, setTurns] = useState<TranscriptTurn[]>([])
  const [connectionState, setConnectionState] =
    useState<ConnectionState>('idle')
  const [error, setError] = useState<string | null>(null)

  const wsRef = useRef<WebSocket | null>(null)
  const roomNameRef = useRef<string | null>(null)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const reconnectCount = useRef(0)
  const isMounted = useRef(true)
  const manualDisconnect = useRef(false)
  const pingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // ── Heartbeat management ────────────────────────────────────────────────
  const startHeartbeat = useCallback(() => {
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current)
      pingIntervalRef.current = null
    }
    // Only start heartbeat if connection is open (called from ws.onopen)
    pingIntervalRef.current = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send('ping')
      }
    }, HEARTBEAT_INTERVAL_MS)
  }, [])

  const stopHeartbeat = useCallback(() => {
    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current)
      pingIntervalRef.current = null
    }
  }, [])

  // ── WebSocket lifecycle ─────────────────────────────────────────────────
  function clearReconnectTimer() {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current)
      reconnectTimerRef.current = null
    }
  }

  function closeWebSocket() {
    // Stop heartbeat before closing
    stopHeartbeat()

    if (wsRef.current) {
      wsRef.current.onopen = null
      wsRef.current.onmessage = null
      wsRef.current.onclose = null
      wsRef.current.onerror = null

      if (
        wsRef.current.readyState === WebSocket.OPEN ||
        wsRef.current.readyState === WebSocket.CONNECTING
      ) {
        wsRef.current.close(1000, 'Component disconnected')
      }

      wsRef.current = null
    }
  }

  function scheduleReconnect(roomName: string) {
    if (!isMounted.current) return

    if (manualDisconnect.current) return

    if (reconnectCount.current >= RECONNECT_MAX_TRIES) {
      setConnectionState('error')
      setError(`Connection failed after ${RECONNECT_MAX_TRIES} attempts`)
      return
    }

    const delay = Math.min(
      RECONNECT_BASE_MS * 2 ** reconnectCount.current,
      RECONNECT_MAX_MS,
    )

    reconnectTimerRef.current = setTimeout(() => {
      if (isMounted.current && roomNameRef.current === roomName) {
        openWebSocket(roomName)
      }
    }, delay)
  }

  function openWebSocket(roomName: string) {
    if (!isMounted.current) return

    clearReconnectTimer()
    closeWebSocket()

    const url = getTranscriptWsUrl(roomName)
    setConnectionState(reconnectCount.current > 0 ? 'connecting' : 'connecting')
    setError(null)

    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => {
      if (!isMounted.current) {
        ws.close()
        return
      }
      reconnectCount.current = 0
      setConnectionState('connected')
      setError(null)
      // Start sending periodic pings to keep connection alive
      startHeartbeat()
    }

    ws.onmessage = (event) => {
      if (!isMounted.current) return

      try {
        const parsed = JSON.parse(event.data as string) as unknown
        const turn = parseTranscriptTurn(parsed)
        if (!turn) {
          return
        }

        setTurns((prev) => {
          const next = [...prev, turn]
          return next.length > maxTurns
            ? next.slice(next.length - maxTurns)
            : next
        })
      } catch {
        console.warn('[WS] Malformed transcript message:', event.data)
      }
    }

    ws.onclose = (event) => {
      if (!isMounted.current) return

      wsRef.current = null

      if (manualDisconnect.current) {
        setConnectionState('disconnected')
        return
      }

      // Normal server/client close: mark disconnected and stop reconnecting only
      // if the caller intentionally changed the room or component is gone.
      if (event.code === 1000 || event.code === 1001) {
        setConnectionState('disconnected')
        return
      }

      reconnectCount.current += 1
      setConnectionState('disconnected')

      if (roomNameRef.current === roomName) {
        scheduleReconnect(roomName)
      }
    }

    ws.onerror = () => {
      if (!isMounted.current) return
      setError('WebSocket connection error')
      // onclose will fire next and handle reconnect behavior.
    }
  }

  // ── Public API ─────────────────────────────────────────────────────────
  const connect = useCallback((roomName: string) => {
    const trimmed = roomName.trim()
    if (!trimmed) {
      setConnectionState('error')
      setError('Missing room name.')
      return
    }

    clearReconnectTimer()
    reconnectCount.current = 0
    roomNameRef.current = trimmed
    manualDisconnect.current = false
    setTurns([])
    setError(null)
    openWebSocket(trimmed)
  }, [])

  const disconnect = useCallback(() => {
    clearReconnectTimer()
    reconnectCount.current = 0
    roomNameRef.current = null
    manualDisconnect.current = true
    closeWebSocket()
    setConnectionState('disconnected')
  }, [])

  const clearTurns = useCallback(() => setTurns([]), [])

  // Cleanup on unmount: close WebSocket and stop heartbeat
  useEffect(() => {
    isMounted.current = true
    return () => {
      isMounted.current = false
      clearReconnectTimer()
      closeWebSocket() // this also stops heartbeat
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return { turns, connectionState, error, connect, disconnect, clearTurns }
}