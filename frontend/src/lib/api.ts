/**
 * src/lib/api.ts
 * ───────────────
 * SWR hooks and fetch helpers for all KoyalAI backend endpoints.
 *
 * Architecture:
 *   Client (browser) → Next.js Route Handler → FastAPI backend
 *   All Route Handlers live in src/app/api/. The browser never calls
 *   FastAPI directly — no CORS configuration needed on the backend.
 *
 * Error handling:
 *   fetcher() throws for non-2xx responses. SWR's `error` state captures it.
 *   Components should render <ErrorBanner> when SWR returns an error.
 */

'use client'

import useSWR, { type SWRConfiguration } from 'swr'
import { API_BASE, POLL_ACTIVE_ROOMS, POLL_TENANT_COSTS, POLL_RAGAS_SCORES, POLL_TELEPHONY_HEALTH } from '@/lib/constants'
import type {
  TenantCosts,
  RoomStatus,
  RoomsResponse,
  TelephonyHealth,
  MultilingualEvalReport,
  CampaignResult,
  CampaignRequest,
  TokenResponse,
  TokenRequest,
  TenantConfig,
  DocumentUploadResult,
  CreateTenantRequest,
} from '@/types'

// ── Core fetcher ─────────────────────────────────────────────────────────────

async function fetcher<T>(url: string): Promise<T> {
  const res = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    cache: 'no-store',
  })
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText)
    throw new Error(`[${res.status}] ${text}`)
  }
  return res.json() as Promise<T>
}

// ── SWR configuration presets ─────────────────────────────────────────────────

const realtimeSWR: SWRConfiguration = {
  refreshInterval: POLL_ACTIVE_ROOMS,
  revalidateOnFocus: true,
  revalidateOnReconnect: true,
}

const costSWR: SWRConfiguration = {
  refreshInterval: POLL_TENANT_COSTS,
  revalidateOnFocus: true,
}

const evalSWR: SWRConfiguration = {
  refreshInterval: POLL_RAGAS_SCORES,
  revalidateOnFocus: false, // Expensive — don't re-fetch on tab focus
}

const healthSWR: SWRConfiguration = {
  refreshInterval: POLL_TELEPHONY_HEALTH,
}

// ── Telephony hooks (Phase 6) ──────────────────────────────────────────────────

/**
 * Poll active call rooms from Phase 6 inbound_handler.
 * Used by the Live Call Monitor page.
 */
export function useActiveRooms() {
  return useSWR<RoomsResponse>(
    `${API_BASE}/telephony/rooms`,
    fetcher,
    realtimeSWR,
  )
}

/**
 * Phase 6 health check — shows active_rooms count in the navbar badge.
 */
export function useTelephonyHealth() {
  return useSWR<TelephonyHealth>(
    `${API_BASE}/telephony/health`,
    fetcher,
    healthSWR,
  )
}

/**
 * Request a LiveKit JWT for browser-based calling.
 * Returns token + ws_url for use with LiveKitRoom component.
 */
export async function requestCallerToken(
  req: TokenRequest,
): Promise<TokenResponse> {
  const res = await fetch(`${API_BASE}/telephony/token`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!res.ok) throw new Error(`Token request failed: ${res.status}`)
  return res.json() as Promise<TokenResponse>
}

// ── Cost hooks (Phase 3 CostTracker) ─────────────────────────────────────────

/**
 * Fetch per-tenant cost breakdown from Redis CostTracker.
 * Polling at 5s so the cost panel feels live during active calls.
 */
export function useTenantCosts(tenantId: string | null) {
  return useSWR<TenantCosts>(
    tenantId ? `${API_BASE}/costs/${tenantId}` : null,
    fetcher,
    costSWR,
  )
}

// ── Eval hooks (Phase 5 RAGAS) ────────────────────────────────────────────────

/**
 * Fetch the latest RAGAS eval report for a tenant.
 * Falls back to null (no report yet) without showing an error.
 */
export function useRagasReport(tenantId: string | null) {
  return useSWR<MultilingualEvalReport | null>(
    tenantId ? `${API_BASE}/evals/ragas?tenant_id=${tenantId}` : null,
    fetcher,
    evalSWR,
  )
}

// ── Tenant mutations ──────────────────────────────────────────────────────────

/**
 * Create a new tenant via POST /tenants/create.
 * Returns TenantConfig including the generated API key (shown once).
 */
export async function createTenant(
  req: CreateTenantRequest,
): Promise<TenantConfig> {
  const res = await fetch(`${API_BASE}/tenants/create`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(error.detail ?? 'Failed to create tenant')
  }
  return res.json() as Promise<TenantConfig>
}

/**
 * Upload a knowledge document for a tenant.
 * Sends multipart/form-data — file + language code + API key header.
 */
export async function uploadDocument(
  apiKey: string,
  file: File,
  language: string,
  tenantId: string,
): Promise<DocumentUploadResult> {
  const form = new FormData()
  form.append('file', file)
  form.append('language', language)
  form.append('tenant_id', tenantId)

  const res = await fetch(`${API_BASE}/documents/upload`, {
    method: 'POST',
    headers: { 'X-API-Key': apiKey },
    body: form,
    // Note: no Content-Type header — browser sets it with boundary for multipart
  })
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(error.detail ?? 'Document upload failed')
  }
  return res.json() as Promise<DocumentUploadResult>
}

// ── Outbound campaign mutation ────────────────────────────────────────────────

/**
 * Launch an outbound SIP campaign.
 * Calls Phase 6's LiveKitSIPOutbound.dial_campaign() via the backend.
 */
export async function launchCampaign(
  req: CampaignRequest,
): Promise<CampaignResult> {
  const res = await fetch(`${API_BASE}/outbound/campaign`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(error.detail ?? 'Campaign launch failed')
  }
  return res.json() as Promise<CampaignResult>
}