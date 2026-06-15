/**
 * src/lib/constants.ts
 * ─────────────────────
 * App-wide constants. No secrets here — all are safe to bundle.
 */

// ── API ──────────────────────────────────────────────────────────────────────

// All API calls go through Next.js Route Handlers, which proxy to BACKEND_URL.
// Client code never calls the FastAPI backend directly.
export const API_BASE = '/api'

// WebSocket URL for live transcript streams (Phase 3 backend).
// Uses NEXT_PUBLIC_ so it's available in browser bundles.
// Default to secure WebSocket in production (wss://) if not provided.
export const WS_BASE = process.env.NEXT_PUBLIC_WS_URL ?? (
  typeof window !== 'undefined' && window.location.protocol === 'https:'
    ? `wss://${window.location.host}/ws`
    : 'ws://localhost:8000'
)

// ── Tenants ───────────────────────────────────────────────────────────────────

export const TENANT_IDS = [
  'tenant_hdfc_bank',
  'tenant_swiggy_support',
] as const

export type TenantId = (typeof TENANT_IDS)[number]

export const TENANT_LABELS: Record<TenantId, string> = {
  tenant_hdfc_bank:     'HDFC Bank',
  tenant_swiggy_support: 'Swiggy Support',
}

// ── Languages ─────────────────────────────────────────────────────────────────

export type LanguageCode = 
  | 'hi-IN'
  | 'en-IN'
  | 'hi-IN+en-IN'
  | 'mr-IN'
  | 'ta-IN'
  | 'te-IN'
  | 'kn-IN'
  | 'bn-IN'

export const LANGUAGE_LABELS: Record<LanguageCode, string> = {
  'hi-IN':       'हिंदी',
  'en-IN':       'English',
  'hi-IN+en-IN': 'Hinglish',
  'mr-IN':       'मराठी',
  'ta-IN':       'தமிழ்',
  'te-IN':       'తెలుగు',
  'kn-IN':       'ಕನ್ನಡ',
  'bn-IN':       'বাংলা',
}

// Tailwind colour class for each language
export const LANGUAGE_COLOR: Record<LanguageCode, string> = {
  'hi-IN':       'bg-orange-500/20 text-orange-300 border-orange-500/30',
  'en-IN':       'bg-blue-500/20 text-blue-300 border-blue-500/30',
  'hi-IN+en-IN': 'bg-violet-500/20 text-violet-300 border-violet-500/30',
  'mr-IN':       'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
  'ta-IN':       'bg-rose-500/20 text-rose-300 border-rose-500/30',
  'te-IN':       'bg-amber-500/20 text-amber-300 border-amber-500/30',
  'kn-IN':       'bg-purple-500/20 text-purple-300 border-purple-500/30',
  'bn-IN':       'bg-sky-500/20 text-sky-300 border-sky-500/30',
}

// ── Pipeline stages ───────────────────────────────────────────────────────────

export type PipelineStage =
  | 'language_detect'
  | 'safety_gate'
  | 'language_bridge'
  | 'retrieval'
  | 'response'
  | 'verification'
  | 'translate_response'
  | 'escalation'

export const PIPELINE_STAGE_LABELS: Record<PipelineStage, string> = {
  language_detect:    'Language Detect',
  safety_gate:        'Safety Gate',
  language_bridge:    'Language Bridge',
  retrieval:          'RAG Retrieval',
  response:           'LLM Response',
  verification:       'Chain of Verification',
  translate_response: 'Translate Response',
  escalation:         'Escalation Handler',
}

// ── SWR polling intervals (ms) ────────────────────────────────────────────────

export const POLL_ACTIVE_ROOMS     =  2_000   // 2s  — live call monitor
export const POLL_TENANT_COSTS     =  5_000   // 5s  — cost panel
export const POLL_TELEPHONY_HEALTH = 10_000   // 10s — health badge
export const POLL_RAGAS_SCORES     = 30_000   // 30s — eval scores (expensive)

// ── RAGAS thresholds (for colour coding — must match Phase 5 backend) ────────

export type LanguageForRagas = 'hi-IN' | 'en-IN' | 'hi-IN+en-IN'

export const RAGAS_FAITHFULNESS_THRESHOLDS: Record<LanguageForRagas, number> = {
  'hi-IN':       0.80,
  'en-IN':       0.82,
  'hi-IN+en-IN': 0.75,
}

export const RAGAS_SHARED_THRESHOLDS = {
  response_relevancy:                      0.75,
  llm_context_precision_without_reference: 0.70,
  context_recall:                          0.65,
} as const