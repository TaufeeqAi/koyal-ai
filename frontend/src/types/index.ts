/**
 * src/types/index.ts
 * ──────────────────
 * Shared TypeScript types for the KoyalAI dashboard.
 * Mirrors Phase 2–6 backend dataclasses and Pydantic models.
 */

// ── Language ────────────────────────────────────────────────────────────────

// Strict language codes for UI constants (imported from constants.ts)
// The | string escape hatch allows future backend languages without breaking TypeScript
export type LanguageCode =
  | 'hi-IN'
  | 'en-IN'
  | 'hi-IN+en-IN'   // Hinglish (code-mixed)
  | 'mr-IN'          // Marathi
  | 'ta-IN'          // Tamil
  | 'te-IN'          // Telugu
  | 'kn-IN'          // Kannada
  | 'bn-IN'          // Bengali
  | string            // unknown/future codes — escape hatch for backend compatibility

// ── Transcript ───────────────────────────────────────────────────────────────

export interface TranscriptTurn {
  speaker: 'caller' | 'agent'
  text: string
  language: LanguageCode
  timestamp: string         // ISO-8601
  is_escalation?: boolean   // true when safety gate triggered
  confidence?: number       // STT confidence (0–1)
}

// ── Telephony (Phase 6 contracts) ────────────────────────────────────────────

export interface TokenRequest {
  tenant_id: string
  room_name?: string
  caller_identity?: string
}

export interface TokenResponse {
  token: string
  ws_url: string
  room_name: string
  tenant_id: string
  identity: string
}

export interface RoomStatus {
  room_name: string
  tenant_id: string
  connected: boolean
}

export interface RoomsResponse {
  count: number
  connected_count: number
  connecting_count: number
  rooms: RoomStatus[]
}

export interface TelephonyHealth {
  status: string
  active_rooms: number
  livekit_url: string
}

// ── Costs (Phase 3 CostTracker) ───────────────────────────────────────────────

export interface TenantCosts {
  tenant_id: string
  stt_cost_inr: number
  tts_cost_inr: number
  llm_tokens: number
  total_cost_inr: number
}

// ── Evaluation (Phase 5 RAGAS) ────────────────────────────────────────────────

export interface LanguageEvalResult {
  language: LanguageCode
  n_cases: number
  faithfulness: number
  faithfulness_threshold: number   // per-language threshold (Phase 5 merged)
  response_relevancy: number
  llm_context_precision: number
  context_recall: number
  passed_faithfulness: boolean
  duration_seconds: number
  error: string | null
}

export interface MultilingualEvalReport {
  timestamp: string
  run_id: string
  all_thresholds_passed: boolean
  failed_languages: string[]
  total_duration_seconds: number
  faithfulness_thresholds: Record<string, number>
  thresholds: Record<string, number>
  results_by_language: Record<string, LanguageEvalResult>
}

// ── Tenants (Phase 7 new) ─────────────────────────────────────────────────────

export interface TenantConfig {
  tenant_id: string
  company_name: string
  primary_language: LanguageCode
  supported_languages: LanguageCode[]
  default_voice: string
  api_key?: string                  // returned only on create; not on GET
}

export interface CreateTenantRequest {
  company_name: string
  primary_language: LanguageCode
}

export interface DocumentUploadResult {
  tenant_id: string
  filename: string
  language: LanguageCode
  chunks_ingested: number
  collection: string
}

// ── Outbound (Phase 6 LiveKitSIPOutbound) ────────────────────────────────────

export interface DialContact {
  phone: string
  name: string
  [key: string]: string             // arbitrary personalisation fields
}

export interface DialResult {
  session_id: string
  phone: string
  room_name: string
  tenant_id: string
  language: LanguageCode
  status: 'dialing' | 'error' | 'skipped' | 'failed'
  sip_call_id: string
  setup_duration_ms: number
  error: string | null
}

export interface CampaignResult {
  total: number
  dialing: number
  failed: number
  skipped: number
  results: DialResult[]
}

export interface CampaignRequest {
  tenant_id: string
  contacts: DialContact[]
  script_template: string
  language: LanguageCode
  max_concurrent?: number
}

// ── Agent pipeline stages (Phase 7 AgentTrace component) ─────────────────────

export type PipelineStage =
  | 'language_detect'
  | 'safety_gate'
  | 'language_bridge'
  | 'retrieval'
  | 'response'
  | 'verification'
  | 'translate_response'
  | 'escalation'

export interface PipelineStageResult {
  stage: PipelineStage
  status: 'pass' | 'fail' | 'skip' | 'pending'
  duration_ms?: number
  label?: string
}

// ── WebSocket JSON messages from /ws/{tenant_id} 
export type WsMessage =
  | { type: 'transcript'; text: string; language: string; confidence: number }
  | { type: 'response';  text: string; language: string; escalated: boolean; verified: boolean }
  | { type: 'escalated'; message: string }
  | { type: 'cost';      tts_chars: number; stt_seconds: number; total_inr: number }
  | { type: 'error';     message: string }