/**
 * src/lib/utils.ts
 * ─────────────────
 * Pure utility functions — no React, no side effects.
 */

import { clsx, type ClassValue } from 'clsx'

/** Merge Tailwind classes safely (handles conditional/array patterns). */
export function cn(...inputs: ClassValue[]): string {
  return clsx(inputs)
}

/** Format a number as INR currency. */
export function formatINR(amount: number): string {
  if (amount < 0.01 && amount > 0) return '< ₹0.01'
  return new Intl.NumberFormat('en-IN', {
    style:                 'currency',
    currency:              'INR',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(amount)
}

/** Format milliseconds as human-readable duration. */
export function formatMs(ms: number): string {
  if (ms < 1_000)  return `${Math.round(ms)}ms`
  if (ms < 60_000) return `${(ms / 1_000).toFixed(1)}s`
  return `${Math.floor(ms / 60_000)}m ${Math.floor((ms % 60_000) / 1_000)}s`
}

/** Format seconds as human-readable duration. */
export function formatSeconds(s: number): string {
  return formatMs(s * 1_000)
}

/** Format a RAGAS score (0–1) as a percentage string. */
export function formatScore(score: number): string {
  return `${Math.round(score * 100)}%`
}

/** Clamp a value between min and max. */
export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}

/** Convert score (0–1) to a Tailwind colour for pass/fail/warning. */
export function scoreToColor(
  score: number,
  threshold: number,
): 'emerald' | 'amber' | 'rose' {
  if (score >= threshold)         return 'emerald'
  if (score >= threshold * 0.9)   return 'amber'
  return 'rose'
}

/**
 * TEMPORARY:
 * Room format:
 *   {tenant_id}-{call_type}-{session_id}
 *
 * When dynamic tenant creation is introduced,
 * replace this with tenant_id from the API response
 * or a backend-provided mapping.
 */
export function tenantFromRoomName(roomName: string): string {
  const idx = roomName.indexOf('-');
  return idx === -1 ? roomName : roomName.slice(0, idx);
}

/** Truncate text to maxLen characters with ellipsis. */
export function truncate(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text
  return `${text.slice(0, maxLen)}…`
}

/** Format ISO timestamp to local time string. */
export function formatTimestamp(iso: string): string {
  return new Date(iso).toLocaleTimeString('en-IN', {
    hour:   '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}