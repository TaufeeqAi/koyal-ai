/**
 * Route Handler: POST /api/telephony/token
 * ─────────────────────────────────────────
 * Proxies token requests to Phase 6 FastAPI backend.
 * The browser never directly calls the backend — no CORS needed.
 */

import { NextRequest, NextResponse } from 'next/server'

const BACKEND = process.env.BACKEND_URL ?? 'http://localhost:8000'

export async function POST(req: NextRequest) {
  try {
    const body = await req.json()
    const res = await fetch(`${BACKEND}/telephony/token`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(10_000),
    })
    const data = await res.json()
    return NextResponse.json(data, { status: res.status })
  } catch (err) {
    return NextResponse.json(
      { detail: err instanceof Error ? err.message : 'Token proxy error' },
      { status: 502 },
    )
  }
}