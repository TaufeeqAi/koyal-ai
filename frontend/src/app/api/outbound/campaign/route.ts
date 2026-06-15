/**
 * Route Handler: POST /api/outbound/campaign
 * ────────────────────────────────────────────
 * Proxies campaign launch to Phase 6 FastAPI backend.
 * Long timeout (5 minutes) — campaigns with many contacts can take time.
 */

import { NextRequest, NextResponse } from 'next/server'

const BACKEND = process.env.BACKEND_URL ?? 'http://localhost:8000'
const CAMPAIGN_TIMEOUT_MS = 5 * 60 * 1_000

async function parseResponseBody(res: Response): Promise<unknown> {
  const contentType = res.headers.get('content-type') ?? ''
  const text = await res.text()

  if (!text) return {}

  if (contentType.includes('application/json')) {
    try {
      return JSON.parse(text)
    } catch {
      return { detail: text }
    }
  }

  return { detail: text }
}

export async function POST(req: NextRequest) {
  let body: unknown

  try {
    body = await req.json()
  } catch {
    return NextResponse.json(
      { detail: 'Invalid JSON body.' },
      { status: 400 },
    )
  }

  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), CAMPAIGN_TIMEOUT_MS)

  try {
    const res = await fetch(`${BACKEND}/telephony/outbound/campaign`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal,
    })

    const data = await parseResponseBody(res)

    if (!res.ok) {
      return NextResponse.json(
        {
          detail:
            (data as { detail?: string }).detail ??
            `Campaign launch failed with HTTP ${res.status}.`,
          upstream_status: res.status,
        },
        { status: res.status },
      )
    }

    return NextResponse.json(data, { status: res.status })
  } catch (err) {
    if (err instanceof Error && err.name === 'AbortError') {
      return NextResponse.json(
        { detail: 'Campaign launch timed out after 5 minutes.' },
        { status: 504 },
      )
    }

    return NextResponse.json(
      {
        detail:
          err instanceof Error
            ? `Campaign proxy error: ${err.message}`
            : 'Campaign proxy error.',
      },
      { status: 502 },
    )
  } finally {
    clearTimeout(timeout)
  }
}