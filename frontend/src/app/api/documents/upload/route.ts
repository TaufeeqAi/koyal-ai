/**
 * Route Handler: POST /api/documents/upload
 * ──────────────────────────────────────────
 * Proxies multipart/form-data document uploads to FastAPI.
 * Forwards the X-API-Key header for tenant authentication.
 */

import { NextRequest, NextResponse } from 'next/server'

const BACKEND = process.env.BACKEND_URL ?? 'http://localhost:8000'

export async function POST(req: NextRequest) {
  try {
    const apiKey = req.headers.get('X-API-Key')
    if (!apiKey) {
      return NextResponse.json({ detail: 'X-API-Key header required' }, { status: 401 })
    }

    const formData = await req.formData()

    const res = await fetch(`${BACKEND}/documents/upload`, {
      method: 'POST',
      headers: {
        'X-API-Key': apiKey,
        // No Content-Type: browser/fetch sets it with boundary for FormData
      },
      body: formData,
      signal: AbortSignal.timeout(60_000),   // Large files may take time
    })
    const data = await res.json()
    return NextResponse.json(data, { status: res.status })
  } catch (err) {
    return NextResponse.json(
      { detail: err instanceof Error ? err.message : 'Upload proxy error' },
      { status: 502 },
    )
  }
}