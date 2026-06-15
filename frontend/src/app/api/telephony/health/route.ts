import { NextResponse } from 'next/server'

const BACKEND = process.env.BACKEND_URL ?? 'http://localhost:8000'

export async function GET() {
  try {
    const res = await fetch(`${BACKEND}/telephony/health`, {
      cache: 'no-store',
      signal: AbortSignal.timeout(5_000),
    })
    const data = await res.json()
    return NextResponse.json(data, { status: res.status })
  } catch {
    // Graceful degradation: return minimal healthy-looking response
    return NextResponse.json(
      { status: 'unreachable', phase: 6, active_rooms: 0, livekit_url: '' },
      { status: 200 },
    )
  }
}