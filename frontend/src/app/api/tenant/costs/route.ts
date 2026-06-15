import { NextRequest, NextResponse } from 'next/server'

const BACKEND = process.env.BACKEND_URL ?? 'http://localhost:8000'

export async function GET(req: NextRequest) {
  const tenantId = req.nextUrl.searchParams.get('tenant_id')
  if (!tenantId) {
    return NextResponse.json({ detail: 'tenant_id required' }, { status: 400 })
  }
  try {
    const res = await fetch(`${BACKEND}/tenant/costs?tenant_id=${tenantId}`, {
      cache: 'no-store',
      signal: AbortSignal.timeout(5_000),
    })
    const data = await res.json()
    return NextResponse.json(data, { status: res.status })
  } catch (err) {
    return NextResponse.json(
      { detail: err instanceof Error ? err.message : 'Costs proxy error' },
      { status: 502 },
    )
  }
}