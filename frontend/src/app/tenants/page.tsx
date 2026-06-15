/**
 * src/app/tenants/page.tsx — Tenant Management
 * ──────────────────────────────────────────────
 * Tenant onboarding, document upload, and cost overview.
 */

import type { Metadata } from 'next'
import { TenantForm } from '@/components/TenantForm'
import { Card, CardHeader, CardTitle } from '@/components/ui/Card'
import { CostPanel } from '@/components/CostPanel'
import { TENANT_IDS, TENANT_LABELS } from '@/lib/constants'

export const metadata: Metadata = {
  title: 'Tenants',
}

export default function TenantsPage() {
  return (
    <div className="space-y-6">
      {/* Page header */}
      <div>
        <h1 className="text-2xl font-bold text-slate-100">Tenant Management</h1>
        <p className="mt-1 text-sm text-slate-500">
          Onboard new tenants · Upload knowledge documents · Track per-tenant costs
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ── Onboarding form ──────────────────────────────────────────── */}
        <Card>
          <CardHeader>
            <CardTitle>Onboard New Tenant</CardTitle>
          </CardHeader>
          <div className="px-5 pb-5">
            <TenantForm />
          </div>
        </Card>

        {/* ── Existing tenant costs ────────────────────────────────────── */}
        <div className="space-y-4">
          <h2 className="text-sm font-semibold uppercase tracking-widest text-slate-400">
            Existing Tenants
          </h2>
          {TENANT_IDS.map((tid) => (
            <Card key={tid}>
              <CardHeader>
                <CardTitle>
                  <span className="text-slate-200">
                    {TENANT_LABELS[tid as keyof typeof TENANT_LABELS]}
                  </span>
                </CardTitle>
              </CardHeader>
              <div className="px-5 pb-5">
                <CostPanel tenantId={tid} />
              </div>
            </Card>
          ))}
        </div>
      </div>
    </div>
  )
}