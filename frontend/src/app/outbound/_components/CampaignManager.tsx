'use client'

import { useState, useCallback } from 'react'
import { launchCampaign } from '@/lib/api'
import { TENANT_IDS, TENANT_LABELS, LANGUAGE_LABELS } from '@/lib/constants'
import { Card, CardHeader, CardTitle } from '@/components/ui/Card'
import { Button } from '@/components/ui/Button'
import { Badge } from '@/components/ui/Badge'
import { ErrorBanner } from '@/components/ui/ErrorBanner'
import { LanguageBadge } from '@/components/LanguageBadge'
import { formatMs } from '@/lib/utils'
import type { CampaignResult, DialContact, LanguageCode } from '@/types'

const SAMPLE_SCRIPT_HI = `नमस्ते {name} जी! HDFC Bank की तरफ से आपकी EMI अनुस्मारक — आपकी EMI 5 तारीख को काटी जाएगी। किसी भी सहायता के लिए 1800-202-6161 पर संपर्क करें।`
const SAMPLE_SCRIPT_EN = `Hello {name}! This is a reminder from HDFC Bank — your EMI is due on the 5th. Please ensure sufficient balance. Call 1800-202-6161 for assistance.`
const SAMPLE_CONTACTS: DialContact[] = [
  { phone: '+919999900001', name: 'Ramesh Kumar' },
  { phone: '+919999900002', name: 'Priya Sharma' },
  { phone: '+919999900003', name: 'Suresh Patel' },
]

type LanguageOption = { code: LanguageCode; script: string }
const LANGUAGE_OPTIONS: LanguageOption[] = [
  { code: 'hi-IN',       script: SAMPLE_SCRIPT_HI },
  { code: 'en-IN',       script: SAMPLE_SCRIPT_EN },
  { code: 'hi-IN+en-IN', script: `Hello {name}! EMI reminder — 5 tarikh ko EMI kategi. Balance ready rakhein.` },
]

export function CampaignManager() {
  const [tenantId,  setTenantId]  = useState<string>(TENANT_IDS[0])
  const [language,  setLanguage]  = useState<LanguageCode>('hi-IN')
  const [script,    setScript]    = useState(SAMPLE_SCRIPT_HI)
  const [contacts,  setContacts]  = useState(JSON.stringify(SAMPLE_CONTACTS, null, 2))
  const [maxConcur, setMaxConcur] = useState(5)
  const [launching, setLaunching] = useState(false)
  const [error,     setError]     = useState<string | null>(null)
  const [result,    setResult]    = useState<CampaignResult | null>(null)

  const handleLanguageChange = useCallback((lang: LanguageCode) => {
    setLanguage(lang)
    const option = LANGUAGE_OPTIONS.find((o) => o.code === lang)
    if (option) setScript(option.script)
  }, [])

  const handleLaunch = useCallback(async () => {
    setError(null)
    let parsedContacts: DialContact[]
    try {
      parsedContacts = JSON.parse(contacts)
      if (!Array.isArray(parsedContacts) || parsedContacts.some((c) => !c.phone)) {
        throw new Error('Each contact must have a "phone" field.')
      }
    } catch (err) {
      setError(`Invalid contacts JSON: ${err instanceof Error ? err.message : String(err)}`)
      return
    }
    setLaunching(true)
    try {
      const res = await launchCampaign({
        tenant_id:       tenantId,
        contacts:        parsedContacts,
        script_template: script,
        language,
        max_concurrent:  maxConcur,
      })
      setResult(res)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Campaign launch failed.')
    } finally {
      setLaunching(false)
    }
  }, [tenantId, language, script, contacts, maxConcur])

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* ── Campaign config form ─────────────────────────────────────────── */}
      <Card>
        <CardHeader>
          <CardTitle>
            Campaign Settings
            {result && (
              <span className="ml-2 text-xs font-normal text-slate-500">
                Last: {result.dialing}/{result.total} dialing
              </span>
            )}
          </CardTitle>
        </CardHeader>

        <div className="space-y-4">
          {/* Tenant */}
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1.5">Tenant</label>
            <select
              value={tenantId}
              onChange={(e) => setTenantId(e.target.value)}
              className="w-full rounded-lg border border-navy-600 bg-navy-800 px-3 py-2 text-sm text-slate-200 focus:border-koyal/50 focus:outline-none"
            >
              {TENANT_IDS.map((tid) => (
                <option key={tid} value={tid}>
                  {TENANT_LABELS[tid as keyof typeof TENANT_LABELS]}
                </option>
              ))}
            </select>
          </div>

          {/* Language */}
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1.5">Language</label>
            <div className="flex flex-wrap gap-2">
              {LANGUAGE_OPTIONS.map(({ code }) => (
                <button
                  key={code}
                  onClick={() => handleLanguageChange(code)}
                  aria-pressed={language === code}
                  className={[
                    'rounded-lg border px-3 py-1.5 text-xs transition-colors',
                    language === code
                      ? 'border-koyal/30 bg-koyal/10 text-koyal'
                      : 'border-navy-600 bg-navy-800 text-slate-400 hover:bg-navy-700',
                  ].join(' ')}
                >
                  {LANGUAGE_LABELS[code as keyof typeof LANGUAGE_LABELS] ?? code}
                </button>
              ))}
            </div>
          </div>

          {/* Script */}
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1.5">
              Script Template
              <span className="ml-1 text-slate-600">Use {'{'}{'{name}'}{'}'} for personalisation</span>
            </label>
            <textarea
              value={script}
              onChange={(e) => setScript(e.target.value)}
              rows={4}
              className={[
                'w-full rounded-lg border border-navy-600 bg-navy-800 px-3 py-2',
                'text-sm text-slate-200 font-mono-data leading-relaxed',
                'focus:border-koyal/50 focus:outline-none resize-none',
              ].join(' ')}
            />
          </div>

          {/* Max concurrent */}
          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1.5">Max Concurrent Dials</label>
            <div className="flex items-center gap-3">
              <input
                type="range"
                min={1} max={10}
                value={maxConcur}
                onChange={(e) => setMaxConcur(Number(e.target.value))}
                className="w-24 accent-koyal"
              />
              <span className="text-sm text-slate-300 font-mono-data">{maxConcur}</span>
            </div>
          </div>

          <Button onClick={handleLaunch} loading={launching} className="w-full">
            🚀 Launch Campaign
          </Button>
        </div>

        {error && <ErrorBanner message={error} className="mt-4" />}
      </Card>

      {/* Contact list editor */}
      <Card>
        <CardHeader>
          <CardTitle>Contact List <span className="text-xs font-normal text-slate-600 ml-1">JSON array</span></CardTitle>
        </CardHeader>
        <textarea
          value={contacts}
          onChange={(e) => setContacts(e.target.value)}
          rows={12}
          spellCheck={false}
          aria-label="Contact list JSON"
          className={[
            'w-full rounded-lg border border-navy-600 bg-navy-800 px-3 py-2',
            'font-mono-data text-xs text-slate-300 leading-relaxed',
            'focus:border-koyal/50 focus:outline-none resize-none',
          ].join(' ')}
        />
      </Card>

      {/* ── Results ──────────────────────────────────────────────────────── */}
      {result && (
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Campaign Results</CardTitle>
          </CardHeader>

          {/* Summary */}
          <div className="grid grid-cols-3 gap-4 mb-4">
            {[
              { label: 'Total',    value: result.total,   color: 'text-slate-300' },
              { label: 'Dialing',  value: result.dialing, color: 'text-emerald-400' },
              { label: 'Errors',   value: result.errors,  color: result.errors > 0 ? 'text-rose-400' : 'text-slate-500' },
            ].map(({ label, value, color }) => (
              <div key={label} className="text-center p-3 rounded-lg bg-navy-800">
                <div className={`text-2xl font-bold font-mono-data ${color}`}>{value}</div>
                <div className="text-xs text-slate-500 mt-1">{label}</div>
              </div>
            ))}
          </div>

          {/* Per-contact results */}
          <ul className="space-y-2">
            {result.results.map((dial, i) => (
              <li
                key={i}
                className={[
                  'flex items-start justify-between rounded-lg px-3 py-2.5 text-xs',
                  dial.status === 'dialing' ? 'bg-emerald-500/5 border border-emerald-500/20' :
                  dial.status === 'error'   ? 'bg-rose-500/5 border border-rose-500/20' :
                                              'bg-navy-700 border border-navy-600',
                ].join(' ')}
              >
                <div>
                  <div className="font-mono-data text-slate-200">
                    {dial.phone}
                    <LanguageBadge language={dial.language} size="xs" className="ml-2" />
                  </div>
                  {dial.error && (
                    <p className="text-rose-400 mt-1">{dial.error}</p>
                  )}
                  {dial.setup_duration_ms > 0 && (
                    <p className="text-slate-500 mt-0.5">
                      setup: {formatMs(dial.setup_duration_ms)}
                    </p>
                  )}
                </div>
                <Badge
                  variant={dial.status === 'dialing' ? 'success' : dial.status === 'error' ? 'error' : 'neutral'}
                  className="shrink-0"
                >
                  {dial.status}
                </Badge>
              </li>
            ))}
          </ul>

          <Button variant="secondary" size="sm" onClick={() => setResult(null)} className="w-full mt-4">
            Clear results
          </Button>
        </Card>
      )}
    </div>
  )
}