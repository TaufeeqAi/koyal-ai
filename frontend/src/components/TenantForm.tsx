/**
 * src/components/TenantForm.tsx
 * ──────────────────────────────
 * Two-step tenant onboarding form:
 *   Step 1: Company name + primary language → POST /tenants/create
 *   Step 2: Document upload (Hindi + English) → POST /documents/upload
 *
 * Uses React 19's useActionState for step-1 server action style,
 * plus standard controlled inputs for the file upload step.
 */

'use client'

import { useState, useCallback, useMemo } from 'react'
import { createTenant, uploadDocument } from '@/lib/api'
import { LANGUAGE_LABELS } from '@/lib/constants'
import { Button } from '@/components/ui/Button'
import { ErrorBanner } from '@/components/ui/ErrorBanner'
import { Badge } from '@/components/ui/Badge'
import type { TenantConfig, LanguageCode } from '@/types'

type Step = 'create' | 'upload' | 'done'

const SUPPORTED_PRIMARY_LANGUAGES: LanguageCode[] = [
  'hi-IN', 'en-IN', 'mr-IN', 'ta-IN', 'te-IN',
]

const UPLOAD_LANGUAGES: { code: LanguageCode; label: string }[] = [
  { code: 'hi-IN', label: 'Hindi Guidelines' },
  { code: 'en-IN', label: 'English Guidelines' },
]

const ACCEPTED_FILE_TYPES = [
  'text/plain',
  'application/pdf',
  'text/markdown',
  '.txt',
  '.pdf',
  '.md',
]

export function TenantForm() {
  const [step, setStep] = useState<Step>('create')
  const [tenant, setTenant] = useState<TenantConfig | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [creating, setCreating] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadResults, setUploadResults] = useState<{ lang: string; chunks: number }[]>([])

  // Precompute step order for indicator
  const stepOrder: Step[] = ['create', 'upload', 'done']
  const stepIndex = useMemo(() => stepOrder.indexOf(step), [step])

  // ── Step 1: Create tenant ─────────────────────────────────────────────────
  const [companyName, setCompanyName] = useState('')
  const [primaryLang, setPrimaryLang] = useState<LanguageCode>('hi-IN')

  const handleCreate = useCallback(async () => {
    if (!companyName.trim()) {
      setError('Company name is required.')
      return
    }
    setError(null)
    setCreating(true)
    try {
      const result = await createTenant({ company_name: companyName.trim(), primary_language: primaryLang })
      setTenant(result)
      setStep('upload')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create tenant.')
    } finally {
      setCreating(false)
    }
  }, [companyName, primaryLang])

  // ── Step 2: Upload documents ───────────────────────────────────────────────
  const [files, setFiles] = useState<Partial<Record<LanguageCode, File>>>({})
  const [fileError, setFileError] = useState<string | null>(null)

  const validateFile = (file: File): string | null => {
    if (file.size > 5 * 1024 * 1024) {
      return `${file.name} exceeds the 5 MB limit.`
    }
    const ext = file.name.split('.').pop()?.toLowerCase()
    const isValidType =
      ACCEPTED_FILE_TYPES.includes(file.type) ||
      (ext && (ext === 'txt' || ext === 'pdf' || ext === 'md'))
    if (!isValidType) {
      return `${file.name} is not a supported file type. Use .txt, .pdf, or .md.`
    }
    return null
  }

  const handleFileChange = (lang: LanguageCode) => (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) {
      setFiles((prev) => {
        const newFiles = { ...prev }
        delete newFiles[lang]
        return newFiles
      })
      setFileError(null)
      return
    }
    const validationError = validateFile(file)
    if (validationError) {
      setFileError(validationError)
      return
    }
    setFileError(null)
    setFiles((prev) => ({ ...prev, [lang]: file }))
  }

  const handleUpload = useCallback(async () => {
    if (!tenant?.api_key) {
      setFileError('Missing API key. Please restart the onboarding process.')
      return
    }
    const selectedFiles = Object.entries(files) as [LanguageCode, File][]
    if (selectedFiles.length === 0) {
      setFileError('Upload at least one document to continue.')
      return
    }
    setFileError(null)
    setUploading(true)

    const results: { lang: string; chunks: number }[] = []
    let lastError: unknown = null

    for (const [lang, file] of selectedFiles) {
      try {
        const result = await uploadDocument(tenant.api_key, file, lang, tenant.tenant_id)
        results.push({ lang: LANGUAGE_LABELS[lang as keyof typeof LANGUAGE_LABELS] ?? lang, chunks: result.chunks_ingested })
      } catch (err) {
        lastError = err
        // Stop on first failure to avoid partial state that's hard to resume
        break
      }
    }

    if (lastError) {
      setFileError(
        lastError instanceof Error ? lastError.message : 'Upload failed. Please retry.'
      )
      // Keep successful results to show what was already uploaded? Better to reset and ask to retry all.
      setUploadResults([])
    } else {
      setUploadResults(results)
      setStep('done')
    }
    setUploading(false)
  }, [files, tenant])

  const handleReset = () => {
    setStep('create')
    setTenant(null)
    setCompanyName('')
    setPrimaryLang('hi-IN')
    setFiles({})
    setError(null)
    setFileError(null)
    setUploadResults([])
  }

  // Helper for step indicator
  const getStepStatus = (s: Step): 'active' | 'past' | 'future' => {
    const idx = stepOrder.indexOf(s)
    if (idx === stepIndex) return 'active'
    if (idx < stepIndex) return 'past'
    return 'future'
  }

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="space-y-6" aria-live="polite">
      {/* Step indicator */}
      <div className="flex items-center gap-2" aria-label="Onboarding steps">
        {stepOrder.map((s, i) => {
          const status = getStepStatus(s)
          const label = i === 0 ? 'Create' : i === 1 ? 'Documents' : 'Done'
          const stepNumber = i + 1
          return (
            <div key={s} className="flex items-center gap-2">
              <span
                className={[
                  'rounded-full px-2.5 py-0.5 text-xs font-medium transition-colors',
                  status === 'active'
                    ? 'bg-koyal/15 text-koyal border border-koyal/30'
                    : status === 'past'
                    ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
                    : 'bg-navy-700 text-slate-500 border border-navy-600',
                ].join(' ')}
                aria-current={status === 'active' ? 'step' : undefined}
              >
                {status === 'past' ? '✓' : `${stepNumber}. ${label}`}
              </span>
              {i < stepOrder.length - 1 && status !== 'future' && (
                <span className="text-slate-600" aria-hidden="true">
                  →
                </span>
              )}
            </div>
          )
        })}
      </div>

      {/* ── Step 1: Create ─────────────────────────────────────────────────── */}
      {step === 'create' && (
        <form
          onSubmit={(e) => {
            e.preventDefault()
            handleCreate()
          }}
          className="space-y-4"
        >
          {error && <ErrorBanner message={error} />}

          <div>
            <label htmlFor="companyName" className="block text-xs font-medium text-slate-400 mb-1.5">
              Company Name <span className="text-rose-400">*</span>
            </label>
            <input
              id="companyName"
              type="text"
              value={companyName}
              onChange={(e) => setCompanyName(e.target.value)}
              placeholder="e.g. HDFC Bank, Swiggy"
              required
              disabled={creating}
              aria-required="true"
              className={[
                'w-full rounded-lg border bg-navy-800 px-3 py-2 text-sm text-slate-200',
                'border-navy-600 placeholder-slate-600',
                'focus:border-koyal/50 focus:outline-none focus:ring-1 focus:ring-koyal/30',
                'transition-colors disabled:opacity-50',
              ].join(' ')}
            />
          </div>

          <div>
            <label htmlFor="primaryLang" className="block text-xs font-medium text-slate-400 mb-1.5">
              Primary Language
            </label>
            <select
              id="primaryLang"
              value={primaryLang}
              onChange={(e) => setPrimaryLang(e.target.value as LanguageCode)}
              disabled={creating}
              className={[
                'w-full rounded-lg border bg-navy-800 px-3 py-2 text-sm text-slate-200',
                'border-navy-600 focus:border-koyal/50 focus:outline-none focus:ring-1 focus:ring-koyal/30',
                'disabled:opacity-50',
              ].join(' ')}
            >
              {SUPPORTED_PRIMARY_LANGUAGES.map((code) => (
                <option key={code} value={code}>
                  {LANGUAGE_LABELS[code as keyof typeof LANGUAGE_LABELS] ?? code}
                </option>
              ))}
            </select>
          </div>

          <Button type="submit" loading={creating} className="w-full">
            Create Tenant →
          </Button>
        </form>
      )}

      {/* ── Step 2: Document upload ────────────────────────────────────────── */}
      {step === 'upload' && tenant && (
        <div className="space-y-4">
          <div className="rounded-lg border border-emerald-500/20 bg-emerald-500/5 p-3">
            <p className="text-sm text-emerald-400 font-medium">✓ Tenant created</p>
            <p className="text-xs text-slate-500 font-mono-data mt-1">
              ID: {tenant.tenant_id}
            </p>
            {tenant.api_key ? (
              <div className="mt-2 p-2 rounded bg-navy-800 border border-koyal/20">
                <p className="text-xs text-koyal font-medium mb-1">
                  API Key (shown once — copy now)
                </p>
                <code className="text-xs text-slate-300 font-mono-data break-all">
                  {tenant.api_key}
                </code>
              </div>
            ) : (
              <p className="text-xs text-rose-400 mt-2">Warning: No API key returned. Please contact support.</p>
            )}
          </div>

          {fileError && <ErrorBanner message={fileError} />}

          <div className="space-y-3">
            {UPLOAD_LANGUAGES.map(({ code, label }) => {
              const file = files[code]
              const inputId = `file-${code}`
              return (
                <div key={code}>
                  <label htmlFor={inputId} className="block text-xs font-medium text-slate-400 mb-1.5">
                    {label}
                  </label>
                  <div
                    className={[
                      'flex items-center gap-3 rounded-lg border px-3 py-2',
                      file ? 'border-koyal/30 bg-koyal/5' : 'border-navy-600 bg-navy-800',
                    ].join(' ')}
                  >
                    <input
                      id={inputId}
                      type="file"
                      accept=".txt,.pdf,.md"
                      onChange={handleFileChange(code)}
                      disabled={uploading}
                      aria-describedby={fileError ? `${inputId}-error` : undefined}
                      className="text-xs text-slate-400 file:mr-3 file:rounded file:border-0 file:bg-navy-700 file:px-2 file:py-1 file:text-xs file:text-slate-300 hover:file:bg-navy-600 disabled:opacity-50"
                    />
                    {file && (
                      <span className="text-xs text-slate-500 font-mono-data">
                        {(file.size / 1024).toFixed(0)} KB
                      </span>
                    )}
                  </div>
                  <div id={`${inputId}-error`} className="text-xs text-rose-400 mt-1" aria-live="polite">
                    {/* error messages displayed globally above */}
                  </div>
                </div>
              )
            })}
          </div>

          <p className="text-xs text-slate-600">
            Accepts .txt, .pdf, .md — max 5 MB each. Content is chunked and embedded into the tenant's Qdrant collection.
          </p>

          <div className="flex gap-3">
            <Button variant="ghost" size="sm" onClick={handleReset} disabled={uploading}>
              Start over
            </Button>
            <Button
              onClick={handleUpload}
              loading={uploading}
              disabled={Object.keys(files).length === 0 || !tenant.api_key}
              className="flex-1"
            >
              Ingest Documents →
            </Button>
          </div>
        </div>
      )}

      {/* ── Step 3: Done ──────────────────────────────────────────────────── */}
      {step === 'done' && (
        <div className="space-y-4 text-center">
          <div className="rounded-full w-12 h-12 bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center mx-auto">
            <span className="text-emerald-400 text-lg" aria-hidden="true">
              ✓
            </span>
          </div>
          <div>
            <p className="text-lg font-semibold text-slate-200">Setup complete!</p>
            <p className="text-sm text-slate-500 mt-1">
              {tenant?.company_name} is ready to receive calls.
            </p>
          </div>
          <div className="space-y-2">
            {uploadResults.map(({ lang, chunks }) => (
              <div
                key={lang}
                className="flex items-center justify-between text-sm px-4 py-2 rounded-lg bg-navy-800 border border-navy-600"
              >
                <span className="text-slate-400">{lang}</span>
                <span className="text-emerald-400 font-mono-data">{chunks} chunks ingested</span>
              </div>
            ))}
          </div>
          <Button variant="secondary" onClick={handleReset}>
            Onboard another tenant
          </Button>
        </div>
      )}
    </div>
  )
}