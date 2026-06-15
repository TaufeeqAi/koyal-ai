'use client'

import React, { useState, useEffect } from 'react'
import { cn } from '@/lib/utils'

interface ErrorBannerProps {
  message: string | null | undefined
  className?: string
  dismissible?: boolean
  autoDismiss?: number // milliseconds
  onDismiss?: () => void
  icon?: React.ReactNode
}

export function ErrorBanner({
  message,
  className,
  dismissible = true,
  autoDismiss,
  onDismiss,
  icon,
}: ErrorBannerProps) {
  const [dismissed, setDismissed] = useState(false)

  useEffect(() => {
    if (autoDismiss && autoDismiss > 0 && message && !dismissed) {
      const timer = setTimeout(() => {
        setDismissed(true)
        onDismiss?.()
      }, autoDismiss)
      return () => clearTimeout(timer)
    }
  }, [autoDismiss, message, dismissed, onDismiss])

  const handleDismiss = () => {
    setDismissed(true)
    onDismiss?.()
  }

  if (!message || dismissed) return null

  return (
    <div
      role="alert"
      className={cn(
        'flex items-start gap-3 rounded-xl border border-rose-500/30',
        'bg-rose-500/10 px-4 py-3 text-sm text-rose-400',
        'animate-fade-in transition-all duration-200 ease-out',
        className,
      )}
    >
      {/* Error icon */}
      {icon !== undefined ? (
        icon
      ) : (
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="16"
          height="16"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="mt-0.5 shrink-0"
          aria-hidden="true"
        >
          <circle cx="12" cy="12" r="10" />
          <line x1="12" x2="12" y1="8" y2="12" />
          <line x1="12" x2="12.01" y1="16" y2="16" />
        </svg>
      )}

      <span className="flex-1">{message}</span>

      {dismissible && (
        <button
          onClick={handleDismiss}
          aria-label="Dismiss error"
          className="shrink-0 opacity-70 hover:opacity-100 transition-opacity"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <line x1="18" x2="6" y1="6" y2="18" />
            <line x1="6" x2="18" y1="6" y2="18" />
          </svg>
        </button>
      )}
    </div>
  )
}