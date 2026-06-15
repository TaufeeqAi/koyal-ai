/**
 * src/components/LanguageBadge.tsx
 * ─────────────────────────────────
 * Coloured pill badge showing the detected language for a call turn.
 * Colours match the KoyalAI language encoding defined in tailwind.config.ts.
 */

import { LANGUAGE_LABELS, LANGUAGE_COLOR } from '@/lib/constants'
import { cn } from '@/lib/utils'
import type { LanguageCode } from '@/types'

interface LanguageBadgeProps {
  language: LanguageCode
  size?: 'xs' | 'sm'
  className?: string
}

export function LanguageBadge({ language, size = 'xs', className }: LanguageBadgeProps) {
  // Type-safe lookup with fallback for unknown/future language codes
  const label = LANGUAGE_LABELS[language as keyof typeof LANGUAGE_LABELS] ?? language
  const colorClass = LANGUAGE_COLOR[language as keyof typeof LANGUAGE_COLOR] 
    ?? 'bg-slate-700/50 text-slate-400 border-slate-600/50'

  return (
    <span
      title={`Detected language: ${label}`}
      aria-label={`Language: ${label}`}
      className={cn(
        'inline-flex items-center rounded-full border font-mono-data font-medium',
        size === 'xs' ? 'px-2 py-0.5 text-xs' : 'px-2.5 py-1 text-xs',
        colorClass,
        className,
      )}
    >
      {label}
    </span>
  )
}