/**
 * src/components/ui/Badge.tsx
 * ────────────────────────────
 * Status badge with semantic colour variants.
 */

import { cn } from '@/lib/utils'
import type { ReactNode } from 'react'

type BadgeVariant =
  | 'default'
  | 'success'
  | 'warning'
  | 'error'
  | 'info'
  | 'neutral'

const VARIANT_CLASSES: Record<BadgeVariant, string> = {
  default: 'bg-koyal/10 text-koyal border-koyal/30',
  success: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
  warning: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
  error:   'bg-rose-500/10 text-rose-400 border-rose-500/30',
  info:    'bg-blue-500/10 text-blue-400 border-blue-500/30',
  neutral: 'bg-slate-700/50 text-slate-400 border-slate-600/50',
}

interface BadgeProps {
  children: ReactNode
  variant?: BadgeVariant
  className?: string
  dot?: boolean
}

export function Badge({ children, variant = 'default', className, dot }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium',
        VARIANT_CLASSES[variant],
        className,
      )}
    >
      {dot === true && (
        <span className={cn(
          'h-1.5 w-1.5 rounded-full',
          variant === 'success' && 'bg-emerald-400 animate-pulse-dot',
          variant === 'error'   && 'bg-rose-400',
          variant === 'warning' && 'bg-amber-400',
          variant === 'default' && 'bg-koyal animate-pulse-dot',
          variant === 'neutral' && 'bg-slate-500',
          variant === 'info'    && 'bg-blue-400',
        )} />
      )}
      {children}
    </span>
  )
}