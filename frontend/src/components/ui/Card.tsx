/**
 * src/components/ui/Card.tsx
 * ───────────────────────────
 * Reusable card container with KoyalAI deep-navy theme.
 * Supports optional header, footer, and hover glow.
 */

import { cn } from '@/lib/utils'
import type { ReactNode } from 'react'

interface CardProps {
  children: ReactNode
  className?: string
  /** Show a subtle cyan glow on hover */
  hoverable?: boolean
  /** Add padding to the content area */
  padded?: boolean
}

export function Card({ children, className, hoverable, padded = true }: CardProps) {
  return (
    <div
      className={cn(
        'rounded-xl border border-navy-600 bg-card-gradient shadow-card',
        padded && 'p-5',
        hoverable && 'transition-shadow duration-200 hover:shadow-card-hover hover:border-koyal/20',
        className,
      )}
    >
      {children}
    </div>
  )
}

export function CardHeader({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={cn('mb-4 flex items-center justify-between', className)}>
      {children}
    </div>
  )
}

export function CardTitle({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <h3 className={cn('text-sm font-semibold uppercase tracking-widest text-slate-400', className)}>
      {children}
    </h3>
  )
}