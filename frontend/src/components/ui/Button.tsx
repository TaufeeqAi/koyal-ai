/**
 * src/components/ui/Button.tsx
 * ─────────────────────────────
 * Button with primary / secondary / ghost / danger variants.
 * Supports loading state with spinner.
 */

import { cn } from '@/lib/utils'
import { LoadingSpinner } from './LoadingSpinner'
import type { ButtonHTMLAttributes, ReactNode } from 'react'

type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger'
type ButtonSize = 'sm' | 'md' | 'lg'

const VARIANT_CLASSES: Record<ButtonVariant, string> = {
  primary: 'bg-koyal text-navy-900 hover:bg-koyal-bright font-semibold shadow-koyal-glow',
  secondary: 'bg-navy-700 text-slate-200 border border-navy-600 hover:bg-navy-600',
  ghost: 'text-slate-400 hover:text-slate-200 hover:bg-navy-700',
  danger: 'bg-rose-500/10 text-rose-400 border border-rose-500/30 hover:bg-rose-500/20',
}

const SIZE_CLASSES: Record<ButtonSize, string> = {
  sm: 'px-3 py-1.5 text-xs rounded-lg',
  md: 'px-4 py-2 text-sm rounded-xl',
  lg: 'px-6 py-3 text-base rounded-xl',
}

const SPINNER_SIZE_MAP: Record<ButtonSize, 'sm' | 'md' | 'lg'> = {
  sm: 'sm',
  md: 'md',
  lg: 'md', // lg button still uses md spinner (proportionally fine)
}

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  children: ReactNode
  variant?: ButtonVariant
  size?: ButtonSize
  loading?: boolean
  icon?: ReactNode
  /** When true, hides children during loading (default: false) */
  hideChildrenWhenLoading?: boolean
}

export function Button({
  children,
  variant = 'primary',
  size = 'md',
  loading = false,
  icon,
  hideChildrenWhenLoading = false,
  disabled,
  className,
  type = 'button', // default to button to prevent accidental form submits
  ...props
}: ButtonProps) {
  const isDisabled = disabled || loading
  const spinnerSize = SPINNER_SIZE_MAP[size]

  // Determine content: show spinner if loading, otherwise icon if present
  const content = loading ? (
    <LoadingSpinner size={spinnerSize} />
  ) : (
    icon
  )

  // Show children unless loading and hideChildrenWhenLoading is true
  const showChildren = !(loading && hideChildrenWhenLoading)

  return (
    <button
      type={type}
      disabled={isDisabled}
      className={cn(
        'inline-flex items-center justify-center gap-2',
        'transition-all duration-150',
        'disabled:opacity-50 disabled:cursor-not-allowed',
        'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-koyal/50',
        VARIANT_CLASSES[variant],
        SIZE_CLASSES[size],
        className,
      )}
      {...props}
    >
      {content}
      {showChildren && children}
    </button>
  )
}