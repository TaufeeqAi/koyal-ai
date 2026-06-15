import { cn } from '@/lib/utils'

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg'
  className?: string
}

const SIZE_CLASSES = {
  sm: 'h-3.5 w-3.5 border-[1.5px]',
  md: 'h-5 w-5 border-2',
  lg: 'h-8 w-8 border-2',
}

export function LoadingSpinner({ size = 'md', className }: LoadingSpinnerProps) {
  return (
    <span
      role="status"
      aria-label="Loading"
      className={cn(
        'inline-block animate-spin rounded-full',
        'border-koyal/30 border-t-koyal',
        SIZE_CLASSES[size],
        className,
      )}
    />
  )
}

interface PageLoaderProps {
  label?: string
  fullScreen?: boolean
}

export function PageLoader({ label = 'Loading…', fullScreen = false }: PageLoaderProps) {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center gap-3',
        fullScreen && 'fixed inset-0 bg-navy-900/80 backdrop-blur-sm z-50',
        !fullScreen && 'py-8'
      )}
    >
      <LoadingSpinner size="lg" />
      <span className="text-sm text-slate-400 font-medium">{label}</span>
    </div>
  )
}