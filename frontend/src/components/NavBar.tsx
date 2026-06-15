/**
 * src/components/NavBar.tsx
 * ──────────────────────────
 * Collapsible left sidebar navigation for the KoyalAI dashboard.
 * Shows the Phase 6 active room count as a live badge.
 *
 * Marked 'use client' because it:
 *   1. Uses pathname for active link highlighting
 *   2. Polls useTelephonyHealth for the active-rooms badge
 *
 * B port changes:
 *   - Inline SVG icons replaced with lucide-react (maintainability)
 *   - Activity icon replaces pulsing dot for status indicator
 */

'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import {
  Monitor,
  Users,
  PhoneOutgoing,
  BookOpenCheck,
  Bird,
  Activity,
} from 'lucide-react'
import { useTelephonyHealth } from '@/lib/api'
import { cn } from '@/lib/utils'

interface NavItem {
  href: string
  label: string
  icon: React.ElementType
  description: string
}

// ── Navigation Configuration ────────────────────────────────────────────────

const NAV_ITEMS: NavItem[] = [
  {
    href: '/',
    label: 'Live Monitor',
    description: 'Real-time call transcripts',
    icon: Monitor,
  },
  {
    href: '/tenants',
    label: 'Tenants',
    description: 'Onboarding + documents',
    icon: Users,
  },
  {
    href: '/outbound',
    label: 'Outbound',
    description: 'Campaign dialing',
    icon: PhoneOutgoing,
  },
  {
    href: '/evals',
    label: 'Evaluations',
    description: 'RAGAS + DeepEval scores',
    icon: BookOpenCheck,
  },
]

// ── Component ─────────────────────────────────────────────────────────────

export function NavBar() {
  const pathname = usePathname()
  const { data: health } = useTelephonyHealth()
  const activeRooms = health?.active_rooms ?? 0

  return (
    <nav
      aria-label="Main navigation"
      className={cn(
        'fixed left-0 top-0 z-40 flex h-screen w-60 flex-col',
        'border-r border-navy-600 bg-navy-900/80 backdrop-blur-xl',
      )}
    >
      {/* ── Brand header ───────────────────────────────────────────────────── */}
      <div className="flex items-center gap-3 px-5 py-6 border-b border-navy-600">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-koyal/10 text-koyal">
          <Bird className="h-6 w-6 text-koyal" strokeWidth={1.5} />
        </div>
        <div className="flex flex-col min-w-0">
          <span className="text-lg font-semibold text-slate-100 tracking-tight leading-none">
            KoyalAI
          </span>
          <span className="mt-1 text-[10px] font-medium uppercase tracking-widest text-slate-500 leading-none">
            Voice Platform
          </span>
        </div>
      </div>

      {/* ── Navigation items ───────────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto px-3 py-4 space-y-1">
        {NAV_ITEMS.map((item) => {
          const isActive = item.href === '/'
            ? pathname === '/'
            : pathname.startsWith(item.href)

          return (
            <Link
              key={item.href}
              href={item.href}
              aria-current={isActive ? 'page' : undefined}
              className={cn(
                'group flex items-start gap-3 rounded-lg px-3 py-2.5',
                'text-sm transition-all duration-150',
                isActive
                  ? 'bg-koyal/10 text-koyal border border-koyal/20'
                  : 'text-slate-400 hover:bg-navy-700 hover:text-slate-200 border border-transparent',
              )}
            >
              <span
                className={cn(
                  'mt-0.5 shrink-0',
                  isActive ? 'text-koyal' : 'text-slate-500 group-hover:text-slate-400',
                )}
              >
                <item.icon className="h-5 w-5" strokeWidth={1.5} />
              </span>
              <div className="flex flex-col min-w-0">
                <span className="font-medium leading-snug">{item.label}</span>
                <span className="text-xs text-slate-500 group-hover:text-slate-400 leading-snug mt-0.5">
                  {item.description}
                </span>
              </div>
            </Link>
          )
        })}
      </div>

      {/* ── Status footer ────────────────────────────────────────────────────── */}
      <div className="border-t border-navy-600 px-5 py-4">
        <div className="flex items-center gap-2.5">
          <span className="relative flex h-2.5 w-2.5 shrink-0">
            {activeRooms > 0 && (
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
            )}
            <span
              className={cn(
                'relative inline-flex h-2.5 w-2.5 rounded-full shrink-0',
                activeRooms > 0 ? 'bg-emerald-400' : 'bg-slate-600',
              )}
            />
          </span>
          <span className="text-xs text-slate-400 leading-none">
            {activeRooms > 0
              ? `${activeRooms} active call${activeRooms !== 1 ? 's' : ''}`
              : 'No active calls'}
          </span>
        </div>
        <div className="mt-2 flex items-center gap-2">
          <Activity className="h-3 w-3 text-slate-600 shrink-0" strokeWidth={2} />
          <span className="text-[10px] text-slate-600 font-mono-data leading-none">
            Phase 6
          </span>
        </div>
      </div>
    </nav>
  )
}