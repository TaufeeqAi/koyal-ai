/**
 * src/app/layout.tsx
 * ───────────────────
 * Root layout for the KoyalAI dashboard.
 *
 * Next.js 16 App Router root layout:
 *   - Sets dark HTML class (always-dark theme)
 *   - Loads Sora + JetBrains Mono from Google Fonts via next/font
 *   - Injects font CSS variables used in globals.css
 *   - Includes NavBar sidebar + main content wrapper
 *   - Viewport metadata for mobile theme color 
 *   - Skip-to-content link for accessibility 
 *
 * NavBar is a Client Component (uses usePathname + SWR).
 * The layout shell itself is a Server Component.
 */

import type { Metadata, Viewport } from 'next'
import { Sora, JetBrains_Mono } from 'next/font/google'
import { NavBar } from '@/components/NavBar'
import './globals.css'

// ── Font loading ─────────────────────────────────────────────────────────────

const sora = Sora({
  subsets:  ['latin'],
  variable: '--font-sora',
  display:  'swap',
  weight:   ['300', '400', '500', '600', '700'],
})

const jbMono = JetBrains_Mono({
  subsets:  ['latin'],
  variable: '--font-jbmono',
  display:  'swap',
  weight:   ['400', '500'],
})

// ── Metadata ──────────────────────────────────────────────────────────────────

export const metadata: Metadata = {
  title: {
    template: '%s | KoyalAI',
    default:  'KoyalAI Dashboard',
  },
  description: 'Multilingual Voice AI Platform for Indian Enterprises — Live call monitor, cost tracking, and RAGAS evaluation.',
  robots: { index: false, follow: false },  // Not for public search indexing
}

// Viewport metadata for mobile theme color and responsive behavior (B port)
export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  themeColor: '#05060d',
}

// ── Root layout ───────────────────────────────────────────────────────────────

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${sora.variable} ${jbMono.variable} dark`}>
      <head>
        {/* Preconnect to Google Fonts for faster Devanagari loading (B port) */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
      </head>
      <body className="antialiased">
        {/* Skip-to-content link for keyboard accessibility (B port) */}
        <a
          href="#main"
          className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-72 focus:z-50 focus:px-4 focus:py-2 focus:bg-koyal focus:text-navy-900 focus:rounded-lg"
        >
          Skip to main content
        </a>

        {/* Left sidebar — fixed 240px */}
        <NavBar />

        {/* Main content — offset by sidebar width */}
        <main className="ml-60 min-h-screen p-6 md:p-8 lg:p-10">
          <div id="main" className="max-w-7xl mx-auto">
            {children}
          </div>
        </main>
      </body>
    </html>
  )
}