import type { Config } from 'tailwindcss'

/**
 * KoyalAI Design System — Tailwind CSS v3.4 Configuration
 *
 * Palette:
 *   navy-950/900/800  → page backgrounds (deep operational dark)
 *   cyan-*            → primary accent ("Koyal blue" — electric, confident)
 *   indigo-*          → secondary accent
 *   saffron (orange)  → Hindi language (hi-IN) — draws from Indian flag palette
 *   blue              → English (en-IN)
 *   violet            → Hinglish (hi-IN+en-IN) — mix of the two
 *   emerald           → pass / connected / success
 *   rose              → fail / error / escalation
 *   amber             → warning / low-confidence
 */
const config: Config = {
  content: ['./src/**/*.{js,ts,jsx,tsx,mdx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // ── Core surface palette ─────────────────────────────────────────
        navy: {
          950: '#05060d',
          900: '#0a0f1e',
          800: '#0f1629',
          700: '#141d35',
          600: '#1a2545',
          500: '#1e2d52',
        },
        // ── Brand accent ─────────────────────────────────────────────────
        // "Koyal" (Indian Nightingale) — bright, alive, precise
        koyal: {
          DEFAULT: '#22d3ee', // cyan-400
          dim:     '#0891b2', // cyan-600
          bright:  '#67e8f9', // cyan-300
          glow:    '#06b6d4', // cyan-500
        },
        // ── Language colour coding ────────────────────────────────────────
        'lang-hi':      '#fb923c', // saffron orange — Hindi
        'lang-en':      '#60a5fa', // sky blue — English
        'lang-hinglish':'#a78bfa', // violet — Hinglish (code-mixed)
        'lang-mr':      '#34d399', // emerald — Marathi
        'lang-ta':      '#f87171', // red-400 — Tamil
        'lang-te':      '#fbbf24', // amber — Telugu
      },
      fontFamily: {
        // Main UI font: Sora — geometric, confident, modern
        sans:  ['var(--font-sora)', 'system-ui', 'sans-serif'],
        // Metrics/code/transcript: JetBrains Mono — terminal authority
        mono:  ['var(--font-jbmono)', 'ui-monospace', 'monospace'],
      },
      boxShadow: {
        'koyal-glow': '0 0 20px rgba(34, 211, 238, 0.15)',
        'card':       '0 1px 3px rgba(0,0,0,0.4), 0 0 1px rgba(34,211,238,0.08)',
        'card-hover': '0 4px 12px rgba(0,0,0,0.5), 0 0 2px rgba(34,211,238,0.15)',
      },
      backgroundImage: {
        // Subtle grid pattern on page backgrounds — mission control aesthetic
        'grid-navy': `
          linear-gradient(rgba(30,45,82,0.3) 1px, transparent 1px),
          linear-gradient(90deg, rgba(30,45,82,0.3) 1px, transparent 1px)
        `,
      },
      backgroundSize: {
        'grid-navy': '40px 40px',
      },
      animation: {
        'pulse-dot':  'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'fade-in':    'fadeIn 0.2s ease-out',
        'slide-up':   'slideUp 0.25s ease-out',
        'glow-pulse': 'glowPulse 3s ease-in-out infinite',
      },
      keyframes: {
        fadeIn:    { '0%': { opacity: '0' }, '100%': { opacity: '1' } },
        slideUp:   { '0%': { transform: 'translateY(8px)', opacity: '0' }, '100%': { transform: 'translateY(0)', opacity: '1' } },
        glowPulse: {
          '0%, 100%': { boxShadow: '0 0 8px rgba(34,211,238,0.1)' },
          '50%':      { boxShadow: '0 0 20px rgba(34,211,238,0.25)' },
        },
      },
    },
  },
  plugins: [],
}

export default config