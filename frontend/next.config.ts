import type { NextConfig } from 'next'

/**
 * KoyalAI Frontend — Next.js 16.2 Configuration
 *
 * Key Next.js 16 changes reflected here:
 *  - `output: 'standalone'`  → Docker multi-stage build copies only what's needed
 *  - Turbopack is the default bundler; no webpack config needed
 *  - `turbopackFileSystemCacheForDev` reduces restart times in large repos
 *  - `images.remotePatterns` replaces deprecated `images.domains`
 *  - `headers()` emits security headers (HSTS, X-Frame-Options, etc.) on every route
 */
const nextConfig: NextConfig = {
  // Standalone mode: copies only the required files for production deployment.
  // Phase 8's Dockerfile.frontend uses this for the minimal Docker image.
  output: 'standalone',

  // Prevent livekit-client from being bundled during SSR
  serverExternalPackages: ['livekit-client'],

  experimental: {
    // Turbopack file-system cache (stable in Next.js 16.1+).
    // Stores compiler artifacts on disk → faster restarts during development.
    turbopackFileSystemCacheForDev: true,
  },

  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'app.sarvam.ai',
        pathname: '/**',
      },
    ],
  },


  // ── Security headers
  // Emitted on every route. Order matters for duplicate-key policy:
  //   1. HSTS forces HTTPS for 2 years (incl. subdomains, preload-eligible)
  //   2. X-Content-Type-Options blocks MIME sniffing
  //   3. X-Frame-Options prevents clickjacking
  //   4. X-XSS-Protection enables legacy browser XSS filter
  //   5. Referrer-Policy limits cross-origin referrer leakage
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key:   'Strict-Transport-Security',
            value: 'max-age=63072000; includeSubDomains; preload',
          },
          { key: 'X-Content-Type-Options', value: 'nosniff' },
          { key: 'X-Frame-Options',        value: 'SAMEORIGIN' },
          { key: 'X-XSS-Protection',       value: '1; mode=block' },
          { key: 'Referrer-Policy',        value: 'strict-origin-when-cross-origin' },
        ],
      },
    ]
  },
  // All API calls from the browser are proxied through Route Handlers.
  // The backend URL is set server-side via BACKEND_URL env var.
  // No rewrites needed — Route Handlers in src/app/api/ do the proxying.
}

export default nextConfig