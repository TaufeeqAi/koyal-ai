// ── frontend/eslint.config.mjs ──────────────────────────────────────────────
/**
 * ESLint 10 flat config for the KoyalAI dashboard.
 *
 * Lint stack:
 *   - typescript-eslint: parses .ts / .tsx + supplies TS-aware rules
 *   - No eslint-config-next (its bundled eslint-plugin-react is broken on
 *     ESLint 10 — see git history; we use plain rules + tsc instead)
 */
import tseslint from 'typescript-eslint'

export default tseslint.config(
  // ── Global ignores ─────────────────────────────────────────────────────
  {
    ignores: [
      'node_modules/**',
      '.next/**',
      'next-env.d.ts',
      'out/**',
      'build/**',
      'dist/**',
      'coverage/**',
      'public/**',
      'eslint.config.mjs',          // don't lint our own config
    ],
  },

  // ── TypeScript recommended ruleset (includes the parser) ───────────────
  ...tseslint.configs.recommended,

  // ── Global language options (applies to .ts, .tsx, .js, .jsx, .mjs) ───
  {
    files: ['**/*.{ts,tsx,js,jsx,mjs}'],
    languageOptions: {
      ecmaVersion: 'latest',
      sourceType: 'module',
      parserOptions: {
        ecmaFeatures: { jsx: true },
      },
      globals: {
        // Browser
        window:    'readonly',
        document:  'readonly',
        navigator: 'readonly',
        // Node (for route.ts files)
        process:     'readonly',
        console:     'readonly',
        fetch:       'readonly',
        AbortSignal: 'readonly',
        URL:         'readonly',
        URLSearchParams: 'readonly',
        // React 19
        React: 'readonly',
        // Next.js
        JSX:          'readonly',
        NextRequest:  'readonly',
        NextResponse: 'readonly',
      },
    },
  },

  // ── KoyalAI project-wide rule overrides ────────────────────────────────
  {
    rules: {
      // Disable base no-unused-vars in favour of TS-aware version
      'no-unused-vars': 'off',
      '@typescript-eslint/no-unused-vars': [
        'error',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
          caughtErrorsIgnorePattern: '^_',
        },
      ],
      '@typescript-eslint/no-explicit-any': 'warn',
      '@typescript-eslint/no-unused-expressions': 'warn',
      'no-console':     'warn',
      'no-debugger':    'error',
    },
  },
)