import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'

export default tseslint.config(
  { ignores: ['dist'] },
  {
    extends: [js.configs.recommended, ...tseslint.configs.recommended],
    files: ['**/*.{ts,tsx}'],
    languageOptions: {
      ecmaVersion: 2022,
      globals: globals.browser,
    },
    plugins: {
      'react-hooks': reactHooks,
      'react-refresh': reactRefresh,
    },
    rules: {
      // Classic react-hooks ruleset. eslint-plugin-react-hooks v7 (required for
      // ESLint 10) ships an expanded `recommended` config; adopting its newer,
      // more opinionated rules (set-state-in-effect, immutability, …) is a
      // separate decision, so we keep the long-standing two rules here.
      'react-hooks/rules-of-hooks': 'error',
      'react-hooks/exhaustive-deps': 'warn',
      'react-refresh/only-export-components': [
        'warn',
        { allowConstantExport: true },
      ],
    },
  },
  {
    // File-based routes must export `Route` next to their component — that is
    // TanStack Router's contract, so the rule can never be satisfied here and
    // 24 permanent warnings only hide the ones worth reading. Same for the
    // generated shadcn/ui files, which pair a component with its `cva` variants
    // and must not be hand-edited, and for the app entry point, which mounts
    // rather than exports.
    files: [
      'src/routes/**/*.tsx',
      'src/components/ui/**/*.tsx',
      'src/main.tsx',
    ],
    rules: {
      'react-refresh/only-export-components': 'off',
    },
  },
  {
    // Playwright end-to-end tests. These are Node scripts driving a browser,
    // not React code: the fixture callback is named `use`, which the
    // rules-of-hooks rule mistakes for React's `use` hook.
    files: ['e2e/**/*.ts', 'playwright.config.ts'],
    languageOptions: {
      globals: { ...globals.node, ...globals.browser },
    },
    rules: {
      'react-hooks/rules-of-hooks': 'off',
      'react-refresh/only-export-components': 'off',
    },
  },
)
