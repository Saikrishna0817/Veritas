import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  globalIgnores(['dist', 'landing']),
  {
    files: ['**/*.{js,jsx}'],
    extends: [
      js.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
      parserOptions: {
        ecmaVersion: 'latest',
        ecmaFeatures: { jsx: true },
        sourceType: 'module',
      },
    },
    rules: {
      'no-unused-vars': ['error', { varsIgnorePattern: '^(motion$|[A-Z_])' }],
    },
  },
  { files: ['vite.config.js', 'playwright.config.js'], languageOptions: { globals: globals.node } },
  // React Three Fiber: three.js objects are made once (useMemo) and their uniforms changed later, outside React's
  // render. The React Compiler rules read that as mutating hook values; the landing page turns them off too.
  { files: ['src/scene/**/*.{js,jsx}'], rules: { 'react-hooks/immutability': 'off', 'react-hooks/refs': 'off' } },
])
