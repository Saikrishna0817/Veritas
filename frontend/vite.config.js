import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],

  // ── Dev server proxy (M3) ──────────────────────────────────────────────────
  // Forwards API and WebSocket requests to the FastAPI backend during development
  // so you don't need to set VITE_API_BASE_URL manually.
  server: {
    port: 5173,
    proxy: {
      // REST API
      '/api': {
        target: 'http://localhost:8001',
        changeOrigin: true,
      },
      // WebSocket detection stream
      '/ws': {
        target: 'ws://localhost:8001',
        ws: true,
        changeOrigin: true,
      },
    },
  },
})
