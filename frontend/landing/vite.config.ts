import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// https://vite.dev/config/
// The built page is served by nginx at /welcome/ (relative paths). The dev server uses the same /welcome/ base so the
// app's dev server can forward /welcome/ to it and a visitor sees the landing page first there too.
export default defineConfig(({ command }) => ({
  base: command === 'serve' ? '/welcome/' : './',
  plugins: [react()],
  build: { target: 'es2022', chunkSizeWarningLimit: 2000 },
  // the shaders shared with the app live in ../shared (frontend/shared/argilla)
  server: { fs: { allow: ['..'] } },
  // the shared code resolves packages from frontend/node_modules; keep one copy of each, this project's
  resolve: { dedupe: ['react', 'react-dom', 'gsap', 'three', '@react-three/fiber'] },
}));
