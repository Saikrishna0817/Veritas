import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';

// https://vite.dev/config/
export default defineConfig({
  base: './',
  plugins: [react()],
  build: { target: 'es2022', chunkSizeWarningLimit: 2000 },
  // the shaders shared with the app live in ../shared (frontend/shared/argilla)
  server: { fs: { allow: ['..'] } },
  // the shared code resolves packages from frontend/node_modules; keep one copy of each, this project's
  resolve: { dedupe: ['react', 'react-dom', 'gsap', 'three', '@react-three/fiber'] },
});
