import { defineConfig } from 'vite'

export default defineConfig({
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      '/voices':     'http://localhost:3776',
      '/synthesize': 'http://localhost:3776',
      '/audio':      'http://localhost:3776',
      '/status':     'http://localhost:3776',
      '/unload':     'http://localhost:3776',
    },
  },
})
