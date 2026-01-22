import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // If the frontend sees a request starting with /api...
      '/api': {
        target: 'http://127.0.0.1:8000', // ...send it to the backend
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '') // remove /api prefix
      }
    }
  }
})