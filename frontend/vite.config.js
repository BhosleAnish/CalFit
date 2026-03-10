import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default {
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:5000',
      '/auth': 'http://localhost:5000',      // ← add this for Google OAuth
      '/static': 'http://localhost:5000',    // ← add this for Flask static files
    }
  }
}