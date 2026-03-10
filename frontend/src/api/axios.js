import axios from 'axios'

const api = axios.create({
  baseURL: '/api',           // proxied to http://localhost:5000 via vite.config.js
  withCredentials: true,     // sends Flask session cookie with every request
})

// Global response interceptor — redirect to login on 401
api.interceptors.response.use(
  response => response,
  error => {
    if (error.response?.status === 401) {
      window.location.href = '/'
    }
    return Promise.reject(error)
  }
)

export default api