export const config = {
  api: {
    baseUrl: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
    wsUrl: import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws',
  },
  app: {
    name: 'Arbitration Platform',
    version: '1.0.0',
  },
} as const;