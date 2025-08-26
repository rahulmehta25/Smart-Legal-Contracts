// Environment configuration
export const config = {
  // API Configuration
  apiUrl: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  wsUrl: import.meta.env.VITE_WS_URL || 'ws://localhost:8000',
  
  // App Configuration
  appName: import.meta.env.VITE_APP_NAME || 'Arbitration Detector',
  appVersion: import.meta.env.VITE_APP_VERSION || '1.0.0',
  
  // Environment
  isDevelopment: import.meta.env.DEV,
  isProduction: import.meta.env.PROD,
  mode: import.meta.env.MODE,
  
  // Feature flags
  enableWebSocket: import.meta.env.VITE_ENABLE_WS !== 'false',
  enableAnalytics: import.meta.env.VITE_ENABLE_ANALYTICS === 'true',
  enableDebugLogs: import.meta.env.VITE_ENABLE_DEBUG === 'true' || import.meta.env.DEV,
  
  // API timeouts and limits
  apiTimeout: parseInt(import.meta.env.VITE_API_TIMEOUT || '10000', 10),
  maxFileSize: parseInt(import.meta.env.VITE_MAX_FILE_SIZE || '10485760', 10), // 10MB default
  maxFilesPerUpload: parseInt(import.meta.env.VITE_MAX_FILES_PER_UPLOAD || '5', 10),
} as const;

// Validate required environment variables
const requiredEnvVars = ['VITE_API_URL'] as const;

export function validateEnvironment(): void {
  const missingVars: string[] = [];
  
  requiredEnvVars.forEach((varName) => {
    if (!import.meta.env[varName] && varName === 'VITE_API_URL' && !config.apiUrl) {
      missingVars.push(varName);
    }
  });
  
  if (missingVars.length > 0) {
    console.warn(`Missing environment variables: ${missingVars.join(', ')}`);
    console.warn('Using default values. Consider creating a .env.local file.');
  }
  
  if (config.enableDebugLogs) {
    console.log('Environment configuration:', {
      mode: config.mode,
      apiUrl: config.apiUrl,
      wsUrl: config.wsUrl,
      isDevelopment: config.isDevelopment,
      isProduction: config.isProduction,
    });
  }
}

// Call validation on import in development
if (config.isDevelopment) {
  validateEnvironment();
}

export default config;