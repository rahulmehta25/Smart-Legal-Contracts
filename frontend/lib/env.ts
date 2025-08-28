/**
 * Environment configuration with type safety and deployment support
 * Updated for Vite environment variables
 */

// Get deployment environment
const isVercel = import.meta.env.VERCEL === '1';
const isProduction = import.meta.env.MODE === 'production';
const isDevelopment = import.meta.env.MODE === 'development';

// Backend URL resolution with fallbacks
function getBackendUrl(): string {
  // Priority order: explicit env var, production default, development default
  if (import.meta.env.VITE_BACKEND_URL) {
    return import.meta.env.VITE_BACKEND_URL;
  }
  
  if (import.meta.env.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL;
  }
  
  if (isProduction && isVercel) {
    // Use environment variable that should be set in Vercel build
    return import.meta.env.VITE_API_BASE_URL || 'https://api.example.com';
  }
  
  // Development fallbacks
  return import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';
}

// WebSocket URL resolution
function getWebSocketUrl(): string {
  if (import.meta.env.VITE_WS_URL) {
    return import.meta.env.VITE_WS_URL;
  }
  
  const backendUrl = getBackendUrl();
  const wsProtocol = backendUrl.startsWith('https://') ? 'wss://' : 'ws://';
  const wsHost = backendUrl.replace(/^https?:\/\//, '').replace(/\/$/, '');
  
  return `${wsProtocol}${wsHost}/ws`;
}

export const env = {
  // API Configuration
  API_URL: getBackendUrl(),
  WS_URL: getWebSocketUrl(),
  BACKEND_URL: getBackendUrl(),
  
  // Environment Detection
  NODE_ENV: import.meta.env.MODE || 'development',
  IS_DEVELOPMENT: isDevelopment,
  IS_PRODUCTION: isProduction,
  IS_VERCEL: isVercel,
  IS_PREVIEW: import.meta.env.VERCEL_ENV === 'preview',
  
  // Vercel specific
  VERCEL_URL: import.meta.env.VERCEL_URL || '',
  VERCEL_ENV: import.meta.env.VERCEL_ENV || '',
  VERCEL_GIT_COMMIT_SHA: import.meta.env.VERCEL_GIT_COMMIT_SHA || '',
  VERCEL_GIT_COMMIT_MESSAGE: import.meta.env.VERCEL_GIT_COMMIT_MESSAGE || '',
  
  // Application
  APP_NAME: import.meta.env.VITE_APP_NAME || 'Arbitration Detector',
  APP_VERSION: import.meta.env.VITE_APP_VERSION || '1.0.0',
  APP_DESCRIPTION: import.meta.env.VITE_APP_DESCRIPTION || 'AI-powered arbitration clause detection in legal documents',
  
  // Feature Flags
  ENABLE_ANALYTICS: import.meta.env.VITE_ENABLE_ANALYTICS === 'true',
  ENABLE_DEBUG_MODE: import.meta.env.VITE_ENABLE_DEBUG === 'true' || isDevelopment,
  ENABLE_ERROR_REPORTING: import.meta.env.VITE_ENABLE_ERROR_REPORTING === 'true',
  
  // API Configuration
  API_TIMEOUT: parseInt(import.meta.env.VITE_API_TIMEOUT || '30000'),
  MAX_FILE_SIZE: parseInt(import.meta.env.VITE_MAX_FILE_SIZE || '10485760'), // 10MB
  
  // Security
  CSP_NONCE: import.meta.env.CSP_NONCE || '',
  
  // Third-party Services
  SENTRY_DSN: import.meta.env.VITE_SENTRY_DSN || '',
  GA_TRACKING_ID: import.meta.env.VITE_GA_TRACKING_ID || '',
} as const;

// Environment validation with detailed error reporting
export function validateEnv(): void {
  const errors: string[] = [];
  const warnings: string[] = [];
  
  // Required environment variables
  const required = [
    { key: 'API_URL', value: env.API_URL },
    { key: 'WS_URL', value: env.WS_URL },
  ] as const;
  
  for (const { key, value } of required) {
    if (!value) {
      errors.push(`Missing required environment variable: ${key}`);
    }
  }
  
  // Production-specific validations
  if (isProduction) {
    if (!process.env.BACKEND_URL && !process.env.NEXT_PUBLIC_BACKEND_URL) {
      warnings.push('BACKEND_URL not set for production deployment');
    }
    
    if (env.API_URL.includes('localhost')) {
      warnings.push('API_URL points to localhost in production environment');
    }
    
    if (!env.SENTRY_DSN && env.ENABLE_ERROR_REPORTING) {
      warnings.push('Error reporting enabled but SENTRY_DSN not configured');
    }
  }
  
  // Log warnings
  if (warnings.length > 0) {
    console.warn('Environment configuration warnings:', warnings);
  }
  
  // Throw errors
  if (errors.length > 0) {
    throw new Error(`Environment validation failed: ${errors.join(', ')}`);
  }
}

// Configuration summary for debugging
export function getConfigSummary() {
  return {
    environment: env.NODE_ENV,
    deployment: {
      isVercel: env.IS_VERCEL,
      isProduction: env.IS_PRODUCTION,
      isPreview: env.IS_PREVIEW,
      vercelEnv: env.VERCEL_ENV,
    },
    api: {
      backendUrl: env.API_URL,
      websocketUrl: env.WS_URL,
      timeout: env.API_TIMEOUT,
    },
    features: {
      analytics: env.ENABLE_ANALYTICS,
      debugMode: env.ENABLE_DEBUG_MODE,
      errorReporting: env.ENABLE_ERROR_REPORTING,
    },
    build: {
      version: env.APP_VERSION,
      commit: env.VERCEL_GIT_COMMIT_SHA,
    },
  };
}

// Auto-validate environment on import in development
if (isDevelopment && typeof window === 'undefined') {
  try {
    validateEnv();
  } catch (error) {
    console.error('Environment validation failed:', error);
  }
}

export default env;