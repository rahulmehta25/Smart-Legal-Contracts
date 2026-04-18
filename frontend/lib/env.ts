/**
 * Environment configuration for Next.js
 */

const isProduction = process.env.NODE_ENV === 'production';
const isDevelopment = process.env.NODE_ENV === 'development';

function getApiUrl(): string {
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL;
  }
  if (isProduction) {
    return 'https://api.smartlegalcontracts.com';
  }
  return 'http://localhost:8000';
}

function getWebSocketUrl(): string {
  if (process.env.NEXT_PUBLIC_WS_URL) {
    return process.env.NEXT_PUBLIC_WS_URL;
  }
  const apiUrl = getApiUrl();
  const wsProtocol = apiUrl.startsWith('https://') ? 'wss://' : 'ws://';
  const wsHost = apiUrl.replace(/^https?:\/\//, '').replace(/\/$/, '');
  return `${wsProtocol}${wsHost}/ws`;
}

export const env = {
  API_URL: getApiUrl(),
  WS_URL: getWebSocketUrl(),

  NODE_ENV: process.env.NODE_ENV || 'development',
  IS_DEVELOPMENT: isDevelopment,
  IS_PRODUCTION: isProduction,

  APP_NAME: 'Smart Legal Contracts',
  APP_VERSION: '2.0.0',

  API_TIMEOUT: parseInt(process.env.NEXT_PUBLIC_API_TIMEOUT || '30000'),
  MAX_FILE_SIZE: parseInt(process.env.NEXT_PUBLIC_MAX_FILE_SIZE || '52428800'), // 50MB
  ENABLE_DEBUG_MODE: process.env.NEXT_PUBLIC_DEBUG === 'true' || isDevelopment,
} as const;

export default env;
