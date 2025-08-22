/**
 * React Native SDK for Arbitration Platform
 * Provides comprehensive arbitration clause detection and analysis
 */

export * from './ArbitrationSDK';
export * from './types';
export * from './hooks';
export * from './components';
export * from './utils';

// Native modules
export { default as ArbitrationDetector } from './ArbitrationDetector';
export { default as OCRModule } from './OCRModule';
export { default as BiometricAuth } from './BiometricAuth';
export { default as OfflineSync } from './OfflineSync';

// Constants
export const SDK_VERSION = '1.0.0';
export const SUPPORTED_DOCUMENT_TYPES = [
  'application/pdf',
  'text/plain',
  'application/msword',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
  'image/jpeg',
  'image/png',
  'image/tiff'
];

// Default configuration
export const DEFAULT_CONFIG = {
  apiBaseUrl: 'https://api.arbitration-platform.com',
  confidenceThreshold: 0.5,
  enableOfflineMode: true,
  enableAnalytics: true,
  cacheSize: 100,
  networkTimeout: 30000
};