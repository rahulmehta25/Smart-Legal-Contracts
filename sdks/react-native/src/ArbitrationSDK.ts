/**
 * Main SDK class for React Native Arbitration Platform
 */

import { NativeModules, NativeEventEmitter, Platform } from 'react-native';
import {
  SDKConfiguration,
  ArbitrationAnalysisResult,
  AnalysisRequest,
  BatchAnalysisResult,
  DocumentType,
  AnalyticsEvent,
  UserPreferences,
  OfflineSyncStatus,
  NetworkInfo,
  OCRResult,
  BiometricAuthResult,
  BiometricType,
  DEFAULT_CONFIG
} from './types';
import { validateConfiguration, validateAnalysisRequest } from './utils/validation';
import OfflineSync from './OfflineSync';
import { EventEmitter } from 'events';

const { ArbitrationDetectorModule } = NativeModules;

/**
 * Main SDK class providing all arbitration detection functionality
 */
export class ArbitrationSDK extends EventEmitter {
  private static instance: ArbitrationSDK | null = null;
  private configuration: SDKConfiguration;
  private nativeEventEmitter: NativeEventEmitter;
  private offlineSync: OfflineSync;
  private isInitialized = false;

  private constructor(config: SDKConfiguration = {}) {
    super();
    this.configuration = { ...DEFAULT_CONFIG, ...config };
    this.nativeEventEmitter = new NativeEventEmitter(ArbitrationDetectorModule);
    this.offlineSync = new OfflineSync(this.configuration);
    this.setupEventListeners();
  }

  /**
   * Get singleton instance of ArbitrationSDK
   */
  public static getInstance(config?: SDKConfiguration): ArbitrationSDK {
    if (!ArbitrationSDK.instance) {
      ArbitrationSDK.instance = new ArbitrationSDK(config);
    }
    return ArbitrationSDK.instance;
  }

  /**
   * Initialize the SDK with configuration
   */
  public async initialize(config?: SDKConfiguration): Promise<void> {
    if (this.isInitialized) {
      return;
    }

    if (config) {
      this.configuration = { ...this.configuration, ...config };
    }

    const validationResult = validateConfiguration(this.configuration);
    if (!validationResult.isValid) {
      throw new Error(`Invalid configuration: ${validationResult.errors.join(', ')}`);
    }

    try {
      // Configure native modules
      await ArbitrationDetectorModule?.configure(this.configuration);
      
      // Initialize offline sync
      if (this.configuration.enableOfflineMode) {
        await this.offlineSync.initialize();
      }

      this.isInitialized = true;
      this.emit('initialized', this.configuration);
    } catch (error) {
      throw new Error(`Failed to initialize SDK: ${error}`);
    }
  }

  /**
   * Update SDK configuration
   */
  public updateConfiguration(config: Partial<SDKConfiguration>): void {
    this.configuration = { ...this.configuration, ...config };
    ArbitrationDetectorModule?.configure(this.configuration);
    this.emit('configurationUpdated', this.configuration);
  }

  /**
   * Get current configuration
   */
  public getConfiguration(): SDKConfiguration {
    return { ...this.configuration };
  }

  // Analysis Methods

  /**
   * Analyze text for arbitration clauses
   */
  public async analyzeText(text: string): Promise<ArbitrationAnalysisResult> {
    this.ensureInitialized();

    const request: AnalysisRequest = { text };
    const validationResult = validateAnalysisRequest(request);
    
    if (!validationResult.isValid) {
      throw new Error(`Invalid analysis request: ${validationResult.errors.join(', ')}`);
    }

    try {
      this.emit('analysisStarted', { text: text.substring(0, 100) + '...' });

      const result = await ArbitrationDetectorModule.analyzeText(text);
      
      // Track analytics
      this.trackEvent({
        eventType: 'analysis_completed',
        timestamp: new Date(),
        properties: {
          hasArbitration: result.hasArbitration.toString(),
          confidence: result.confidence.toString(),
          riskLevel: result.riskLevel
        }
      });

      this.emit('analysisCompleted', result);
      return result;
    } catch (error) {
      this.trackEvent({
        eventType: 'analysis_failed',
        timestamp: new Date(),
        properties: {
          error: error.toString()
        }
      });

      this.emit('analysisError', error);
      throw error;
    }
  }

  /**
   * Analyze document from file
   */
  public async analyzeDocument(
    fileUri: string,
    documentType: DocumentType
  ): Promise<ArbitrationAnalysisResult> {
    this.ensureInitialized();

    try {
      this.emit('analysisStarted', { fileUri, documentType });

      const result = await ArbitrationDetectorModule.analyzeDocument(fileUri, documentType);

      this.trackEvent({
        eventType: 'document_analyzed',
        timestamp: new Date(),
        properties: {
          documentType,
          hasArbitration: result.hasArbitration.toString()
        }
      });

      this.emit('analysisCompleted', result);
      return result;
    } catch (error) {
      this.emit('analysisError', error);
      throw error;
    }
  }

  /**
   * Analyze image with OCR
   */
  public async analyzeImage(imageUri: string): Promise<ArbitrationAnalysisResult> {
    this.ensureInitialized();

    try {
      this.emit('analysisStarted', { imageUri });

      // First extract text using OCR
      const ocrResult = await this.extractTextFromImage(imageUri);
      
      // Then analyze the extracted text
      const result = await this.analyzeText(ocrResult.text);

      this.trackEvent({
        eventType: 'image_analyzed',
        timestamp: new Date(),
        properties: {
          ocrConfidence: ocrResult.confidence.toString(),
          hasArbitration: result.hasArbitration.toString()
        }
      });

      this.emit('analysisCompleted', result);
      return result;
    } catch (error) {
      this.emit('analysisError', error);
      throw error;
    }
  }

  /**
   * Batch analyze multiple requests
   */
  public async batchAnalyze(requests: AnalysisRequest[]): Promise<BatchAnalysisResult> {
    this.ensureInitialized();

    if (requests.length === 0) {
      throw new Error('No requests provided for batch analysis');
    }

    try {
      this.emit('batchAnalysisStarted', { count: requests.length });

      const result = await ArbitrationDetectorModule.batchAnalyze(requests);

      this.trackEvent({
        eventType: 'batch_analysis_completed',
        timestamp: new Date(),
        properties: {
          totalDocuments: result.summary.totalDocuments.toString(),
          documentsWithArbitration: result.summary.documentsWithArbitration.toString()
        }
      });

      this.emit('batchAnalysisCompleted', result);
      return result;
    } catch (error) {
      this.emit('batchAnalysisError', error);
      throw error;
    }
  }

  /**
   * Cancel ongoing analysis
   */
  public cancelAnalysis(): void {
    ArbitrationDetectorModule?.cancelAnalysis();
    this.emit('analysisCancelled');
  }

  // OCR Methods

  /**
   * Extract text from image using OCR
   */
  public async extractTextFromImage(imageUri: string): Promise<OCRResult> {
    this.ensureInitialized();

    try {
      const { OCRModule } = NativeModules;
      const result = await OCRModule.extractTextFromImage(imageUri);

      this.trackEvent({
        eventType: 'ocr_completed',
        timestamp: new Date(),
        properties: {
          textLength: result.text.length.toString(),
          confidence: result.confidence.toString()
        }
      });

      return result;
    } catch (error) {
      this.emit('ocrError', error);
      throw error;
    }
  }

  /**
   * Extract text from camera
   */
  public async extractTextFromCamera(): Promise<OCRResult> {
    this.ensureInitialized();

    try {
      const { OCRModule } = NativeModules;
      return await OCRModule.extractTextFromCamera();
    } catch (error) {
      this.emit('ocrError', error);
      throw error;
    }
  }

  // Biometric Authentication

  /**
   * Check if biometric authentication is available
   */
  public async isBiometricAvailable(): Promise<boolean> {
    try {
      const { BiometricAuthModule } = NativeModules;
      return await BiometricAuthModule.isAvailable();
    } catch (error) {
      return false;
    }
  }

  /**
   * Get available biometric type
   */
  public async getBiometricType(): Promise<BiometricType> {
    try {
      const { BiometricAuthModule } = NativeModules;
      return await BiometricAuthModule.getBiometricType();
    } catch (error) {
      return BiometricType.NONE;
    }
  }

  /**
   * Authenticate with biometrics
   */
  public async authenticateWithBiometrics(reason: string): Promise<BiometricAuthResult> {
    try {
      const { BiometricAuthModule } = NativeModules;
      const result = await BiometricAuthModule.authenticate(reason);

      this.trackEvent({
        eventType: 'biometric_auth_attempt',
        timestamp: new Date(),
        properties: {
          success: result.success.toString(),
          biometricType: result.biometricType || 'unknown'
        }
      });

      return result;
    } catch (error) {
      throw new Error(`Biometric authentication failed: ${error}`);
    }
  }

  // Offline & Sync

  /**
   * Enable or disable offline mode
   */
  public enableOfflineMode(enabled: boolean): void {
    this.configuration.enableOfflineMode = enabled;
    this.offlineSync.setEnabled(enabled);
    this.emit('offlineModeChanged', enabled);
  }

  /**
   * Sync offline data
   */
  public async syncOfflineData(): Promise<void> {
    if (!this.configuration.enableOfflineMode) {
      throw new Error('Offline mode is not enabled');
    }

    try {
      await this.offlineSync.sync();
      this.emit('syncCompleted');
    } catch (error) {
      this.emit('syncError', error);
      throw error;
    }
  }

  /**
   * Get offline sync status
   */
  public async getSyncStatus(): Promise<OfflineSyncStatus> {
    return await this.offlineSync.getStatus();
  }

  // Analytics & Tracking

  /**
   * Track analytics event
   */
  public trackEvent(event: AnalyticsEvent): void {
    if (!this.configuration.enableAnalytics) {
      return;
    }

    try {
      this.offlineSync.queueAnalyticsEvent(event);
      this.emit('eventTracked', event);
    } catch (error) {
      console.warn('Failed to track event:', error);
    }
  }

  /**
   * Track analysis result
   */
  public trackAnalysis(result: ArbitrationAnalysisResult): void {
    this.trackEvent({
      eventType: 'analysis_result_tracked',
      timestamp: new Date(),
      properties: {
        analysisId: result.id,
        hasArbitration: result.hasArbitration.toString(),
        confidence: result.confidence.toString(),
        riskLevel: result.riskLevel,
        keywordCount: result.keywordMatches.length.toString()
      }
    });
  }

  // User Preferences

  /**
   * Save user preferences
   */
  public async saveUserPreferences(preferences: UserPreferences): Promise<void> {
    try {
      await this.offlineSync.saveUserPreferences(preferences);
      this.emit('preferencesUpdated', preferences);
    } catch (error) {
      throw new Error(`Failed to save preferences: ${error}`);
    }
  }

  /**
   * Get user preferences
   */
  public async getUserPreferences(): Promise<UserPreferences | null> {
    try {
      return await this.offlineSync.getUserPreferences();
    } catch (error) {
      return null;
    }
  }

  // Network & Connectivity

  /**
   * Get network information
   */
  public async getNetworkInfo(): Promise<NetworkInfo> {
    try {
      const NetInfo = require('@react-native-community/netinfo');
      const state = await NetInfo.fetch();
      
      return {
        isConnected: state.isConnected || false,
        type: state.type || 'unknown',
        isInternetReachable: state.isInternetReachable || false
      };
    } catch (error) {
      return {
        isConnected: false,
        type: 'unknown',
        isInternetReachable: false
      };
    }
  }

  // Utility Methods

  /**
   * Clear all cached data
   */
  public async clearCache(): Promise<void> {
    try {
      await this.offlineSync.clearCache();
      this.emit('cacheCleared');
    } catch (error) {
      throw new Error(`Failed to clear cache: ${error}`);
    }
  }

  /**
   * Get storage statistics
   */
  public async getStorageStatistics(): Promise<any> {
    try {
      return await this.offlineSync.getStorageStatistics();
    } catch (error) {
      throw new Error(`Failed to get storage statistics: ${error}`);
    }
  }

  /**
   * Export analysis data
   */
  public async exportAnalysisData(format: 'json' | 'csv' = 'json'): Promise<string> {
    try {
      return await this.offlineSync.exportData(format);
    } catch (error) {
      throw new Error(`Failed to export data: ${error}`);
    }
  }

  // Private Methods

  private ensureInitialized(): void {
    if (!this.isInitialized) {
      throw new Error('SDK not initialized. Call initialize() first.');
    }
  }

  private setupEventListeners(): void {
    // Listen to native events
    this.nativeEventEmitter.addListener('AnalysisProgress', (progress: number) => {
      this.emit('analysisProgress', progress);
    });

    this.nativeEventEmitter.addListener('NetworkStateChanged', (networkInfo: NetworkInfo) => {
      this.emit('networkStateChanged', networkInfo);
    });

    this.nativeEventEmitter.addListener('SyncStatusChanged', (status: OfflineSyncStatus) => {
      this.emit('syncStatusChanged', status);
    });
  }

  /**
   * Cleanup resources
   */
  public destroy(): void {
    this.removeAllListeners();
    this.nativeEventEmitter.removeAllListeners('AnalysisProgress');
    this.nativeEventEmitter.removeAllListeners('NetworkStateChanged');
    this.nativeEventEmitter.removeAllListeners('SyncStatusChanged');
    this.offlineSync.destroy();
    ArbitrationSDK.instance = null;
  }
}

// Export default instance
export default ArbitrationSDK.getInstance();