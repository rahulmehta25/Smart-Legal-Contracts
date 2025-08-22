/**
 * TypeScript definitions for React Native Arbitration SDK
 */

export interface ArbitrationAnalysisResult {
  id: string;
  hasArbitration: boolean;
  confidence: number;
  keywordMatches: KeywordMatch[];
  exclusionMatches: KeywordMatch[];
  riskLevel: RiskLevel;
  recommendations: string[];
  metadata: AnalysisMetadata;
}

export interface KeywordMatch {
  keyword: string;
  range: string;
  context: string;
  confidence: number;
}

export enum RiskLevel {
  MINIMAL = 'minimal',
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high'
}

export interface AnalysisMetadata {
  textLength: number;
  processingTime: number;
  sdkVersion: string;
  modelVersion: string;
  analysisDate: Date;
}

export interface SDKConfiguration {
  apiBaseUrl?: string;
  apiKey?: string;
  confidenceThreshold?: number;
  enableOfflineMode?: boolean;
  enableAnalytics?: boolean;
  cacheSize?: number;
  networkTimeout?: number;
}

export interface AnalysisRequest {
  text: string;
  requestId?: string;
  options?: AnalysisOptions;
}

export interface AnalysisOptions {
  includeConfidenceScore?: boolean;
  includeKeywordMatches?: boolean;
  includeRecommendations?: boolean;
  language?: string;
  strictMode?: boolean;
}

export interface BatchAnalysisResult {
  batchId: string;
  results: ArbitrationAnalysisResult[];
  summary: BatchSummary;
  processingTime: number;
}

export interface BatchSummary {
  totalDocuments: number;
  documentsWithArbitration: number;
  averageConfidence: number;
  highRiskDocuments: number;
  processingErrors: number;
}

export enum DocumentType {
  PDF = 'pdf',
  WORD = 'word',
  TEXT = 'text',
  RTF = 'rtf',
  HTML = 'html',
  IMAGE = 'image'
}

export interface DocumentUploadResult {
  success: boolean;
  result?: ArbitrationAnalysisResult;
  error?: string;
}

export interface OCRResult {
  text: string;
  confidence: number;
  blocks: TextBlock[];
}

export interface TextBlock {
  text: string;
  boundingBox: BoundingBox;
  confidence: number;
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface BiometricAuthResult {
  success: boolean;
  error?: string;
  biometricType?: BiometricType;
}

export enum BiometricType {
  NONE = 'None',
  TOUCH_ID = 'TouchID',
  FACE_ID = 'FaceID',
  FINGERPRINT = 'Fingerprint',
  FACE = 'Face',
  IRIS = 'Iris'
}

export interface NotificationConfig {
  enablePushNotifications?: boolean;
  enableAnalysisComplete?: boolean;
  enableHighRiskAlerts?: boolean;
  enableWeeklyReports?: boolean;
}

export interface UserPreferences {
  language?: string;
  theme?: AppTheme;
  notifications?: NotificationConfig;
  autoSave?: boolean;
  biometricAuth?: boolean;
}

export enum AppTheme {
  LIGHT = 'light',
  DARK = 'dark',
  SYSTEM = 'system'
}

export interface AnalyticsEvent {
  eventType: AnalyticsEventType;
  timestamp: Date;
  properties?: Record<string, string>;
}

export enum AnalyticsEventType {
  ANALYSIS_STARTED = 'analysis_started',
  ANALYSIS_COMPLETED = 'analysis_completed',
  ANALYSIS_FAILED = 'analysis_failed',
  DOCUMENT_SCANNED = 'document_scanned',
  OFFLINE_MODE_ENABLED = 'offline_mode_enabled',
  CACHE_HIT = 'cache_hit',
  API_CALL_MADE = 'api_call_made'
}

export interface OfflineSyncStatus {
  isOnline: boolean;
  pendingUploads: number;
  lastSyncTime?: Date;
  syncInProgress: boolean;
}

export interface CacheInfo {
  totalItems: number;
  sizeInBytes: number;
  hitRate: number;
  lastCleanup: Date;
}

export interface StorageStatistics {
  totalAnalyses: number;
  analysesWithArbitration: number;
  totalEvents: number;
  unsyncedEvents: number;
  cacheHitRate: number;
  databaseSizeBytes: number;
}

export interface APIError {
  code: string;
  message: string;
  details?: any;
}

export interface NetworkInfo {
  isConnected: boolean;
  type: string;
  isInternetReachable: boolean;
}

// Hook return types
export interface UseArbitrationDetectorResult {
  analyzeText: (text: string) => Promise<ArbitrationAnalysisResult>;
  analyzeDocument: (uri: string, type: DocumentType) => Promise<ArbitrationAnalysisResult>;
  analyzeImage: (uri: string) => Promise<ArbitrationAnalysisResult>;
  batchAnalyze: (requests: AnalysisRequest[]) => Promise<BatchAnalysisResult>;
  isAnalyzing: boolean;
  progress: number;
  lastResult: ArbitrationAnalysisResult | null;
  error: string | null;
  cancelAnalysis: () => void;
}

export interface UseOfflineModeResult {
  isOffline: boolean;
  syncStatus: OfflineSyncStatus;
  sync: () => Promise<void>;
  enableOfflineMode: (enabled: boolean) => void;
}

export interface UseBiometricAuthResult {
  isAvailable: boolean;
  biometricType: BiometricType;
  authenticate: (reason: string) => Promise<BiometricAuthResult>;
  isEnrolled: boolean;
}

export interface UseAnalyticsResult {
  trackEvent: (event: AnalyticsEvent) => void;
  trackAnalysis: (result: ArbitrationAnalysisResult) => void;
  getStatistics: () => Promise<StorageStatistics>;
}

// Component props
export interface DocumentAnalyzerProps {
  onAnalysisComplete?: (result: ArbitrationAnalysisResult) => void;
  onAnalysisError?: (error: string) => void;
  configuration?: SDKConfiguration;
  theme?: AppTheme;
  showProgress?: boolean;
  allowFileUpload?: boolean;
  allowImageScanning?: boolean;
  maxTextLength?: number;
}

export interface AnalysisResultViewProps {
  result: ArbitrationAnalysisResult;
  onClose?: () => void;
  onShare?: (result: ArbitrationAnalysisResult) => void;
  showMetadata?: boolean;
  showRecommendations?: boolean;
}

export interface CameraViewProps {
  onOCRComplete?: (result: OCRResult) => void;
  onAnalysisComplete?: (result: ArbitrationAnalysisResult) => void;
  onError?: (error: string) => void;
  showOverlay?: boolean;
  captureButtonPosition?: 'bottom' | 'right';
}

export interface HistoryViewProps {
  onItemSelect?: (result: ArbitrationAnalysisResult) => void;
  filter?: ResultFilter;
  sortBy?: SortOption;
  limit?: number;
}

export interface ResultFilter {
  hasArbitration?: boolean;
  minConfidence?: number;
  maxConfidence?: number;
  riskLevel?: RiskLevel;
  dateFrom?: Date;
  dateTo?: Date;
}

export enum SortOption {
  DATE_ASC = 'date_asc',
  DATE_DESC = 'date_desc',
  CONFIDENCE_ASC = 'confidence_asc',
  CONFIDENCE_DESC = 'confidence_desc'
}

// Native module interfaces
export interface ArbitrationDetectorModule {
  analyzeText(text: string, options?: AnalysisOptions): Promise<ArbitrationAnalysisResult>;
  analyzeDocument(uri: string, type: DocumentType): Promise<ArbitrationAnalysisResult>;
  batchAnalyze(requests: AnalysisRequest[]): Promise<BatchAnalysisResult>;
  cancelAnalysis(): void;
  configure(config: SDKConfiguration): void;
}

export interface OCRModuleInterface {
  extractTextFromImage(uri: string): Promise<OCRResult>;
  extractTextFromCamera(): Promise<OCRResult>;
  isAvailable(): Promise<boolean>;
}

export interface BiometricAuthModule {
  isAvailable(): Promise<boolean>;
  getBiometricType(): Promise<BiometricType>;
  authenticate(reason: string): Promise<BiometricAuthResult>;
  isEnrolled(): Promise<boolean>;
}

export interface OfflineSyncModule {
  enableOfflineMode(enabled: boolean): void;
  sync(): Promise<void>;
  getSyncStatus(): Promise<OfflineSyncStatus>;
  getPendingItems(): Promise<any[]>;
}

// Event listeners
export type AnalysisProgressListener = (progress: number) => void;
export type NetworkStatusListener = (info: NetworkInfo) => void;
export type SyncStatusListener = (status: OfflineSyncStatus) => void;
export type ErrorListener = (error: APIError) => void;