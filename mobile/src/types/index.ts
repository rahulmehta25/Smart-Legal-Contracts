export interface Document {
  id: string;
  name: string;
  content: string;
  extractedText: string;
  createdAt: Date;
  updatedAt: Date;
  imagePath?: string;
  size: number;
  type: string;
  analysisStatus: AnalysisStatus;
  syncStatus: SyncStatus;
}

export interface ArbitrationAnalysis {
  id: string;
  documentId: string;
  hasArbitrationClause: boolean;
  confidence: number;
  detectedClauses: ArbitrationClause[];
  riskLevel: RiskLevel;
  recommendations: string[];
  createdAt: Date;
  processingTime: number;
}

export interface ArbitrationClause {
  id: string;
  text: string;
  type: ClauseType;
  startPosition: number;
  endPosition: number;
  confidence: number;
  severity: SeverityLevel;
  explanation: string;
}

export enum AnalysisStatus {
  PENDING = 'pending',
  PROCESSING = 'processing',
  COMPLETED = 'completed',
  FAILED = 'failed'
}

export enum SyncStatus {
  SYNCED = 'synced',
  PENDING = 'pending',
  FAILED = 'failed',
  OFFLINE = 'offline'
}

export enum ClauseType {
  MANDATORY_ARBITRATION = 'mandatory_arbitration',
  VOLUNTARY_ARBITRATION = 'voluntary_arbitration',
  CLASS_ACTION_WAIVER = 'class_action_waiver',
  JURISDICTION_CLAUSE = 'jurisdiction_clause',
  DISPUTE_RESOLUTION = 'dispute_resolution'
}

export enum RiskLevel {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical'
}

export enum SeverityLevel {
  INFO = 'info',
  WARNING = 'warning',
  CRITICAL = 'critical'
}

export interface User {
  id: string;
  email: string;
  name: string;
  avatar?: string;
  preferences: UserPreferences;
  subscription: SubscriptionType;
}

export interface UserPreferences {
  darkMode: boolean;
  biometricAuth: boolean;
  notifications: NotificationSettings;
  language: string;
  autoSync: boolean;
  offlineMode: boolean;
}

export interface NotificationSettings {
  analysisComplete: boolean;
  criticalClauses: boolean;
  teamUpdates: boolean;
  weeklyReports: boolean;
}

export enum SubscriptionType {
  FREE = 'free',
  PREMIUM = 'premium',
  ENTERPRISE = 'enterprise'
}

export interface CameraPermissions {
  camera: boolean;
  storage: boolean;
  microphone: boolean;
}

export interface ScanResult {
  text: string;
  confidence: number;
  boundingBoxes: BoundingBox[];
  imageUri: string;
  processingTime: number;
}

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
  text: string;
  confidence: number;
}

export interface ApiResponse<T> {
  data: T;
  success: boolean;
  message?: string;
  errors?: string[];
}

export interface SyncQueueItem {
  id: string;
  type: 'document' | 'analysis' | 'user_preferences';
  action: 'create' | 'update' | 'delete';
  data: any;
  timestamp: Date;
  retryCount: number;
  maxRetries: number;
}

export interface Theme {
  colors: {
    primary: string;
    secondary: string;
    background: string;
    surface: string;
    text: string;
    textSecondary: string;
    error: string;
    warning: string;
    success: string;
    info: string;
    border: string;
    shadow: string;
  };
  spacing: {
    xs: number;
    sm: number;
    md: number;
    lg: number;
    xl: number;
  };
  typography: {
    fontFamily: string;
    fontSize: {
      xs: number;
      sm: number;
      md: number;
      lg: number;
      xl: number;
      xxl: number;
    };
    fontWeight: {
      light: string;
      regular: string;
      medium: string;
      bold: string;
    };
  };
  borderRadius: {
    sm: number;
    md: number;
    lg: number;
    xl: number;
  };
}

export type RootStackParamList = {
  Home: undefined;
  Scanner: undefined;
  Analysis: { documentId: string };
  History: undefined;
  Settings: undefined;
  DocumentDetails: { documentId: string };
  BiometricSetup: undefined;
  OnboardingFlow: undefined;
};

export type BottomTabParamList = {
  HomeTab: undefined;
  ScannerTab: undefined;
  HistoryTab: undefined;
  SettingsTab: undefined;
};