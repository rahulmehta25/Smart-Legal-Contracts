// API Response Types
export interface ApiResponse<T = any> {
  data?: T;
  message?: string;
  status?: string;
  error?: string;
}

// Health Check Types
export interface HealthCheckResponse {
  status: string;
  service: string;
  version: string;
}

// WebSocket Message Types
export interface WebSocketMessage {
  type: string;
  message?: string;
  data?: any;
  server_time?: string;
  status?: string;
  original_message?: any;
}

// WebSocket Stats Types
export interface WebSocketStats {
  active_connections: number;
  server_status: string;
  features: string[];
}

// Document Analysis Types
export interface DocumentAnalysis {
  id: string;
  document_id: string;
  text: string;
  has_arbitration: boolean;
  confidence: number;
  patterns_found: ArbitrationPattern[];
  analysis_metadata: AnalysisMetadata;
  created_at: string;
}

export interface ArbitrationPattern {
  pattern_type: string;
  pattern_text: string;
  confidence: number;
  position: {
    start: number;
    end: number;
  };
}

export interface AnalysisMetadata {
  document_length: number;
  processing_time: number;
  model_version: string;
  language: string;
}

// Upload Types
export interface DocumentUpload {
  id: string;
  filename: string;
  content_type: string;
  size: number;
  status: 'pending' | 'processing' | 'completed' | 'error';
  created_at: string;
  analysis?: DocumentAnalysis;
}

// API Error Types
export interface ApiError {
  detail: string;
  code?: string;
  field?: string;
  status?: number;
}

// Connection Status Types
export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

// API Endpoints Configuration
export interface ApiEndpoints {
  health: string;
  documents: string;
  analysis: string;
  upload: string;
  websocket: string;
  websocketStats: string;
}