// ============================================
// Core API Types
// ============================================

export interface ApiResponse<T = unknown> {
  data?: T;
  message?: string;
  status?: string;
  error?: string;
}

export interface ApiError {
  detail: string;
  code?: string;
  field?: string;
  status?: number;
}

export interface PaginationParams {
  skip?: number;
  limit?: number;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  skip: number;
  limit: number;
  hasMore: boolean;
}

// ============================================
// Health Check Types
// ============================================

export interface HealthCheckResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  service: string;
  version: string;
  timestamp?: string;
  components?: {
    database: 'up' | 'down';
    vectorStore: 'up' | 'down';
    cache: 'up' | 'down';
  };
}

// ============================================
// Document Types
// ============================================

export type DocumentStatus = 'pending' | 'processing' | 'completed' | 'error';
export type RiskLevel = 'high' | 'medium' | 'low';
export type ClauseType =
  | 'mandatory_arbitration'
  | 'binding_arbitration'
  | 'jury_waiver'
  | 'class_action_waiver'
  | 'escalation_clause'
  | 'mediation_first'
  | 'forum_selection'
  | 'other';

export interface Document {
  id: number;
  filename: string;
  content: string;
  content_type: string;
  file_size?: number;
  page_count?: number;
  processed: boolean;
  created_at: string;
  updated_at?: string;
}

export interface DocumentCreate {
  filename: string;
  content: string;
  content_type: string;
}

export interface DocumentUploadResponse {
  message: string;
  document_id: number;
  filename: string;
  chunks_created: number;
}

export interface DocumentChunk {
  id: number;
  document_id: number;
  content: string;
  chunk_index: number;
  embedding_id?: string;
  created_at: string;
}

export interface DocumentWithAnalysis extends Document {
  latest_analysis?: ArbitrationAnalysis;
}

// ============================================
// Analysis Types
// ============================================

export interface ArbitrationClause {
  id: number;
  analysis_id: number;
  clause_text: string;
  clause_type: ClauseType;
  confidence_score: number;
  risk_level: RiskLevel;
  start_position?: number;
  end_position?: number;
  section_reference?: string;
  impact_summary?: string;
  recommendations?: string[];
  metadata?: Record<string, unknown>;
}

export interface ArbitrationAnalysis {
  id: number;
  document_id: number;
  has_arbitration_clause: boolean;
  confidence_score: number;
  analysis_summary: string;
  analyzed_at: string;
  analysis_version: string;
  processing_time_ms: number;
  clauses: ArbitrationClause[];
  risk_level?: RiskLevel;
}

export interface AnalysisRequest {
  document_id: number;
  options?: {
    include_recommendations?: boolean;
    detailed_analysis?: boolean;
  };
}

export interface QuickAnalysisRequest {
  text: string;
  options?: {
    include_recommendations?: boolean;
  };
}

export interface QuickAnalysisResponse {
  has_arbitration_clause: boolean;
  confidence_score: number;
  clauses_found: number;
  summary: string;
  processing_time_ms: number;
  clauses?: ArbitrationClause[];
}

export interface AnalysisStatistics {
  total_analyses: number;
  analyses_with_arbitration: number;
  average_confidence: number;
  clause_types_distribution: Record<string, number>;
  risk_level_distribution: Record<RiskLevel, number>;
  processing_time_stats: {
    average_ms: number;
    min_ms: number;
    max_ms: number;
  };
}

// ============================================
// Batch Analysis Types
// ============================================

export type BatchStatus = 'pending' | 'processing' | 'completed' | 'partial' | 'failed';

export interface BatchAnalysisJob {
  id: string;
  status: BatchStatus;
  total_documents: number;
  processed_documents: number;
  failed_documents: number;
  created_at: string;
  completed_at?: string;
  results?: BatchAnalysisResult[];
}

export interface BatchAnalysisResult {
  document_id: number;
  filename: string;
  status: 'success' | 'error';
  analysis?: ArbitrationAnalysis;
  error?: string;
}

export interface BatchAnalysisRequest {
  document_ids: number[];
  options?: {
    parallel?: boolean;
    max_concurrent?: number;
  };
}

// ============================================
// Comparison Types
// ============================================

export interface DocumentComparison {
  document_a: DocumentWithAnalysis;
  document_b: DocumentWithAnalysis;
  similarity_score: number;
  clause_comparison: ClauseComparison[];
  summary: string;
}

export interface ClauseComparison {
  clause_type: ClauseType;
  in_document_a: boolean;
  in_document_b: boolean;
  clause_a?: ArbitrationClause;
  clause_b?: ArbitrationClause;
  similarity?: number;
  differences?: string[];
}

// ============================================
// Search Types
// ============================================

export interface SearchResult {
  document_id: number;
  filename: string;
  chunk_content: string;
  similarity_score: number;
  chunk_index: number;
}

export interface SearchRequest {
  query: string;
  limit?: number;
  filters?: {
    has_arbitration?: boolean;
    risk_level?: RiskLevel;
    date_from?: string;
    date_to?: string;
  };
}

// ============================================
// Settings Types
// ============================================

export interface UserSettings {
  api_key?: string;
  notifications: {
    email_enabled: boolean;
    high_risk_alerts: boolean;
    analysis_complete: boolean;
  };
  export_preferences: {
    default_format: 'pdf' | 'json' | 'csv';
    include_highlights: boolean;
    include_recommendations: boolean;
  };
  analysis_defaults: {
    detailed_analysis: boolean;
    include_recommendations: boolean;
    auto_analyze_on_upload: boolean;
  };
}

// ============================================
// WebSocket Types
// ============================================

export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

export interface WebSocketMessage {
  type: 'analysis_progress' | 'analysis_complete' | 'error' | 'ping';
  payload?: unknown;
  timestamp: string;
}

export interface AnalysisProgressMessage {
  document_id: number;
  stage: 'parsing' | 'chunking' | 'embedding' | 'analyzing' | 'complete';
  progress: number;
  message?: string;
}

// ============================================
// Filter Types for History
// ============================================

export interface AnalysisFilters {
  dateFrom?: string;
  dateTo?: string;
  riskLevel?: RiskLevel | 'all';
  hasArbitration?: boolean | 'all';
  documentType?: string;
  searchQuery?: string;
}

export interface SortOptions {
  field: 'analyzed_at' | 'confidence_score' | 'filename' | 'risk_level';
  direction: 'asc' | 'desc';
}
