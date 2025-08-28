import axios, { AxiosInstance, AxiosResponse, AxiosError, AxiosRequestConfig } from 'axios';
import env from '@/lib/env';
import type {
  ApiResponse,
  HealthCheckResponse,
  WebSocketStats,
  DocumentAnalysis,
  DocumentUpload,
  ApiError,
} from '@/types/api';

/**
 * API Service Class for backend integration with Vercel deployment support
 */
class ApiService {
  private client: AxiosInstance;
  private retryAttempts = 3;
  private retryDelay = 1000;
  
  constructor() {
    this.client = axios.create({
      baseURL: env.API_URL,
      timeout: env.API_TIMEOUT,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
      },
      // Enable cookies for CORS
      withCredentials: false,
      // Validate status
      validateStatus: (status) => status < 500,
    });
    
    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        // Add auth token if available
        const token = this.getAuthToken();
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        
        // Add CORS headers for cross-origin requests
        if (config.headers) {
          config.headers['Access-Control-Allow-Origin'] = '*';
          config.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS, PATCH';
          config.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With';
        }
        
        // Add request timestamp for debugging
        if (env.ENABLE_DEBUG_MODE) {
          config.headers['X-Request-Time'] = new Date().toISOString();
          config.headers['X-Client-Version'] = env.APP_VERSION;
        }
        
        return config;
      },
      (error) => Promise.reject(error)
    );
    
    // Response interceptor with retry logic
    this.client.interceptors.response.use(
      (response) => {
        // Log successful responses in debug mode
        if (env.ENABLE_DEBUG_MODE) {
          console.log(`API Response [${response.config.method?.toUpperCase()}] ${response.config.url}:`, {
            status: response.status,
            headers: response.headers,
            data: response.data,
          });
        }
        return response;
      },
      async (error: AxiosError) => {
        const config = error.config as AxiosRequestConfig & { _retry?: number };
        
        // Retry logic for network errors
        if (
          error.code === 'NETWORK_ERROR' ||
          error.code === 'ECONNABORTED' ||
          (error.response?.status && error.response.status >= 500)
        ) {
          config._retry = config._retry || 0;
          
          if (config._retry < this.retryAttempts) {
            config._retry += 1;
            
            // Exponential backoff
            const delay = this.retryDelay * Math.pow(2, config._retry - 1);
            await new Promise(resolve => setTimeout(resolve, delay));
            
            console.warn(`Retrying API request (${config._retry}/${this.retryAttempts}):`, config.url);
            return this.client.request(config);
          }
        }
        
        // Format error response
        const apiError: ApiError = {
          detail: error.message,
          code: error.code,
          status: error.response?.status,
        };
        
        if (error.response?.data) {
          const errorData = error.response.data as any;
          apiError.detail = errorData.detail || errorData.message || error.message;
          apiError.status = error.response.status;
        }
        
        // Log errors in debug mode
        if (env.ENABLE_DEBUG_MODE) {
          console.error('API Error:', {
            url: config.url,
            method: config.method,
            status: error.response?.status,
            error: apiError,
          });
        }
        
        return Promise.reject(apiError);
      }
    );
  }
  
  /**
   * Get authentication token from localStorage
   */
  private getAuthToken(): string | null {
    if (typeof window === 'undefined') return null;
    return localStorage.getItem('auth_token');
  }
  
  /**
   * Set authentication token
   */
  public setAuthToken(token: string): void {
    if (typeof window !== 'undefined') {
      localStorage.setItem('auth_token', token);
    }
  }
  
  /**
   * Clear authentication token
   */
  public clearAuthToken(): void {
    if (typeof window !== 'undefined') {
      localStorage.removeItem('auth_token');
    }
  }
  
  // === Health Check ===
  
  /**
   * Check API health status
   */
  async healthCheck(): Promise<HealthCheckResponse> {
    const response = await this.client.get<HealthCheckResponse>('/health');
    return response.data;
  }
  
  // === WebSocket Stats ===
  
  /**
   * Get WebSocket connection statistics
   */
  async getWebSocketStats(): Promise<WebSocketStats> {
    const response = await this.client.get<WebSocketStats>('/api/websocket/stats');
    return response.data;
  }
  
  // === Document Management ===
  
  /**
   * Upload a document for analysis
   */
  async uploadDocument(file: File): Promise<DocumentUpload> {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await this.client.post<DocumentUpload>('/api/v1/documents/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    
    return response.data;
  }
  
  /**
   * Get document by ID
   */
  async getDocument(documentId: string): Promise<DocumentUpload> {
    const response = await this.client.get<DocumentUpload>(`/api/v1/documents/${documentId}`);
    return response.data;
  }
  
  /**
   * List all documents
   */
  async listDocuments(): Promise<DocumentUpload[]> {
    const response = await this.client.get<DocumentUpload[]>('/api/v1/documents');
    return response.data;
  }
  
  /**
   * Delete a document
   */
  async deleteDocument(documentId: string): Promise<void> {
    await this.client.delete(`/api/v1/documents/${documentId}`);
  }
  
  // === Document Analysis ===
  
  /**
   * Analyze document text for arbitration clauses
   */
  async analyzeText(text: string): Promise<DocumentAnalysis> {
    const response = await this.client.post<DocumentAnalysis>('/api/v1/analysis/text', {
      text,
    });
    return response.data;
  }
  
  /**
   * Get analysis by document ID
   */
  async getAnalysis(documentId: string): Promise<DocumentAnalysis> {
    const response = await this.client.get<DocumentAnalysis>(`/api/v1/analysis/${documentId}`);
    return response.data;
  }
  
  /**
   * List all analyses
   */
  async listAnalyses(): Promise<DocumentAnalysis[]> {
    const response = await this.client.get<DocumentAnalysis[]>('/api/v1/analysis');
    return response.data;
  }
  
  // === API Information ===
  
  /**
   * Get API overview and available endpoints
   */
  async getApiOverview(): Promise<any> {
    const response = await this.client.get('/api/v1');
    return response.data;
  }
  
  /**
   * Get root API information
   */
  async getRootInfo(): Promise<any> {
    const response = await this.client.get('/');
    return response.data;
  }
}

// Create singleton instance
const apiService = new ApiService();

export default apiService;

// Export types for convenience
export type { HealthCheckResponse, WebSocketStats, DocumentAnalysis, DocumentUpload, ApiError };