import axios, { AxiosInstance, AxiosError, AxiosRequestConfig } from "axios";
import env from "@/lib/env";
import type {
  ApiError,
  Document,
  DocumentUploadResponse,
  DocumentChunk,
  ArbitrationAnalysis,
  AnalysisRequest,
  QuickAnalysisRequest,
  QuickAnalysisResponse,
  AnalysisStatistics,
  BatchAnalysisJob,
  BatchAnalysisRequest,
  SearchResult,
  SearchRequest,
  HealthCheckResponse,
  PaginationParams,
} from "@/types/api";

class ApiClient {
  private client: AxiosInstance;
  private maxRetries = 3;
  private retryDelay = 1000;

  constructor() {
    this.client = axios.create({
      baseURL: env.API_URL,
      timeout: env.API_TIMEOUT,
      headers: {
        "Content-Type": "application/json",
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    this.client.interceptors.request.use(
      (config) => {
        const token = this.getAuthToken();
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    this.client.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        const config = error.config as AxiosRequestConfig & { _retry?: number };

        if (this.shouldRetry(error) && config) {
          config._retry = (config._retry || 0) + 1;

          if (config._retry <= this.maxRetries) {
            const delay = this.retryDelay * Math.pow(2, config._retry - 1);
            await new Promise((resolve) => setTimeout(resolve, delay));
            return this.client.request(config);
          }
        }

        return Promise.reject(this.formatError(error));
      }
    );
  }

  private shouldRetry(error: AxiosError): boolean {
    if (!error.response) return true;
    const status = error.response.status;
    return status === 429 || status >= 500;
  }

  private formatError(error: AxiosError): ApiError {
    const response = error.response;

    if (!response) {
      return {
        detail: "Network error. Please check your connection.",
        code: "NETWORK_ERROR",
      };
    }

    const data = response.data as Record<string, unknown>;

    return {
      detail: (data?.detail as string) || (data?.message as string) || error.message,
      code: (data?.code as string) || undefined,
      status: response.status,
    };
  }

  private getAuthToken(): string | null {
    if (typeof window === "undefined") return null;
    return localStorage.getItem("auth_token");
  }

  setAuthToken(token: string): void {
    if (typeof window !== "undefined") {
      localStorage.setItem("auth_token", token);
    }
  }

  clearAuthToken(): void {
    if (typeof window !== "undefined") {
      localStorage.removeItem("auth_token");
    }
  }

  // Health
  async healthCheck(): Promise<HealthCheckResponse> {
    const response = await this.client.get<HealthCheckResponse>("/health");
    return response.data;
  }

  // Documents
  async uploadDocument(file: File, onProgress?: (progress: number) => void): Promise<DocumentUploadResponse> {
    const formData = new FormData();
    formData.append("file", file);

    const response = await this.client.post<DocumentUploadResponse>("/api/v1/documents/upload", formData, {
      headers: { "Content-Type": "multipart/form-data" },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(progress);
        }
      },
    });

    return response.data;
  }

  async getDocuments(params?: PaginationParams & { processed_only?: boolean }): Promise<Document[]> {
    const response = await this.client.get<Document[]>("/api/v1/documents", { params });
    return response.data;
  }

  async getDocument(id: number): Promise<Document> {
    const response = await this.client.get<Document>(`/api/v1/documents/${id}`);
    return response.data;
  }

  async getDocumentChunks(documentId: number): Promise<DocumentChunk[]> {
    const response = await this.client.get<DocumentChunk[]>(`/api/v1/documents/${documentId}/chunks`);
    return response.data;
  }

  async deleteDocument(id: number): Promise<void> {
    await this.client.delete(`/api/v1/documents/${id}`);
  }

  async searchDocuments(request: SearchRequest): Promise<{ query: string; total_results: number; results: SearchResult[] }> {
    const response = await this.client.get("/api/v1/documents/search/", {
      params: { query: request.query, limit: request.limit || 10 },
    });
    return response.data;
  }

  async getDocumentStatistics(): Promise<{
    total_documents: number;
    processed_documents: number;
    total_chunks: number;
  }> {
    const response = await this.client.get("/api/v1/documents/stats/overview");
    return response.data;
  }

  // Analysis
  async analyzeDocument(request: AnalysisRequest): Promise<ArbitrationAnalysis> {
    const response = await this.client.post<ArbitrationAnalysis>("/api/v1/analysis/analyze", request);
    return response.data;
  }

  async quickAnalyze(request: QuickAnalysisRequest): Promise<QuickAnalysisResponse> {
    const response = await this.client.post<QuickAnalysisResponse>("/api/v1/analysis/quick-analyze", request);
    return response.data;
  }

  async getAnalyses(params?: PaginationParams & { has_arbitration_only?: boolean }): Promise<ArbitrationAnalysis[]> {
    const response = await this.client.get<ArbitrationAnalysis[]>("/api/v1/analysis", { params });
    return response.data;
  }

  async getAnalysis(id: number): Promise<ArbitrationAnalysis> {
    const response = await this.client.get<ArbitrationAnalysis>(`/api/v1/analysis/${id}`);
    return response.data;
  }

  async getDocumentAnalyses(documentId: number): Promise<ArbitrationAnalysis[]> {
    const response = await this.client.get<ArbitrationAnalysis[]>(`/api/v1/analysis/document/${documentId}`);
    return response.data;
  }

  async getLatestAnalysis(documentId: number): Promise<ArbitrationAnalysis> {
    const response = await this.client.get<ArbitrationAnalysis>(`/api/v1/analysis/document/${documentId}/latest`);
    return response.data;
  }

  async deleteAnalysis(id: number): Promise<void> {
    await this.client.delete(`/api/v1/analysis/${id}`);
  }

  async getAnalysisStatistics(): Promise<AnalysisStatistics> {
    const response = await this.client.get<AnalysisStatistics>("/api/v1/analysis/stats/overview");
    return response.data;
  }

  async getClauseTypesSummary(): Promise<{
    clause_types: Record<string, number>;
    total_types: number;
    total_clauses: number;
  }> {
    const response = await this.client.get("/api/v1/analysis/stats/clause-types");
    return response.data;
  }

  // Batch Analysis
  async createBatchAnalysis(request: BatchAnalysisRequest): Promise<BatchAnalysisJob> {
    const response = await this.client.post<BatchAnalysisJob>("/api/v1/batch/analyze", request);
    return response.data;
  }

  async getBatchJob(jobId: string): Promise<BatchAnalysisJob> {
    const response = await this.client.get<BatchAnalysisJob>(`/api/v1/batch/${jobId}`);
    return response.data;
  }

  async getBatchJobs(): Promise<BatchAnalysisJob[]> {
    const response = await this.client.get<BatchAnalysisJob[]>("/api/v1/batch");
    return response.data;
  }
}

export const apiClient = new ApiClient();
export default apiClient;
