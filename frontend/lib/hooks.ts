import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import apiClient from "@/lib/api-client";
import type {
  Document,
  ArbitrationAnalysis,
  AnalysisRequest,
  QuickAnalysisRequest,
  BatchAnalysisRequest,
  PaginationParams,
} from "@/types/api";

// Query Keys
export const queryKeys = {
  health: ["health"] as const,
  documents: (params?: PaginationParams) => ["documents", params] as const,
  document: (id: number) => ["document", id] as const,
  documentChunks: (id: number) => ["document-chunks", id] as const,
  documentStats: ["document-stats"] as const,
  analyses: (params?: PaginationParams) => ["analyses", params] as const,
  analysis: (id: number) => ["analysis", id] as const,
  documentAnalyses: (documentId: number) => ["document-analyses", documentId] as const,
  latestAnalysis: (documentId: number) => ["latest-analysis", documentId] as const,
  analysisStats: ["analysis-stats"] as const,
  clauseTypes: ["clause-types"] as const,
  batchJobs: ["batch-jobs"] as const,
  batchJob: (id: string) => ["batch-job", id] as const,
};

// Health
export function useHealthCheck() {
  return useQuery({
    queryKey: queryKeys.health,
    queryFn: () => apiClient.healthCheck(),
    refetchInterval: 30000,
    retry: false,
  });
}

// Documents
export function useDocuments(params?: PaginationParams & { processed_only?: boolean }) {
  return useQuery({
    queryKey: queryKeys.documents(params),
    queryFn: () => apiClient.getDocuments(params),
  });
}

export function useDocument(id: number) {
  return useQuery({
    queryKey: queryKeys.document(id),
    queryFn: () => apiClient.getDocument(id),
    enabled: !!id,
  });
}

export function useDocumentChunks(documentId: number) {
  return useQuery({
    queryKey: queryKeys.documentChunks(documentId),
    queryFn: () => apiClient.getDocumentChunks(documentId),
    enabled: !!documentId,
  });
}

export function useDocumentStatistics() {
  return useQuery({
    queryKey: queryKeys.documentStats,
    queryFn: () => apiClient.getDocumentStatistics(),
  });
}

export function useUploadDocument() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ file, onProgress }: { file: File; onProgress?: (progress: number) => void }) =>
      apiClient.uploadDocument(file, onProgress),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      queryClient.invalidateQueries({ queryKey: queryKeys.documentStats });
    },
  });
}

export function useDeleteDocument() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: number) => apiClient.deleteDocument(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      queryClient.invalidateQueries({ queryKey: queryKeys.documentStats });
    },
  });
}

// Analysis
export function useAnalyses(params?: PaginationParams & { has_arbitration_only?: boolean }) {
  return useQuery({
    queryKey: queryKeys.analyses(params),
    queryFn: () => apiClient.getAnalyses(params),
  });
}

export function useAnalysis(id: number) {
  return useQuery({
    queryKey: queryKeys.analysis(id),
    queryFn: () => apiClient.getAnalysis(id),
    enabled: !!id,
  });
}

export function useDocumentAnalyses(documentId: number) {
  return useQuery({
    queryKey: queryKeys.documentAnalyses(documentId),
    queryFn: () => apiClient.getDocumentAnalyses(documentId),
    enabled: !!documentId,
  });
}

export function useLatestAnalysis(documentId: number) {
  return useQuery({
    queryKey: queryKeys.latestAnalysis(documentId),
    queryFn: () => apiClient.getLatestAnalysis(documentId),
    enabled: !!documentId,
    retry: false,
  });
}

export function useAnalysisStatistics() {
  return useQuery({
    queryKey: queryKeys.analysisStats,
    queryFn: () => apiClient.getAnalysisStatistics(),
  });
}

export function useClauseTypesSummary() {
  return useQuery({
    queryKey: queryKeys.clauseTypes,
    queryFn: () => apiClient.getClauseTypesSummary(),
  });
}

export function useAnalyzeDocument() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: AnalysisRequest) => apiClient.analyzeDocument(request),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ["analyses"] });
      queryClient.invalidateQueries({ queryKey: queryKeys.analysisStats });
      queryClient.setQueryData(queryKeys.analysis(data.id), data);
    },
  });
}

export function useQuickAnalyze() {
  return useMutation({
    mutationFn: (request: QuickAnalysisRequest) => apiClient.quickAnalyze(request),
  });
}

export function useDeleteAnalysis() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (id: number) => apiClient.deleteAnalysis(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["analyses"] });
      queryClient.invalidateQueries({ queryKey: queryKeys.analysisStats });
    },
  });
}

// Batch Analysis
export function useBatchJobs() {
  return useQuery({
    queryKey: queryKeys.batchJobs,
    queryFn: () => apiClient.getBatchJobs(),
  });
}

export function useBatchJob(jobId: string) {
  return useQuery({
    queryKey: queryKeys.batchJob(jobId),
    queryFn: () => apiClient.getBatchJob(jobId),
    enabled: !!jobId,
    refetchInterval: (query) => {
      const data = query.state.data;
      if (data?.status === "processing") return 2000;
      return false;
    },
  });
}

export function useCreateBatchAnalysis() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (request: BatchAnalysisRequest) => apiClient.createBatchAnalysis(request),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.batchJobs });
    },
  });
}
