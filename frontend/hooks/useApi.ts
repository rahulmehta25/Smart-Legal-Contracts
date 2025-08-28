'use client';

import { useState, useCallback } from 'react';
import apiService from '@/services/api';
import type { ApiError } from '@/types/api';

export interface UseApiState<T> {
  data: T | null;
  loading: boolean;
  error: ApiError | null;
}

export interface UseApiReturn<T> extends UseApiState<T> {
  execute: (...args: any[]) => Promise<T | null>;
  reset: () => void;
}

/**
 * Custom hook for API calls with loading and error states
 */
export function useApi<T>(
  apiFunction: (...args: any[]) => Promise<T>
): UseApiReturn<T> {
  const [state, setState] = useState<UseApiState<T>>({
    data: null,
    loading: false,
    error: null,
  });

  const execute = useCallback(
    async (...args: any[]): Promise<T | null> => {
      setState({ data: null, loading: true, error: null });

      try {
        const result = await apiFunction(...args);
        setState({ data: result, loading: false, error: null });
        return result;
      } catch (error) {
        const apiError = error as ApiError;
        setState({ data: null, loading: false, error: apiError });
        return null;
      }
    },
    [apiFunction]
  );

  const reset = useCallback(() => {
    setState({ data: null, loading: false, error: null });
  }, []);

  return {
    ...state,
    execute,
    reset,
  };
}

/**
 * Hook for health check API
 */
export function useHealthCheck() {
  return useApi(apiService.healthCheck);
}

/**
 * Hook for WebSocket stats API
 */
export function useWebSocketStats() {
  return useApi(apiService.getWebSocketStats);
}

/**
 * Hook for document upload API
 */
export function useDocumentUpload() {
  return useApi(apiService.uploadDocument);
}

/**
 * Hook for text analysis API
 */
export function useTextAnalysis() {
  return useApi(apiService.analyzeText);
}

/**
 * Hook for getting API overview
 */
export function useApiOverview() {
  return useApi(apiService.getApiOverview);
}

export default useApi;