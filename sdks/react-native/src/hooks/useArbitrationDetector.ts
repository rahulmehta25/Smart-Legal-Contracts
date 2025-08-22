/**
 * Hook for arbitration detection functionality
 */

import { useState, useCallback, useEffect, useRef } from 'react';
import ArbitrationSDK from '../ArbitrationSDK';
import {
  ArbitrationAnalysisResult,
  AnalysisRequest,
  BatchAnalysisResult,
  DocumentType,
  UseArbitrationDetectorResult
} from '../types';

export function useArbitrationDetector(): UseArbitrationDetectorResult {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [lastResult, setLastResult] = useState<ArbitrationAnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  const analysisRef = useRef<Promise<any> | null>(null);

  useEffect(() => {
    const sdk = ArbitrationSDK.getInstance();

    const handleAnalysisStarted = () => {
      setIsAnalyzing(true);
      setProgress(0);
      setError(null);
    };

    const handleAnalysisProgress = (progressValue: number) => {
      setProgress(progressValue);
    };

    const handleAnalysisCompleted = (result: ArbitrationAnalysisResult) => {
      setIsAnalyzing(false);
      setProgress(1);
      setLastResult(result);
      setError(null);
    };

    const handleAnalysisError = (err: Error) => {
      setIsAnalyzing(false);
      setProgress(0);
      setError(err.message);
    };

    const handleAnalysisCancelled = () => {
      setIsAnalyzing(false);
      setProgress(0);
    };

    // Setup event listeners
    sdk.on('analysisStarted', handleAnalysisStarted);
    sdk.on('analysisProgress', handleAnalysisProgress);
    sdk.on('analysisCompleted', handleAnalysisCompleted);
    sdk.on('analysisError', handleAnalysisError);
    sdk.on('analysisCancelled', handleAnalysisCancelled);

    return () => {
      sdk.off('analysisStarted', handleAnalysisStarted);
      sdk.off('analysisProgress', handleAnalysisProgress);
      sdk.off('analysisCompleted', handleAnalysisCompleted);
      sdk.off('analysisError', handleAnalysisError);
      sdk.off('analysisCancelled', handleAnalysisCancelled);
    };
  }, []);

  const analyzeText = useCallback(async (text: string): Promise<ArbitrationAnalysisResult> => {
    const sdk = ArbitrationSDK.getInstance();
    
    try {
      const promise = sdk.analyzeText(text);
      analysisRef.current = promise;
      const result = await promise;
      analysisRef.current = null;
      return result;
    } catch (err) {
      analysisRef.current = null;
      throw err;
    }
  }, []);

  const analyzeDocument = useCallback(async (
    uri: string,
    type: DocumentType
  ): Promise<ArbitrationAnalysisResult> => {
    const sdk = ArbitrationSDK.getInstance();
    
    try {
      const promise = sdk.analyzeDocument(uri, type);
      analysisRef.current = promise;
      const result = await promise;
      analysisRef.current = null;
      return result;
    } catch (err) {
      analysisRef.current = null;
      throw err;
    }
  }, []);

  const analyzeImage = useCallback(async (uri: string): Promise<ArbitrationAnalysisResult> => {
    const sdk = ArbitrationSDK.getInstance();
    
    try {
      const promise = sdk.analyzeImage(uri);
      analysisRef.current = promise;
      const result = await promise;
      analysisRef.current = null;
      return result;
    } catch (err) {
      analysisRef.current = null;
      throw err;
    }
  }, []);

  const batchAnalyze = useCallback(async (
    requests: AnalysisRequest[]
  ): Promise<BatchAnalysisResult> => {
    const sdk = ArbitrationSDK.getInstance();
    
    try {
      const promise = sdk.batchAnalyze(requests);
      analysisRef.current = promise;
      const result = await promise;
      analysisRef.current = null;
      return result;
    } catch (err) {
      analysisRef.current = null;
      throw err;
    }
  }, []);

  const cancelAnalysis = useCallback(() => {
    const sdk = ArbitrationSDK.getInstance();
    sdk.cancelAnalysis();
    analysisRef.current = null;
  }, []);

  return {
    analyzeText,
    analyzeDocument,
    analyzeImage,
    batchAnalyze,
    isAnalyzing,
    progress,
    lastResult,
    error,
    cancelAnalysis
  };
}