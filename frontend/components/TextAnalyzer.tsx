'use client';

import { useState } from 'react';
import { useTextAnalysis } from '@/hooks/useApi';
import { cn } from '@/lib/utils';
import { 
  FileText, 
  Send, 
  AlertTriangle,
  CheckCircle,
  XCircle,
  Loader2,
  Copy,
  BarChart3
} from 'lucide-react';

export interface TextAnalyzerProps {
  id?: string;
  className?: string;
}

/**
 * Text Analyzer Component
 * Allows users to input text and analyze it for arbitration clauses
 */
export function TextAnalyzer({
  id = 'text-analyzer',
  className,
}: TextAnalyzerProps) {
  const [text, setText] = useState('');
  const { data: analysis, loading, error, execute, reset } = useTextAnalysis();
  
  const handleAnalyze = async () => {
    if (!text.trim()) return;
    await execute(text);
  };

  const handleReset = () => {
    setText('');
    reset();
  };

  const handleCopyResults = async () => {
    if (analysis) {
      const results = JSON.stringify(analysis, null, 2);
      try {
        await navigator.clipboard.writeText(results);
        // You could add a toast notification here
      } catch (err) {
        console.error('Failed to copy results:', err);
      }
    }
  };

  const sampleTexts = [
    {
      name: 'Terms with Arbitration',
      text: 'Any dispute arising out of or relating to this agreement shall be resolved by binding arbitration administered by the American Arbitration Association.',
    },
    {
      name: 'Terms without Arbitration',
      text: 'This software is provided as-is without warranty. Users may contact support for assistance with technical issues.',
    },
  ];

  return (
    <div id={id} className={cn('glass-card p-6', className)}>
      <div id={`${id}-header`} className="flex items-center justify-between mb-6">
        <div id={`${id}-title-section`}>
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white flex items-center space-x-2">
            <FileText className="h-5 w-5" />
            <span>Text Analyzer</span>
          </h3>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
            Analyze text for arbitration clauses using AI
          </p>
        </div>
        
        {analysis && (
          <button
            id={`${id}-reset-btn`}
            onClick={handleReset}
            className="px-3 py-1.5 text-sm bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300 rounded-md hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
          >
            Reset
          </button>
        )}
      </div>

      {/* Sample Text Buttons */}
      <div id={`${id}-samples`} className="mb-4">
        <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Quick samples:</p>
        <div className="flex flex-wrap gap-2">
          {sampleTexts.map((sample, index) => (
            <button
              key={index}
              id={`${id}-sample-${index}`}
              onClick={() => setText(sample.text)}
              className="px-3 py-1.5 text-xs bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400 rounded-md hover:bg-blue-200 dark:hover:bg-blue-900/30 transition-colors"
            >
              {sample.name}
            </button>
          ))}
        </div>
      </div>

      {/* Text Input */}
      <div id={`${id}-input-section`} className="mb-6">
        <label id={`${id}-input-label`} htmlFor={`${id}-textarea`} className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Enter text to analyze:
        </label>
        <textarea
          id={`${id}-textarea`}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter terms of service, contract text, or any document content to analyze for arbitration clauses..."
          className="w-full h-32 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-800 text-gray-900 dark:text-white resize-none"
          disabled={loading}
        />
        <div id={`${id}-input-info`} className="flex justify-between items-center mt-2 text-xs text-gray-500 dark:text-gray-400">
          <span>{text.length} characters</span>
          <span>Max recommended: 5000 characters</span>
        </div>
      </div>

      {/* Analyze Button */}
      <div id={`${id}-action-section`} className="mb-6">
        <button
          id={`${id}-analyze-btn`}
          onClick={handleAnalyze}
          disabled={!text.trim() || loading}
          className="w-full flex items-center justify-center space-x-2 px-4 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors font-medium"
        >
          {loading ? (
            <>
              <Loader2 className="h-5 w-5 animate-spin" />
              <span>Analyzing...</span>
            </>
          ) : (
            <>
              <Send className="h-5 w-5" />
              <span>Analyze Text</span>
            </>
          )}
        </button>
      </div>

      {/* Results Section */}
      {(analysis || error) && (
        <div id={`${id}-results`} className="space-y-4">
          <div id={`${id}-results-header`} className="flex items-center justify-between">
            <h4 className="text-lg font-semibold text-gray-900 dark:text-white">
              Analysis Results
            </h4>
            {analysis && (
              <button
                id={`${id}-copy-btn`}
                onClick={handleCopyResults}
                className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                aria-label="Copy results"
              >
                <Copy className="h-4 w-4" />
              </button>
            )}
          </div>

          {error && (
            <div id={`${id}-error-result`} className="p-4 bg-red-50 dark:bg-red-900/20 rounded-md border border-red-200 dark:border-red-800">
              <div className="flex items-start space-x-2">
                <XCircle className="h-5 w-5 text-red-500 flex-shrink-0 mt-0.5" />
                <div>
                  <h5 className="font-medium text-red-800 dark:text-red-400">Analysis Failed</h5>
                  <p className="text-sm text-red-700 dark:text-red-300 mt-1">
                    {error.detail}
                  </p>
                </div>
              </div>
            </div>
          )}

          {analysis && (
            <div id={`${id}-success-result`} className="space-y-4">
              {/* Main Result */}
              <div id={`${id}-main-result`} className={cn(
                'p-4 rounded-md border',
                analysis.has_arbitration
                  ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800'
                  : 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800'
              )}>
                <div className="flex items-start space-x-3">
                  {analysis.has_arbitration ? (
                    <AlertTriangle className="h-6 w-6 text-red-500 flex-shrink-0" />
                  ) : (
                    <CheckCircle className="h-6 w-6 text-green-500 flex-shrink-0" />
                  )}
                  <div className="flex-1">
                    <h5 className={cn(
                      'font-semibold text-lg',
                      analysis.has_arbitration 
                        ? 'text-red-800 dark:text-red-400' 
                        : 'text-green-800 dark:text-green-400'
                    )}>
                      {analysis.has_arbitration ? 'Arbitration Clause Detected' : 'No Arbitration Clause Found'}
                    </h5>
                    <div className="flex items-center space-x-4 mt-2 text-sm">
                      <div className="flex items-center space-x-1">
                        <BarChart3 className="h-4 w-4" />
                        <span className="font-medium">Confidence:</span>
                        <span>{(analysis.confidence * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Patterns Found */}
              {analysis.patterns_found && analysis.patterns_found.length > 0 && (
                <div id={`${id}-patterns`} className="p-4 bg-gray-50 dark:bg-gray-800 rounded-md">
                  <h6 className="font-medium text-gray-900 dark:text-white mb-3">
                    Detected Patterns ({analysis.patterns_found.length})
                  </h6>
                  <div className="space-y-2">
                    {analysis.patterns_found.map((pattern, index) => (
                      <div
                        key={index}
                        id={`${id}-pattern-${index}`}
                        className="p-3 bg-white dark:bg-gray-700 rounded border border-gray-200 dark:border-gray-600"
                      >
                        <div className="flex justify-between items-start mb-2">
                          <span className="text-sm font-medium text-gray-900 dark:text-white">
                            {pattern.pattern_type}
                          </span>
                          <span className="text-xs bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400 px-2 py-1 rounded">
                            {(pattern.confidence * 100).toFixed(1)}% confidence
                          </span>
                        </div>
                        <p className="text-sm text-gray-700 dark:text-gray-300 italic">
                          "{pattern.pattern_text}"
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Analysis Metadata */}
              {analysis.analysis_metadata && (
                <div id={`${id}-metadata`} className="p-4 bg-gray-50 dark:bg-gray-800 rounded-md">
                  <h6 className="font-medium text-gray-900 dark:text-white mb-3">Analysis Details</h6>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <span className="text-gray-600 dark:text-gray-400">Document Length:</span>
                      <span className="ml-2 font-medium">{analysis.analysis_metadata.document_length} chars</span>
                    </div>
                    <div>
                      <span className="text-gray-600 dark:text-gray-400">Processing Time:</span>
                      <span className="ml-2 font-medium">{analysis.analysis_metadata.processing_time}ms</span>
                    </div>
                    <div>
                      <span className="text-gray-600 dark:text-gray-400">Model Version:</span>
                      <span className="ml-2 font-medium">{analysis.analysis_metadata.model_version}</span>
                    </div>
                    <div>
                      <span className="text-gray-600 dark:text-gray-400">Language:</span>
                      <span className="ml-2 font-medium">{analysis.analysis_metadata.language}</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default TextAnalyzer;