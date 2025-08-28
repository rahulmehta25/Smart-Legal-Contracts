'use client';

import { useEffect, useState } from 'react';
import { useApiOverview } from '@/hooks/useApi';
import { cn } from '@/lib/utils';
import { 
  RefreshCw, 
  Server, 
  Layers, 
  Code,
  CheckCircle,
  XCircle,
  ExternalLink 
} from 'lucide-react';

export interface ApiOverviewProps {
  id?: string;
  className?: string;
}

/**
 * API Overview Component
 * Displays available API endpoints and features
 */
export function ApiOverview({
  id = 'api-overview',
  className,
}: ApiOverviewProps) {
  const { data, loading, error, execute } = useApiOverview();
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({});

  useEffect(() => {
    execute();
  }, [execute]);

  const toggleSection = (section: string) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section],
    }));
  };

  const handleRefresh = () => {
    execute();
  };

  return (
    <div id={id} className={cn('glass-card p-4', className)}>
      <div id={`${id}-header`} className="flex items-center justify-between mb-4">
        <h3 id={`${id}-title`} className="text-lg font-semibold text-gray-900 dark:text-white flex items-center space-x-2">
          <Server className="h-5 w-5" />
          <span>API Overview</span>
        </h3>
        <button
          id={`${id}-refresh-btn`}
          onClick={handleRefresh}
          disabled={loading}
          className="p-1.5 rounded-md hover:bg-gray-100 dark:hover:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed"
          aria-label="Refresh API overview"
        >
          <RefreshCw className={cn('h-4 w-4', loading && 'animate-spin')} />
        </button>
      </div>

      {loading && (
        <div id={`${id}-loading`} className="flex items-center justify-center py-8">
          <RefreshCw className="h-6 w-6 animate-spin text-blue-500" />
          <span className="ml-2 text-gray-600 dark:text-gray-400">Loading API information...</span>
        </div>
      )}

      {error && (
        <div id={`${id}-error`} className="p-4 bg-red-50 dark:bg-red-900/20 rounded-md">
          <div className="flex items-center space-x-2 mb-2">
            <XCircle className="h-5 w-5 text-red-500" />
            <span className="font-medium text-red-800 dark:text-red-400">
              Failed to load API information
            </span>
          </div>
          <p className="text-sm text-red-700 dark:text-red-300">
            {error.detail}
          </p>
        </div>
      )}

      {data && (
        <div id={`${id}-content`} className="space-y-4">
          {/* API Version */}
          <div id={`${id}-version`} className="flex items-center space-x-2 text-sm">
            <span className="text-gray-600 dark:text-gray-400">Version:</span>
            <span className="font-medium bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400 px-2 py-1 rounded">
              {data.version}
            </span>
          </div>

          {/* Endpoints Section */}
          {data.endpoints && (
            <div id={`${id}-endpoints`} className="space-y-2">
              <button
                id={`${id}-endpoints-toggle`}
                onClick={() => toggleSection('endpoints')}
                className="flex items-center space-x-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white"
              >
                <Code className="h-4 w-4" />
                <span>Available Endpoints ({Object.keys(data.endpoints).length})</span>
                <span className={cn('transition-transform', expandedSections.endpoints && 'rotate-90')}>
                  →
                </span>
              </button>
              
              {expandedSections.endpoints && (
                <div id={`${id}-endpoints-list`} className="ml-6 space-y-2">
                  {Object.entries(data.endpoints).map(([name, endpoint]) => (
                    <div
                      key={name}
                      id={`${id}-endpoint-${name}`}
                      className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-800 rounded text-sm"
                    >
                      <div className="space-y-1">
                        <div className="font-medium capitalize text-gray-900 dark:text-white">
                          {name.replace(/_/g, ' ')}
                        </div>
                        <div className="font-mono text-xs text-gray-600 dark:text-gray-400">
                          {endpoint as string}
                        </div>
                      </div>
                      <ExternalLink className="h-3 w-3 text-gray-400" />
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Features Section */}
          {data.features && (
            <div id={`${id}-features`} className="space-y-2">
              <button
                id={`${id}-features-toggle`}
                onClick={() => toggleSection('features')}
                className="flex items-center space-x-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white"
              >
                <Layers className="h-4 w-4" />
                <span>Features ({data.features.length})</span>
                <span className={cn('transition-transform', expandedSections.features && 'rotate-90')}>
                  →
                </span>
              </button>
              
              {expandedSections.features && (
                <div id={`${id}-features-list`} className="ml-6 space-y-1">
                  {data.features.map((feature: string, index: number) => (
                    <div
                      key={index}
                      id={`${id}-feature-${index}`}
                      className="flex items-start space-x-2 text-sm"
                    >
                      <CheckCircle className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                      <span className="text-gray-700 dark:text-gray-300">{feature}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Quick Test Buttons */}
          <div id={`${id}-test-buttons`} className="pt-4 border-t border-gray-200 dark:border-gray-700">
            <div className="flex flex-wrap gap-2">
              <button
                id={`${id}-test-health`}
                onClick={() => window.open(`${window.location.origin}/api/backend/health`, '_blank')}
                className="px-3 py-1.5 text-xs bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400 rounded-md hover:bg-green-200 dark:hover:bg-green-900/30 transition-colors"
              >
                Test Health
              </button>
              <button
                id={`${id}-test-docs`}
                onClick={() => window.open(`${window.location.origin}/api/backend/docs`, '_blank')}
                className="px-3 py-1.5 text-xs bg-blue-100 text-blue-800 dark:bg-blue-900/20 dark:text-blue-400 rounded-md hover:bg-blue-200 dark:hover:bg-blue-900/30 transition-colors"
              >
                View Docs
              </button>
              <button
                id={`${id}-test-api-v1`}
                onClick={() => window.open(`${window.location.origin}/api/backend/api/v1`, '_blank')}
                className="px-3 py-1.5 text-xs bg-purple-100 text-purple-800 dark:bg-purple-900/20 dark:text-purple-400 rounded-md hover:bg-purple-200 dark:hover:bg-purple-900/30 transition-colors"
              >
                API v1
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default ApiOverview;