'use client';

import { useEffect, useState } from 'react';
import { useHealthCheck } from '@/hooks/useApi';
import { cn } from '@/lib/utils';
import { RefreshCw, CheckCircle, XCircle, AlertCircle } from 'lucide-react';

export interface HealthCheckProps {
  id?: string;
  className?: string;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

/**
 * Health Check Component
 * Displays the current status of the backend API
 */
export function HealthCheck({
  id = 'health-check',
  className,
  autoRefresh = true,
  refreshInterval = 30000, // 30 seconds
}: HealthCheckProps) {
  const { data, loading, error, execute } = useHealthCheck();
  const [lastChecked, setLastChecked] = useState<Date | null>(null);

  // Auto-refresh functionality
  useEffect(() => {
    if (autoRefresh && refreshInterval > 0) {
      const interval = setInterval(() => {
        execute();
      }, refreshInterval);

      return () => clearInterval(interval);
    }
  }, [autoRefresh, refreshInterval, execute]);

  // Initial health check
  useEffect(() => {
    const checkHealth = async () => {
      await execute();
      setLastChecked(new Date());
    };
    
    checkHealth();
  }, [execute]);

  const handleRefresh = async () => {
    await execute();
    setLastChecked(new Date());
  };

  const getStatusIcon = () => {
    if (loading) {
      return <RefreshCw id={`${id}-loading-icon`} className="h-4 w-4 animate-spin text-blue-500" />;
    }
    
    if (error) {
      return <XCircle id={`${id}-error-icon`} className="h-4 w-4 text-red-500" />;
    }
    
    if (data?.status === 'healthy') {
      return <CheckCircle id={`${id}-success-icon`} className="h-4 w-4 text-green-500" />;
    }
    
    return <AlertCircle id={`${id}-warning-icon`} className="h-4 w-4 text-yellow-500" />;
  };

  const getStatusText = () => {
    if (loading) return 'Checking...';
    if (error) return 'Error';
    if (data?.status === 'healthy') return 'Healthy';
    return 'Unknown';
  };

  const getStatusColor = () => {
    if (loading) return 'status-connecting';
    if (error) return 'status-error';
    if (data?.status === 'healthy') return 'status-healthy';
    return 'status-error';
  };

  return (
    <div id={id} className={cn('glass-card p-4', className)}>
      <div id={`${id}-header`} className="flex items-center justify-between mb-3">
        <h3 id={`${id}-title`} className="text-lg font-semibold text-gray-900 dark:text-white">
          API Health Status
        </h3>
        <button
          id={`${id}-refresh-btn`}
          onClick={handleRefresh}
          disabled={loading}
          className="p-1.5 rounded-md hover:bg-gray-100 dark:hover:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed"
          aria-label="Refresh health status"
        >
          <RefreshCw className={cn('h-4 w-4', loading && 'animate-spin')} />
        </button>
      </div>

      <div id={`${id}-content`} className="space-y-3">
        {/* Status Indicator */}
        <div id={`${id}-status`} className="flex items-center space-x-2">
          {getStatusIcon()}
          <span id={`${id}-status-badge`} className={cn('status-indicator', getStatusColor())}>
            {getStatusText()}
          </span>
        </div>

        {/* Service Information */}
        {data && (
          <div id={`${id}-service-info`} className="space-y-2 text-sm">
            <div id={`${id}-service-name`} className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Service:</span>
              <span className="font-medium">{data.service}</span>
            </div>
            <div id={`${id}-service-version`} className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">Version:</span>
              <span className="font-medium">{data.version}</span>
            </div>
          </div>
        )}

        {/* Error Information */}
        {error && (
          <div id={`${id}-error-info`} className="mt-3 p-3 bg-red-50 dark:bg-red-900/20 rounded-md">
            <p id={`${id}-error-message`} className="text-sm text-red-800 dark:text-red-400">
              <strong>Error:</strong> {error.detail}
            </p>
            {error.code && (
              <p id={`${id}-error-code`} className="text-xs text-red-600 dark:text-red-500 mt-1">
                Code: {error.code}
              </p>
            )}
          </div>
        )}

        {/* Last Checked */}
        {lastChecked && (
          <div id={`${id}-last-checked`} className="text-xs text-gray-500 dark:text-gray-400 mt-3">
            Last checked: {lastChecked.toLocaleTimeString()}
          </div>
        )}
      </div>
    </div>
  );
}

export default HealthCheck;