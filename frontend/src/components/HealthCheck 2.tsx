import React from 'react';
import { RefreshCw, CheckCircle, XCircle, Clock } from 'lucide-react';
import { useHealthCheck } from '@/hooks/useApi';
import { Button } from '@/components/ui/Button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { formatDate } from '@/lib/utils';

interface HealthCheckProps {
  id?: string;
}

export function HealthCheck({ id = 'health-check' }: HealthCheckProps) {
  const { data, loading, error, refetch } = useHealthCheck();

  const getStatusIcon = () => {
    if (loading) return <Clock className="h-4 w-4" />;
    if (error) return <XCircle className="h-4 w-4" />;
    if (data?.status === 'healthy') return <CheckCircle className="h-4 w-4" />;
    return <XCircle className="h-4 w-4" />;
  };

  const getStatusBadge = () => {
    if (loading) return <Badge variant="secondary">Checking...</Badge>;
    if (error) return <Badge variant="destructive">Error</Badge>;
    if (data?.status === 'healthy') return <Badge variant="default">Healthy</Badge>;
    return <Badge variant="destructive">Unhealthy</Badge>;
  };

  const getStatusColor = () => {
    if (loading) return 'text-yellow-500';
    if (error) return 'text-red-500';
    if (data?.status === 'healthy') return 'text-green-500';
    return 'text-red-500';
  };

  return (
    <Card id={id} className="w-full">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-lg font-semibold">API Health Check</CardTitle>
        <Button
          variant="outline"
          size="sm"
          onClick={() => refetch()}
          disabled={loading}
          className="gap-2"
        >
          <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className={getStatusColor()}>
              {getStatusIcon()}
            </div>
            <span className="font-medium">Backend Status</span>
          </div>
          {getStatusBadge()}
        </div>

        {error && (
          <div id={`${id}-error`} className="rounded-md bg-red-50 p-3 border border-red-200">
            <p className="text-sm text-red-700">
              <strong>Error:</strong> {error}
            </p>
          </div>
        )}

        {data && (
          <div id={`${id}-details`} className="space-y-2">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-muted-foreground">Status:</span>
                <span className="ml-2 font-medium">{data.status}</span>
              </div>
              <div>
                <span className="text-muted-foreground">Timestamp:</span>
                <span className="ml-2 font-medium">
                  {formatDate(new Date(data.timestamp))}
                </span>
              </div>
              {data.version && (
                <div>
                  <span className="text-muted-foreground">Version:</span>
                  <span className="ml-2 font-medium">{data.version}</span>
                </div>
              )}
            </div>
          </div>
        )}

        <div id={`${id}-info`} className="text-xs text-muted-foreground">
          This component checks the /health endpoint of your FastAPI backend
        </div>
      </CardContent>
    </Card>
  );
}