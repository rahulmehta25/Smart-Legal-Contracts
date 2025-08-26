import React, { useEffect, useState } from 'react';
import { Button } from './ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Alert, AlertDescription } from './ui/alert';
import { apiService } from '../services/api';
import config from '../config/env';
import { CheckCircle, XCircle, Loader2, Wifi, WifiOff } from 'lucide-react';

interface ConnectionStatus {
  api: 'connected' | 'disconnected' | 'testing';
  websocket: 'connected' | 'disconnected' | 'testing';
}

const ApiTestComponent: React.FC = () => {
  const [status, setStatus] = useState<ConnectionStatus>({
    api: 'disconnected',
    websocket: 'disconnected'
  });
  const [apiResponse, setApiResponse] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const testApiConnection = async () => {
    setStatus(prev => ({ ...prev, api: 'testing' }));
    setError(null);
    
    try {
      // Test the health endpoint
      const response = await apiService.get('/health');
      setApiResponse(response);
      setStatus(prev => ({ ...prev, api: 'connected' }));
    } catch (err: any) {
      setError(err?.response?.data?.detail || err?.message || 'API connection failed');
      setStatus(prev => ({ ...prev, api: 'disconnected' }));
    }
  };

  const testWebSocketConnection = () => {
    if (!config.enableWebSocket) {
      setError('WebSocket is disabled in configuration');
      return;
    }

    setStatus(prev => ({ ...prev, websocket: 'testing' }));
    setError(null);

    try {
      const ws = new WebSocket(`${config.wsUrl}/ws/test`);
      
      ws.onopen = () => {
        setStatus(prev => ({ ...prev, websocket: 'connected' }));
        ws.close();
      };
      
      ws.onerror = () => {
        setStatus(prev => ({ ...prev, websocket: 'disconnected' }));
        setError('WebSocket connection failed');
      };
      
      ws.onclose = () => {
        if (status.websocket === 'testing') {
          setStatus(prev => ({ ...prev, websocket: 'disconnected' }));
        }
      };
    } catch (err: any) {
      setStatus(prev => ({ ...prev, websocket: 'disconnected' }));
      setError(`WebSocket error: ${err.message}`);
    }
  };

  useEffect(() => {
    // Test API connection on component mount
    testApiConnection();
  }, []);

  const getStatusIcon = (connectionStatus: string) => {
    switch (connectionStatus) {
      case 'connected':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'testing':
        return <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />;
      default:
        return <XCircle className="h-4 w-4 text-red-500" />;
    }
  };

  const getStatusColor = (connectionStatus: string) => {
    switch (connectionStatus) {
      case 'connected':
        return 'bg-green-100 text-green-800 border-green-200';
      case 'testing':
        return 'bg-blue-100 text-blue-800 border-blue-200';
      default:
        return 'bg-red-100 text-red-800 border-red-200';
    }
  };

  return (
    <Card className="w-full max-w-2xl mx-auto">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Wifi className="h-5 w-5" />
          Backend Connection Test
        </CardTitle>
        <CardDescription>
          Test the connection to the backend API and WebSocket services
        </CardDescription>
      </CardHeader>
      
      <CardContent className="space-y-6">
        {/* Configuration Info */}
        <div className="space-y-2">
          <h3 className="font-semibold">Configuration</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600">API URL:</span>
              <Badge variant="outline">{config.apiUrl}</Badge>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">WebSocket URL:</span>
              <Badge variant="outline">{config.wsUrl}</Badge>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Environment:</span>
              <Badge variant={config.isDevelopment ? "secondary" : "default"}>
                {config.mode}
              </Badge>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">WebSocket Enabled:</span>
              <Badge variant={config.enableWebSocket ? "default" : "destructive"}>
                {config.enableWebSocket ? 'Yes' : 'No'}
              </Badge>
            </div>
          </div>
        </div>

        {/* Connection Status */}
        <div className="space-y-4">
          <h3 className="font-semibold">Connection Status</h3>
          
          {/* API Status */}
          <div className="flex items-center justify-between p-3 border rounded-lg">
            <div className="flex items-center gap-2">
              {getStatusIcon(status.api)}
              <span className="font-medium">API Connection</span>
            </div>
            <Badge className={getStatusColor(status.api)}>
              {status.api.toUpperCase()}
            </Badge>
          </div>

          {/* WebSocket Status */}
          <div className="flex items-center justify-between p-3 border rounded-lg">
            <div className="flex items-center gap-2">
              {config.enableWebSocket ? (
                getStatusIcon(status.websocket)
              ) : (
                <WifiOff className="h-4 w-4 text-gray-400" />
              )}
              <span className="font-medium">WebSocket Connection</span>
            </div>
            <Badge className={config.enableWebSocket ? getStatusColor(status.websocket) : 'bg-gray-100 text-gray-500'}>
              {config.enableWebSocket ? status.websocket.toUpperCase() : 'DISABLED'}
            </Badge>
          </div>
        </div>

        {/* Test Buttons */}
        <div className="flex gap-2">
          <Button 
            onClick={testApiConnection}
            disabled={status.api === 'testing'}
            variant="outline"
          >
            {status.api === 'testing' && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            Test API
          </Button>
          
          {config.enableWebSocket && (
            <Button 
              onClick={testWebSocketConnection}
              disabled={status.websocket === 'testing'}
              variant="outline"
            >
              {status.websocket === 'testing' && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Test WebSocket
            </Button>
          )}
        </div>

        {/* Error Display */}
        {error && (
          <Alert variant="destructive">
            <XCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* API Response */}
        {apiResponse && (
          <div className="space-y-2">
            <h3 className="font-semibold">API Response</h3>
            <pre className="bg-gray-100 p-3 rounded text-sm overflow-x-auto">
              {JSON.stringify(apiResponse, null, 2)}
            </pre>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default ApiTestComponent;