import { useEffect, useRef, useState, useCallback } from 'react';
import { createWebSocketConnection } from '../services/api';
import config from '../config/env';

export interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: number;
}

export interface UseWebSocketOptions {
  autoConnect?: boolean;
  reconnectAttempts?: number;
  reconnectInterval?: number;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Event) => void;
  onMessage?: (message: WebSocketMessage) => void;
}

export function useWebSocket(endpoint: string, options: UseWebSocketOptions = {}) {
  const {
    autoConnect = true,
    reconnectAttempts = 5,
    reconnectInterval = 3000,
    onConnect,
    onDisconnect,
    onError,
    onMessage,
  } = options;

  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const messageQueueRef = useRef<any[]>([]);

  const connect = useCallback(() => {
    if (!config.enableWebSocket) {
      console.warn('WebSocket is disabled in configuration');
      return;
    }

    if (wsRef.current?.readyState === WebSocket.CONNECTING) {
      return;
    }

    setIsConnecting(true);
    setError(null);

    try {
      const ws = createWebSocketConnection(endpoint);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsConnected(true);
        setIsConnecting(false);
        setError(null);
        reconnectAttemptsRef.current = 0;
        
        // Send any queued messages
        while (messageQueueRef.current.length > 0) {
          const message = messageQueueRef.current.shift();
          ws.send(JSON.stringify(message));
        }
        
        onConnect?.();
        
        if (config.enableDebugLogs) {
          console.log(`WebSocket connected to ${endpoint}`);
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        setIsConnecting(false);
        onDisconnect?.();
        
        if (config.enableDebugLogs) {
          console.log(`WebSocket disconnected from ${endpoint}`);
        }

        // Attempt to reconnect if we haven't exceeded the limit
        if (reconnectAttemptsRef.current < reconnectAttempts) {
          reconnectAttemptsRef.current++;
          reconnectTimeoutRef.current = setTimeout(() => {
            if (config.enableDebugLogs) {
              console.log(`Reconnecting to WebSocket (attempt ${reconnectAttemptsRef.current}/${reconnectAttempts})`);
            }
            connect();
          }, reconnectInterval);
        }
      };

      ws.onerror = (event) => {
        setError('WebSocket connection error');
        setIsConnecting(false);
        onError?.(event);
        
        if (config.enableDebugLogs) {
          console.error('WebSocket error:', event);
        }
      };

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          setLastMessage(message);
          onMessage?.(message);
          
          if (config.enableDebugLogs) {
            console.log('WebSocket message received:', message);
          }
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };
    } catch (error) {
      setError('Failed to create WebSocket connection');
      setIsConnecting(false);
      
      if (config.enableDebugLogs) {
        console.error('WebSocket connection error:', error);
      }
    }
  }, [endpoint, reconnectAttempts, reconnectInterval, onConnect, onDisconnect, onError, onMessage]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setIsConnected(false);
    setIsConnecting(false);
    reconnectAttemptsRef.current = reconnectAttempts; // Prevent auto-reconnect
  }, [reconnectAttempts]);

  const sendMessage = useCallback((message: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
      
      if (config.enableDebugLogs) {
        console.log('WebSocket message sent:', message);
      }
    } else {
      // Queue the message for when connection is established
      messageQueueRef.current.push(message);
      
      if (config.enableDebugLogs) {
        console.log('WebSocket message queued:', message);
      }
    }
  }, []);

  const sendTypedMessage = useCallback((type: string, data: any) => {
    sendMessage({
      type,
      data,
      timestamp: Date.now(),
    });
  }, [sendMessage]);

  // Auto-connect on mount if enabled
  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    return () => {
      disconnect();
    };
  }, [autoConnect, connect, disconnect]);

  return {
    isConnected,
    isConnecting,
    error,
    lastMessage,
    connect,
    disconnect,
    sendMessage,
    sendTypedMessage,
  };
}

// Specialized WebSocket hooks for different features
export function useArbitrationWebSocket() {
  return useWebSocket('/ws/arbitration', {
    onConnect: () => {
      console.log('Connected to arbitration WebSocket');
    },
    onMessage: (message) => {
      console.log('Arbitration update:', message);
    },
  });
}

export function useCollaborationWebSocket(workspaceId: string) {
  return useWebSocket(`/ws/collaboration/${workspaceId}`, {
    onConnect: () => {
      console.log(`Connected to collaboration WebSocket for workspace ${workspaceId}`);
    },
    onMessage: (message) => {
      console.log('Collaboration update:', message);
    },
  });
}

export function useNotificationWebSocket() {
  return useWebSocket('/ws/notifications', {
    onConnect: () => {
      console.log('Connected to notification WebSocket');
    },
    onMessage: (message) => {
      console.log('Notification received:', message);
    },
  });
}