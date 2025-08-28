'use client';

import { useState, useRef } from 'react';
import { useWebSocket } from '@/hooks/useWebSocket';
import { useWebSocketStats } from '@/hooks/useApi';
import { cn, formatDate } from '@/lib/utils';
import { 
  Send, 
  Wifi, 
  WifiOff, 
  Loader2, 
  MessageCircle,
  Users,
  Activity 
} from 'lucide-react';
import type { WebSocketMessage } from '@/types/api';

export interface WebSocketTestProps {
  id?: string;
  className?: string;
}

/**
 * WebSocket Test Component
 * Tests WebSocket connectivity and allows sending/receiving messages
 */
export function WebSocketTest({
  id = 'websocket-test',
  className,
}: WebSocketTestProps) {
  const [message, setMessage] = useState('');
  const [messageHistory, setMessageHistory] = useState<Array<{
    id: string;
    type: 'sent' | 'received';
    content: WebSocketMessage | string;
    timestamp: Date;
  }>>([]);
  
  const messageInputRef = useRef<HTMLInputElement>(null);
  const { data: stats, loading: statsLoading, execute: fetchStats } = useWebSocketStats();

  const {
    connectionStatus,
    lastMessage,
    sendMessage,
    connect,
    disconnect,
    isConnected,
  } = useWebSocket({
    onMessage: (msg) => {
      setMessageHistory(prev => [
        ...prev,
        {
          id: `${Date.now()}-${Math.random()}`,
          type: 'received',
          content: msg,
          timestamp: new Date(),
        },
      ]);
      // Refresh stats when we receive messages
      fetchStats();
    },
    onConnect: () => {
      setMessageHistory(prev => [
        ...prev,
        {
          id: `${Date.now()}-connect`,
          type: 'received',
          content: 'Connected to WebSocket server',
          timestamp: new Date(),
        },
      ]);
      fetchStats();
    },
    onDisconnect: () => {
      setMessageHistory(prev => [
        ...prev,
        {
          id: `${Date.now()}-disconnect`,
          type: 'received',
          content: 'Disconnected from WebSocket server',
          timestamp: new Date(),
        },
      ]);
    },
  });

  const handleSendMessage = () => {
    if (!message.trim() || !isConnected) return;

    const messageToSend = {
      type: 'test_message',
      content: message,
      timestamp: new Date().toISOString(),
    };

    sendMessage(messageToSend);
    
    setMessageHistory(prev => [
      ...prev,
      {
        id: `${Date.now()}-sent`,
        type: 'sent',
        content: messageToSend,
        timestamp: new Date(),
      },
    ]);
    
    setMessage('');
    messageInputRef.current?.focus();
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const getConnectionIcon = () => {
    switch (connectionStatus) {
      case 'connecting':
        return <Loader2 className="h-4 w-4 animate-spin text-yellow-500" />;
      case 'connected':
        return <Wifi className="h-4 w-4 text-green-500" />;
      case 'error':
        return <WifiOff className="h-4 w-4 text-red-500" />;
      default:
        return <WifiOff className="h-4 w-4 text-gray-400" />;
    }
  };

  const getConnectionStatusText = () => {
    switch (connectionStatus) {
      case 'connecting':
        return 'Connecting...';
      case 'connected':
        return 'Connected';
      case 'error':
        return 'Error';
      default:
        return 'Disconnected';
    }
  };

  const getConnectionStatusColor = () => {
    switch (connectionStatus) {
      case 'connecting':
        return 'status-connecting';
      case 'connected':
        return 'status-healthy';
      case 'error':
        return 'status-error';
      default:
        return 'status-error';
    }
  };

  return (
    <div id={id} className={cn('glass-card p-4', className)}>
      <div id={`${id}-header`} className="flex items-center justify-between mb-4">
        <h3 id={`${id}-title`} className="text-lg font-semibold text-gray-900 dark:text-white">
          WebSocket Test
        </h3>
        <div id={`${id}-actions`} className="flex items-center space-x-2">
          <button
            id={`${id}-toggle-btn`}
            onClick={isConnected ? disconnect : connect}
            className={cn(
              'px-3 py-1.5 rounded-md text-sm font-medium transition-colors',
              isConnected
                ? 'bg-red-100 text-red-800 hover:bg-red-200 dark:bg-red-900/20 dark:text-red-400'
                : 'bg-green-100 text-green-800 hover:bg-green-200 dark:bg-green-900/20 dark:text-green-400'
            )}
          >
            {isConnected ? 'Disconnect' : 'Connect'}
          </button>
        </div>
      </div>

      {/* Connection Status */}
      <div id={`${id}-connection-status`} className="mb-4">
        <div className="flex items-center space-x-2 mb-2">
          {getConnectionIcon()}
          <span className={cn('status-indicator', getConnectionStatusColor())}>
            {getConnectionStatusText()}
          </span>
        </div>
        
        {/* WebSocket Stats */}
        {stats && (
          <div id={`${id}-stats`} className="grid grid-cols-3 gap-4 mt-3 text-sm">
            <div id={`${id}-active-connections`} className="flex items-center space-x-2">
              <Users className="h-4 w-4 text-blue-500" />
              <span className="text-gray-600 dark:text-gray-400">
                {stats.active_connections} connection{stats.active_connections !== 1 ? 's' : ''}
              </span>
            </div>
            <div id={`${id}-server-status`} className="flex items-center space-x-2">
              <Activity className="h-4 w-4 text-green-500" />
              <span className="text-gray-600 dark:text-gray-400">
                {stats.server_status}
              </span>
            </div>
            <div id={`${id}-features-count`} className="flex items-center space-x-2">
              <MessageCircle className="h-4 w-4 text-purple-500" />
              <span className="text-gray-600 dark:text-gray-400">
                {stats.features.length} feature{stats.features.length !== 1 ? 's' : ''}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Message Input */}
      <div id={`${id}-message-input`} className="mb-4">
        <div className="flex space-x-2">
          <input
            ref={messageInputRef}
            id={`${id}-input-field`}
            type="text"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Enter a test message..."
            disabled={!isConnected}
            className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 disabled:cursor-not-allowed dark:bg-gray-800 dark:border-gray-600 dark:text-white"
          />
          <button
            id={`${id}-send-btn`}
            onClick={handleSendMessage}
            disabled={!message.trim() || !isConnected}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
            aria-label="Send message"
          >
            <Send className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Message History */}
      <div id={`${id}-message-history`} className="space-y-2">
        <h4 id={`${id}-history-title`} className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Message History
        </h4>
        <div 
          id={`${id}-history-container`}
          className="max-h-64 overflow-y-auto space-y-2 p-3 bg-gray-50 dark:bg-gray-800 rounded-md"
        >
          {messageHistory.length === 0 ? (
            <p id={`${id}-no-messages`} className="text-sm text-gray-500 dark:text-gray-400 text-center py-4">
              No messages yet. Connect and send a test message.
            </p>
          ) : (
            messageHistory.map((item) => (
              <div
                key={item.id}
                id={`${id}-message-${item.id}`}
                className={cn(
                  'p-2 rounded text-sm',
                  item.type === 'sent'
                    ? 'bg-blue-100 text-blue-800 ml-4 dark:bg-blue-900/20 dark:text-blue-400'
                    : 'bg-green-100 text-green-800 mr-4 dark:bg-green-900/20 dark:text-green-400'
                )}
              >
                <div className="flex justify-between items-start mb-1">
                  <span className="font-medium">
                    {item.type === 'sent' ? 'Sent' : 'Received'}
                  </span>
                  <span className="text-xs opacity-70">
                    {formatDate(item.timestamp)}
                  </span>
                </div>
                <div className="break-words">
                  {typeof item.content === 'string' ? (
                    item.content
                  ) : (
                    <pre className="whitespace-pre-wrap text-xs">
                      {JSON.stringify(item.content, null, 2)}
                    </pre>
                  )}
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

export default WebSocketTest;