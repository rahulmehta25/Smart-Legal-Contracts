import { useState } from 'react';
import { Send, Wifi, WifiOff, Trash2 } from 'lucide-react';
import { useWebSocket } from '@/hooks/useWebSocket';
import { Button } from '@/components/ui/Button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { formatDate } from '@/lib/utils';
import type { WebSocketMessage } from '@/types/api';

interface WebSocketTestProps {
  id?: string;
}

export function WebSocketTest({ id = 'websocket-test' }: WebSocketTestProps) {
  const [messageInput, setMessageInput] = useState('');
  
  const {
    connectionStatus,
    messages,
    sendMessage,
    connect,
    disconnect,
    clearMessages,
  } = useWebSocket({
    onMessage: (message: WebSocketMessage) => {
      console.log('Received message:', message);
    },
    onConnect: () => {
      console.log('WebSocket connected');
    },
    onDisconnect: () => {
      console.log('WebSocket disconnected');
    },
    onError: (error) => {
      console.error('WebSocket error:', error);
    },
  });

  const handleSendMessage = () => {
    if (messageInput.trim() && connectionStatus.connected) {
      const message: WebSocketMessage = {
        type: 'test_message',
        data: messageInput.trim(),
        timestamp: new Date().toISOString(),
      };
      
      const sent = sendMessage(message);
      if (sent) {
        setMessageInput('');
      }
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const getConnectionBadge = () => {
    if (connectionStatus.connected) {
      return <Badge variant="default" className="gap-1">
        <Wifi className="h-3 w-3" />
        Connected
      </Badge>;
    }
    return <Badge variant="destructive" className="gap-1">
      <WifiOff className="h-3 w-3" />
      Disconnected
    </Badge>;
  };

  return (
    <Card id={id} className="w-full">
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-lg font-semibold">WebSocket Connection</CardTitle>
        <div className="flex items-center gap-2">
          {getConnectionBadge()}
          <Button
            variant="outline"
            size="sm"
            onClick={connectionStatus.connected ? disconnect : connect}
          >
            {connectionStatus.connected ? 'Disconnect' : 'Connect'}
          </Button>
        </div>
      </CardHeader>
      
      <CardContent className="space-y-4">
        {/* Connection Status Details */}
        <div id={`${id}-status`} className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-muted-foreground">Status:</span>
            <span className="ml-2 font-medium">
              {connectionStatus.connected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
          {connectionStatus.reconnectAttempts && connectionStatus.reconnectAttempts > 0 && (
            <div>
              <span className="text-muted-foreground">Reconnect Attempts:</span>
              <span className="ml-2 font-medium">{connectionStatus.reconnectAttempts}</span>
            </div>
          )}
          {connectionStatus.lastConnected && (
            <div>
              <span className="text-muted-foreground">Last Connected:</span>
              <span className="ml-2 font-medium">
                {formatDate(connectionStatus.lastConnected)}
              </span>
            </div>
          )}
          {connectionStatus.lastDisconnected && (
            <div>
              <span className="text-muted-foreground">Last Disconnected:</span>
              <span className="ml-2 font-medium">
                {formatDate(connectionStatus.lastDisconnected)}
              </span>
            </div>
          )}
        </div>

        {/* Message Input */}
        <div id={`${id}-input`} className="space-y-2">
          <label htmlFor={`${id}-message-input`} className="text-sm font-medium">
            Send Test Message
          </label>
          <div className="flex gap-2">
            <input
              id={`${id}-message-input`}
              type="text"
              value={messageInput}
              onChange={(e) => setMessageInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Enter message to send..."
              className="flex-1 px-3 py-2 border border-input rounded-md bg-background text-sm ring-offset-background focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
              disabled={!connectionStatus.connected}
            />
            <Button
              onClick={handleSendMessage}
              disabled={!connectionStatus.connected || !messageInput.trim()}
              className="gap-2"
            >
              <Send className="h-4 w-4" />
              Send
            </Button>
          </div>
        </div>

        {/* Messages */}
        <div id={`${id}-messages`} className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium">
              Messages ({messages.length})
            </span>
            {messages.length > 0 && (
              <Button
                variant="outline"
                size="sm"
                onClick={clearMessages}
                className="gap-2"
              >
                <Trash2 className="h-4 w-4" />
                Clear
              </Button>
            )}
          </div>
          
          <div className="max-h-64 overflow-y-auto space-y-2 border rounded-md p-3 bg-muted/50">
            {messages.length === 0 ? (
              <p className="text-sm text-muted-foreground text-center py-4">
                No messages yet. Connect and send a test message!
              </p>
            ) : (
              messages.map((message, index) => (
                <div
                  key={index}
                  className="p-2 rounded border bg-background text-sm"
                >
                  <div className="flex items-center justify-between mb-1">
                    <Badge variant="outline" className="text-xs">
                      {message.type}
                    </Badge>
                    {message.timestamp && (
                      <span className="text-xs text-muted-foreground">
                        {formatDate(new Date(message.timestamp))}
                      </span>
                    )}
                  </div>
                  <div className="font-mono text-sm">
                    {typeof message.data === 'string' 
                      ? message.data 
                      : JSON.stringify(message.data, null, 2)}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="text-xs text-muted-foreground">
          This component connects to the /ws endpoint of your FastAPI backend
        </div>
      </CardContent>
    </Card>
  );
}