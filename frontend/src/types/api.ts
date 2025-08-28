export interface HealthResponse {
  status: string;
  timestamp: string;
  version?: string;
}

export interface ApiError {
  detail: string;
  code?: string;
}

export interface WebSocketMessage {
  type: string;
  data?: any;
  timestamp?: string;
}

export interface ConnectionStatus {
  connected: boolean;
  lastConnected?: Date;
  lastDisconnected?: Date;
  reconnectAttempts?: number;
}