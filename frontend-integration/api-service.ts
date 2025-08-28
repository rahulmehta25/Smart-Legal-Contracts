/**
 * API Service for Arbitration Frontend
 * Handles all backend communication
 */

import axios, { AxiosInstance, AxiosError } from 'axios';

// API Configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const API_VERSION = process.env.NEXT_PUBLIC_API_VERSION || 'v1';
const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';

// Create axios instance
const apiClient: AxiosInstance = axios.create({
  baseURL: `${API_BASE_URL}/api/${API_VERSION}`,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for authentication
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    if (error.response?.status === 401) {
      // Handle token expiration
      localStorage.removeItem('auth_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Authentication Services
export const authService = {
  async login(email: string, password: string) {
    const response = await apiClient.post('/auth/login', { email, password });
    const { access_token, user } = response.data;
    localStorage.setItem('auth_token', access_token);
    return { token: access_token, user };
  },

  async register(email: string, password: string, name: string) {
    const response = await apiClient.post('/auth/register', { 
      email, 
      password, 
      name 
    });
    return response.data;
  },

  async logout() {
    localStorage.removeItem('auth_token');
    await apiClient.post('/auth/logout');
  },

  async getCurrentUser() {
    const response = await apiClient.get('/auth/me');
    return response.data;
  },
};

// Document Analysis Services
export const analysisService = {
  async analyzeText(text: string) {
    const response = await apiClient.post('/analyze/text', { text });
    return response.data;
  },

  async analyzeFile(file: File, onProgress?: (progress: number) => void) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await apiClient.post('/analyze/file', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(progress);
        }
      },
    });
    return response.data;
  },

  async batchAnalyze(files: File[]) {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file);
    });

    const response = await apiClient.post('/batch/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  async getJobStatus(jobId: string) {
    const response = await apiClient.get(`/jobs/${jobId}/status`);
    return response.data;
  },
};

// Document Management Services
export const documentService = {
  async getDocuments(page = 1, limit = 10) {
    const response = await apiClient.get('/documents', {
      params: { page, limit },
    });
    return response.data;
  },

  async getDocument(id: string) {
    const response = await apiClient.get(`/documents/${id}`);
    return response.data;
  },

  async deleteDocument(id: string) {
    const response = await apiClient.delete(`/documents/${id}`);
    return response.data;
  },

  async downloadDocument(id: string) {
    const response = await apiClient.get(`/documents/${id}/download`, {
      responseType: 'blob',
    });
    return response.data;
  },

  async searchDocuments(query: string) {
    const response = await apiClient.get('/documents/search', {
      params: { q: query },
    });
    return response.data;
  },
};

// Statistics Services
export const statsService = {
  async getOverview() {
    const response = await apiClient.get('/stats/overview');
    return response.data;
  },

  async getAnalysisHistory(days = 30) {
    const response = await apiClient.get('/stats/history', {
      params: { days },
    });
    return response.data;
  },

  async getClauseDistribution() {
    const response = await apiClient.get('/stats/clauses');
    return response.data;
  },
};

// WebSocket Connection
export class WebSocketService {
  private socket: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private listeners: Map<string, Set<Function>> = new Map();

  connect(token?: string) {
    const wsUrl = token ? `${WS_URL}/ws?token=${token}` : `${WS_URL}/ws`;
    
    this.socket = new WebSocket(wsUrl);

    this.socket.onopen = () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
      this.emit('connected');
    };

    this.socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.emit(data.event, data.data);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };

    this.socket.onerror = (error) => {
      console.error('WebSocket error:', error);
      this.emit('error', error);
    };

    this.socket.onclose = () => {
      console.log('WebSocket disconnected');
      this.emit('disconnected');
      this.attemptReconnect();
    };
  }

  private attemptReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      setTimeout(() => {
        console.log(`Reconnecting... (attempt ${this.reconnectAttempts})`);
        this.connect();
      }, this.reconnectDelay * this.reconnectAttempts);
    }
  }

  send(event: string, data: any) {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({ event, data }));
    } else {
      console.error('WebSocket is not connected');
    }
  }

  on(event: string, callback: Function) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(callback);
  }

  off(event: string, callback: Function) {
    const callbacks = this.listeners.get(event);
    if (callbacks) {
      callbacks.delete(callback);
    }
  }

  private emit(event: string, data?: any) {
    const callbacks = this.listeners.get(event);
    if (callbacks) {
      callbacks.forEach(callback => callback(data));
    }
  }

  disconnect() {
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
  }
}

// Export singleton WebSocket instance
export const wsService = new WebSocketService();

// Health Check
export const healthCheck = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/health`);
    return response.data;
  } catch (error) {
    return { status: 'unhealthy', error: error.message };
  }
};

// Export default API client for custom requests
export default apiClient;