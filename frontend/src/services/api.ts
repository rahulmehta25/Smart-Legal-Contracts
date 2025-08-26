import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import config from '../config/env';

// Get API URL from centralized configuration
const API_BASE_URL = config.apiUrl;
const WS_BASE_URL = config.wsUrl;

// Create axios instance with default configuration
const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: config.apiTimeout,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Token storage utilities
export const tokenStorage = {
  getToken: (): string | null => {
    return localStorage.getItem('auth_token');
  },
  
  setToken: (token: string): void => {
    localStorage.setItem('auth_token', token);
  },
  
  removeToken: (): void => {
    localStorage.removeItem('auth_token');
  },
  
  getRefreshToken: (): string | null => {
    return localStorage.getItem('refresh_token');
  },
  
  setRefreshToken: (token: string): void => {
    localStorage.setItem('refresh_token', token);
  },
  
  removeRefreshToken: (): void => {
    localStorage.removeItem('refresh_token');
  }
};

// Request interceptor to add auth token
apiClient.interceptors.request.use(
  (config) => {
    const token = tokenStorage.getToken();
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor to handle token refresh and errors
apiClient.interceptors.response.use(
  (response: AxiosResponse) => {
    return response;
  },
  async (error) => {
    const originalRequest = error.config;
    
    // Handle 401 unauthorized errors
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;
      
      const refreshToken = tokenStorage.getRefreshToken();
      if (refreshToken) {
        try {
          const response = await axios.post(`${API_BASE_URL}/auth/refresh`, {
            refresh_token: refreshToken
          });
          
          const { access_token, refresh_token: newRefreshToken } = response.data;
          tokenStorage.setToken(access_token);
          
          if (newRefreshToken) {
            tokenStorage.setRefreshToken(newRefreshToken);
          }
          
          // Retry the original request with new token
          originalRequest.headers.Authorization = `Bearer ${access_token}`;
          return apiClient(originalRequest);
        } catch (refreshError) {
          // Refresh failed, clear tokens and redirect to login
          tokenStorage.removeToken();
          tokenStorage.removeRefreshToken();
          window.location.href = '/login';
          return Promise.reject(refreshError);
        }
      } else {
        // No refresh token, redirect to login
        tokenStorage.removeToken();
        window.location.href = '/login';
      }
    }
    
    return Promise.reject(error);
  }
);

// API service methods
export const apiService = {
  // Auth endpoints
  auth: {
    login: async (credentials: { email: string; password: string }) => {
      const response = await apiClient.post('/auth/login', credentials);
      return response.data;
    },
    
    register: async (userData: { email: string; password: string; username: string }) => {
      const response = await apiClient.post('/auth/register', userData);
      return response.data;
    },
    
    logout: async () => {
      const response = await apiClient.post('/auth/logout');
      tokenStorage.removeToken();
      tokenStorage.removeRefreshToken();
      return response.data;
    },
    
    refreshToken: async () => {
      const refreshToken = tokenStorage.getRefreshToken();
      if (!refreshToken) throw new Error('No refresh token available');
      
      const response = await apiClient.post('/auth/refresh', {
        refresh_token: refreshToken
      });
      return response.data;
    }
  },
  
  // User profile endpoints
  user: {
    getProfile: async () => {
      const response = await apiClient.get('/user/profile');
      return response.data;
    },
    
    updateProfile: async (userData: any) => {
      const response = await apiClient.put('/user/profile', userData);
      return response.data;
    }
  },
  
  // Arbitration detection endpoints
  arbitration: {
    detectOpportunities: async (params?: any) => {
      const response = await apiClient.get('/arbitration/detect', { params });
      return response.data;
    },
    
    getHistory: async (params?: any) => {
      const response = await apiClient.get('/arbitration/history', { params });
      return response.data;
    },
    
    getStats: async () => {
      const response = await apiClient.get('/arbitration/stats');
      return response.data;
    }
  },
  
  // AI marketplace endpoints
  ai: {
    getModels: async () => {
      const response = await apiClient.get('/ai/models');
      return response.data;
    },
    
    createStrategy: async (strategyData: any) => {
      const response = await apiClient.post('/ai/strategies', strategyData);
      return response.data;
    },
    
    getStrategies: async () => {
      const response = await apiClient.get('/ai/strategies');
      return response.data;
    }
  },
  
  // Collaboration endpoints
  collaboration: {
    getWorkspaces: async () => {
      const response = await apiClient.get('/collaboration/workspaces');
      return response.data;
    },
    
    createWorkspace: async (workspaceData: any) => {
      const response = await apiClient.post('/collaboration/workspaces', workspaceData);
      return response.data;
    },
    
    joinWorkspace: async (workspaceId: string) => {
      const response = await apiClient.post(`/collaboration/workspaces/${workspaceId}/join`);
      return response.data;
    }
  },
  
  // Generic CRUD operations
  get: async (url: string, config?: AxiosRequestConfig) => {
    const response = await apiClient.get(url, config);
    return response.data;
  },
  
  post: async (url: string, data?: any, config?: AxiosRequestConfig) => {
    const response = await apiClient.post(url, data, config);
    return response.data;
  },
  
  put: async (url: string, data?: any, config?: AxiosRequestConfig) => {
    const response = await apiClient.put(url, data, config);
    return response.data;
  },
  
  delete: async (url: string, config?: AxiosRequestConfig) => {
    const response = await apiClient.delete(url, config);
    return response.data;
  }
};

// WebSocket connection utility
export const createWebSocketConnection = (endpoint: string): WebSocket => {
  const wsUrl = `${WS_BASE_URL}${endpoint}`;
  const token = tokenStorage.getToken();
  
  // Add token to WebSocket URL if available
  const urlWithAuth = token ? `${wsUrl}?token=${token}` : wsUrl;
  
  return new WebSocket(urlWithAuth);
};

// Export the configured axios instance for direct use if needed
export { apiClient, API_BASE_URL, WS_BASE_URL };

export default apiService;