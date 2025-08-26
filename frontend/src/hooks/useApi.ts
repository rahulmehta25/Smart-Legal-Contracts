import { useState, useCallback } from 'react';
import { apiService } from '../services/api';
import { useToast } from './use-toast';

interface ApiState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

interface UseApiOptions {
  showErrorToast?: boolean;
  showSuccessToast?: boolean;
  successMessage?: string;
}

export function useApi<T = any>(options: UseApiOptions = {}) {
  const { showErrorToast = true, showSuccessToast = false, successMessage } = options;
  const { toast } = useToast();
  
  const [state, setState] = useState<ApiState<T>>({
    data: null,
    loading: false,
    error: null,
  });

  const execute = useCallback(async (
    apiCall: () => Promise<T>,
    customOptions?: UseApiOptions
  ): Promise<T | null> => {
    setState({ data: null, loading: true, error: null });
    
    try {
      const result = await apiCall();
      setState({ data: result, loading: false, error: null });
      
      if ((customOptions?.showSuccessToast ?? showSuccessToast) && successMessage) {
        toast({
          title: 'Success',
          description: customOptions?.successMessage || successMessage,
        });
      }
      
      return result;
    } catch (error: any) {
      const errorMessage = error?.response?.data?.detail || error?.message || 'An error occurred';
      setState({ data: null, loading: false, error: errorMessage });
      
      if (customOptions?.showErrorToast ?? showErrorToast) {
        toast({
          title: 'Error',
          description: errorMessage,
          variant: 'destructive',
        });
      }
      
      return null;
    }
  }, [showErrorToast, showSuccessToast, successMessage, toast]);

  const reset = useCallback(() => {
    setState({ data: null, loading: false, error: null });
  }, []);

  return {
    ...state,
    execute,
    reset,
  };
}

// Specialized hooks for common API operations
export function useAuth() {
  const api = useApi();
  
  const login = useCallback(async (credentials: { email: string; password: string }) => {
    return api.execute(() => apiService.auth.login(credentials), {
      showSuccessToast: true,
      successMessage: 'Login successful',
    });
  }, [api]);
  
  const register = useCallback(async (userData: { email: string; password: string; username: string }) => {
    return api.execute(() => apiService.auth.register(userData), {
      showSuccessToast: true,
      successMessage: 'Registration successful',
    });
  }, [api]);
  
  const logout = useCallback(async () => {
    return api.execute(() => apiService.auth.logout(), {
      showSuccessToast: true,
      successMessage: 'Logout successful',
    });
  }, [api]);
  
  return {
    ...api,
    login,
    register,
    logout,
  };
}

export function useArbitration() {
  const api = useApi();
  
  const detectOpportunities = useCallback(async (params?: any) => {
    return api.execute(() => apiService.arbitration.detectOpportunities(params));
  }, [api]);
  
  const getHistory = useCallback(async (params?: any) => {
    return api.execute(() => apiService.arbitration.getHistory(params));
  }, [api]);
  
  const getStats = useCallback(async () => {
    return api.execute(() => apiService.arbitration.getStats());
  }, [api]);
  
  return {
    ...api,
    detectOpportunities,
    getHistory,
    getStats,
  };
}

export function useUser() {
  const api = useApi();
  
  const getProfile = useCallback(async () => {
    return api.execute(() => apiService.user.getProfile());
  }, [api]);
  
  const updateProfile = useCallback(async (userData: any) => {
    return api.execute(() => apiService.user.updateProfile(userData), {
      showSuccessToast: true,
      successMessage: 'Profile updated successfully',
    });
  }, [api]);
  
  return {
    ...api,
    getProfile,
    updateProfile,
  };
}