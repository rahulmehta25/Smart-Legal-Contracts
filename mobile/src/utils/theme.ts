import { Theme } from '@types/index';

export const lightTheme: Theme = {
  colors: {
    primary: '#2563EB',
    secondary: '#7C3AED',
    background: '#FFFFFF',
    surface: '#F8FAFC',
    text: '#1E293B',
    textSecondary: '#64748B',
    error: '#DC2626',
    warning: '#D97706',
    success: '#059669',
    info: '#0284C7',
    border: '#E2E8F0',
    shadow: '#000000',
  },
  spacing: {
    xs: 4,
    sm: 8,
    md: 16,
    lg: 24,
    xl: 32,
  },
  typography: {
    fontFamily: 'Inter',
    fontSize: {
      xs: 12,
      sm: 14,
      md: 16,
      lg: 18,
      xl: 24,
      xxl: 32,
    },
    fontWeight: {
      light: '300',
      regular: '400',
      medium: '500',
      bold: '700',
    },
  },
  borderRadius: {
    sm: 4,
    md: 8,
    lg: 12,
    xl: 16,
  },
};

export const darkTheme: Theme = {
  ...lightTheme,
  colors: {
    primary: '#3B82F6',
    secondary: '#8B5CF6',
    background: '#0F172A',
    surface: '#1E293B',
    text: '#F1F5F9',
    textSecondary: '#94A3B8',
    error: '#EF4444',
    warning: '#F59E0B',
    success: '#10B981',
    info: '#06B6D4',
    border: '#334155',
    shadow: '#000000',
  },
};

export const getTheme = (isDark: boolean): Theme => {
  return isDark ? darkTheme : lightTheme;
};