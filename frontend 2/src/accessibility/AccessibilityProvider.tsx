/**
 * Accessibility Provider
 * React context provider for comprehensive accessibility features including
 * screen reader support, keyboard navigation, theme management, and WCAG compliance
 */

import React, { createContext, useContext, useEffect, useReducer, useCallback, useRef } from 'react';
import {
  IAccessibilityConfig,
  IAccessibilityContextValue,
  IAccessibilityProviderProps,
  IAccessibilityTheme,
  AccessibilityThemeType,
  IAccessibilityViolation,
  IKeyboardNavigationConfig,
  IFocusManager,
  FontSize
} from '@/types/accessibility';

import { AccessibilityHelper } from './AccessibilityHelper';
import { FocusManager } from './FocusManager';
import { ScreenReaderManager } from './ScreenReaderManager';
import { ThemeManager } from './ThemeManager';
import { AccessibilityAuditor } from './AccessibilityAuditor';

// Accessibility themes
const themes: Record<AccessibilityThemeType, IAccessibilityTheme> = {
  default: {
    id: 'default',
    name: 'default',
    displayName: 'Default Theme',
    description: 'Standard theme with balanced colors and typography',
    colors: {
      primary: '#2563eb',
      secondary: '#64748b',
      background: '#ffffff',
      surface: '#f8fafc',
      text: '#1e293b',
      textSecondary: '#64748b',
      border: '#e2e8f0',
      focus: '#3b82f6',
      error: '#dc2626',
      warning: '#d97706',
      success: '#059669',
      info: '#0284c7',
      disabled: '#94a3b8'
    },
    typography: {
      fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      fontSize: {
        xs: '0.75rem',
        sm: '0.875rem',
        base: '1rem',
        lg: '1.125rem',
        xl: '1.25rem',
        '2xl': '1.5rem',
        '3xl': '1.875rem',
        '4xl': '2.25rem'
      },
      fontWeight: {
        light: 300,
        normal: 400,
        medium: 500,
        semibold: 600,
        bold: 700
      },
      lineHeight: {
        tight: 1.25,
        normal: 1.5,
        relaxed: 1.625,
        loose: 2
      },
      letterSpacing: {
        tight: '-0.025em',
        normal: '0',
        wide: '0.025em'
      }
    },
    spacing: {
      xs: '0.25rem',
      sm: '0.5rem',
      md: '1rem',
      lg: '1.5rem',
      xl: '2rem',
      '2xl': '2.5rem',
      '3xl': '3rem',
      '4xl': '4rem'
    },
    contrast: {
      level: 'AA',
      ratios: {
        normal: 4.5,
        large: 3,
        interactive: 3
      },
      enhanced: false
    },
    colorBlind: {
      protanopia: false,
      deuteranopia: false,
      tritanopia: false,
      achromatopsia: false,
      patterns: false,
      textures: false
    }
  },
  'high-contrast': {
    id: 'high-contrast',
    name: 'high-contrast',
    displayName: 'High Contrast',
    description: 'High contrast theme for improved visibility',
    colors: {
      primary: '#0066cc',
      secondary: '#333333',
      background: '#ffffff',
      surface: '#ffffff',
      text: '#000000',
      textSecondary: '#333333',
      border: '#000000',
      focus: '#ff6600',
      error: '#cc0000',
      warning: '#ff6600',
      success: '#009900',
      info: '#0066cc',
      disabled: '#666666'
    },
    typography: {
      fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      fontSize: {
        xs: '0.875rem',
        sm: '1rem',
        base: '1.125rem',
        lg: '1.25rem',
        xl: '1.375rem',
        '2xl': '1.625rem',
        '3xl': '2rem',
        '4xl': '2.5rem'
      },
      fontWeight: {
        light: 400,
        normal: 500,
        medium: 600,
        semibold: 700,
        bold: 800
      },
      lineHeight: {
        tight: 1.375,
        normal: 1.625,
        relaxed: 1.75,
        loose: 2.25
      },
      letterSpacing: {
        tight: '0',
        normal: '0.025em',
        wide: '0.05em'
      }
    },
    spacing: {
      xs: '0.375rem',
      sm: '0.625rem',
      md: '1.25rem',
      lg: '1.875rem',
      xl: '2.5rem',
      '2xl': '3.125rem',
      '3xl': '3.75rem',
      '4xl': '5rem'
    },
    contrast: {
      level: 'AAA',
      ratios: {
        normal: 7,
        large: 4.5,
        interactive: 4.5
      },
      enhanced: true
    },
    colorBlind: {
      protanopia: true,
      deuteranopia: true,
      tritanopia: true,
      achromatopsia: false,
      patterns: true,
      textures: true
    }
  },
  dark: {
    id: 'dark',
    name: 'dark',
    displayName: 'Dark Theme',
    description: 'Dark theme for reduced eye strain',
    colors: {
      primary: '#3b82f6',
      secondary: '#9ca3af',
      background: '#111827',
      surface: '#1f2937',
      text: '#f9fafb',
      textSecondary: '#d1d5db',
      border: '#374151',
      focus: '#60a5fa',
      error: '#f87171',
      warning: '#fbbf24',
      success: '#34d399',
      info: '#38bdf8',
      disabled: '#6b7280'
    },
    typography: {
      fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      fontSize: {
        xs: '0.75rem',
        sm: '0.875rem',
        base: '1rem',
        lg: '1.125rem',
        xl: '1.25rem',
        '2xl': '1.5rem',
        '3xl': '1.875rem',
        '4xl': '2.25rem'
      },
      fontWeight: {
        light: 300,
        normal: 400,
        medium: 500,
        semibold: 600,
        bold: 700
      },
      lineHeight: {
        tight: 1.25,
        normal: 1.5,
        relaxed: 1.625,
        loose: 2
      },
      letterSpacing: {
        tight: '-0.025em',
        normal: '0',
        wide: '0.025em'
      }
    },
    spacing: {
      xs: '0.25rem',
      sm: '0.5rem',
      md: '1rem',
      lg: '1.5rem',
      xl: '2rem',
      '2xl': '2.5rem',
      '3xl': '3rem',
      '4xl': '4rem'
    },
    contrast: {
      level: 'AA',
      ratios: {
        normal: 4.5,
        large: 3,
        interactive: 3
      },
      enhanced: false
    },
    colorBlind: {
      protanopia: false,
      deuteranopia: false,
      tritanopia: false,
      achromatopsia: false,
      patterns: false,
      textures: false
    }
  },
  'high-contrast-dark': {
    id: 'high-contrast-dark',
    name: 'high-contrast-dark',
    displayName: 'High Contrast Dark',
    description: 'High contrast dark theme for improved visibility in low light',
    colors: {
      primary: '#66b3ff',
      secondary: '#cccccc',
      background: '#000000',
      surface: '#000000',
      text: '#ffffff',
      textSecondary: '#cccccc',
      border: '#ffffff',
      focus: '#ffcc00',
      error: '#ff6666',
      warning: '#ffcc00',
      success: '#66ff66',
      info: '#66b3ff',
      disabled: '#999999'
    },
    typography: {
      fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      fontSize: {
        xs: '0.875rem',
        sm: '1rem',
        base: '1.125rem',
        lg: '1.25rem',
        xl: '1.375rem',
        '2xl': '1.625rem',
        '3xl': '2rem',
        '4xl': '2.5rem'
      },
      fontWeight: {
        light: 400,
        normal: 500,
        medium: 600,
        semibold: 700,
        bold: 800
      },
      lineHeight: {
        tight: 1.375,
        normal: 1.625,
        relaxed: 1.75,
        loose: 2.25
      },
      letterSpacing: {
        tight: '0',
        normal: '0.025em',
        wide: '0.05em'
      }
    },
    spacing: {
      xs: '0.375rem',
      sm: '0.625rem',
      md: '1.25rem',
      lg: '1.875rem',
      xl: '2.5rem',
      '2xl': '3.125rem',
      '3xl': '3.75rem',
      '4xl': '5rem'
    },
    contrast: {
      level: 'AAA',
      ratios: {
        normal: 7,
        large: 4.5,
        interactive: 4.5
      },
      enhanced: true
    },
    colorBlind: {
      protanopia: true,
      deuteranopia: true,
      tritanopia: true,
      achromatopsia: true,
      patterns: true,
      textures: true
    }
  },
  'low-vision': {
    id: 'low-vision',
    name: 'low-vision',
    displayName: 'Low Vision',
    description: 'Optimized theme for users with low vision',
    colors: {
      primary: '#0066cc',
      secondary: '#444444',
      background: '#fffef7',
      surface: '#fffef7',
      text: '#000000',
      textSecondary: '#2d2d2d',
      border: '#333333',
      focus: '#ff8800',
      error: '#d63384',
      warning: '#fd7e14',
      success: '#198754',
      info: '#0dcaf0',
      disabled: '#777777'
    },
    typography: {
      fontFamily: '"Open Sans", Verdana, Arial, sans-serif',
      fontSize: {
        xs: '1rem',
        sm: '1.125rem',
        base: '1.25rem',
        lg: '1.5rem',
        xl: '1.75rem',
        '2xl': '2rem',
        '3xl': '2.5rem',
        '4xl': '3rem'
      },
      fontWeight: {
        light: 500,
        normal: 600,
        medium: 700,
        semibold: 800,
        bold: 900
      },
      lineHeight: {
        tight: 1.5,
        normal: 1.75,
        relaxed: 2,
        loose: 2.5
      },
      letterSpacing: {
        tight: '0.025em',
        normal: '0.05em',
        wide: '0.1em'
      }
    },
    spacing: {
      xs: '0.5rem',
      sm: '0.75rem',
      md: '1.5rem',
      lg: '2.25rem',
      xl: '3rem',
      '2xl': '3.75rem',
      '3xl': '4.5rem',
      '4xl': '6rem'
    },
    contrast: {
      level: 'AAA',
      ratios: {
        normal: 7,
        large: 4.5,
        interactive: 4.5
      },
      enhanced: true
    },
    colorBlind: {
      protanopia: true,
      deuteranopia: true,
      tritanopia: true,
      achromatopsia: false,
      patterns: true,
      textures: true
    }
  },
  protanopia: {
    id: 'protanopia',
    name: 'protanopia',
    displayName: 'Protanopia Support',
    description: 'Theme optimized for red-blind users',
    colors: {
      primary: '#0066ff',
      secondary: '#666666',
      background: '#ffffff',
      surface: '#f8fafc',
      text: '#1e293b',
      textSecondary: '#64748b',
      border: '#e2e8f0',
      focus: '#0066ff',
      error: '#ffaa00',
      warning: '#ff6600',
      success: '#0099cc',
      info: '#0066ff',
      disabled: '#94a3b8'
    },
    typography: {
      fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      fontSize: {
        xs: '0.75rem',
        sm: '0.875rem',
        base: '1rem',
        lg: '1.125rem',
        xl: '1.25rem',
        '2xl': '1.5rem',
        '3xl': '1.875rem',
        '4xl': '2.25rem'
      },
      fontWeight: {
        light: 300,
        normal: 400,
        medium: 500,
        semibold: 600,
        bold: 700
      },
      lineHeight: {
        tight: 1.25,
        normal: 1.5,
        relaxed: 1.625,
        loose: 2
      },
      letterSpacing: {
        tight: '-0.025em',
        normal: '0',
        wide: '0.025em'
      }
    },
    spacing: {
      xs: '0.25rem',
      sm: '0.5rem',
      md: '1rem',
      lg: '1.5rem',
      xl: '2rem',
      '2xl': '2.5rem',
      '3xl': '3rem',
      '4xl': '4rem'
    },
    contrast: {
      level: 'AA',
      ratios: {
        normal: 4.5,
        large: 3,
        interactive: 3
      },
      enhanced: false
    },
    colorBlind: {
      protanopia: true,
      deuteranopia: false,
      tritanopia: false,
      achromatopsia: false,
      patterns: true,
      textures: false
    }
  },
  deuteranopia: {
    id: 'deuteranopia',
    name: 'deuteranopia',
    displayName: 'Deuteranopia Support',
    description: 'Theme optimized for green-blind users',
    colors: {
      primary: '#0066ff',
      secondary: '#666666',
      background: '#ffffff',
      surface: '#f8fafc',
      text: '#1e293b',
      textSecondary: '#64748b',
      border: '#e2e8f0',
      focus: '#0066ff',
      error: '#ff3366',
      warning: '#ffaa00',
      success: '#0066cc',
      info: '#0066ff',
      disabled: '#94a3b8'
    },
    typography: {
      fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      fontSize: {
        xs: '0.75rem',
        sm: '0.875rem',
        base: '1rem',
        lg: '1.125rem',
        xl: '1.25rem',
        '2xl': '1.5rem',
        '3xl': '1.875rem',
        '4xl': '2.25rem'
      },
      fontWeight: {
        light: 300,
        normal: 400,
        medium: 500,
        semibold: 600,
        bold: 700
      },
      lineHeight: {
        tight: 1.25,
        normal: 1.5,
        relaxed: 1.625,
        loose: 2
      },
      letterSpacing: {
        tight: '-0.025em',
        normal: '0',
        wide: '0.025em'
      }
    },
    spacing: {
      xs: '0.25rem',
      sm: '0.5rem',
      md: '1rem',
      lg: '1.5rem',
      xl: '2rem',
      '2xl': '2.5rem',
      '3xl': '3rem',
      '4xl': '4rem'
    },
    contrast: {
      level: 'AA',
      ratios: {
        normal: 4.5,
        large: 3,
        interactive: 3
      },
      enhanced: false
    },
    colorBlind: {
      protanopia: false,
      deuteranopia: true,
      tritanopia: false,
      achromatopsia: false,
      patterns: true,
      textures: false
    }
  },
  tritanopia: {
    id: 'tritanopia',
    name: 'tritanopia',
    displayName: 'Tritanopia Support',
    description: 'Theme optimized for blue-blind users',
    colors: {
      primary: '#cc3366',
      secondary: '#666666',
      background: '#ffffff',
      surface: '#f8fafc',
      text: '#1e293b',
      textSecondary: '#64748b',
      border: '#e2e8f0',
      focus: '#cc3366',
      error: '#cc0033',
      warning: '#ff9900',
      success: '#339900',
      info: '#cc3366',
      disabled: '#94a3b8'
    },
    typography: {
      fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      fontSize: {
        xs: '0.75rem',
        sm: '0.875rem',
        base: '1rem',
        lg: '1.125rem',
        xl: '1.25rem',
        '2xl': '1.5rem',
        '3xl': '1.875rem',
        '4xl': '2.25rem'
      },
      fontWeight: {
        light: 300,
        normal: 400,
        medium: 500,
        semibold: 600,
        bold: 700
      },
      lineHeight: {
        tight: 1.25,
        normal: 1.5,
        relaxed: 1.625,
        loose: 2
      },
      letterSpacing: {
        tight: '-0.025em',
        normal: '0',
        wide: '0.025em'
      }
    },
    spacing: {
      xs: '0.25rem',
      sm: '0.5rem',
      md: '1rem',
      lg: '1.5rem',
      xl: '2rem',
      '2xl': '2.5rem',
      '3xl': '3rem',
      '4xl': '4rem'
    },
    contrast: {
      level: 'AA',
      ratios: {
        normal: 4.5,
        large: 3,
        interactive: 3
      },
      enhanced: false
    },
    colorBlind: {
      protanopia: false,
      deuteranopia: false,
      tritanopia: true,
      achromatopsia: false,
      patterns: true,
      textures: false
    }
  }
};

// Default accessibility configuration
const defaultConfig: IAccessibilityConfig = {
  screenReader: {
    enabled: true,
    announcePageChanges: true,
    announceFormErrors: true,
    announceLoadingStates: true,
    verbosity: 'normal',
    skipToContentEnabled: true,
    landmarkNavigation: true
  },
  keyboard: {
    enabled: true,
    focusVisible: true,
    skipLinks: true,
    tabOrder: 'sequential',
    arrowKeyNavigation: true,
    escapeKeyHandling: true,
    enterKeyActivation: true,
    spaceKeyActivation: true,
    customKeyBindings: {}
  },
  motor: {
    stickyKeys: false,
    slowKeys: false,
    bounceKeys: false,
    mouseKeys: false,
    clickAssist: false,
    dragAssist: false,
    hoverDelay: 500,
    clickDelay: 0,
    doubleClickSpeed: 500,
    targetSize: 'medium'
  },
  cognitive: {
    simplifiedInterface: false,
    reducedClutter: false,
    breadcrumbs: true,
    progressIndicators: true,
    confirmationDialogs: true,
    timeoutWarnings: true,
    autoSave: true,
    errorPrevention: true
  },
  display: {
    reducedMotion: false,
    highContrast: false,
    invertColors: false,
    grayscale: false,
    zoom: 1,
    cursorSize: 'normal',
    cursorColor: '#000000'
  },
  font: {
    size: 'base',
    family: 'system',
    weight: 'normal',
    lineHeight: 'normal',
    letterSpacing: 'normal',
    dyslexiaFriendly: false
  },
  wcag: {
    level: 'AA',
    guidelines: {
      perceivable: {
        textAlternatives: true,
        timeBasedMedia: true,
        adaptable: true,
        distinguishable: true
      },
      operable: {
        keyboardAccessible: true,
        seizuresAndPhysicalReactions: true,
        navigable: true,
        inputModalities: true
      },
      understandable: {
        readable: true,
        predictable: true,
        inputAssistance: true
      },
      robust: {
        compatible: true
      }
    },
    audit: {
      enabled: true,
      autoCheck: true,
      reportViolations: true,
      fixSuggestions: true
    }
  },
  voice: {
    enabled: false,
    commands: false,
    feedback: false
  }
};

// Accessibility state type
interface AccessibilityState {
  config: IAccessibilityConfig;
  theme: IAccessibilityTheme;
  violations: IAccessibilityViolation[];
  focusManager: IFocusManager;
}

// Action types
type AccessibilityAction =
  | { type: 'UPDATE_CONFIG'; payload: Partial<IAccessibilityConfig> }
  | { type: 'UPDATE_THEME'; payload: AccessibilityThemeType }
  | { type: 'SET_VIOLATIONS'; payload: IAccessibilityViolation[] }
  | { type: 'ADD_VIOLATION'; payload: IAccessibilityViolation }
  | { type: 'REMOVE_VIOLATION'; payload: string }
  | { type: 'CLEAR_VIOLATIONS' };

// Reducer function
function accessibilityReducer(state: AccessibilityState, action: AccessibilityAction): AccessibilityState {
  switch (action.type) {
    case 'UPDATE_CONFIG':
      return {
        ...state,
        config: { ...state.config, ...action.payload }
      };
      
    case 'UPDATE_THEME':
      return {
        ...state,
        theme: themes[action.payload]
      };
      
    case 'SET_VIOLATIONS':
      return {
        ...state,
        violations: action.payload
      };
      
    case 'ADD_VIOLATION':
      return {
        ...state,
        violations: [...state.violations, action.payload]
      };
      
    case 'REMOVE_VIOLATION':
      return {
        ...state,
        violations: state.violations.filter(v => v.id !== action.payload)
      };
      
    case 'CLEAR_VIOLATIONS':
      return {
        ...state,
        violations: []
      };
      
    default:
      return state;
  }
}

// Create context
const AccessibilityContext = createContext<IAccessibilityContextValue | null>(null);

// Provider component
export const AccessibilityProvider: React.FC<IAccessibilityProviderProps> = ({
  children,
  initialConfig,
  initialTheme = 'default'
}) => {
  // Initialize state
  const [state, dispatch] = useReducer(accessibilityReducer, {
    config: { ...defaultConfig, ...initialConfig },
    theme: themes[initialTheme],
    violations: [],
    focusManager: new FocusManager()
  });

  // Managers and services
  const screenReaderManager = useRef(new ScreenReaderManager()).current;
  const themeManager = useRef(new ThemeManager()).current;
  const accessibilityAuditor = useRef(new AccessibilityAuditor()).current;
  const helper = useRef(new AccessibilityHelper()).current;

  // Media query for reduced motion preference
  const prefersReducedMotion = useRef(
    window.matchMedia('(prefers-reduced-motion: reduce)').matches
  ).current;

  const isHighContrast = useRef(
    window.matchMedia('(prefers-contrast: high)').matches
  ).current;

  /**
   * Update configuration
   */
  const updateConfig = useCallback((newConfig: Partial<IAccessibilityConfig>) => {
    dispatch({ type: 'UPDATE_CONFIG', payload: newConfig });
    
    // Apply configuration changes
    if (newConfig.screenReader) {
      screenReaderManager.updateConfig(newConfig.screenReader);
    }
    
    if (newConfig.keyboard) {
      state.focusManager.updateConfig(newConfig.keyboard);
    }
    
    if (newConfig.display) {
      themeManager.applyDisplaySettings(newConfig.display);
    }
    
    if (newConfig.font) {
      themeManager.applyFontSettings(newConfig.font);
    }
  }, [screenReaderManager, state.focusManager, themeManager]);

  /**
   * Update theme
   */
  const updateTheme = useCallback((themeType: AccessibilityThemeType) => {
    dispatch({ type: 'UPDATE_THEME', payload: themeType });
    themeManager.applyTheme(themes[themeType]);
  }, [themeManager]);

  /**
   * Announce to screen reader
   */
  const announceToScreenReader = useCallback((
    message: string, 
    priority: 'polite' | 'assertive' = 'polite'
  ) => {
    screenReaderManager.announce(message, priority);
  }, [screenReaderManager]);

  // Initialize accessibility features
  useEffect(() => {
    // Apply initial theme
    themeManager.applyTheme(state.theme);
    
    // Initialize screen reader
    screenReaderManager.initialize(state.config.screenReader);
    
    // Start accessibility auditing if enabled
    if (state.config.wcag.audit.enabled && state.config.wcag.audit.autoCheck) {
      const runAudit = async () => {
        const violations = await accessibilityAuditor.auditPage();
        dispatch({ type: 'SET_VIOLATIONS', payload: violations });
      };
      
      // Run initial audit
      runAudit();
      
      // Set up mutation observer for dynamic content
      const observer = new MutationObserver(() => {
        runAudit();
      });
      
      observer.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ['aria-*', 'role', 'tabindex', 'alt', 'title']
      });
      
      return () => observer.disconnect();
    }
  }, [state.theme, state.config, screenReaderManager, themeManager, accessibilityAuditor]);

  // Handle keyboard events
  useEffect(() => {
    if (!state.config.keyboard.enabled) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      state.focusManager.handleKeyboardEvent({
        key: event.key,
        code: event.code,
        altKey: event.altKey,
        ctrlKey: event.ctrlKey,
        metaKey: event.metaKey,
        shiftKey: event.shiftKey,
        target: event.target as HTMLElement,
        preventDefault: () => event.preventDefault(),
        stopPropagation: () => event.stopPropagation()
      });
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [state.config.keyboard, state.focusManager]);

  // Apply theme classes to document
  useEffect(() => {
    const root = document.documentElement;
    
    // Remove existing theme classes
    Object.keys(themes).forEach(themeKey => {
      root.classList.remove(`theme-${themeKey}`);
    });
    
    // Apply current theme class
    root.classList.add(`theme-${state.theme.name}`);
    
    // Apply CSS custom properties
    const colors = state.theme.colors;
    Object.entries(colors).forEach(([key, value]) => {
      root.style.setProperty(`--color-${key}`, value);
    });
    
    const typography = state.theme.typography;
    root.style.setProperty('--font-family', typography.fontFamily);
    
    Object.entries(typography.fontSize).forEach(([key, value]) => {
      root.style.setProperty(`--font-size-${key}`, value);
    });
    
    Object.entries(typography.fontWeight).forEach(([key, value]) => {
      root.style.setProperty(`--font-weight-${key}`, value.toString());
    });
    
    Object.entries(typography.lineHeight).forEach(([key, value]) => {
      root.style.setProperty(`--line-height-${key}`, value.toString());
    });
    
    Object.entries(typography.letterSpacing).forEach(([key, value]) => {
      root.style.setProperty(`--letter-spacing-${key}`, value);
    });
    
    const spacing = state.theme.spacing;
    Object.entries(spacing).forEach(([key, value]) => {
      root.style.setProperty(`--spacing-${key}`, value);
    });
    
  }, [state.theme]);

  // Context value
  const contextValue: IAccessibilityContextValue = {
    config: state.config,
    theme: state.theme,
    updateConfig,
    updateTheme,
    announceToScreenReader,
    focusManager: state.focusManager,
    isReducedMotion: prefersReducedMotion,
    isHighContrast,
    currentFontSize: state.config.font.size,
    violations: state.violations
  };

  return (
    <AccessibilityContext.Provider value={contextValue}>
      {children}
    </AccessibilityContext.Provider>
  );
};

// Custom hook to use accessibility context
export const useAccessibility = (): IAccessibilityContextValue => {
  const context = useContext(AccessibilityContext);
  if (!context) {
    throw new Error('useAccessibility must be used within an AccessibilityProvider');
  }
  return context;
};

export default AccessibilityProvider;