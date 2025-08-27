import React, { createContext, useContext, useEffect, useState } from 'react';

interface AccessibilityContextType {
  reducedMotion: boolean;
  highContrast: boolean;
  fontSize: 'small' | 'medium' | 'large';
  focusVisible: boolean;
  announcements: string[];
  announce: (message: string) => void;
  toggleReducedMotion: () => void;
  toggleHighContrast: () => void;
  setFontSize: (size: 'small' | 'medium' | 'large') => void;
}

const AccessibilityContext = createContext<AccessibilityContextType | undefined>(undefined);

export const useAccessibility = () => {
  const context = useContext(AccessibilityContext);
  if (context === undefined) {
    throw new Error('useAccessibility must be used within an AccessibilityProvider');
  }
  return context;
};

interface AccessibilityProviderProps {
  children: React.ReactNode;
}

export const AccessibilityProvider: React.FC<AccessibilityProviderProps> = ({ children }) => {
  const [reducedMotion, setReducedMotion] = useState(false);
  const [highContrast, setHighContrast] = useState(false);
  const [fontSize, setFontSizeState] = useState<'small' | 'medium' | 'large'>('medium');
  const [focusVisible, setFocusVisible] = useState(false);
  const [announcements, setAnnouncements] = useState<string[]>([]);

  // Check for system preferences on load
  useEffect(() => {
    // Check for reduced motion preference
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    setReducedMotion(mediaQuery.matches);
    
    const handleChange = (e: MediaQueryListEvent) => {
      setReducedMotion(e.matches);
    };
    
    mediaQuery.addEventListener('change', handleChange);
    
    // Check for high contrast preference
    const contrastQuery = window.matchMedia('(prefers-contrast: high)');
    setHighContrast(contrastQuery.matches);
    
    const handleContrastChange = (e: MediaQueryListEvent) => {
      setHighContrast(e.matches);
    };
    
    contrastQuery.addEventListener('change', handleContrastChange);

    // Load saved preferences
    const savedReducedMotion = localStorage.getItem('accessibility-reduced-motion');
    const savedHighContrast = localStorage.getItem('accessibility-high-contrast');
    const savedFontSize = localStorage.getItem('accessibility-font-size') as 'small' | 'medium' | 'large';
    
    if (savedReducedMotion !== null) {
      setReducedMotion(savedReducedMotion === 'true');
    }
    
    if (savedHighContrast !== null) {
      setHighContrast(savedHighContrast === 'true');
    }
    
    if (savedFontSize) {
      setFontSizeState(savedFontSize);
    }

    return () => {
      mediaQuery.removeEventListener('change', handleChange);
      contrastQuery.removeEventListener('change', handleContrastChange);
    };
  }, []);

  // Apply accessibility settings to document
  useEffect(() => {
    const root = document.documentElement;
    
    // Apply reduced motion
    if (reducedMotion) {
      root.style.setProperty('--transition-smooth', 'none');
      root.style.setProperty('--transition-bounce', 'none');
      root.style.setProperty('--transition-magnetic', 'none');
      root.classList.add('reduce-motion');
    } else {
      root.style.removeProperty('--transition-smooth');
      root.style.removeProperty('--transition-bounce');
      root.style.removeProperty('--transition-magnetic');
      root.classList.remove('reduce-motion');
    }
    
    // Apply high contrast
    if (highContrast) {
      root.classList.add('high-contrast');
    } else {
      root.classList.remove('high-contrast');
    }
    
    // Apply font size
    root.classList.remove('font-small', 'font-medium', 'font-large');
    root.classList.add(`font-${fontSize}`);
  }, [reducedMotion, highContrast, fontSize]);

  // Focus management
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Tab') {
        setFocusVisible(true);
      }
    };

    const handleMouseDown = () => {
      setFocusVisible(false);
    };

    document.addEventListener('keydown', handleKeyDown);
    document.addEventListener('mousedown', handleMouseDown);

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.removeEventListener('mousedown', handleMouseDown);
    };
  }, []);

  const announce = (message: string) => {
    setAnnouncements(prev => [...prev, message]);
    
    // Remove announcement after it's been read
    setTimeout(() => {
      setAnnouncements(prev => prev.slice(1));
    }, 1000);
  };

  const toggleReducedMotion = () => {
    const newValue = !reducedMotion;
    setReducedMotion(newValue);
    localStorage.setItem('accessibility-reduced-motion', newValue.toString());
    announce(newValue ? 'Reduced motion enabled' : 'Reduced motion disabled');
  };

  const toggleHighContrast = () => {
    const newValue = !highContrast;
    setHighContrast(newValue);
    localStorage.setItem('accessibility-high-contrast', newValue.toString());
    announce(newValue ? 'High contrast enabled' : 'High contrast disabled');
  };

  const setFontSize = (size: 'small' | 'medium' | 'large') => {
    setFontSizeState(size);
    localStorage.setItem('accessibility-font-size', size);
    announce(`Font size set to ${size}`);
  };

  const contextValue: AccessibilityContextType = {
    reducedMotion,
    highContrast,
    fontSize,
    focusVisible,
    announcements,
    announce,
    toggleReducedMotion,
    toggleHighContrast,
    setFontSize,
  };

  return (
    <AccessibilityContext.Provider value={contextValue}>
      {children}
      
      {/* Screen reader announcements */}
      <div
        aria-live="polite"
        aria-atomic="true"
        className="sr-only"
        role="status"
      >
        {announcements.map((announcement, index) => (
          <div key={index}>{announcement}</div>
        ))}
      </div>
      
      {/* Skip to main content link */}
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-50 bg-primary text-primary-foreground px-4 py-2 rounded-md focus-ring"
      >
        Skip to main content
      </a>
    </AccessibilityContext.Provider>
  );
};

// Accessibility Controls Component
export const AccessibilityControls: React.FC<{ className?: string }> = ({ className = '' }) => {
  const { 
    reducedMotion, 
    highContrast, 
    fontSize, 
    toggleReducedMotion, 
    toggleHighContrast, 
    setFontSize 
  } = useAccessibility();

  return (
    <div className={`space-y-4 ${className}`}>
      <h3 className="text-lg font-semibold">Accessibility Settings</h3>
      
      {/* Reduced Motion Toggle */}
      <div className="flex items-center justify-between">
        <label htmlFor="reduced-motion" className="text-sm font-medium">
          Reduce animations and motion
        </label>
        <button
          id="reduced-motion"
          role="switch"
          aria-checked={reducedMotion}
          onClick={toggleReducedMotion}
          className={`
            relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2
            ${reducedMotion ? 'bg-primary' : 'bg-muted'}
          `}
        >
          <span
            className={`
              inline-block h-4 w-4 transform rounded-full bg-background transition-transform
              ${reducedMotion ? 'translate-x-6' : 'translate-x-1'}
            `}
          />
        </button>
      </div>

      {/* High Contrast Toggle */}
      <div className="flex items-center justify-between">
        <label htmlFor="high-contrast" className="text-sm font-medium">
          High contrast mode
        </label>
        <button
          id="high-contrast"
          role="switch"
          aria-checked={highContrast}
          onClick={toggleHighContrast}
          className={`
            relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2
            ${highContrast ? 'bg-primary' : 'bg-muted'}
          `}
        >
          <span
            className={`
              inline-block h-4 w-4 transform rounded-full bg-background transition-transform
              ${highContrast ? 'translate-x-6' : 'translate-x-1'}
            `}
          />
        </button>
      </div>

      {/* Font Size Controls */}
      <div className="space-y-2">
        <label className="text-sm font-medium">Text size</label>
        <div className="flex space-x-2">
          {(['small', 'medium', 'large'] as const).map((size) => (
            <button
              key={size}
              onClick={() => setFontSize(size)}
              className={`
                px-3 py-1 text-xs rounded-md transition-colors focus-ring
                ${fontSize === size 
                  ? 'bg-primary text-primary-foreground' 
                  : 'bg-muted text-muted-foreground hover:bg-muted/80'
                }
              `}
              aria-pressed={fontSize === size}
            >
              {size.charAt(0).toUpperCase() + size.slice(1)}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default AccessibilityProvider;