/**
 * Theme Manager
 * Advanced theme management for accessibility with high contrast modes,
 * color blindness support, font scaling, and display preferences
 */

import {
  IAccessibilityTheme,
  IDisplaySettings,
  IFontSettings,
  AccessibilityThemeType
} from '@/types/accessibility';

export class ThemeManager {
  private currentTheme: IAccessibilityTheme | null = null;
  private baseStyles: CSSStyleSheet | null = null;
  private themeStyles: CSSStyleSheet | null = null;

  constructor() {
    this.initializeBaseStyles();
    this.detectSystemPreferences();
  }

  /**
   * Initialize base accessibility styles
   */
  private initializeBaseStyles(): void {
    const style = document.createElement('style');
    style.id = 'accessibility-base-styles';
    
    style.textContent = `
      /* Focus Management */
      .keyboard-navigation *:focus {
        outline: 3px solid var(--color-focus, #3b82f6);
        outline-offset: 2px;
      }
      
      .mouse-navigation *:focus {
        outline: none;
      }
      
      .focus-visible {
        outline: 3px solid var(--color-focus, #3b82f6);
        outline-offset: 2px;
      }
      
      /* Skip Links */
      .skip-navigation {
        position: absolute;
        top: -50px;
        left: 8px;
        z-index: 9999;
      }
      
      .skip-link {
        position: absolute;
        left: -10000px;
        top: auto;
        width: 1px;
        height: 1px;
        overflow: hidden;
        background: var(--color-primary, #000);
        color: var(--color-background, #fff);
        padding: 12px 24px;
        text-decoration: none;
        font-weight: 600;
        border-radius: 6px;
        font-size: 16px;
        line-height: 1.4;
        transition: all 0.2s ease;
      }
      
      .skip-link:focus {
        position: static;
        left: auto;
        top: auto;
        width: auto;
        height: auto;
        overflow: visible;
        transform: translateY(0);
      }
      
      /* Screen Reader Only */
      .sr-only {
        position: absolute;
        left: -10000px;
        top: auto;
        width: 1px;
        height: 1px;
        overflow: hidden;
      }
      
      .sr-only-focusable:focus,
      .sr-only-focusable:active {
        position: static;
        left: auto;
        top: auto;
        width: auto;
        height: auto;
        overflow: visible;
      }
      
      /* Reduced Motion */
      @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
          animation-duration: 0.01ms !important;
          animation-iteration-count: 1 !important;
          transition-duration: 0.01ms !important;
          scroll-behavior: auto !important;
        }
      }
      
      .reduced-motion *, 
      .reduced-motion *::before, 
      .reduced-motion *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
        scroll-behavior: auto !important;
      }
      
      /* High Contrast Mode */
      .high-contrast {
        filter: contrast(150%);
      }
      
      .high-contrast img,
      .high-contrast video {
        filter: contrast(120%);
      }
      
      /* Color Inversion */
      .invert-colors {
        filter: invert(1) hue-rotate(180deg);
      }
      
      .invert-colors img,
      .invert-colors video,
      .invert-colors canvas {
        filter: invert(1) hue-rotate(180deg);
      }
      
      /* Grayscale */
      .grayscale {
        filter: grayscale(1);
      }
      
      /* Font Scaling */
      .font-size-xs { font-size: var(--font-size-xs, 0.75rem) !important; }
      .font-size-sm { font-size: var(--font-size-sm, 0.875rem) !important; }
      .font-size-base { font-size: var(--font-size-base, 1rem) !important; }
      .font-size-lg { font-size: var(--font-size-lg, 1.125rem) !important; }
      .font-size-xl { font-size: var(--font-size-xl, 1.25rem) !important; }
      .font-size-2xl { font-size: var(--font-size-2xl, 1.5rem) !important; }
      .font-size-3xl { font-size: var(--font-size-3xl, 1.875rem) !important; }
      .font-size-4xl { font-size: var(--font-size-4xl, 2.25rem) !important; }
      
      /* Dyslexia-Friendly Fonts */
      .dyslexia-friendly,
      .dyslexia-friendly * {
        font-family: 'OpenDyslexic', 'Comic Sans MS', 'Arial', sans-serif !important;
      }
      
      /* Target Size Enhancement */
      .large-targets button,
      .large-targets a,
      .large-targets input,
      .large-targets select,
      .large-targets textarea {
        min-height: 44px;
        min-width: 44px;
        padding: 12px 16px;
      }
      
      /* Cursor Enhancement */
      .large-cursor {
        cursor: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32"><polygon points="0,0 0,22 6,16 11,21 15,17 10,12 16,6" fill="black"/><polygon points="2,2 2,18 8,14 11,17 13,15 10,12 14,8" fill="white"/></svg>') 0 0, pointer;
      }
      
      .extra-large-cursor {
        cursor: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 48 48"><polygon points="0,0 0,33 9,24 16,31 22,25 15,18 24,9" fill="black"/><polygon points="3,3 3,27 12,21 16,25 19,22 15,18 21,12" fill="white"/></svg>') 0 0, pointer;
      }
      
      /* Loading States */
      [aria-busy="true"] {
        cursor: wait;
        pointer-events: none;
      }
      
      [aria-busy="true"]::after {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 20px;
        height: 20px;
        margin: -10px 0 0 -10px;
        border: 2px solid var(--color-primary, #3b82f6);
        border-radius: 50%;
        border-top-color: transparent;
        animation: spin 1s linear infinite;
      }
      
      @keyframes spin {
        to { transform: rotate(360deg); }
      }
      
      /* Error States */
      [aria-invalid="true"] {
        border-color: var(--color-error, #dc2626) !important;
        box-shadow: 0 0 0 2px rgba(220, 38, 38, 0.2);
      }
      
      /* Required Fields */
      [aria-required="true"]::after {
        content: ' *';
        color: var(--color-error, #dc2626);
        font-weight: bold;
      }
      
      /* Hidden Content */
      [aria-hidden="true"] {
        display: none !important;
      }
      
      /* Expanded/Collapsed States */
      [aria-expanded="false"] + * {
        display: none;
      }
      
      /* Selection States */
      [aria-selected="true"] {
        background-color: var(--color-primary, #3b82f6);
        color: var(--color-background, white);
      }
      
      /* Pressed States */
      [aria-pressed="true"] {
        background-color: var(--color-primary, #3b82f6);
        color: var(--color-background, white);
      }
      
      /* Disabled States */
      [aria-disabled="true"],
      :disabled {
        opacity: 0.6;
        cursor: not-allowed;
        pointer-events: none;
      }
    `;

    document.head.appendChild(style);
  }

  /**
   * Detect system accessibility preferences
   */
  private detectSystemPreferences(): void {
    const root = document.documentElement;

    // Detect reduced motion preference
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    this.updateReducedMotion(prefersReducedMotion.matches);
    prefersReducedMotion.addListener((e) => this.updateReducedMotion(e.matches));

    // Detect high contrast preference
    const prefersHighContrast = window.matchMedia('(prefers-contrast: high)');
    this.updateHighContrast(prefersHighContrast.matches);
    prefersHighContrast.addListener((e) => this.updateHighContrast(e.matches));

    // Detect color scheme preference
    const prefersDarkScheme = window.matchMedia('(prefers-color-scheme: dark)');
    root.setAttribute('data-color-scheme', prefersDarkScheme.matches ? 'dark' : 'light');
    prefersDarkScheme.addListener((e) => {
      root.setAttribute('data-color-scheme', e.matches ? 'dark' : 'light');
    });
  }

  /**
   * Update reduced motion setting
   */
  private updateReducedMotion(enabled: boolean): void {
    document.body.classList.toggle('reduced-motion', enabled);
  }

  /**
   * Update high contrast setting
   */
  private updateHighContrast(enabled: boolean): void {
    document.body.classList.toggle('high-contrast', enabled);
  }

  /**
   * Apply accessibility theme
   */
  public applyTheme(theme: IAccessibilityTheme): void {
    this.currentTheme = theme;
    this.updateCSSCustomProperties(theme);
    this.updateThemeClass(theme.name);
    this.applyColorBlindSupport(theme);
  }

  /**
   * Update CSS custom properties with theme values
   */
  private updateCSSCustomProperties(theme: IAccessibilityTheme): void {
    const root = document.documentElement;

    // Apply color properties
    Object.entries(theme.colors).forEach(([key, value]) => {
      root.style.setProperty(`--color-${key}`, value);
    });

    // Apply typography properties
    root.style.setProperty('--font-family', theme.typography.fontFamily);
    
    Object.entries(theme.typography.fontSize).forEach(([key, value]) => {
      root.style.setProperty(`--font-size-${key}`, value);
    });

    Object.entries(theme.typography.fontWeight).forEach(([key, value]) => {
      root.style.setProperty(`--font-weight-${key}`, value.toString());
    });

    Object.entries(theme.typography.lineHeight).forEach(([key, value]) => {
      root.style.setProperty(`--line-height-${key}`, value.toString());
    });

    Object.entries(theme.typography.letterSpacing).forEach(([key, value]) => {
      root.style.setProperty(`--letter-spacing-${key}`, value);
    });

    // Apply spacing properties
    Object.entries(theme.spacing).forEach(([key, value]) => {
      root.style.setProperty(`--spacing-${key}`, value);
    });
  }

  /**
   * Update theme class on document element
   */
  private updateThemeClass(themeName: string): void {
    const root = document.documentElement;
    
    // Remove existing theme classes
    root.className = root.className.replace(/theme-[\w-]+/g, '');
    
    // Add new theme class
    root.classList.add(`theme-${themeName}`);
  }

  /**
   * Apply color blind support
   */
  private applyColorBlindSupport(theme: IAccessibilityTheme): void {
    const { colorBlind } = theme;
    const body = document.body;

    // Reset color blind classes
    body.classList.remove('protanopia', 'deuteranopia', 'tritanopia', 'achromatopsia');
    body.classList.remove('use-patterns', 'use-textures');

    // Apply color blind support
    if (colorBlind.protanopia) body.classList.add('protanopia');
    if (colorBlind.deuteranopia) body.classList.add('deuteranopia');
    if (colorBlind.tritanopia) body.classList.add('tritanopia');
    if (colorBlind.achromatopsia) body.classList.add('achromatopsia');
    if (colorBlind.patterns) body.classList.add('use-patterns');
    if (colorBlind.textures) body.classList.add('use-textures');
  }

  /**
   * Apply display settings
   */
  public applyDisplaySettings(settings: IDisplaySettings): void {
    const body = document.body;
    const root = document.documentElement;

    // Apply class toggles
    body.classList.toggle('reduced-motion', settings.reducedMotion);
    body.classList.toggle('high-contrast', settings.highContrast);
    body.classList.toggle('invert-colors', settings.invertColors);
    body.classList.toggle('grayscale', settings.grayscale);

    // Apply zoom
    root.style.setProperty('--zoom-level', settings.zoom.toString());
    root.style.fontSize = `${16 * settings.zoom}px`;

    // Apply cursor settings
    body.classList.remove('large-cursor', 'extra-large-cursor');
    if (settings.cursorSize === 'large') {
      body.classList.add('large-cursor');
    } else if (settings.cursorSize === 'extra-large') {
      body.classList.add('extra-large-cursor');
    }

    // Apply cursor color
    if (settings.cursorColor !== '#000000') {
      root.style.setProperty('--cursor-color', settings.cursorColor);
    }
  }

  /**
   * Apply font settings
   */
  public applyFontSettings(settings: IFontSettings): void {
    const root = document.documentElement;
    const body = document.body;

    // Apply font family
    if (settings.family === 'dyslexic') {
      body.classList.add('dyslexia-friendly');
      // Load OpenDyslexic font if not already loaded
      this.loadDyslexicFont();
    } else {
      body.classList.remove('dyslexia-friendly');
      const fontFamilies = {
        system: 'system-ui, -apple-system, sans-serif',
        serif: 'Georgia, "Times New Roman", serif',
        mono: '"SF Mono", Consolas, monospace'
      };
      root.style.setProperty('--font-family', fontFamilies[settings.family] || fontFamilies.system);
    }

    // Apply font size scaling
    const sizeMultipliers = {
      xs: 0.75,
      sm: 0.875,
      base: 1,
      lg: 1.125,
      xl: 1.25,
      '2xl': 1.5,
      '3xl': 1.875,
      '4xl': 2.25
    };

    const multiplier = sizeMultipliers[settings.size] || 1;
    root.style.setProperty('--font-size-multiplier', multiplier.toString());

    // Apply to all font sizes
    const baseSizes = {
      xs: 0.75,
      sm: 0.875,
      base: 1,
      lg: 1.125,
      xl: 1.25,
      '2xl': 1.5,
      '3xl': 1.875,
      '4xl': 2.25
    };

    Object.entries(baseSizes).forEach(([key, baseSize]) => {
      root.style.setProperty(`--font-size-${key}`, `${baseSize * multiplier}rem`);
    });

    // Apply font weight
    const weightMap = {
      light: '300',
      normal: '400',
      medium: '500',
      semibold: '600',
      bold: '700'
    };
    root.style.setProperty('--font-weight-base', weightMap[settings.weight] || '400');

    // Apply line height
    const lineHeightMap = {
      tight: '1.25',
      normal: '1.5',
      relaxed: '1.625',
      loose: '2'
    };
    root.style.setProperty('--line-height-base', lineHeightMap[settings.lineHeight] || '1.5');

    // Apply letter spacing
    const letterSpacingMap = {
      tight: '-0.025em',
      normal: '0',
      wide: '0.025em'
    };
    root.style.setProperty('--letter-spacing-base', letterSpacingMap[settings.letterSpacing] || '0');
  }

  /**
   * Load dyslexia-friendly font
   */
  private loadDyslexicFont(): void {
    if (document.getElementById('dyslexic-font')) return;

    const link = document.createElement('link');
    link.id = 'dyslexic-font';
    link.href = 'https://fonts.googleapis.com/css2?family=OpenDyslexic:wght@400;700&display=swap';
    link.rel = 'stylesheet';
    
    // Fallback to system fonts if external font fails
    link.onerror = () => {
      const style = document.createElement('style');
      style.textContent = `
        .dyslexia-friendly,
        .dyslexia-friendly * {
          font-family: 'Comic Sans MS', 'Trebuchet MS', Verdana, sans-serif !important;
        }
      `;
      document.head.appendChild(style);
    };

    document.head.appendChild(link);
  }

  /**
   * Generate theme preview
   */
  public generateThemePreview(theme: IAccessibilityTheme): HTMLElement {
    const preview = document.createElement('div');
    preview.className = 'theme-preview';
    preview.style.cssText = `
      padding: 16px;
      border-radius: 8px;
      background: ${theme.colors.background};
      color: ${theme.colors.text};
      font-family: ${theme.typography.fontFamily};
      border: 2px solid ${theme.colors.border};
    `;

    preview.innerHTML = `
      <h3 style="color: ${theme.colors.primary}; margin: 0 0 8px 0; font-size: ${theme.typography.fontSize.lg};">
        ${theme.displayName}
      </h3>
      <p style="margin: 0 0 12px 0; color: ${theme.colors.textSecondary}; font-size: ${theme.typography.fontSize.sm};">
        ${theme.description}
      </p>
      <div style="display: flex; gap: 8px; align-items: center;">
        <button style="
          background: ${theme.colors.primary};
          color: ${theme.colors.background};
          border: none;
          padding: 8px 16px;
          border-radius: 4px;
          font-size: ${theme.typography.fontSize.sm};
          cursor: pointer;
        ">Primary</button>
        <span style="
          color: ${theme.colors.success};
          font-size: ${theme.typography.fontSize.sm};
        ">Success</span>
        <span style="
          color: ${theme.colors.error};
          font-size: ${theme.typography.fontSize.sm};
        ">Error</span>
      </div>
    `;

    return preview;
  }

  /**
   * Get current theme
   */
  public getCurrentTheme(): IAccessibilityTheme | null {
    return this.currentTheme;
  }

  /**
   * Reset to system defaults
   */
  public resetToSystemDefaults(): void {
    const root = document.documentElement;
    const body = document.body;

    // Remove all theme classes
    body.className = body.className.replace(/\b(theme-[\w-]+|reduced-motion|high-contrast|invert-colors|grayscale|dyslexia-friendly|large-cursor|extra-large-cursor|use-patterns|use-textures|protanopia|deuteranopia|tritanopia|achromatopsia)\b/g, '').trim();

    // Reset custom properties
    const customProperties = Array.from(root.style).filter(prop => prop.startsWith('--'));
    customProperties.forEach(prop => root.style.removeProperty(prop));

    // Redetect system preferences
    this.detectSystemPreferences();
  }

  /**
   * Export current theme settings
   */
  public exportThemeSettings(): {
    theme: IAccessibilityTheme | null;
    customProperties: Record<string, string>;
    classes: string[];
  } {
    const root = document.documentElement;
    const body = document.body;

    // Get custom properties
    const customProperties: Record<string, string> = {};
    Array.from(root.style).forEach(prop => {
      if (prop.startsWith('--')) {
        customProperties[prop] = root.style.getPropertyValue(prop);
      }
    });

    // Get relevant classes
    const relevantClasses = Array.from(body.classList).filter(cls => 
      cls.includes('theme-') || 
      cls.includes('reduced-motion') ||
      cls.includes('high-contrast') ||
      cls.includes('invert-colors') ||
      cls.includes('grayscale') ||
      cls.includes('dyslexia-friendly') ||
      cls.includes('cursor') ||
      cls.includes('protanopia') ||
      cls.includes('deuteranopia') ||
      cls.includes('tritanopia') ||
      cls.includes('achromatopsia')
    );

    return {
      theme: this.currentTheme,
      customProperties,
      classes: relevantClasses
    };
  }

  /**
   * Calculate color contrast ratio
   */
  public calculateContrastRatio(color1: string, color2: string): number {
    const rgb1 = this.hexToRgb(color1);
    const rgb2 = this.hexToRgb(color2);
    
    if (!rgb1 || !rgb2) return 1;

    const l1 = this.getLuminance(rgb1);
    const l2 = this.getLuminance(rgb2);
    
    const lighter = Math.max(l1, l2);
    const darker = Math.min(l1, l2);
    
    return (lighter + 0.05) / (darker + 0.05);
  }

  /**
   * Convert hex to RGB
   */
  private hexToRgb(hex: string): { r: number; g: number; b: number } | null {
    const match = hex.match(/^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i);
    return match ? {
      r: parseInt(match[1], 16),
      g: parseInt(match[2], 16),
      b: parseInt(match[3], 16)
    } : null;
  }

  /**
   * Get relative luminance
   */
  private getLuminance(rgb: { r: number; g: number; b: number }): number {
    const [r, g, b] = [rgb.r, rgb.g, rgb.b].map(c => {
      c = c / 255;
      return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
    });

    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  }

  /**
   * Cleanup resources
   */
  public destroy(): void {
    const baseStyles = document.getElementById('accessibility-base-styles');
    if (baseStyles) {
      document.head.removeChild(baseStyles);
    }

    const dyslexicFont = document.getElementById('dyslexic-font');
    if (dyslexicFont) {
      document.head.removeChild(dyslexicFont);
    }

    this.resetToSystemDefaults();
  }
}