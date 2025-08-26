export interface IAccessibilityOptions {
  highContrast: boolean;
  reducedMotion: boolean;
  screenReader: boolean;
  keyboardNavigation: boolean;
  fontSize: FontSize;
  colorBlindMode: string | null;
  focusIndicators: boolean;
  autoAnnounce: boolean;
}

export interface IKeyboardNavigationConfig {
  skipLinks: boolean;
  focusTrap: boolean;
  customShortcuts: boolean;
  arrowNavigation: boolean;
}

export type FontSize = 'small' | 'medium' | 'large' | 'extra-large';

export interface IAriaLiveRegion {
  politeness: 'polite' | 'assertive' | 'off';
  atomic: boolean;
  relevant: 'additions' | 'removals' | 'text' | 'all';
}

export interface IColorBlindnessOptions {
  type: 'protanopia' | 'deuteranopia' | 'tritanopia' | 'achromatopsia' | null;
  severity: 'mild' | 'moderate' | 'severe';
}

export interface IAccessibilityAuditResult {
  score: number;
  issues: IAccessibilityIssue[];
  recommendations: string[];
  timestamp: Date;
}

export interface IAccessibilityIssue {
  type: 'error' | 'warning' | 'notice';
  element: string;
  description: string;
  wcagLevel: 'A' | 'AA' | 'AAA';
  impact: 'low' | 'medium' | 'high' | 'critical';
  fix: string;
}