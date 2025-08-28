/**
 * Accessibility Types
 * Comprehensive type definitions for accessibility features including screen reader support,
 * keyboard navigation, themes, and WCAG compliance
 */

// Screen Reader Types
export interface IScreenReaderConfig {
  enabled: boolean;
  announcePageChanges: boolean;
  announceFormErrors: boolean;
  announceLoadingStates: boolean;
  verbosity: ScreenReaderVerbosity;
  skipToContentEnabled: boolean;
  landmarkNavigation: boolean;
}

export type ScreenReaderVerbosity = 'minimal' | 'normal' | 'verbose';

export interface IAriaAttributes {
  'aria-label'?: string;
  'aria-labelledby'?: string;
  'aria-describedby'?: string;
  'aria-expanded'?: boolean;
  'aria-hidden'?: boolean;
  'aria-live'?: 'off' | 'polite' | 'assertive';
  'aria-atomic'?: boolean;
  'aria-busy'?: boolean;
  'aria-controls'?: string;
  'aria-current'?: boolean | 'page' | 'step' | 'location' | 'date' | 'time';
  'aria-disabled'?: boolean;
  'aria-invalid'?: boolean | 'grammar' | 'spelling';
  'aria-multiselectable'?: boolean;
  'aria-orientation'?: 'horizontal' | 'vertical';
  'aria-pressed'?: boolean;
  'aria-readonly'?: boolean;
  'aria-required'?: boolean;
  'aria-selected'?: boolean;
  'aria-sort'?: 'none' | 'ascending' | 'descending' | 'other';
  'aria-valuemax'?: number;
  'aria-valuemin'?: number;
  'aria-valuenow'?: number;
  'aria-valuetext'?: string;
  role?: AriaRole;
}

export type AriaRole =
  | 'alert'
  | 'application'
  | 'article'
  | 'banner'
  | 'button'
  | 'checkbox'
  | 'complementary'
  | 'contentinfo'
  | 'dialog'
  | 'document'
  | 'form'
  | 'grid'
  | 'gridcell'
  | 'heading'
  | 'img'
  | 'link'
  | 'list'
  | 'listitem'
  | 'main'
  | 'menu'
  | 'menuitem'
  | 'navigation'
  | 'option'
  | 'progressbar'
  | 'radio'
  | 'region'
  | 'search'
  | 'slider'
  | 'spinbutton'
  | 'status'
  | 'tab'
  | 'tablist'
  | 'tabpanel'
  | 'textbox'
  | 'timer'
  | 'tooltip'
  | 'tree'
  | 'treeitem';

export interface ILiveRegionConfig {
  id: string;
  politeness: 'polite' | 'assertive';
  atomic: boolean;
  relevant: 'additions' | 'removals' | 'text' | 'all';
  autoAnnounce: boolean;
}

// Keyboard Navigation Types
export interface IKeyboardNavigationConfig {
  enabled: boolean;
  focusVisible: boolean;
  skipLinks: boolean;
  tabOrder: 'sequential' | 'logical' | 'custom';
  arrowKeyNavigation: boolean;
  escapeKeyHandling: boolean;
  enterKeyActivation: boolean;
  spaceKeyActivation: boolean;
  customKeyBindings: Record<string, KeyboardAction>;
}

export type KeyboardAction =
  | 'focus_next'
  | 'focus_previous'
  | 'focus_first'
  | 'focus_last'
  | 'activate'
  | 'close'
  | 'expand'
  | 'collapse'
  | 'select'
  | 'navigate_up'
  | 'navigate_down'
  | 'navigate_left'
  | 'navigate_right'
  | 'page_up'
  | 'page_down'
  | 'home'
  | 'end';

export interface IKeyboardEvent {
  key: string;
  code: string;
  altKey: boolean;
  ctrlKey: boolean;
  metaKey: boolean;
  shiftKey: boolean;
  target: HTMLElement;
  preventDefault: () => void;
  stopPropagation: () => void;
}

export interface IFocusableElement {
  element: HTMLElement;
  tabIndex: number;
  visible: boolean;
  disabled: boolean;
  ariaHidden: boolean;
}

export interface IFocusManager {
  getFocusableElements: () => IFocusableElement[];
  setFocus: (element: HTMLElement) => void;
  focusNext: () => void;
  focusPrevious: () => void;
  focusFirst: () => void;
  focusLast: () => void;
  createFocusTrap: (container: HTMLElement) => IFocusTrap;
}

export interface IFocusTrap {
  activate: () => void;
  deactivate: () => void;
  update: () => void;
  isActive: boolean;
}

export interface ISkipLink {
  id: string;
  href: string;
  text: string;
  visible: boolean;
  order: number;
}

// Theme and Visual Accessibility Types
export interface IAccessibilityTheme {
  id: string;
  name: string;
  displayName: string;
  description: string;
  colors: IThemeColors;
  typography: IThemeTypography;
  spacing: IThemeSpacing;
  contrast: IThemeContrast;
  colorBlind: IColorBlindSupport;
}

export interface IThemeColors {
  primary: string;
  secondary: string;
  background: string;
  surface: string;
  text: string;
  textSecondary: string;
  border: string;
  focus: string;
  error: string;
  warning: string;
  success: string;
  info: string;
  disabled: string;
}

export interface IThemeTypography {
  fontFamily: string;
  fontSize: {
    xs: string;
    sm: string;
    base: string;
    lg: string;
    xl: string;
    '2xl': string;
    '3xl': string;
    '4xl': string;
  };
  fontWeight: {
    light: number;
    normal: number;
    medium: number;
    semibold: number;
    bold: number;
  };
  lineHeight: {
    tight: number;
    normal: number;
    relaxed: number;
    loose: number;
  };
  letterSpacing: {
    tight: string;
    normal: string;
    wide: string;
  };
}

export interface IThemeSpacing {
  xs: string;
  sm: string;
  md: string;
  lg: string;
  xl: string;
  '2xl': string;
  '3xl': string;
  '4xl': string;
}

export interface IThemeContrast {
  level: 'AA' | 'AAA';
  ratios: {
    normal: number;
    large: number;
    interactive: number;
  };
  enhanced: boolean;
}

export interface IColorBlindSupport {
  protanopia: boolean;
  deuteranopia: boolean;
  tritanopia: boolean;
  achromatopsia: boolean;
  patterns: boolean;
  textures: boolean;
}

export type AccessibilityThemeType = 
  | 'default'
  | 'high-contrast'
  | 'dark'
  | 'high-contrast-dark'
  | 'low-vision'
  | 'protanopia'
  | 'deuteranopia'
  | 'tritanopia';

// Font and Display Types
export interface IFontSettings {
  size: FontSize;
  family: FontFamily;
  weight: FontWeight;
  lineHeight: LineHeight;
  letterSpacing: LetterSpacing;
  dyslexiaFriendly: boolean;
}

export type FontSize = 'xs' | 'sm' | 'base' | 'lg' | 'xl' | '2xl' | '3xl' | '4xl';
export type FontFamily = 'system' | 'serif' | 'mono' | 'dyslexic';
export type FontWeight = 'light' | 'normal' | 'medium' | 'semibold' | 'bold';
export type LineHeight = 'tight' | 'normal' | 'relaxed' | 'loose';
export type LetterSpacing = 'tight' | 'normal' | 'wide';

export interface IDisplaySettings {
  reducedMotion: boolean;
  highContrast: boolean;
  invertColors: boolean;
  grayscale: boolean;
  zoom: number;
  cursorSize: CursorSize;
  cursorColor: string;
}

export type CursorSize = 'normal' | 'large' | 'extra-large';

// Motor Accessibility Types
export interface IMotorAccessibilityConfig {
  stickyKeys: boolean;
  slowKeys: boolean;
  bounceKeys: boolean;
  mouseKeys: boolean;
  clickAssist: boolean;
  dragAssist: boolean;
  hoverDelay: number;
  clickDelay: number;
  doubleClickSpeed: number;
  targetSize: TargetSize;
}

export type TargetSize = 'small' | 'medium' | 'large' | 'extra-large';

export interface IClickAssistConfig {
  enabled: boolean;
  dwellTime: number;
  showIndicator: boolean;
  soundFeedback: boolean;
}

export interface IDragAssistConfig {
  enabled: boolean;
  snapToGrid: boolean;
  gridSize: number;
  magneticEdges: boolean;
  dropZoneHighlight: boolean;
}

// Cognitive Accessibility Types
export interface ICognitiveAccessibilityConfig {
  simplifiedInterface: boolean;
  reducedClutter: boolean;
  breadcrumbs: boolean;
  progressIndicators: boolean;
  confirmationDialogs: boolean;
  timeoutWarnings: boolean;
  autoSave: boolean;
  errorPrevention: boolean;
}

export interface ITimeoutConfig {
  enabled: boolean;
  duration: number;
  warningTime: number;
  extendable: boolean;
  showProgress: boolean;
}

export interface IErrorPreventionConfig {
  validateOnBlur: boolean;
  validateOnChange: boolean;
  showInlineHelp: boolean;
  requireConfirmation: boolean;
  preventAccidentalSubmission: boolean;
}

// WCAG Compliance Types
export interface IWCAGComplianceConfig {
  level: 'A' | 'AA' | 'AAA';
  guidelines: {
    perceivable: IPerceivableGuidelines;
    operable: IOperableGuidelines;
    understandable: IUnderstandableGuidelines;
    robust: IRobustGuidelines;
  };
  audit: {
    enabled: boolean;
    autoCheck: boolean;
    reportViolations: boolean;
    fixSuggestions: boolean;
  };
}

export interface IPerceivableGuidelines {
  textAlternatives: boolean;
  timeBasedMedia: boolean;
  adaptable: boolean;
  distinguishable: boolean;
}

export interface IOperableGuidelines {
  keyboardAccessible: boolean;
  seizuresAndPhysicalReactions: boolean;
  navigable: boolean;
  inputModalities: boolean;
}

export interface IUnderstandableGuidelines {
  readable: boolean;
  predictable: boolean;
  inputAssistance: boolean;
}

export interface IRobustGuidelines {
  compatible: boolean;
}

export interface IAccessibilityViolation {
  id: string;
  severity: 'error' | 'warning' | 'info';
  wcagLevel: 'A' | 'AA' | 'AAA';
  guideline: string;
  description: string;
  element: string;
  xpath: string;
  suggestions: string[];
  canAutoFix: boolean;
}

// Accessibility Context Types
export interface IAccessibilityContextValue {
  config: IAccessibilityConfig;
  theme: IAccessibilityTheme;
  updateConfig: (config: Partial<IAccessibilityConfig>) => void;
  updateTheme: (theme: AccessibilityThemeType) => void;
  announceToScreenReader: (message: string, priority?: 'polite' | 'assertive') => void;
  focusManager: IFocusManager;
  isReducedMotion: boolean;
  isHighContrast: boolean;
  currentFontSize: FontSize;
  violations: IAccessibilityViolation[];
}

export interface IAccessibilityConfig {
  screenReader: IScreenReaderConfig;
  keyboard: IKeyboardNavigationConfig;
  motor: IMotorAccessibilityConfig;
  cognitive: ICognitiveAccessibilityConfig;
  display: IDisplaySettings;
  font: IFontSettings;
  wcag: IWCAGComplianceConfig;
  voice: {
    enabled: boolean;
    commands: boolean;
    feedback: boolean;
  };
}

// Component Props Types
export interface IAccessibilityProviderProps {
  children: React.ReactNode;
  initialConfig?: Partial<IAccessibilityConfig>;
  initialTheme?: AccessibilityThemeType;
}

export interface ISkipNavigationProps {
  links: ISkipLink[];
  className?: string;
  id?: string;
}

export interface IFocusIndicatorProps {
  visible: boolean;
  target: HTMLElement | null;
  className?: string;
}

export interface ILiveRegionProps {
  children: React.ReactNode;
  politeness?: 'polite' | 'assertive';
  atomic?: boolean;
  relevant?: 'additions' | 'removals' | 'text' | 'all';
  className?: string;
  id?: string;
}

export interface IAccessibilityControlsProps {
  position?: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';
  collapsible?: boolean;
  className?: string;
  id?: string;
}

// Hooks Types
export interface IUseAccessibilityReturn {
  config: IAccessibilityConfig;
  theme: IAccessibilityTheme;
  updateConfig: (config: Partial<IAccessibilityConfig>) => void;
  updateTheme: (theme: AccessibilityThemeType) => void;
  announceToScreenReader: (message: string, priority?: 'polite' | 'assertive') => void;
  violations: IAccessibilityViolation[];
}

export interface IUseKeyboardNavigationReturn {
  handleKeyDown: (event: React.KeyboardEvent) => void;
  focusManager: IFocusManager;
  currentFocus: HTMLElement | null;
  trapFocus: (container: HTMLElement) => IFocusTrap;
}

export interface IUseScreenReaderReturn {
  announce: (message: string, priority?: 'polite' | 'assertive') => void;
  isScreenReaderActive: boolean;
  addLiveRegion: (config: ILiveRegionConfig) => string;
  removeLiveRegion: (id: string) => void;
}

export interface IUseReducedMotionReturn {
  prefersReducedMotion: boolean;
  shouldAnimate: boolean;
}

export interface IUseColorContrastReturn {
  checkContrast: (foreground: string, background: string) => number;
  meetsWCAG: (ratio: number, level: 'A' | 'AA' | 'AAA', size: 'normal' | 'large') => boolean;
  suggestColors: (baseColor: string) => string[];
}

// Utility Types
export interface IAccessibilityHelper {
  isElementVisible: (element: HTMLElement) => boolean;
  isElementFocusable: (element: HTMLElement) => boolean;
  getAccessibleName: (element: HTMLElement) => string;
  getAccessibleDescription: (element: HTMLElement) => string;
  getRole: (element: HTMLElement) => string | null;
  validateTabIndex: (element: HTMLElement) => boolean;
  checkColorContrast: (foreground: string, background: string) => number;
  generateUniqueId: (prefix?: string) => string;
}

export type AccessibilityEventType =
  | 'config_changed'
  | 'theme_changed'
  | 'focus_changed'
  | 'violation_detected'
  | 'screen_reader_announcement'
  | 'keyboard_navigation'
  | 'error_occurred';

export interface IAccessibilityEvent {
  type: AccessibilityEventType;
  payload: Record<string, unknown>;
  timestamp: Date;
  element?: HTMLElement;
}

// Error Types
export interface IAccessibilityError {
  code: AccessibilityErrorCode;
  message: string;
  element?: HTMLElement;
  guideline?: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  fixable: boolean;
}

export type AccessibilityErrorCode =
  | 'MISSING_ARIA_LABEL'
  | 'INVALID_ARIA_ATTRIBUTE'
  | 'INSUFFICIENT_COLOR_CONTRAST'
  | 'MISSING_HEADING_STRUCTURE'
  | 'DUPLICATE_ID'
  | 'MISSING_ALT_TEXT'
  | 'KEYBOARD_TRAP'
  | 'FOCUS_ORDER_ISSUE'
  | 'MISSING_LIVE_REGION'
  | 'INVALID_ROLE'
  | 'MISSING_FORM_LABELS'
  | 'TIMEOUT_TOO_SHORT';