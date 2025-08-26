/**
 * Accessibility Helper
 * Utility functions for accessibility features, ARIA support,
 * color contrast calculations, and element accessibility checks
 */

import {
  IAccessibilityHelper,
  IAriaAttributes,
  AccessibilityErrorCode,
  IAccessibilityError
} from '@/types/accessibility';

export class AccessibilityHelper implements IAccessibilityHelper {
  private uniqueIdCounter: number = 0;
  private readonly WCAG_AA_NORMAL_RATIO = 4.5;
  private readonly WCAG_AA_LARGE_RATIO = 3;
  private readonly WCAG_AAA_NORMAL_RATIO = 7;
  private readonly WCAG_AAA_LARGE_RATIO = 4.5;

  /**
   * Check if element is visible to users
   */
  public isElementVisible(element: HTMLElement): boolean {
    if (!element) return false;

    const style = window.getComputedStyle(element);
    
    // Check CSS visibility
    if (style.visibility === 'hidden' || style.display === 'none') {
      return false;
    }

    // Check if element has zero dimensions
    const rect = element.getBoundingClientRect();
    if (rect.width === 0 && rect.height === 0) {
      return false;
    }

    // Check if element is off-screen
    if (rect.right < 0 || rect.bottom < 0) {
      return false;
    }

    // Check opacity
    if (parseFloat(style.opacity) === 0) {
      return false;
    }

    // Check if element is hidden with clip
    if (style.clip === 'rect(0px, 0px, 0px, 0px)' || style.clipPath === 'inset(100%)') {
      return false;
    }

    return true;
  }

  /**
   * Check if element can receive focus
   */
  public isElementFocusable(element: HTMLElement): boolean {
    if (!element || !this.isElementVisible(element)) {
      return false;
    }

    // Check if element is disabled
    if ('disabled' in element && (element as any).disabled) {
      return false;
    }

    // Check aria-hidden
    if (element.getAttribute('aria-hidden') === 'true') {
      return false;
    }

    // Check tabindex
    const tabIndex = element.tabIndex;
    if (tabIndex < 0) {
      return false;
    }

    // Naturally focusable elements
    const focusableSelectors = [
      'a[href]',
      'button',
      'input:not([type="hidden"])',
      'select',
      'textarea',
      'details',
      '[tabindex]:not([tabindex="-1"])',
      '[contenteditable="true"]'
    ];

    const isFocusableElement = focusableSelectors.some(selector => {
      try {
        return element.matches(selector);
      } catch {
        return false;
      }
    });

    return isFocusableElement || tabIndex >= 0;
  }

  /**
   * Get accessible name for element
   */
  public getAccessibleName(element: HTMLElement): string {
    if (!element) return '';

    // Check aria-labelledby
    const labelledBy = element.getAttribute('aria-labelledby');
    if (labelledBy) {
      const referencedElements = labelledBy.split(/\s+/)
        .map(id => document.getElementById(id))
        .filter(el => el !== null);
      
      if (referencedElements.length > 0) {
        return referencedElements.map(el => el!.textContent?.trim() || '').join(' ');
      }
    }

    // Check aria-label
    const ariaLabel = element.getAttribute('aria-label');
    if (ariaLabel) {
      return ariaLabel.trim();
    }

    // Check associated label elements
    if (element.id) {
      const label = document.querySelector(`label[for="${element.id}"]`);
      if (label) {
        return label.textContent?.trim() || '';
      }
    }

    // Check if element is inside a label
    const parentLabel = element.closest('label');
    if (parentLabel) {
      return parentLabel.textContent?.trim() || '';
    }

    // Check alt attribute for images
    if (element.tagName.toLowerCase() === 'img') {
      const alt = element.getAttribute('alt');
      if (alt !== null) {
        return alt.trim();
      }
    }

    // Check title attribute
    const title = element.getAttribute('title');
    if (title) {
      return title.trim();
    }

    // Check placeholder for input elements
    if (['input', 'textarea'].includes(element.tagName.toLowerCase())) {
      const placeholder = element.getAttribute('placeholder');
      if (placeholder) {
        return placeholder.trim();
      }
    }

    // For buttons and links, use text content
    if (['button', 'a'].includes(element.tagName.toLowerCase())) {
      return element.textContent?.trim() || '';
    }

    // Check value attribute for input buttons
    if (element.tagName.toLowerCase() === 'input' && 
        ['button', 'submit', 'reset'].includes((element as HTMLInputElement).type)) {
      return (element as HTMLInputElement).value || '';
    }

    return element.textContent?.trim() || '';
  }

  /**
   * Get accessible description for element
   */
  public getAccessibleDescription(element: HTMLElement): string {
    if (!element) return '';

    // Check aria-describedby
    const describedBy = element.getAttribute('aria-describedby');
    if (describedBy) {
      const referencedElements = describedBy.split(/\s+/)
        .map(id => document.getElementById(id))
        .filter(el => el !== null);
      
      if (referencedElements.length > 0) {
        return referencedElements.map(el => el!.textContent?.trim() || '').join(' ');
      }
    }

    // Check title attribute (if not already used for name)
    const title = element.getAttribute('title');
    const accessibleName = this.getAccessibleName(element);
    if (title && title !== accessibleName) {
      return title.trim();
    }

    return '';
  }

  /**
   * Get role for element
   */
  public getRole(element: HTMLElement): string | null {
    if (!element) return null;

    // Explicit role attribute
    const explicitRole = element.getAttribute('role');
    if (explicitRole) {
      return explicitRole;
    }

    // Implicit roles based on element type
    const tagName = element.tagName.toLowerCase();
    const implicitRoles: Record<string, string | null> = {
      'button': 'button',
      'a': element.hasAttribute('href') ? 'link' : null,
      'img': element.hasAttribute('alt') ? 'img' : 'presentation',
      'input': this.getInputRole(element as HTMLInputElement),
      'textarea': 'textbox',
      'select': element.hasAttribute('multiple') ? 'listbox' : 'combobox',
      'h1': 'heading',
      'h2': 'heading',
      'h3': 'heading',
      'h4': 'heading',
      'h5': 'heading',
      'h6': 'heading',
      'nav': 'navigation',
      'main': 'main',
      'header': 'banner',
      'footer': 'contentinfo',
      'aside': 'complementary',
      'section': 'region',
      'article': 'article',
      'form': 'form',
      'table': 'table',
      'tbody': 'rowgroup',
      'thead': 'rowgroup',
      'tfoot': 'rowgroup',
      'tr': 'row',
      'td': 'cell',
      'th': 'columnheader',
      'ul': 'list',
      'ol': 'list',
      'li': 'listitem',
      'dl': 'list',
      'dt': 'term',
      'dd': 'definition'
    };

    return implicitRoles[tagName] || null;
  }

  /**
   * Get role for input elements
   */
  private getInputRole(element: HTMLInputElement): string {
    const type = element.type.toLowerCase();
    const roleMap: Record<string, string> = {
      'button': 'button',
      'submit': 'button',
      'reset': 'button',
      'checkbox': 'checkbox',
      'radio': 'radio',
      'range': 'slider',
      'email': 'textbox',
      'password': 'textbox',
      'search': 'searchbox',
      'tel': 'textbox',
      'text': 'textbox',
      'url': 'textbox'
    };

    return roleMap[type] || 'textbox';
  }

  /**
   * Validate tabindex value
   */
  public validateTabIndex(element: HTMLElement): boolean {
    if (!element) return false;

    const tabIndex = element.tabIndex;
    
    // Check if element should not have tabindex
    const nonFocusableElements = ['div', 'span', 'p', 'img'];
    if (nonFocusableElements.includes(element.tagName.toLowerCase()) && 
        tabIndex > 0) {
      return false;
    }

    // Check for valid tabindex values
    if (tabIndex < -1) {
      return false;
    }

    return true;
  }

  /**
   * Check color contrast ratio
   */
  public checkColorContrast(foreground: string, background: string): number {
    const fgLuminance = this.getLuminance(foreground);
    const bgLuminance = this.getLuminance(background);
    
    const lighter = Math.max(fgLuminance, bgLuminance);
    const darker = Math.min(fgLuminance, bgLuminance);
    
    return (lighter + 0.05) / (darker + 0.05);
  }

  /**
   * Calculate relative luminance of a color
   */
  private getLuminance(color: string): number {
    // Convert color to RGB
    const rgb = this.hexToRgb(color) || this.parseRgb(color);
    if (!rgb) return 0;

    // Convert to relative luminance
    const r = rgb.r / 255;
    const g = rgb.g / 255;
    const b = rgb.b / 255;
    
    const rLuminance = r <= 0.03928 ? r / 12.92 : Math.pow((r + 0.055) / 1.055, 2.4);
    const gLuminance = g <= 0.03928 ? g / 12.92 : Math.pow((g + 0.055) / 1.055, 2.4);
    const bLuminance = b <= 0.03928 ? b / 12.92 : Math.pow((b + 0.055) / 1.055, 2.4);

    return 0.2126 * rLuminance + 0.7152 * gLuminance + 0.0722 * bLuminance;
  }

  /**
   * Convert hex color to RGB
   */
  private hexToRgb(hex: string): { r: number; g: number; b: number } | null {
    const match = hex.match(/^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i);
    if (!match || !match[1] || !match[2] || !match[3]) return null;

    return {
      r: parseInt(match[1], 16),
      g: parseInt(match[2], 16),
      b: parseInt(match[3], 16)
    };
  }

  /**
   * Parse RGB color string
   */
  private parseRgb(color: string): { r: number; g: number; b: number } | null {
    const match = color.match(/rgba?\((\d+),\s*(\d+),\s*(\d+)/);
    if (!match || !match[1] || !match[2] || !match[3]) return null;

    return {
      r: parseInt(match[1], 10),
      g: parseInt(match[2], 10),
      b: parseInt(match[3], 10)
    };
  }

  /**
   * Check if contrast meets WCAG guidelines
   */
  public meetsWCAGContrast(
    ratio: number, 
    level: 'A' | 'AA' | 'AAA', 
    size: 'normal' | 'large'
  ): boolean {
    if (level === 'A') return true; // Level A has no contrast requirements

    const thresholds = {
      'AA': {
        normal: this.WCAG_AA_NORMAL_RATIO,
        large: this.WCAG_AA_LARGE_RATIO
      },
      'AAA': {
        normal: this.WCAG_AAA_NORMAL_RATIO,
        large: this.WCAG_AAA_LARGE_RATIO
      }
    };

    return ratio >= thresholds[level][size];
  }

  /**
   * Generate unique ID
   */
  public generateUniqueId(prefix: string = 'a11y'): string {
    return `${prefix}-${++this.uniqueIdCounter}-${Date.now().toString(36)}`;
  }

  /**
   * Set ARIA attributes on element
   */
  public setAriaAttributes(element: HTMLElement, attributes: IAriaAttributes): void {
    Object.entries(attributes).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        element.setAttribute(key, String(value));
      }
    });
  }

  /**
   * Announce text to screen readers
   */
  public announceToScreenReader(text: string, priority: 'polite' | 'assertive' = 'polite'): void {
    const announcement = document.createElement('div');
    announcement.setAttribute('aria-live', priority);
    announcement.setAttribute('aria-atomic', 'true');
    announcement.className = 'sr-only';
    announcement.textContent = text;
    
    // Add to document
    document.body.appendChild(announcement);
    
    // Remove after announcement
    setTimeout(() => {
      if (document.body.contains(announcement)) {
        document.body.removeChild(announcement);
      }
    }, 1000);
  }

  /**
   * Create error for accessibility violation
   */
  public createAccessibilityError(
    code: AccessibilityErrorCode,
    message: string,
    element?: HTMLElement,
    severity: 'low' | 'medium' | 'high' | 'critical' = 'medium'
  ): IAccessibilityError {
    return {
      code,
      message,
      ...(element && { element }),
      severity,
      fixable: this.isFixable(code),
      guideline: this.getWCAGGuideline(code)
    };
  }

  /**
   * Check if error is automatically fixable
   */
  private isFixable(code: AccessibilityErrorCode): boolean {
    const autoFixable = [
      'MISSING_ARIA_LABEL',
      'DUPLICATE_ID',
      'MISSING_FORM_LABELS',
      'INVALID_ROLE'
    ];
    
    return autoFixable.includes(code);
  }

  /**
   * Get WCAG guideline for error code
   */
  private getWCAGGuideline(code: AccessibilityErrorCode): string {
    const guidelines: Record<AccessibilityErrorCode, string> = {
      'MISSING_ARIA_LABEL': 'WCAG 2.1 - 4.1.2 Name, Role, Value',
      'INVALID_ARIA_ATTRIBUTE': 'WCAG 2.1 - 4.1.2 Name, Role, Value',
      'INSUFFICIENT_COLOR_CONTRAST': 'WCAG 2.1 - 1.4.3 Contrast (Minimum)',
      'MISSING_HEADING_STRUCTURE': 'WCAG 2.1 - 1.3.1 Info and Relationships',
      'DUPLICATE_ID': 'WCAG 2.1 - 4.1.1 Parsing',
      'MISSING_ALT_TEXT': 'WCAG 2.1 - 1.1.1 Non-text Content',
      'KEYBOARD_TRAP': 'WCAG 2.1 - 2.1.2 No Keyboard Trap',
      'FOCUS_ORDER_ISSUE': 'WCAG 2.1 - 2.4.3 Focus Order',
      'MISSING_LIVE_REGION': 'WCAG 2.1 - 4.1.3 Status Messages',
      'INVALID_ROLE': 'WCAG 2.1 - 4.1.2 Name, Role, Value',
      'MISSING_FORM_LABELS': 'WCAG 2.1 - 1.3.1 Info and Relationships',
      'TIMEOUT_TOO_SHORT': 'WCAG 2.1 - 2.2.1 Timing Adjustable'
    };

    return guidelines[code] || 'WCAG 2.1 - General';
  }

  /**
   * Get all focusable elements in container
   */
  public getFocusableElements(container: HTMLElement = document.body): HTMLElement[] {
    const focusableSelector = [
      'a[href]',
      'button:not([disabled])',
      'input:not([disabled]):not([type="hidden"])',
      'select:not([disabled])',
      'textarea:not([disabled])',
      'details',
      '[tabindex]:not([tabindex="-1"])',
      '[contenteditable="true"]'
    ].join(', ');

    const elements = container.querySelectorAll(focusableSelector);
    return Array.from(elements)
      .filter(el => this.isElementFocusable(el as HTMLElement))
      .sort((a, b) => {
        const aTabIndex = (a as HTMLElement).tabIndex;
        const bTabIndex = (b as HTMLElement).tabIndex;
        
        if (aTabIndex === bTabIndex) return 0;
        if (aTabIndex === 0) return 1;
        if (bTabIndex === 0) return -1;
        if (aTabIndex < 0) return 1;
        if (bTabIndex < 0) return -1;
        
        return aTabIndex - bTabIndex;
      }) as HTMLElement[];
  }

  /**
   * Check if element has sufficient color contrast
   */
  public checkElementContrast(element: HTMLElement): {
    ratio: number;
    passes: boolean;
    level: 'AA' | 'AAA' | null;
  } {
    const style = window.getComputedStyle(element);
    const color = style.color;
    const backgroundColor = style.backgroundColor;
    
    // If background color is transparent, find parent with background
    let bgColor = backgroundColor;
    let parent = element.parentElement;
    
    while (parent && (bgColor === 'rgba(0, 0, 0, 0)' || bgColor === 'transparent')) {
      const parentStyle = window.getComputedStyle(parent);
      bgColor = parentStyle.backgroundColor;
      parent = parent.parentElement;
    }
    
    if (bgColor === 'rgba(0, 0, 0, 0)' || bgColor === 'transparent') {
      bgColor = '#ffffff'; // Default to white
    }

    const ratio = this.checkColorContrast(color, bgColor);
    
    // Determine if text is large
    const fontSize = parseFloat(style.fontSize);
    const fontWeight = style.fontWeight;
    const isLarge = fontSize >= 18 || (fontSize >= 14 && (fontWeight === 'bold' || parseInt(fontWeight) >= 700));
    
    const passesAA = this.meetsWCAGContrast(ratio, 'AA', isLarge ? 'large' : 'normal');
    const passesAAA = this.meetsWCAGContrast(ratio, 'AAA', isLarge ? 'large' : 'normal');

    return {
      ratio,
      passes: passesAA,
      level: passesAAA ? 'AAA' : passesAA ? 'AA' : null
    };
  }

  /**
   * Fix common accessibility issues automatically
   */
  public autoFixElement(element: HTMLElement): IAccessibilityError[] {
    const fixes: IAccessibilityError[] = [];

    // Fix missing form labels
    if (['input', 'select', 'textarea'].includes(element.tagName.toLowerCase())) {
      const hasLabel = this.getAccessibleName(element);
      if (!hasLabel) {
        const id = element.id || this.generateUniqueId('form-element');
        element.id = id;
        
        // Try to find nearby text that could be a label
        const labelText = this.findNearbyLabelText(element);
        if (labelText) {
          element.setAttribute('aria-label', labelText);
          fixes.push(this.createAccessibilityError(
            'MISSING_FORM_LABELS',
            `Added aria-label: "${labelText}"`,
            element,
            'medium'
          ));
        }
      }
    }

    // Fix missing alt text for images
    if (element.tagName.toLowerCase() === 'img' && !element.hasAttribute('alt')) {
      element.setAttribute('alt', '');
      fixes.push(this.createAccessibilityError(
        'MISSING_ALT_TEXT',
        'Added empty alt attribute for decorative image',
        element,
        'low'
      ));
    }

    // Fix duplicate IDs
    if (element.id) {
      const duplicates = document.querySelectorAll(`#${element.id}`);
      if (duplicates.length > 1) {
        const newId = this.generateUniqueId('element');
        element.id = newId;
        fixes.push(this.createAccessibilityError(
          'DUPLICATE_ID',
          `Changed duplicate ID to: ${newId}`,
          element,
          'high'
        ));
      }
    }

    return fixes;
  }

  /**
   * Find nearby text that could serve as a label
   */
  private findNearbyLabelText(element: HTMLElement): string | null {
    // Check previous sibling text
    let sibling = element.previousSibling;
    while (sibling) {
      if (sibling.nodeType === Node.TEXT_NODE) {
        const text = sibling.textContent?.trim();
        if (text && text.length > 0) {
          return text;
        }
      } else if (sibling.nodeType === Node.ELEMENT_NODE) {
        const text = (sibling as Element).textContent?.trim();
        if (text && text.length > 0 && text.length < 100) {
          return text;
        }
      }
      sibling = sibling.previousSibling;
    }

    // Check parent element text
    const parent = element.parentElement;
    if (parent) {
      const clone = parent.cloneNode(true) as HTMLElement;
      // Remove the input element from clone
      const inputClone = clone.querySelector(element.tagName.toLowerCase());
      if (inputClone) {
        inputClone.remove();
      }
      
      const text = clone.textContent?.trim();
      if (text && text.length > 0 && text.length < 100) {
        return text;
      }
    }

    return null;
  }
}