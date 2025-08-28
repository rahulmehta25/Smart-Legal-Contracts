/**
 * Accessibility Auditor
 * Automated WCAG compliance checking and accessibility violation detection
 */

import {
  IAccessibilityViolation,
  IWCAGComplianceConfig,
  AccessibilityErrorCode
} from '@/types/accessibility';
import { AccessibilityHelper } from './AccessibilityHelper';

export class AccessibilityAuditor {
  private helper: AccessibilityHelper;
  private config: IWCAGComplianceConfig['audit'];

  constructor() {
    this.helper = new AccessibilityHelper();
    this.config = {
      enabled: true,
      autoCheck: true,
      reportViolations: true,
      fixSuggestions: true
    };
  }

  /**
   * Audit entire page for accessibility violations
   */
  public async auditPage(): Promise<IAccessibilityViolation[]> {
    const violations: IAccessibilityViolation[] = [];

    if (!this.config.enabled) return violations;

    // Run all audit checks
    violations.push(...this.checkColorContrast());
    violations.push(...this.checkMissingAltText());
    violations.push(...this.checkFormLabels());
    violations.push(...this.checkHeadingStructure());
    violations.push(...this.checkAriaAttributes());
    violations.push(...this.checkKeyboardAccessibility());
    violations.push(...this.checkDuplicateIds());

    return violations;
  }

  /**
   * Check color contrast violations
   */
  private checkColorContrast(): IAccessibilityViolation[] {
    const violations: IAccessibilityViolation[] = [];
    
    const textElements = document.querySelectorAll('p, span, div, h1, h2, h3, h4, h5, h6, a, button, label, input, textarea');
    
    textElements.forEach(element => {
      const htmlElement = element as HTMLElement;
      const contrast = this.helper.checkElementContrast(htmlElement);
      
      if (!contrast.passes) {
        violations.push({
          id: `contrast-${Date.now()}-${Math.random()}`,
          severity: 'error',
          wcagLevel: 'AA',
          guideline: 'WCAG 2.1 - 1.4.3 Contrast (Minimum)',
          description: `Insufficient color contrast ratio: ${contrast.ratio.toFixed(2)}:1`,
          element: htmlElement.outerHTML.substring(0, 200),
          xpath: this.getXPath(htmlElement),
          suggestions: [
            'Increase contrast between text and background colors',
            'Use darker text on light backgrounds',
            'Use lighter text on dark backgrounds',
            'Consider using high contrast theme'
          ],
          canAutoFix: false
        });
      }
    });

    return violations;
  }

  /**
   * Check for missing alt text on images
   */
  private checkMissingAltText(): IAccessibilityViolation[] {
    const violations: IAccessibilityViolation[] = [];
    
    const images = document.querySelectorAll('img');
    
    images.forEach(img => {
      const alt = img.getAttribute('alt');
      
      if (alt === null) {
        violations.push({
          id: `alt-text-${Date.now()}-${Math.random()}`,
          severity: 'error',
          wcagLevel: 'A',
          guideline: 'WCAG 2.1 - 1.1.1 Non-text Content',
          description: 'Image missing alt attribute',
          element: img.outerHTML,
          xpath: this.getXPath(img),
          suggestions: [
            'Add alt attribute describing the image content',
            'Use empty alt="" for decorative images',
            'Use aria-label for complex images',
            'Consider using figure/figcaption for detailed descriptions'
          ],
          canAutoFix: true
        });
      }
    });

    return violations;
  }

  /**
   * Check for form labels
   */
  private checkFormLabels(): IAccessibilityViolation[] {
    const violations: IAccessibilityViolation[] = [];
    
    const formControls = document.querySelectorAll('input:not([type="hidden"]), select, textarea');
    
    formControls.forEach(control => {
      const htmlControl = control as HTMLElement;
      const accessibleName = this.helper.getAccessibleName(htmlControl);
      
      if (!accessibleName) {
        violations.push({
          id: `form-label-${Date.now()}-${Math.random()}`,
          severity: 'error',
          wcagLevel: 'A',
          guideline: 'WCAG 2.1 - 1.3.1 Info and Relationships',
          description: 'Form control missing accessible name',
          element: htmlControl.outerHTML.substring(0, 200),
          xpath: this.getXPath(htmlControl),
          suggestions: [
            'Add a label element associated with this control',
            'Add aria-label attribute',
            'Add aria-labelledby pointing to descriptive text',
            'Wrap control in a label element'
          ],
          canAutoFix: true
        });
      }
    });

    return violations;
  }

  /**
   * Check heading structure
   */
  private checkHeadingStructure(): IAccessibilityViolation[] {
    const violations: IAccessibilityViolation[] = [];
    
    const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
    let previousLevel = 0;
    let hasH1 = false;

    headings.forEach(heading => {
      const level = parseInt(heading.tagName.charAt(1));
      
      if (level === 1) {
        if (hasH1) {
          violations.push({
            id: `heading-h1-${Date.now()}-${Math.random()}`,
            severity: 'warning',
            wcagLevel: 'AA',
            guideline: 'WCAG 2.1 - 1.3.1 Info and Relationships',
            description: 'Multiple H1 elements found',
            element: heading.outerHTML,
            xpath: this.getXPath(heading as HTMLElement),
            suggestions: [
              'Use only one H1 per page',
              'Convert additional H1s to H2 or appropriate level',
              'Consider using aria-labelledby for sections'
            ],
            canAutoFix: false
          });
        }
        hasH1 = true;
      }

      if (previousLevel > 0 && level > previousLevel + 1) {
        violations.push({
          id: `heading-skip-${Date.now()}-${Math.random()}`,
          severity: 'error',
          wcagLevel: 'AA',
          guideline: 'WCAG 2.1 - 1.3.1 Info and Relationships',
          description: `Heading level skipped: H${previousLevel} to H${level}`,
          element: heading.outerHTML,
          xpath: this.getXPath(heading as HTMLElement),
          suggestions: [
            'Use sequential heading levels (H1, H2, H3...)',
            'Do not skip heading levels',
            'Consider restructuring content hierarchy'
          ],
          canAutoFix: false
        });
      }

      previousLevel = level;
    });

    if (!hasH1 && headings.length > 0) {
      violations.push({
        id: `heading-no-h1-${Date.now()}-${Math.random()}`,
        severity: 'warning',
        wcagLevel: 'AA',
        guideline: 'WCAG 2.1 - 1.3.1 Info and Relationships',
        description: 'No H1 heading found on page',
        element: '',
        xpath: '',
        suggestions: [
          'Add an H1 heading as the main page title',
          'Ensure page has clear content hierarchy',
          'Consider using aria-labelledby on main element'
        ],
        canAutoFix: false
      });
    }

    return violations;
  }

  /**
   * Check ARIA attributes
   */
  private checkAriaAttributes(): IAccessibilityViolation[] {
    const violations: IAccessibilityViolation[] = [];
    
    const ariaElements = document.querySelectorAll('[aria-labelledby], [aria-describedby], [role]');
    
    ariaElements.forEach(element => {
      const htmlElement = element as HTMLElement;
      
      // Check aria-labelledby references
      const labelledBy = htmlElement.getAttribute('aria-labelledby');
      if (labelledBy) {
        const ids = labelledBy.split(/\s+/);
        ids.forEach(id => {
          if (!document.getElementById(id)) {
            violations.push({
              id: `aria-labelledby-${Date.now()}-${Math.random()}`,
              severity: 'error',
              wcagLevel: 'A',
              guideline: 'WCAG 2.1 - 4.1.2 Name, Role, Value',
              description: `aria-labelledby references non-existent element: ${id}`,
              element: htmlElement.outerHTML.substring(0, 200),
              xpath: this.getXPath(htmlElement),
              suggestions: [
                'Ensure referenced element exists',
                'Check element ID spelling',
                'Use aria-label instead if appropriate'
              ],
              canAutoFix: false
            });
          }
        });
      }

      // Check aria-describedby references
      const describedBy = htmlElement.getAttribute('aria-describedby');
      if (describedBy) {
        const ids = describedBy.split(/\s+/);
        ids.forEach(id => {
          if (!document.getElementById(id)) {
            violations.push({
              id: `aria-describedby-${Date.now()}-${Math.random()}`,
              severity: 'warning',
              wcagLevel: 'AA',
              guideline: 'WCAG 2.1 - 4.1.2 Name, Role, Value',
              description: `aria-describedby references non-existent element: ${id}`,
              element: htmlElement.outerHTML.substring(0, 200),
              xpath: this.getXPath(htmlElement),
              suggestions: [
                'Ensure referenced element exists',
                'Check element ID spelling',
                'Remove invalid reference'
              ],
              canAutoFix: false
            });
          }
        });
      }

      // Check invalid roles
      const role = htmlElement.getAttribute('role');
      if (role && !this.isValidRole(role)) {
        violations.push({
          id: `invalid-role-${Date.now()}-${Math.random()}`,
          severity: 'error',
          wcagLevel: 'A',
          guideline: 'WCAG 2.1 - 4.1.2 Name, Role, Value',
          description: `Invalid ARIA role: ${role}`,
          element: htmlElement.outerHTML.substring(0, 200),
          xpath: this.getXPath(htmlElement),
          suggestions: [
            'Use valid ARIA role values',
            'Check ARIA specification for valid roles',
            'Remove role attribute if not needed'
          ],
          canAutoFix: true
        });
      }
    });

    return violations;
  }

  /**
   * Check keyboard accessibility
   */
  private checkKeyboardAccessibility(): IAccessibilityViolation[] {
    const violations: IAccessibilityViolation[] = [];
    
    const interactiveElements = document.querySelectorAll('button, a, input, select, textarea, [role="button"], [role="link"], [role="menuitem"], [role="tab"]');
    
    interactiveElements.forEach(element => {
      const htmlElement = element as HTMLElement;
      
      if (!this.helper.isElementFocusable(htmlElement)) {
        violations.push({
          id: `keyboard-${Date.now()}-${Math.random()}`,
          severity: 'error',
          wcagLevel: 'A',
          guideline: 'WCAG 2.1 - 2.1.1 Keyboard',
          description: 'Interactive element not keyboard accessible',
          element: htmlElement.outerHTML.substring(0, 200),
          xpath: this.getXPath(htmlElement),
          suggestions: [
            'Ensure element has tabindex="0" or is naturally focusable',
            'Remove tabindex="-1" if not intended to be non-focusable',
            'Add keyboard event handlers for custom interactive elements'
          ],
          canAutoFix: true
        });
      }
    });

    return violations;
  }

  /**
   * Check for duplicate IDs
   */
  private checkDuplicateIds(): IAccessibilityViolation[] {
    const violations: IAccessibilityViolation[] = [];
    const idMap = new Map<string, HTMLElement[]>();
    
    const elementsWithIds = document.querySelectorAll('[id]');
    
    elementsWithIds.forEach(element => {
      const htmlElement = element as HTMLElement;
      const id = htmlElement.id;
      
      if (!idMap.has(id)) {
        idMap.set(id, []);
      }
      idMap.get(id)!.push(htmlElement);
    });

    idMap.forEach((elements, id) => {
      if (elements.length > 1) {
        elements.forEach((element, index) => {
          violations.push({
            id: `duplicate-id-${Date.now()}-${Math.random()}`,
            severity: 'error',
            wcagLevel: 'A',
            guideline: 'WCAG 2.1 - 4.1.1 Parsing',
            description: `Duplicate ID found: ${id} (occurrence ${index + 1} of ${elements.length})`,
            element: element.outerHTML.substring(0, 200),
            xpath: this.getXPath(element),
            suggestions: [
              'Ensure all IDs are unique on the page',
              'Use class instead of ID for styling',
              'Generate unique IDs programmatically'
            ],
            canAutoFix: true
          });
        });
      }
    });

    return violations;
  }

  /**
   * Get XPath for element
   */
  private getXPath(element: HTMLElement): string {
    if (element.id) {
      return `//*[@id="${element.id}"]`;
    }

    const parts: string[] = [];
    let current: HTMLElement | null = element;

    while (current && current.nodeType === Node.ELEMENT_NODE && current !== document.body) {
      let selector = current.tagName.toLowerCase();
      
      if (current.className) {
        selector += '.' + Array.from(current.classList).join('.');
      }
      
      const parent = current.parentElement;
      if (parent) {
        const siblings = Array.from(parent.children).filter(child => 
          child.tagName === current!.tagName
        );
        
        if (siblings.length > 1) {
          const index = siblings.indexOf(current) + 1;
          selector += `[${index}]`;
        }
      }
      
      parts.unshift(selector);
      current = parent;
    }

    return '/' + parts.join('/');
  }

  /**
   * Check if role is valid
   */
  private isValidRole(role: string): boolean {
    const validRoles = [
      'alert', 'alertdialog', 'application', 'article', 'banner', 'button',
      'cell', 'checkbox', 'columnheader', 'combobox', 'complementary',
      'contentinfo', 'definition', 'dialog', 'directory', 'document',
      'feed', 'figure', 'form', 'grid', 'gridcell', 'group', 'heading',
      'img', 'link', 'list', 'listbox', 'listitem', 'log', 'main',
      'marquee', 'math', 'menu', 'menubar', 'menuitem', 'menuitemcheckbox',
      'menuitemradio', 'navigation', 'none', 'note', 'option', 'presentation',
      'progressbar', 'radio', 'radiogroup', 'region', 'row', 'rowgroup',
      'rowheader', 'scrollbar', 'search', 'searchbox', 'separator',
      'slider', 'spinbutton', 'status', 'switch', 'tab', 'table',
      'tablist', 'tabpanel', 'term', 'textbox', 'timer', 'toolbar',
      'tooltip', 'tree', 'treegrid', 'treeitem'
    ];

    return validRoles.includes(role);
  }

  /**
   * Update configuration
   */
  public updateConfig(config: Partial<IWCAGComplianceConfig['audit']>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Get audit statistics
   */
  public getAuditStatistics(violations: IAccessibilityViolation[]): {
    total: number;
    bySeverity: Record<string, number>;
    byWCAGLevel: Record<string, number>;
    fixable: number;
  } {
    const stats = {
      total: violations.length,
      bySeverity: { error: 0, warning: 0, info: 0 },
      byWCAGLevel: { A: 0, AA: 0, AAA: 0 },
      fixable: 0
    };

    violations.forEach(violation => {
      stats.bySeverity[violation.severity]++;
      stats.byWCAGLevel[violation.wcagLevel]++;
      if (violation.canAutoFix) {
        stats.fixable++;
      }
    });

    return stats;
  }
}