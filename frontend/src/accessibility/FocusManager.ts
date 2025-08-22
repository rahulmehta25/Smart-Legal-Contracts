/**
 * Focus Manager
 * Advanced focus management with keyboard navigation, focus trapping,
 * and custom navigation patterns for complex UI components
 */

import {
  IFocusManager,
  IFocusableElement,
  IKeyboardEvent,
  IKeyboardNavigationConfig,
  IFocusTrap,
  KeyboardAction
} from '@/types/accessibility';

export class FocusManager implements IFocusManager {
  private config: IKeyboardNavigationConfig;
  private currentFocus: HTMLElement | null = null;
  private focusTraps: Map<HTMLElement, FocusTrap> = new Map();
  private navigationHistory: HTMLElement[] = [];
  private keyBindings: Map<string, KeyboardAction> = new Map();

  // Default configuration
  private defaultConfig: IKeyboardNavigationConfig = {
    enabled: true,
    focusVisible: true,
    skipLinks: true,
    tabOrder: 'sequential',
    arrowKeyNavigation: true,
    escapeKeyHandling: true,
    enterKeyActivation: true,
    spaceKeyActivation: true,
    customKeyBindings: {}
  };

  constructor(config?: Partial<IKeyboardNavigationConfig>) {
    this.config = { ...this.defaultConfig, ...config };
    this.initializeKeyBindings();
    this.setupEventListeners();
  }

  /**
   * Initialize default key bindings
   */
  private initializeKeyBindings(): void {
    const defaultBindings: Record<string, KeyboardAction> = {
      'Tab': 'focus_next',
      'Shift+Tab': 'focus_previous',
      'Home': 'focus_first',
      'End': 'focus_last',
      'ArrowDown': 'navigate_down',
      'ArrowUp': 'navigate_up',
      'ArrowLeft': 'navigate_left',
      'ArrowRight': 'navigate_right',
      'Enter': 'activate',
      'Space': 'activate',
      'Escape': 'close'
    };

    // Apply default bindings
    Object.entries(defaultBindings).forEach(([key, action]) => {
      this.keyBindings.set(key, action);
    });

    // Apply custom bindings
    Object.entries(this.config.customKeyBindings).forEach(([key, action]) => {
      this.keyBindings.set(key, action);
    });
  }

  /**
   * Setup global event listeners
   */
  private setupEventListeners(): void {
    if (!this.config.enabled) return;

    // Focus events
    document.addEventListener('focusin', this.handleFocusIn.bind(this));
    document.addEventListener('focusout', this.handleFocusOut.bind(this));

    // Keyboard events
    document.addEventListener('keydown', this.handleKeyDown.bind(this));

    // Mouse events (for focus visible)
    if (this.config.focusVisible) {
      document.addEventListener('mousedown', this.handleMouseDown.bind(this));
      document.addEventListener('pointerdown', this.handlePointerDown.bind(this));
    }
  }

  /**
   * Handle focus in events
   */
  private handleFocusIn(event: FocusEvent): void {
    const target = event.target as HTMLElement;
    if (target && this.isElementFocusable(target)) {
      this.currentFocus = target;
      this.addToHistory(target);
      this.showFocusIndicator(target);
    }
  }

  /**
   * Handle focus out events
   */
  private handleFocusOut(event: FocusEvent): void {
    const target = event.target as HTMLElement;
    if (target) {
      this.hideFocusIndicator(target);
    }
  }

  /**
   * Handle mouse down events
   */
  private handleMouseDown(event: MouseEvent): void {
    if (this.config.focusVisible) {
      document.body.classList.add('mouse-navigation');
      document.body.classList.remove('keyboard-navigation');
    }
  }

  /**
   * Handle pointer down events
   */
  private handlePointerDown(event: PointerEvent): void {
    if (event.pointerType !== 'mouse' && this.config.focusVisible) {
      document.body.classList.add('keyboard-navigation');
      document.body.classList.remove('mouse-navigation');
    }
  }

  /**
   * Handle keyboard events
   */
  public handleKeyboardEvent(event: IKeyboardEvent): void {
    if (!this.config.enabled) return;

    const keyCombo = this.getKeyCombo(event);
    const action = this.keyBindings.get(keyCombo);

    if (!action) return;

    // Handle global actions
    switch (action) {
      case 'focus_next':
        if (event.key === 'Tab' && !event.shiftKey) {
          this.focusNext();
          event.preventDefault();
        }
        break;

      case 'focus_previous':
        if (event.key === 'Tab' && event.shiftKey) {
          this.focusPrevious();
          event.preventDefault();
        }
        break;

      case 'focus_first':
        if (this.config.arrowKeyNavigation) {
          this.focusFirst();
          event.preventDefault();
        }
        break;

      case 'focus_last':
        if (this.config.arrowKeyNavigation) {
          this.focusLast();
          event.preventDefault();
        }
        break;

      case 'navigate_down':
      case 'navigate_up':
      case 'navigate_left':
      case 'navigate_right':
        if (this.config.arrowKeyNavigation) {
          this.handleArrowNavigation(action, event);
        }
        break;

      case 'activate':
        if (this.handleActivation(event)) {
          event.preventDefault();
        }
        break;

      case 'close':
        if (this.config.escapeKeyHandling) {
          this.handleEscape(event);
        }
        break;

      default:
        break;
    }

    // Set keyboard navigation mode
    if (this.config.focusVisible) {
      document.body.classList.add('keyboard-navigation');
      document.body.classList.remove('mouse-navigation');
    }
  }

  /**
   * Handle keyboard down events (legacy support)
   */
  private handleKeyDown(event: KeyboardEvent): void {
    this.handleKeyboardEvent({
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
  }

  /**
   * Get key combination string
   */
  private getKeyCombo(event: IKeyboardEvent): string {
    const modifiers: string[] = [];
    if (event.ctrlKey) modifiers.push('Ctrl');
    if (event.altKey) modifiers.push('Alt');
    if (event.metaKey) modifiers.push('Meta');
    if (event.shiftKey) modifiers.push('Shift');

    return [...modifiers, event.key].join('+');
  }

  /**
   * Handle arrow key navigation
   */
  private handleArrowNavigation(action: KeyboardAction, event: IKeyboardEvent): void {
    if (!this.currentFocus) return;

    const container = this.findNavigationContainer(this.currentFocus);
    if (!container) return;

    const elements = this.getFocusableElements(container);
    const currentIndex = elements.findIndex(el => el.element === this.currentFocus);

    if (currentIndex === -1) return;

    let targetIndex = -1;

    // Determine target based on container type and action
    const containerRole = container.getAttribute('role') || '';
    
    if (this.isGridContainer(container)) {
      targetIndex = this.handleGridNavigation(elements, currentIndex, action, container);
    } else if (['menu', 'menubar', 'listbox', 'tree'].includes(containerRole)) {
      targetIndex = this.handleMenuNavigation(elements, currentIndex, action);
    } else if (['tablist'].includes(containerRole)) {
      targetIndex = this.handleTabNavigation(elements, currentIndex, action);
    } else {
      // Default linear navigation
      targetIndex = this.handleLinearNavigation(elements, currentIndex, action);
    }

    if (targetIndex >= 0 && targetIndex < elements.length) {
      this.setFocus(elements[targetIndex].element);
      event.preventDefault();
    }
  }

  /**
   * Check if container is a grid
   */
  private isGridContainer(container: HTMLElement): boolean {
    const role = container.getAttribute('role') || '';
    return ['grid', 'treegrid'].includes(role) || 
           container.tagName.toLowerCase() === 'table';
  }

  /**
   * Handle grid navigation
   */
  private handleGridNavigation(
    elements: IFocusableElement[], 
    currentIndex: number, 
    action: KeyboardAction,
    container: HTMLElement
  ): number {
    const gridInfo = this.calculateGridDimensions(elements, container);
    const { rows, cols } = gridInfo;
    const currentRow = Math.floor(currentIndex / cols);
    const currentCol = currentIndex % cols;

    switch (action) {
      case 'navigate_up':
        return currentRow > 0 ? (currentRow - 1) * cols + currentCol : -1;
      
      case 'navigate_down':
        return currentRow < rows - 1 ? (currentRow + 1) * cols + currentCol : -1;
      
      case 'navigate_left':
        return currentCol > 0 ? currentRow * cols + (currentCol - 1) : -1;
      
      case 'navigate_right':
        return currentCol < cols - 1 ? currentRow * cols + (currentCol + 1) : -1;
      
      default:
        return -1;
    }
  }

  /**
   * Calculate grid dimensions
   */
  private calculateGridDimensions(elements: IFocusableElement[], container: HTMLElement): {
    rows: number;
    cols: number;
  } {
    // For tables, use actual structure
    if (container.tagName.toLowerCase() === 'table') {
      const rows = container.querySelectorAll('tr').length;
      const firstRow = container.querySelector('tr');
      const cols = firstRow ? firstRow.querySelectorAll('td, th').length : 1;
      return { rows, cols };
    }

    // For grids with explicit structure
    const gridRows = container.querySelectorAll('[role="row"]');
    if (gridRows.length > 0) {
      const firstRow = gridRows[0];
      const cols = firstRow.querySelectorAll('[role="gridcell"], [role="columnheader"]').length;
      return { rows: gridRows.length, cols };
    }

    // Estimate based on CSS or default to single row
    const cols = Math.ceil(Math.sqrt(elements.length));
    const rows = Math.ceil(elements.length / cols);
    return { rows, cols };
  }

  /**
   * Handle menu navigation
   */
  private handleMenuNavigation(
    elements: IFocusableElement[], 
    currentIndex: number, 
    action: KeyboardAction
  ): number {
    switch (action) {
      case 'navigate_up':
        return currentIndex > 0 ? currentIndex - 1 : elements.length - 1;
      
      case 'navigate_down':
        return currentIndex < elements.length - 1 ? currentIndex + 1 : 0;
      
      default:
        return -1;
    }
  }

  /**
   * Handle tab navigation
   */
  private handleTabNavigation(
    elements: IFocusableElement[], 
    currentIndex: number, 
    action: KeyboardAction
  ): number {
    switch (action) {
      case 'navigate_left':
        return currentIndex > 0 ? currentIndex - 1 : elements.length - 1;
      
      case 'navigate_right':
        return currentIndex < elements.length - 1 ? currentIndex + 1 : 0;
      
      default:
        return -1;
    }
  }

  /**
   * Handle linear navigation
   */
  private handleLinearNavigation(
    elements: IFocusableElement[], 
    currentIndex: number, 
    action: KeyboardAction
  ): number {
    switch (action) {
      case 'navigate_up':
      case 'navigate_left':
        return currentIndex > 0 ? currentIndex - 1 : -1;
      
      case 'navigate_down':
      case 'navigate_right':
        return currentIndex < elements.length - 1 ? currentIndex + 1 : -1;
      
      default:
        return -1;
    }
  }

  /**
   * Find navigation container for element
   */
  private findNavigationContainer(element: HTMLElement): HTMLElement | null {
    const navigationRoles = [
      'grid', 'treegrid', 'menu', 'menubar', 'listbox', 
      'tree', 'tablist', 'radiogroup', 'toolbar'
    ];

    let current: HTMLElement | null = element;
    while (current) {
      const role = current.getAttribute('role') || '';
      if (navigationRoles.includes(role) || 
          ['table', 'ul', 'ol'].includes(current.tagName.toLowerCase())) {
        return current;
      }
      current = current.parentElement;
    }

    return null;
  }

  /**
   * Handle activation (Enter/Space)
   */
  private handleActivation(event: IKeyboardEvent): boolean {
    if (!this.currentFocus) return false;

    const tagName = this.currentFocus.tagName.toLowerCase();
    const role = this.currentFocus.getAttribute('role') || '';

    // Elements that should be activated with Enter
    const enterActivatable = ['button', 'a', 'menuitem', 'tab', 'treeitem'];
    
    // Elements that should be activated with Space
    const spaceActivatable = ['button', 'checkbox', 'radio', 'menuitem', 'tab'];

    if (event.key === 'Enter' && 
        (enterActivatable.includes(tagName) || enterActivatable.includes(role))) {
      this.currentFocus.click();
      return true;
    }

    if (event.key === ' ' && 
        (spaceActivatable.includes(tagName) || spaceActivatable.includes(role))) {
      this.currentFocus.click();
      return true;
    }

    return false;
  }

  /**
   * Handle Escape key
   */
  private handleEscape(event: IKeyboardEvent): void {
    if (!this.currentFocus) return;

    // Close active focus trap
    const activeTrap = Array.from(this.focusTraps.values()).find(trap => trap.isActive);
    if (activeTrap) {
      activeTrap.deactivate();
      return;
    }

    // Close modals, dialogs, menus
    const closeable = this.currentFocus.closest('[role="dialog"], [role="menu"], .modal, .dropdown');
    if (closeable) {
      const closeButton = closeable.querySelector('[aria-label*="close"], .close, .btn-close');
      if (closeButton instanceof HTMLElement) {
        closeButton.click();
        return;
      }
    }

    // Clear selection in lists
    const selectable = this.currentFocus.closest('[role="listbox"], [role="tree"], [role="grid"]');
    if (selectable) {
      const selected = selectable.querySelectorAll('[aria-selected="true"]');
      selected.forEach(el => el.setAttribute('aria-selected', 'false'));
    }
  }

  /**
   * Get all focusable elements
   */
  public getFocusableElements(container: HTMLElement = document.body): IFocusableElement[] {
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
      .map(el => {
        const element = el as HTMLElement;
        return {
          element,
          tabIndex: element.tabIndex,
          visible: this.isElementVisible(element),
          disabled: this.isElementDisabled(element),
          ariaHidden: element.getAttribute('aria-hidden') === 'true'
        };
      })
      .filter(item => item.visible && !item.disabled && !item.ariaHidden)
      .sort((a, b) => {
        if (a.tabIndex === b.tabIndex) return 0;
        if (a.tabIndex === 0) return 1;
        if (b.tabIndex === 0) return -1;
        if (a.tabIndex < 0) return 1;
        if (b.tabIndex < 0) return -1;
        return a.tabIndex - b.tabIndex;
      });
  }

  /**
   * Set focus to element
   */
  public setFocus(element: HTMLElement): void {
    if (this.isElementFocusable(element)) {
      element.focus();
      this.currentFocus = element;
      this.addToHistory(element);
    }
  }

  /**
   * Focus next element
   */
  public focusNext(): void {
    const focusable = this.getFocusableElements();
    if (focusable.length === 0) return;

    const currentIndex = this.currentFocus ? 
      focusable.findIndex(item => item.element === this.currentFocus) : -1;
    
    const nextIndex = currentIndex < focusable.length - 1 ? currentIndex + 1 : 0;
    this.setFocus(focusable[nextIndex].element);
  }

  /**
   * Focus previous element
   */
  public focusPrevious(): void {
    const focusable = this.getFocusableElements();
    if (focusable.length === 0) return;

    const currentIndex = this.currentFocus ? 
      focusable.findIndex(item => item.element === this.currentFocus) : 0;
    
    const prevIndex = currentIndex > 0 ? currentIndex - 1 : focusable.length - 1;
    this.setFocus(focusable[prevIndex].element);
  }

  /**
   * Focus first element
   */
  public focusFirst(): void {
    const focusable = this.getFocusableElements();
    if (focusable.length > 0) {
      this.setFocus(focusable[0].element);
    }
  }

  /**
   * Focus last element
   */
  public focusLast(): void {
    const focusable = this.getFocusableElements();
    if (focusable.length > 0) {
      this.setFocus(focusable[focusable.length - 1].element);
    }
  }

  /**
   * Create focus trap
   */
  public createFocusTrap(container: HTMLElement): IFocusTrap {
    const trap = new FocusTrap(container, this);
    this.focusTraps.set(container, trap);
    return trap;
  }

  /**
   * Remove focus trap
   */
  public removeFocusTrap(container: HTMLElement): void {
    const trap = this.focusTraps.get(container);
    if (trap) {
      trap.deactivate();
      this.focusTraps.delete(container);
    }
  }

  /**
   * Update configuration
   */
  public updateConfig(config: Partial<IKeyboardNavigationConfig>): void {
    this.config = { ...this.config, ...config };
    this.initializeKeyBindings();
  }

  /**
   * Add element to navigation history
   */
  private addToHistory(element: HTMLElement): void {
    this.navigationHistory = this.navigationHistory.filter(el => el !== element);
    this.navigationHistory.push(element);
    
    // Keep history limited
    if (this.navigationHistory.length > 10) {
      this.navigationHistory.shift();
    }
  }

  /**
   * Show focus indicator
   */
  private showFocusIndicator(element: HTMLElement): void {
    if (this.config.focusVisible) {
      element.classList.add('focus-visible');
    }
  }

  /**
   * Hide focus indicator
   */
  private hideFocusIndicator(element: HTMLElement): void {
    element.classList.remove('focus-visible');
  }

  /**
   * Check if element is visible
   */
  private isElementVisible(element: HTMLElement): boolean {
    const style = window.getComputedStyle(element);
    return !(
      style.display === 'none' ||
      style.visibility === 'hidden' ||
      style.opacity === '0' ||
      element.offsetParent === null
    );
  }

  /**
   * Check if element is disabled
   */
  private isElementDisabled(element: HTMLElement): boolean {
    return 'disabled' in element && (element as any).disabled === true;
  }

  /**
   * Check if element is focusable
   */
  private isElementFocusable(element: HTMLElement): boolean {
    if (!this.isElementVisible(element) || this.isElementDisabled(element)) {
      return false;
    }

    const tabIndex = element.tabIndex;
    if (tabIndex < 0) return false;

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

    return focusableSelectors.some(selector => {
      try {
        return element.matches(selector);
      } catch {
        return false;
      }
    }) || tabIndex >= 0;
  }
}

/**
 * Focus Trap implementation
 */
class FocusTrap implements IFocusTrap {
  private container: HTMLElement;
  private focusManager: FocusManager;
  private _isActive: boolean = false;
  private previousFocus: HTMLElement | null = null;
  private focusableElements: IFocusableElement[] = [];

  constructor(container: HTMLElement, focusManager: FocusManager) {
    this.container = container;
    this.focusManager = focusManager;
  }

  public get isActive(): boolean {
    return this._isActive;
  }

  public activate(): void {
    if (this._isActive) return;

    this.previousFocus = document.activeElement as HTMLElement;
    this.update();
    
    if (this.focusableElements.length > 0) {
      this.focusManager.setFocus(this.focusableElements[0].element);
    }

    this._isActive = true;
    this.container.addEventListener('keydown', this.handleKeyDown.bind(this));
  }

  public deactivate(): void {
    if (!this._isActive) return;

    this._isActive = false;
    this.container.removeEventListener('keydown', this.handleKeyDown.bind(this));

    if (this.previousFocus) {
      this.focusManager.setFocus(this.previousFocus);
    }
  }

  public update(): void {
    this.focusableElements = this.focusManager.getFocusableElements(this.container);
  }

  private handleKeyDown(event: KeyboardEvent): void {
    if (!this._isActive || event.key !== 'Tab') return;

    const currentFocus = document.activeElement as HTMLElement;
    const currentIndex = this.focusableElements.findIndex(item => item.element === currentFocus);

    if (currentIndex === -1) return;

    if (event.shiftKey) {
      // Shift+Tab (backward)
      if (currentIndex === 0) {
        event.preventDefault();
        this.focusManager.setFocus(this.focusableElements[this.focusableElements.length - 1].element);
      }
    } else {
      // Tab (forward)
      if (currentIndex === this.focusableElements.length - 1) {
        event.preventDefault();
        this.focusManager.setFocus(this.focusableElements[0].element);
      }
    }
  }
}