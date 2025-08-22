/**
 * Screen Reader Manager
 * Comprehensive screen reader support with live regions, ARIA management,
 * and intelligent announcement system
 */

import {
  IScreenReaderConfig,
  ILiveRegionConfig,
  ScreenReaderVerbosity
} from '@/types/accessibility';

export class ScreenReaderManager {
  private config: IScreenReaderConfig;
  private liveRegions: Map<string, HTMLElement> = new Map();
  private announcementQueue: Array<{
    message: string;
    priority: 'polite' | 'assertive';
    timestamp: number;
  }> = [];
  private isProcessingQueue: boolean = false;

  // Default configuration
  private defaultConfig: IScreenReaderConfig = {
    enabled: true,
    announcePageChanges: true,
    announceFormErrors: true,
    announceLoadingStates: true,
    verbosity: 'normal',
    skipToContentEnabled: true,
    landmarkNavigation: true
  };

  constructor(config?: Partial<IScreenReaderConfig>) {
    this.config = { ...this.defaultConfig, ...config };
  }

  /**
   * Initialize screen reader support
   */
  public initialize(config?: Partial<IScreenReaderConfig>): void {
    if (config) {
      this.updateConfig(config);
    }

    this.setupLiveRegions();
    this.setupSkipNavigation();
    this.setupPageChangeAnnouncement();
    this.setupFormErrorAnnouncement();
    this.setupLoadingStateAnnouncement();
    this.enhanceExistingContent();
  }

  /**
   * Update configuration
   */
  public updateConfig(config: Partial<IScreenReaderConfig>): void {
    this.config = { ...this.config, ...config };
  }

  /**
   * Announce message to screen readers
   */
  public announce(message: string, priority: 'polite' | 'assertive' = 'polite'): void {
    if (!this.config.enabled || !message.trim()) {
      return;
    }

    const processedMessage = this.processMessage(message);
    
    this.announcementQueue.push({
      message: processedMessage,
      priority,
      timestamp: Date.now()
    });

    if (!this.isProcessingQueue) {
      this.processAnnouncementQueue();
    }
  }

  /**
   * Process message based on verbosity settings
   */
  private processMessage(message: string): string {
    switch (this.config.verbosity) {
      case 'minimal':
        // Remove extra details and keep only essential information
        return message
          .replace(/\b(please|kindly|exactly|precisely)\b/gi, '')
          .replace(/\s+/g, ' ')
          .trim();

      case 'verbose':
        // Add contextual information
        const context = this.getCurrentContext();
        return context ? `${context}: ${message}` : message;

      case 'normal':
      default:
        return message;
    }
  }

  /**
   * Get current context for verbose announcements
   */
  private getCurrentContext(): string | null {
    const activeElement = document.activeElement;
    if (!activeElement) return null;

    // Get landmark or section context
    const landmark = activeElement.closest('[role="main"], [role="banner"], [role="navigation"], [role="complementary"], [role="contentinfo"], main, nav, header, footer, aside');
    if (landmark) {
      const role = landmark.getAttribute('role') || landmark.tagName.toLowerCase();
      const label = landmark.getAttribute('aria-label') || 
                   landmark.getAttribute('aria-labelledby');
      
      if (label) {
        return `In ${role} ${label}`;
      }
      return `In ${role}`;
    }

    // Get form context
    const form = activeElement.closest('form');
    if (form) {
      const formName = form.getAttribute('aria-label') || 
                      form.getAttribute('name') || 
                      'form';
      return `In ${formName}`;
    }

    return null;
  }

  /**
   * Process announcement queue
   */
  private async processAnnouncementQueue(): Promise<void> {
    if (this.isProcessingQueue || this.announcementQueue.length === 0) {
      return;
    }

    this.isProcessingQueue = true;

    while (this.announcementQueue.length > 0) {
      const announcement = this.announcementQueue.shift();
      if (announcement) {
        await this.makeAnnouncement(announcement.message, announcement.priority);
        // Brief delay between announcements
        await this.delay(100);
      }
    }

    this.isProcessingQueue = false;
  }

  /**
   * Make announcement using live regions
   */
  private async makeAnnouncement(message: string, priority: 'polite' | 'assertive'): Promise<void> {
    const regionId = priority === 'assertive' ? 'assertive-live-region' : 'polite-live-region';
    let liveRegion = this.liveRegions.get(regionId);

    if (!liveRegion) {
      liveRegion = this.createLiveRegion({
        id: regionId,
        politeness: priority,
        atomic: true,
        relevant: 'all',
        autoAnnounce: true
      });
    }

    // Clear previous message
    liveRegion.textContent = '';
    
    // Brief delay to ensure screen readers notice the change
    await this.delay(10);
    
    // Set new message
    liveRegion.textContent = message;
  }

  /**
   * Create live region
   */
  public createLiveRegion(config: ILiveRegionConfig): HTMLElement {
    const existing = this.liveRegions.get(config.id);
    if (existing) {
      return existing;
    }

    const liveRegion = document.createElement('div');
    liveRegion.id = config.id;
    liveRegion.className = 'sr-only live-region';
    liveRegion.setAttribute('aria-live', config.politeness);
    liveRegion.setAttribute('aria-atomic', config.atomic.toString());
    liveRegion.setAttribute('aria-relevant', config.relevant);

    // Add to document
    document.body.appendChild(liveRegion);
    this.liveRegions.set(config.id, liveRegion);

    return liveRegion;
  }

  /**
   * Remove live region
   */
  public removeLiveRegion(id: string): void {
    const liveRegion = this.liveRegions.get(id);
    if (liveRegion) {
      document.body.removeChild(liveRegion);
      this.liveRegions.delete(id);
    }
  }

  /**
   * Setup default live regions
   */
  private setupLiveRegions(): void {
    // Polite live region for general announcements
    this.createLiveRegion({
      id: 'polite-live-region',
      politeness: 'polite',
      atomic: true,
      relevant: 'all',
      autoAnnounce: true
    });

    // Assertive live region for urgent announcements
    this.createLiveRegion({
      id: 'assertive-live-region',
      politeness: 'assertive',
      atomic: true,
      relevant: 'all',
      autoAnnounce: true
    });

    // Status region for form validation and state changes
    this.createLiveRegion({
      id: 'status-live-region',
      politeness: 'polite',
      atomic: false,
      relevant: 'additions text',
      autoAnnounce: true
    });
  }

  /**
   * Setup skip navigation
   */
  private setupSkipNavigation(): void {
    if (!this.config.skipToContentEnabled) return;

    const existingSkipNav = document.getElementById('skip-navigation');
    if (existingSkipNav) return;

    const skipNav = document.createElement('div');
    skipNav.id = 'skip-navigation';
    skipNav.className = 'skip-navigation';
    
    // Find main content
    const mainContent = document.querySelector('main, [role="main"], #main, .main');
    const mainId = mainContent?.id || 'main-content';
    
    if (mainContent && !mainContent.id) {
      mainContent.id = mainId;
    }

    skipNav.innerHTML = `
      <a href="#${mainId}" class="skip-link">
        Skip to main content
      </a>
    `;

    // Insert at beginning of body
    document.body.insertBefore(skipNav, document.body.firstChild);

    // Add CSS for skip links
    this.addSkipLinkStyles();
  }

  /**
   * Add CSS styles for skip links
   */
  private addSkipLinkStyles(): void {
    const existingStyles = document.getElementById('skip-link-styles');
    if (existingStyles) return;

    const style = document.createElement('style');
    style.id = 'skip-link-styles';
    style.textContent = `
      .skip-navigation {
        position: absolute;
        top: -40px;
        left: 6px;
        z-index: 1000;
      }
      
      .skip-link {
        position: absolute;
        left: -10000px;
        top: auto;
        width: 1px;
        height: 1px;
        overflow: hidden;
        background: #000;
        color: #fff;
        padding: 8px 24px;
        text-decoration: none;
        font-weight: 600;
        border-radius: 4px;
        font-size: 14px;
        line-height: 1.4;
      }
      
      .skip-link:focus {
        position: static;
        left: auto;
        top: auto;
        width: auto;
        height: auto;
        overflow: visible;
      }
      
      .sr-only {
        position: absolute;
        left: -10000px;
        top: auto;
        width: 1px;
        height: 1px;
        overflow: hidden;
      }
      
      .live-region {
        position: absolute;
        left: -10000px;
        top: auto;
        width: 1px;
        height: 1px;
        overflow: hidden;
      }
    `;

    document.head.appendChild(style);
  }

  /**
   * Setup page change announcement
   */
  private setupPageChangeAnnouncement(): void {
    if (!this.config.announcePageChanges) return;

    // Listen for URL changes (SPA navigation)
    let currentUrl = window.location.href;
    
    const checkUrlChange = () => {
      if (window.location.href !== currentUrl) {
        currentUrl = window.location.href;
        this.announcePageChange();
      }
    };

    // Check for URL changes
    setInterval(checkUrlChange, 100);

    // Listen for popstate events
    window.addEventListener('popstate', () => {
      setTimeout(() => this.announcePageChange(), 100);
    });

    // Override pushState and replaceState
    const originalPushState = history.pushState.bind(history);
    const originalReplaceState = history.replaceState.bind(history);

    history.pushState = (...args) => {
      originalPushState(...args);
      setTimeout(() => this.announcePageChange(), 100);
    };

    history.replaceState = (...args) => {
      originalReplaceState(...args);
      setTimeout(() => this.announcePageChange(), 100);
    };
  }

  /**
   * Announce page change
   */
  private announcePageChange(): void {
    const title = document.title;
    const heading = document.querySelector('h1');
    const headingText = heading?.textContent?.trim();

    let announcement = '';
    
    if (headingText && headingText !== title) {
      announcement = `Page changed to ${title}. Main heading: ${headingText}`;
    } else {
      announcement = `Page changed to ${title}`;
    }

    this.announce(announcement, 'assertive');
  }

  /**
   * Setup form error announcement
   */
  private setupFormErrorAnnouncement(): void {
    if (!this.config.announceFormErrors) return;

    document.addEventListener('invalid', (event) => {
      const element = event.target as HTMLElement;
      if (element && 'validity' in element) {
        const validity = (element as any).validity;
        const message = this.getValidationMessage(element, validity);
        this.announce(`Error: ${message}`, 'assertive');
      }
    });

    // Listen for custom error announcements
    document.addEventListener('form-error', (event) => {
      const customEvent = event as CustomEvent;
      if (customEvent.detail?.message) {
        this.announce(`Error: ${customEvent.detail.message}`, 'assertive');
      }
    });
  }

  /**
   * Get validation message for form field
   */
  private getValidationMessage(element: HTMLElement, validity: ValidityState): string {
    const fieldName = this.getFieldName(element);

    if (validity.valueMissing) {
      return `${fieldName} is required`;
    }
    if (validity.typeMismatch) {
      const type = (element as HTMLInputElement).type;
      return `Please enter a valid ${type} for ${fieldName}`;
    }
    if (validity.tooShort) {
      const minLength = (element as HTMLInputElement).minLength;
      return `${fieldName} must be at least ${minLength} characters`;
    }
    if (validity.tooLong) {
      const maxLength = (element as HTMLInputElement).maxLength;
      return `${fieldName} must be no more than ${maxLength} characters`;
    }
    if (validity.rangeUnderflow) {
      const min = (element as HTMLInputElement).min;
      return `${fieldName} must be at least ${min}`;
    }
    if (validity.rangeOverflow) {
      const max = (element as HTMLInputElement).max;
      return `${fieldName} must be no more than ${max}`;
    }
    if (validity.patternMismatch) {
      return `${fieldName} format is not valid`;
    }

    return `${fieldName} is not valid`;
  }

  /**
   * Get field name for error messages
   */
  private getFieldName(element: HTMLElement): string {
    // Check aria-label
    const ariaLabel = element.getAttribute('aria-label');
    if (ariaLabel) return ariaLabel;

    // Check associated label
    if (element.id) {
      const label = document.querySelector(`label[for="${element.id}"]`);
      if (label?.textContent) return label.textContent.trim();
    }

    // Check parent label
    const parentLabel = element.closest('label');
    if (parentLabel?.textContent) return parentLabel.textContent.trim();

    // Check name attribute
    const name = element.getAttribute('name');
    if (name) return name.replace(/[_-]/g, ' ');

    // Check placeholder
    const placeholder = element.getAttribute('placeholder');
    if (placeholder) return placeholder;

    return 'This field';
  }

  /**
   * Setup loading state announcement
   */
  private setupLoadingStateAnnouncement(): void {
    if (!this.config.announceLoadingStates) return;

    // Watch for aria-busy changes
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.type === 'attributes' && mutation.attributeName === 'aria-busy') {
          const element = mutation.target as HTMLElement;
          const isBusy = element.getAttribute('aria-busy') === 'true';
          
          if (isBusy) {
            this.announce('Loading', 'polite');
          } else {
            this.announce('Loading complete', 'polite');
          }
        }
      });
    });

    observer.observe(document.body, {
      attributes: true,
      attributeFilter: ['aria-busy'],
      subtree: true
    });

    // Watch for loading indicators
    document.addEventListener('loading-start', () => {
      this.announce('Loading', 'polite');
    });

    document.addEventListener('loading-complete', () => {
      this.announce('Loading complete', 'polite');
    });
  }

  /**
   * Enhance existing content with ARIA attributes
   */
  private enhanceExistingContent(): void {
    // Enhance headings with level information
    const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
    headings.forEach((heading) => {
      const level = parseInt(heading.tagName.charAt(1));
      heading.setAttribute('aria-level', level.toString());
    });

    // Enhance lists
    const lists = document.querySelectorAll('ul, ol');
    lists.forEach((list) => {
      if (!list.getAttribute('role')) {
        list.setAttribute('role', 'list');
      }
      
      const items = list.querySelectorAll('li');
      items.forEach((item) => {
        if (!item.getAttribute('role')) {
          item.setAttribute('role', 'listitem');
        }
      });
    });

    // Enhance buttons without accessible names
    const buttons = document.querySelectorAll('button');
    buttons.forEach((button) => {
      if (!this.hasAccessibleName(button as HTMLElement)) {
        const text = button.textContent?.trim();
        if (text) {
          button.setAttribute('aria-label', text);
        }
      }
    });

    // Enhance images without alt text
    const images = document.querySelectorAll('img:not([alt])');
    images.forEach((img) => {
      img.setAttribute('alt', '');
    });
  }

  /**
   * Check if element has accessible name
   */
  private hasAccessibleName(element: HTMLElement): boolean {
    return !!(
      element.getAttribute('aria-label') ||
      element.getAttribute('aria-labelledby') ||
      element.getAttribute('title') ||
      element.textContent?.trim()
    );
  }

  /**
   * Utility delay function
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Check if screen reader is active
   */
  public isScreenReaderActive(): boolean {
    // This is a heuristic approach - not 100% reliable
    return !!(
      navigator.userAgent.includes('NVDA') ||
      navigator.userAgent.includes('JAWS') ||
      navigator.userAgent.includes('VoiceOver') ||
      navigator.userAgent.includes('TalkBack') ||
      window.speechSynthesis ||
      (window as any).navigator.tts
    );
  }

  /**
   * Get screen reader statistics
   */
  public getStatistics(): {
    liveRegionsCount: number;
    announcementsQueued: number;
    config: IScreenReaderConfig;
  } {
    return {
      liveRegionsCount: this.liveRegions.size,
      announcementsQueued: this.announcementQueue.length,
      config: { ...this.config }
    };
  }

  /**
   * Cleanup resources
   */
  public destroy(): void {
    // Remove live regions
    this.liveRegions.forEach((region) => {
      if (document.body.contains(region)) {
        document.body.removeChild(region);
      }
    });
    this.liveRegions.clear();

    // Clear announcement queue
    this.announcementQueue = [];

    // Remove skip navigation
    const skipNav = document.getElementById('skip-navigation');
    if (skipNav) {
      document.body.removeChild(skipNav);
    }

    // Remove styles
    const styles = document.getElementById('skip-link-styles');
    if (styles) {
      document.head.removeChild(styles);
    }
  }
}