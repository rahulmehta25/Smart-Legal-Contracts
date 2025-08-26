import { Page, expect, Locator } from '@playwright/test';
import { TestData } from '../fixtures/test-data';

/**
 * Test utility functions for common operations
 */
export class TestHelpers {
  constructor(private page: Page) {}

  /**
   * Wait for API response and return the data
   */
  async waitForAPIResponse(urlPattern: string | RegExp, timeout = 10000) {
    return this.page.waitForResponse(
      response => {
        const url = response.url();
        return typeof urlPattern === 'string' 
          ? url.includes(urlPattern)
          : urlPattern.test(url);
      },
      { timeout }
    );
  }

  /**
   * Wait for WebSocket connection establishment
   */
  async waitForWebSocketConnection(timeout = 10000) {
    return this.page.waitForFunction(() => {
      return (window as any).wsConnection?.readyState === WebSocket.OPEN;
    }, undefined, { timeout });
  }

  /**
   * Upload a file to the application
   */
  async uploadFile(fileInputSelector: string, filePath: string) {
    const fileInput = this.page.locator(fileInputSelector);
    await fileInput.setInputFiles(filePath);
  }

  /**
   * Wait for element to be visible with custom timeout
   */
  async waitForVisible(selector: string, timeout = 10000) {
    return this.page.waitForSelector(selector, { 
      state: 'visible', 
      timeout 
    });
  }

  /**
   * Wait for element to contain specific text
   */
  async waitForText(selector: string, text: string, timeout = 10000) {
    await this.page.waitForFunction(
      ({ selector, text }) => {
        const element = document.querySelector(selector);
        return element?.textContent?.includes(text);
      },
      { selector, text },
      { timeout }
    );
  }

  /**
   * Check if element exists without waiting
   */
  async elementExists(selector: string): Promise<boolean> {
    try {
      await this.page.locator(selector).first().waitFor({ timeout: 1000 });
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Scroll element into view
   */
  async scrollIntoView(selector: string) {
    await this.page.locator(selector).scrollIntoViewIfNeeded();
  }

  /**
   * Take screenshot with custom naming
   */
  async takeScreenshot(name: string, options?: Parameters<Page['screenshot']>[0]) {
    return this.page.screenshot({
      path: `test-results/screenshots/${name}.png`,
      ...options
    });
  }

  /**
   * Check page accessibility
   */
  async checkAccessibility() {
    // Check for basic accessibility attributes
    const issues = [];

    // Check for images without alt text
    const imagesWithoutAlt = await this.page.locator('img:not([alt])').count();
    if (imagesWithoutAlt > 0) {
      issues.push(`${imagesWithoutAlt} images missing alt text`);
    }

    // Check for form inputs without labels
    const inputsWithoutLabels = await this.page.locator('input:not([aria-label]):not([aria-labelledby])').count();
    if (inputsWithoutLabels > 0) {
      issues.push(`${inputsWithoutLabels} inputs missing labels`);
    }

    // Check for buttons without accessible names
    const buttonsWithoutNames = await this.page.locator('button:not([aria-label]):not([aria-labelledby])').filter({
      hasNotText: /.+/
    }).count();
    if (buttonsWithoutNames > 0) {
      issues.push(`${buttonsWithoutNames} buttons missing accessible names`);
    }

    return issues;
  }

  /**
   * Mock API responses for testing
   */
  async mockAPIResponse(url: string | RegExp, responseData: any, status = 200) {
    await this.page.route(url, route => {
      route.fulfill({
        status,
        contentType: 'application/json',
        body: JSON.stringify(responseData)
      });
    });
  }

  /**
   * Mock WebSocket connection
   */
  async mockWebSocket(url: string = 'ws://localhost:8000/ws') {
    await this.page.addInitScript((wsUrl) => {
      class MockWebSocket extends EventTarget {
        public readyState = WebSocket.OPEN;
        public url = wsUrl;
        
        constructor(url: string) {
          super();
          // Simulate connection
          setTimeout(() => {
            this.dispatchEvent(new Event('open'));
          }, 100);
        }

        send(data: string) {
          // Echo back the data
          setTimeout(() => {
            const event = new MessageEvent('message', {
              data: JSON.stringify({
                type: 'echo',
                original_message: JSON.parse(data),
                server_time: new Date().toISOString(),
                status: 'received'
              })
            });
            this.dispatchEvent(event);
          }, 200);
        }

        close() {
          this.readyState = WebSocket.CLOSED;
          this.dispatchEvent(new Event('close'));
        }
      }

      (window as any).WebSocket = MockWebSocket;
    }, url);
  }

  /**
   * Simulate network conditions
   */
  async simulateSlowNetwork() {
    await this.page.route('**/*', route => {
      setTimeout(() => route.continue(), Math.random() * 2000 + 1000);
    });
  }

  /**
   * Simulate offline condition
   */
  async simulateOffline() {
    await this.page.context().setOffline(true);
  }

  /**
   * Restore online condition
   */
  async goOnline() {
    await this.page.context().setOffline(false);
  }

  /**
   * Check console for errors
   */
  async getConsoleErrors(): Promise<string[]> {
    const errors: string[] = [];
    
    this.page.on('console', msg => {
      if (msg.type() === 'error') {
        errors.push(msg.text());
      }
    });

    return errors;
  }

  /**
   * Wait for loading to complete
   */
  async waitForLoadingComplete() {
    // Wait for common loading indicators to disappear
    await Promise.race([
      this.page.waitForSelector('[data-testid="loading"]', { state: 'detached', timeout: 15000 }),
      this.page.waitForSelector('.loading', { state: 'detached', timeout: 15000 }),
      this.page.waitForSelector('[aria-busy="true"]', { state: 'detached', timeout: 15000 }),
      this.page.waitForTimeout(1000) // Fallback timeout
    ]);
  }

  /**
   * Check if page is responsive
   */
  async checkResponsive(breakpoints = [320, 768, 1024, 1440]) {
    const results: { width: number; height: number; issues: string[] }[] = [];

    for (const width of breakpoints) {
      await this.page.setViewportSize({ width, height: 800 });
      await this.page.waitForTimeout(500); // Allow layout to adjust

      const issues = [];

      // Check for horizontal overflow
      const hasOverflow = await this.page.evaluate(() => {
        return document.documentElement.scrollWidth > document.documentElement.clientWidth;
      });

      if (hasOverflow) {
        issues.push('Horizontal overflow detected');
      }

      // Check for overlapping elements (basic check)
      const overlappingElements = await this.page.evaluate(() => {
        const elements = Array.from(document.querySelectorAll('*'));
        let overlaps = 0;
        
        for (let i = 0; i < elements.length; i++) {
          const rect1 = elements[i].getBoundingClientRect();
          if (rect1.width === 0 || rect1.height === 0) continue;
          
          for (let j = i + 1; j < elements.length; j++) {
            const rect2 = elements[j].getBoundingClientRect();
            if (rect2.width === 0 || rect2.height === 0) continue;
            
            if (rect1.left < rect2.right && rect2.left < rect1.right &&
                rect1.top < rect2.bottom && rect2.top < rect1.bottom) {
              overlaps++;
            }
          }
        }
        
        return overlaps;
      });

      if (overlappingElements > 5) { // Allow for some normal overlaps
        issues.push(`${overlappingElements} potentially overlapping elements`);
      }

      results.push({ width, height: 800, issues });
    }

    return results;
  }
}

/**
 * Custom expect matchers for Playwright
 */
export const customExpect = {
  /**
   * Expect API response to match schema
   */
  async toMatchAPISchema(response: any, expectedKeys: string[]) {
    for (const key of expectedKeys) {
      expect(response).toHaveProperty(key);
    }
  },

  /**
   * Expect element to be accessible
   */
  async toBeAccessible(locator: Locator) {
    // Check if element has proper ARIA attributes
    const element = locator.first();
    
    // Check for aria-label or aria-labelledby
    const hasAriaLabel = await element.getAttribute('aria-label');
    const hasAriaLabelledBy = await element.getAttribute('aria-labelledby');
    const hasText = await element.textContent();
    
    expect(hasAriaLabel || hasAriaLabelledBy || hasText).toBeTruthy();
  },

  /**
   * Expect WebSocket message format
   */
  toMatchWebSocketFormat(message: any, expectedType?: string) {
    expect(message).toHaveProperty('type');
    if (expectedType) {
      expect(message.type).toBe(expectedType);
    }
    expect(message).toHaveProperty('timestamp');
  }
};