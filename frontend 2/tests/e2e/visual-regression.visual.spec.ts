import { test, expect } from '@playwright/test';
import { TestHelpers } from '../utils/test-helpers';
import { TestData } from '../fixtures/test-data';

/**
 * Visual Regression Tests
 * 
 * Tests visual consistency across different browsers, screen sizes,
 * and application states to catch visual bugs and layout issues.
 */

test.describe('Visual Regression - Homepage', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
  });

  test('should match homepage baseline on desktop', async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 720 });
    await page.goto('/');
    
    // Wait for page to fully load
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000);
    
    // Take full page screenshot
    await expect(page).toHaveScreenshot('homepage-desktop.png', {
      fullPage: true,
      animations: 'disabled'
    });
  });

  test('should match homepage baseline on tablet', async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.goto('/');
    
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000);
    
    await expect(page).toHaveScreenshot('homepage-tablet.png', {
      fullPage: true,
      animations: 'disabled'
    });
  });

  test('should match homepage baseline on mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');
    
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000);
    
    await expect(page).toHaveScreenshot('homepage-mobile.png', {
      fullPage: true,
      animations: 'disabled'
    });
  });

  test('should maintain visual consistency with loading states', async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 720 });
    
    // Mock slow API response to capture loading state
    await page.route('**/api/**', route => {
      setTimeout(() => {
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ status: 'healthy' })
        });
      }, 3000);
    });

    await page.goto('/');
    
    // Wait for loading state to appear
    await page.waitForTimeout(1000);
    
    // Take screenshot during loading
    await expect(page).toHaveScreenshot('homepage-loading-state.png', {
      animations: 'disabled'
    });
  });
});

test.describe('Visual Regression - Analysis Interface', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
    await page.goto('/');
  });

  test('should match analysis interface with content', async ({ page }) => {
    const textInput = page.locator('[data-testid="text-input"], textarea').first();
    
    if (await textInput.count() > 0) {
      await textInput.fill(TestData.documents.withArbitration().content);
      await page.waitForTimeout(1000);
      
      await expect(page).toHaveScreenshot('analysis-interface-with-content.png', {
        animations: 'disabled'
      });
    } else {
      console.log('ℹ️ No text input found for analysis interface test');
    }
  });

  test('should match positive analysis results', async ({ page }) => {
    await helpers.mockAPIResponse('**/api/v1/analysis/**', TestData.api.analysisResponse(true));

    const textInput = page.locator('[data-testid="text-input"], textarea').first();
    
    if (await textInput.count() > 0) {
      await textInput.fill(TestData.documents.withArbitration().content);
      
      const submitButton = page.locator('[data-testid="analyze-button"], button:has-text("Analyze")').first();
      if (await submitButton.count() > 0) {
        await submitButton.click();
        
        // Wait for results to appear
        await page.waitForTimeout(3000);
        
        await expect(page).toHaveScreenshot('analysis-results-positive.png', {
          animations: 'disabled'
        });
      }
    }
  });

  test('should match negative analysis results', async ({ page }) => {
    await helpers.mockAPIResponse('**/api/v1/analysis/**', TestData.api.analysisResponse(false));

    const textInput = page.locator('[data-testid="text-input"], textarea').first();
    
    if (await textInput.count() > 0) {
      await textInput.fill(TestData.documents.withoutArbitration().content);
      
      const submitButton = page.locator('[data-testid="analyze-button"], button:has-text("Analyze")').first();
      if (await submitButton.count() > 0) {
        await submitButton.click();
        
        await page.waitForTimeout(3000);
        
        await expect(page).toHaveScreenshot('analysis-results-negative.png', {
          animations: 'disabled'
        });
      }
    }
  });

  test('should match error states visually', async ({ page }) => {
    await helpers.mockAPIResponse('**/api/v1/analysis/**', 
      TestData.api.errorResponse(500, 'Analysis service unavailable'), 
      500
    );

    const textInput = page.locator('[data-testid="text-input"], textarea').first();
    
    if (await textInput.count() > 0) {
      await textInput.fill('Test content for error state');
      
      const submitButton = page.locator('[data-testid="analyze-button"], button:has-text("Analyze")').first();
      if (await submitButton.count() > 0) {
        await submitButton.click();
        
        await page.waitForTimeout(3000);
        
        await expect(page).toHaveScreenshot('analysis-error-state.png', {
          animations: 'disabled'
        });
      }
    }
  });
});

test.describe('Visual Regression - Component States', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
    await page.goto('/');
  });

  test('should match button hover states', async ({ page }) => {
    const buttons = page.locator('button').first();
    
    if (await buttons.count() > 0) {
      // Hover over button
      await buttons.hover();
      await page.waitForTimeout(500);
      
      await expect(buttons).toHaveScreenshot('button-hover-state.png');
    }
  });

  test('should match form focus states', async ({ page }) => {
    const inputs = page.locator('input, textarea').first();
    
    if (await inputs.count() > 0) {
      await inputs.focus();
      await page.waitForTimeout(500);
      
      await expect(inputs).toHaveScreenshot('input-focus-state.png');
    }
  });

  test('should match disabled states', async ({ page }) => {
    // Look for disabled elements or create them
    const submitButton = page.locator('[data-testid="analyze-button"], button:has-text("Analyze")').first();
    
    if (await submitButton.count() > 0) {
      // Disable button via JavaScript if not already disabled
      await page.evaluate(() => {
        const button = document.querySelector('[data-testid="analyze-button"], button') as HTMLButtonElement;
        if (button) {
          button.disabled = true;
        }
      });
      
      await page.waitForTimeout(500);
      
      await expect(submitButton).toHaveScreenshot('button-disabled-state.png');
    }
  });

  test('should match loading spinner states', async ({ page }) => {
    // Mock slow response to trigger loading state
    await page.route('**/api/**', route => {
      setTimeout(() => {
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ status: 'healthy' })
        });
      }, 5000);
    });

    const textInput = page.locator('[data-testid="text-input"], textarea').first();
    
    if (await textInput.count() > 0) {
      await textInput.fill('Test content');
      
      const submitButton = page.locator('[data-testid="analyze-button"], button:has-text("Analyze")').first();
      if (await submitButton.count() > 0) {
        await submitButton.click();
        
        // Wait for loading state to appear
        await page.waitForTimeout(1000);
        
        // Screenshot loading state
        const loadingElement = page.locator('[data-testid="loading"], .loading, .spinner').first();
        
        if (await loadingElement.count() > 0) {
          await expect(loadingElement).toHaveScreenshot('loading-spinner-state.png');
        } else {
          // Take full page screenshot if no specific loading element
          await expect(page).toHaveScreenshot('page-loading-state.png', {
            animations: 'allow' // Allow loading animations
          });
        }
      }
    }
  });
});

test.describe('Visual Regression - Dark Mode', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
  });

  test('should match dark mode if supported', async ({ page }) => {
    // Check for dark mode toggle
    await page.goto('/');
    
    const darkModeToggle = page.locator('[data-testid="dark-mode"], [data-testid="theme-toggle"], button:has-text("Dark")');
    
    if (await darkModeToggle.count() > 0) {
      // Switch to dark mode
      await darkModeToggle.click();
      await page.waitForTimeout(1000);
      
      await expect(page).toHaveScreenshot('homepage-dark-mode.png', {
        fullPage: true,
        animations: 'disabled'
      });
    } else {
      // Try system dark mode preference
      await page.emulateMedia({ colorScheme: 'dark' });
      await page.goto('/');
      await page.waitForTimeout(1000);
      
      await expect(page).toHaveScreenshot('homepage-system-dark.png', {
        fullPage: true,
        animations: 'disabled'
      });
    }
  });

  test('should match light mode explicitly', async ({ page }) => {
    await page.emulateMedia({ colorScheme: 'light' });
    await page.goto('/');
    await page.waitForTimeout(1000);
    
    await expect(page).toHaveScreenshot('homepage-light-mode.png', {
      fullPage: true,
      animations: 'disabled'
    });
  });
});

test.describe('Visual Regression - Cross-Browser', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
    await page.goto('/');
  });

  test('should maintain visual consistency across browsers', async ({ page, browserName }) => {
    await page.setViewportSize({ width: 1280, height: 720 });
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000);
    
    // Take browser-specific screenshot
    await expect(page).toHaveScreenshot(`homepage-${browserName}.png`, {
      fullPage: true,
      animations: 'disabled'
    });
    
    // Test key UI elements exist
    const keyElements = [
      'header, .header, nav',
      'main, .main, .content',
      'footer, .footer'
    ];

    for (const selector of keyElements) {
      const element = page.locator(selector).first();
      if (await element.count() > 0) {
        await expect(element).toBeVisible();
      }
    }
  });

  test('should handle browser-specific CSS features', async ({ page, browserName }) => {
    await page.goto('/');
    
    // Check for CSS Grid/Flexbox support
    const layoutSupport = await page.evaluate(() => {
      const testDiv = document.createElement('div');
      document.body.appendChild(testDiv);
      
      const support = {
        grid: CSS.supports('display', 'grid'),
        flexbox: CSS.supports('display', 'flex'),
        customProperties: CSS.supports('color', 'var(--test-color)'),
        backdropFilter: CSS.supports('backdrop-filter', 'blur(5px)')
      };
      
      document.body.removeChild(testDiv);
      return support;
    });

    console.log(`${browserName} CSS support:`, layoutSupport);
    
    // Modern browsers should support these features
    expect(layoutSupport.flexbox).toBe(true);
    expect(layoutSupport.customProperties).toBe(true);
    
    if (browserName !== 'webkit') { // Safari has limited support
      expect(layoutSupport.grid).toBe(true);
    }
  });
});

test.describe('Visual Regression - Animation States', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
  });

  test('should capture animations if present', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    // Look for animated elements
    const animatedElements = page.locator('[data-testid*="animation"], .animate, .transition');
    
    if (await animatedElements.count() > 0) {
      // Capture animation start state
      await expect(page).toHaveScreenshot('animations-start-state.png', {
        animations: 'allow'
      });
      
      // Wait for animation to complete
      await page.waitForTimeout(3000);
      
      // Capture animation end state
      await expect(page).toHaveScreenshot('animations-end-state.png', {
        animations: 'disabled'
      });
    } else {
      console.log('ℹ️ No animated elements detected');
    }
  });

  test('should handle reduced motion preference', async ({ page }) => {
    // Set reduced motion preference
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    await expect(page).toHaveScreenshot('homepage-reduced-motion.png', {
      fullPage: true,
      animations: 'disabled'
    });
  });
});

test.describe('Visual Regression - Component Library', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
    await page.goto('/');
  });

  test('should capture button variations', async ({ page }) => {
    // Test different button states if they exist
    const buttons = page.locator('button');
    const buttonCount = await buttons.count();
    
    if (buttonCount > 0) {
      for (let i = 0; i < Math.min(5, buttonCount); i++) {
        const button = buttons.nth(i);
        
        // Default state
        await expect(button).toHaveScreenshot(`button-${i}-default.png`);
        
        // Hover state
        await button.hover();
        await page.waitForTimeout(300);
        await expect(button).toHaveScreenshot(`button-${i}-hover.png`);
        
        // Focus state
        await button.focus();
        await page.waitForTimeout(300);
        await expect(button).toHaveScreenshot(`button-${i}-focus.png`);
      }
    }
  });

  test('should capture form elements', async ({ page }) => {
    const formElements = [
      'input[type="text"]',
      'input[type="email"]', 
      'input[type="password"]',
      'textarea',
      'select'
    ];

    for (const selector of formElements) {
      const elements = page.locator(selector);
      
      if (await elements.count() > 0) {
        const element = elements.first();
        
        // Default state
        await expect(element).toHaveScreenshot(`form-${selector.replace(/[^a-z]/gi, '')}-default.png`);
        
        // Focus state
        await element.focus();
        await page.waitForTimeout(300);
        await expect(element).toHaveScreenshot(`form-${selector.replace(/[^a-z]/gi, '')}-focus.png`);
        
        // With content
        if (await element.getAttribute('type') !== 'file') {
          await element.fill('Sample content');
          await page.waitForTimeout(300);
          await expect(element).toHaveScreenshot(`form-${selector.replace(/[^a-z]/gi, '')}-filled.png`);
        }
      }
    }
  });

  test('should capture card/container components', async ({ page }) => {
    const containers = page.locator('.card, .container, .box, [data-testid*="card"]');
    const containerCount = await containers.count();
    
    if (containerCount > 0) {
      for (let i = 0; i < Math.min(3, containerCount); i++) {
        const container = containers.nth(i);
        await expect(container).toHaveScreenshot(`container-${i}.png`);
      }
    }
  });
});

test.describe('Visual Regression - Error States', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
  });

  test('should capture network error state', async ({ page }) => {
    // Mock network failure
    await page.route('**/api/**', route => route.abort());
    
    await page.goto('/');
    await page.waitForTimeout(3000);
    
    await expect(page).toHaveScreenshot('error-state-network.png', {
      fullPage: true,
      animations: 'disabled'
    });
  });

  test('should capture API error state', async ({ page }) => {
    await helpers.mockAPIResponse('**/api/v1/analysis/**', 
      TestData.api.errorResponse(500, 'Internal Server Error'), 
      500
    );

    const textInput = page.locator('[data-testid="text-input"], textarea').first();
    
    if (await textInput.count() > 0) {
      await textInput.fill('Test error scenario');
      
      const submitButton = page.locator('[data-testid="analyze-button"], button:has-text("Analyze")').first();
      if (await submitButton.count() > 0) {
        await submitButton.click();
        await page.waitForTimeout(3000);
        
        await expect(page).toHaveScreenshot('error-state-api.png', {
          animations: 'disabled'
        });
      }
    }
  });

  test('should capture validation error state', async ({ page }) => {
    // Try to submit empty form to trigger validation
    const submitButton = page.locator('[data-testid="analyze-button"], button:has-text("Analyze"), button[type="submit"]').first();
    
    if (await submitButton.count() > 0) {
      await submitButton.click();
      await page.waitForTimeout(2000);
      
      await expect(page).toHaveScreenshot('error-state-validation.png', {
        animations: 'disabled'
      });
    }
  });
});

test.describe('Visual Regression - Accessibility Features', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
  });

  test('should capture high contrast mode', async ({ page }) => {
    // Simulate high contrast mode
    await page.emulateMedia({ colorScheme: 'dark', forcedColors: 'active' });
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    await expect(page).toHaveScreenshot('accessibility-high-contrast.png', {
      fullPage: true,
      animations: 'disabled'
    });
  });

  test('should capture focus indicators', async ({ page }) => {
    await page.goto('/');
    
    // Tab through focusable elements and capture focus states
    const focusableElements = page.locator('button, a, input, textarea, select, [tabindex="0"]');
    const elementCount = await focusableElements.count();
    
    if (elementCount > 0) {
      for (let i = 0; i < Math.min(3, elementCount); i++) {
        const element = focusableElements.nth(i);
        await element.focus();
        await page.waitForTimeout(300);
        
        await expect(element).toHaveScreenshot(`focus-indicator-${i}.png`);
      }
    }
  });

  test('should capture large text scaling', async ({ page }) => {
    // Simulate browser zoom or large text preference
    await page.evaluate(() => {
      document.documentElement.style.fontSize = '20px'; // Increase base font size
    });
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    await expect(page).toHaveScreenshot('accessibility-large-text.png', {
      fullPage: true,
      animations: 'disabled'
    });
  });
});

test.describe('Visual Regression - Print Styles', () => {
  test('should capture print layout', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    // Simulate print media
    await page.emulateMedia({ media: 'print' });
    await page.waitForTimeout(1000);
    
    await expect(page).toHaveScreenshot('print-layout.png', {
      fullPage: true,
      animations: 'disabled'
    });
  });
});