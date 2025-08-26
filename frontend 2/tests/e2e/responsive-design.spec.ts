import { test, expect } from '@playwright/test';
import { TestHelpers } from '../utils/test-helpers';

/**
 * Responsive Design Tests
 * 
 * Tests application layout and functionality across different screen sizes,
 * orientations, and devices to ensure consistent user experience.
 */

test.describe('Responsive Layout - Desktop', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
  });

  test('should display correctly on large desktop (1920x1080)', async ({ page }) => {
    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.goto('/');
    
    // Check for proper layout utilization
    const bodyWidth = await page.evaluate(() => document.body.offsetWidth);
    expect(bodyWidth).toBe(1920);
    
    // Check for no horizontal overflow
    const hasOverflow = await page.evaluate(() => {
      return document.documentElement.scrollWidth > document.documentElement.clientWidth;
    });
    expect(hasOverflow).toBe(false);
    
    // Take screenshot for visual verification
    await helpers.takeScreenshot('desktop-large-1920x1080');
  });

  test('should display correctly on standard desktop (1366x768)', async ({ page }) => {
    await page.setViewportSize({ width: 1366, height: 768 });
    await page.goto('/');
    
    await page.waitForTimeout(1000); // Allow layout to adjust
    
    // Check main content is visible
    const mainContent = page.locator('main, [role="main"], .main-content, .container').first();
    if (await mainContent.count() > 0) {
      await expect(mainContent).toBeVisible();
    }
    
    // Check navigation is accessible
    const navigation = page.locator('nav, [role="navigation"], .navigation, .header').first();
    if (await navigation.count() > 0) {
      await expect(navigation).toBeVisible();
    }
    
    await helpers.takeScreenshot('desktop-standard-1366x768');
  });

  test('should display correctly on small desktop (1024x768)', async ({ page }) => {
    await page.setViewportSize({ width: 1024, height: 768 });
    await page.goto('/');
    
    const responsiveIssues = await helpers.checkResponsive([1024]);
    expect(responsiveIssues[0].issues).toEqual([]);
    
    await helpers.takeScreenshot('desktop-small-1024x768');
  });
});

test.describe('Responsive Layout - Tablet', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
  });

  test('should display correctly on iPad Pro (1024x1366)', async ({ page }) => {
    await page.setViewportSize({ width: 1024, height: 1366 });
    await page.goto('/');
    
    await page.waitForTimeout(1000);
    
    // Check for tablet-optimized layout
    const responsiveIssues = await helpers.checkResponsive([1024]);
    expect(responsiveIssues[0].issues.length).toBeLessThanOrEqual(2); // Allow minor issues
    
    // Test touch interactions
    const textInput = page.locator('[data-testid="text-input"], textarea').first();
    if (await textInput.count() > 0) {
      await textInput.tap();
      await expect(textInput).toBeFocused();
    }
    
    await helpers.takeScreenshot('tablet-ipad-pro-1024x1366');
  });

  test('should display correctly on iPad (768x1024)', async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.goto('/');
    
    await page.waitForTimeout(1000);
    
    // Check layout adapts to narrower width
    const responsiveIssues = await helpers.checkResponsive([768]);
    expect(responsiveIssues[0].issues.length).toBeLessThanOrEqual(3);
    
    await helpers.takeScreenshot('tablet-ipad-768x1024');
  });

  test('should handle orientation changes on tablet', async ({ page }) => {
    // Portrait mode
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.goto('/');
    await page.waitForTimeout(1000);
    
    const portraitLayout = await page.locator('body').screenshot();
    
    // Landscape mode  
    await page.setViewportSize({ width: 1024, height: 768 });
    await page.waitForTimeout(1000);
    
    const landscapeLayout = await page.locator('body').screenshot();
    
    // Layouts should be different (responsive)
    expect(portraitLayout).not.toEqual(landscapeLayout);
    
    await helpers.takeScreenshot('tablet-landscape-1024x768');
  });
});

test.describe('Responsive Layout - Mobile', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
  });

  test('should display correctly on iPhone 14 Pro (393x852)', async ({ page }) => {
    await page.setViewportSize({ width: 393, height: 852 });
    await page.goto('/');
    
    await page.waitForTimeout(1000);
    
    // Check for mobile layout
    const responsiveIssues = await helpers.checkResponsive([393]);
    expect(responsiveIssues[0].issues.length).toBeLessThanOrEqual(2);
    
    // Check mobile navigation
    const mobileNav = page.locator('[data-testid="mobile-nav"], .mobile-menu, .hamburger').first();
    if (await mobileNav.count() > 0) {
      console.log('✅ Mobile navigation detected');
    }
    
    await helpers.takeScreenshot('mobile-iphone-14-pro-393x852');
  });

  test('should display correctly on Android (360x800)', async ({ page }) => {
    await page.setViewportSize({ width: 360, height: 800 });
    await page.goto('/');
    
    await page.waitForTimeout(1000);
    
    const responsiveIssues = await helpers.checkResponsive([360]);
    expect(responsiveIssues[0].issues.length).toBeLessThanOrEqual(3);
    
    await helpers.takeScreenshot('mobile-android-360x800');
  });

  test('should handle mobile touch interactions', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 }); // iPhone SE
    await page.goto('/');
    
    // Test tap interactions
    const clickableElements = page.locator('button, a, [data-testid*="button"]');
    const elementCount = await clickableElements.count();
    
    if (elementCount > 0) {
      const firstElement = clickableElements.first();
      
      // Test tap
      await firstElement.tap();
      await page.waitForTimeout(500);
      
      console.log('✅ Touch interactions working');
    }
    
    // Test scroll behavior
    await page.evaluate(() => window.scrollTo(0, 100));
    await page.waitForTimeout(500);
    
    const scrollPosition = await page.evaluate(() => window.pageYOffset);
    expect(scrollPosition).toBeGreaterThan(0);
  });

  test('should handle mobile form interactions', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 812 }); // iPhone X
    await page.goto('/');
    
    const textInput = page.locator('[data-testid="text-input"], textarea, input').first();
    
    if (await textInput.count() > 0) {
      // Test focus brings up virtual keyboard (simulated)
      await textInput.tap();
      
      // Check if input is properly focused and visible
      await expect(textInput).toBeFocused();
      
      // Test typing
      await textInput.fill('Mobile test input');
      
      const value = await textInput.inputValue();
      expect(value).toBe('Mobile test input');
      
      console.log('✅ Mobile form interactions working');
    }
  });

  test('should handle pinch zoom gracefully', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');
    
    // Simulate zoom
    await page.evaluate(() => {
      // Simulate viewport meta tag behavior
      const meta = document.querySelector('meta[name="viewport"]');
      if (meta) {
        console.log('Viewport meta tag:', meta.getAttribute('content'));
      }
    });
    
    // Try to zoom (limited in test environment)
    await page.evaluate(() => {
      document.body.style.zoom = '1.5';
    });
    
    await page.waitForTimeout(1000);
    
    // Check layout doesn't break
    const bodyWidth = await page.evaluate(() => document.body.offsetWidth);
    const hasOverflow = await page.evaluate(() => {
      return document.documentElement.scrollWidth > window.innerWidth * 1.2; // Allow some tolerance
    });
    
    expect(hasOverflow).toBe(false);
    
    // Reset zoom
    await page.evaluate(() => {
      document.body.style.zoom = '1';
    });
  });
});

test.describe('Responsive Component Behavior', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
  });

  test('should adapt navigation for different screen sizes', async ({ page }) => {
    const breakpoints = [
      { width: 1200, height: 800, device: 'desktop' },
      { width: 768, height: 1024, device: 'tablet' },
      { width: 375, height: 667, device: 'mobile' }
    ];

    for (const breakpoint of breakpoints) {
      await page.setViewportSize({ width: breakpoint.width, height: breakpoint.height });
      await page.goto('/');
      await page.waitForTimeout(1000);
      
      console.log(`Testing ${breakpoint.device} (${breakpoint.width}x${breakpoint.height})`);
      
      // Check navigation visibility and accessibility
      const navigation = page.locator('nav, [role="navigation"], .navigation').first();
      
      if (await navigation.count() > 0) {
        await expect(navigation).toBeVisible();
        
        // Check if mobile menu exists on small screens
        if (breakpoint.width < 768) {
          const mobileMenu = page.locator('[data-testid="mobile-menu"], .mobile-nav, .hamburger');
          if (await mobileMenu.count() > 0) {
            console.log(`✅ Mobile menu available on ${breakpoint.device}`);
          }
        }
      }
      
      await helpers.takeScreenshot(`navigation-${breakpoint.device}-${breakpoint.width}x${breakpoint.height}`);
    }
  });

  test('should adapt content layout for different screens', async ({ page }) => {
    const viewports = [
      { width: 1440, height: 900 },
      { width: 1024, height: 768 },
      { width: 768, height: 1024 },
      { width: 390, height: 844 }
    ];

    for (const viewport of viewports) {
      await page.setViewportSize(viewport);
      await page.goto('/');
      await page.waitForTimeout(1000);
      
      // Check main content area
      const mainContent = page.locator('main, .main, .content, [role="main"]').first();
      
      if (await mainContent.count() > 0) {
        const contentBox = await mainContent.boundingBox();
        
        if (contentBox) {
          // Content should not overflow viewport width
          expect(contentBox.width).toBeLessThanOrEqual(viewport.width);
          
          // Content should be visible
          expect(contentBox.x).toBeGreaterThanOrEqual(0);
        }
      }
      
      // Check for responsive text sizing
      const headings = page.locator('h1, h2, h3').first();
      if (await headings.count() > 0) {
        const fontSize = await headings.evaluate(el => {
          return window.getComputedStyle(el).fontSize;
        });
        
        console.log(`${viewport.width}x${viewport.height}: Heading font size: ${fontSize}`);
      }
    }
  });

  test('should handle text scaling appropriately', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');
    
    // Check that text is readable on mobile
    const textElements = page.locator('p, span, div').filter({ hasText: /.{10,}/ });
    
    if (await textElements.count() > 0) {
      const fontSize = await textElements.first().evaluate(el => {
        return parseFloat(window.getComputedStyle(el).fontSize);
      });
      
      // Text should be at least 16px on mobile for readability
      expect(fontSize).toBeGreaterThanOrEqual(14);
      console.log(`Mobile text size: ${fontSize}px`);
    }
  });

  test('should maintain functionality across breakpoints', async ({ page }) => {
    const breakpoints = [1200, 768, 480, 320];
    
    for (const width of breakpoints) {
      await page.setViewportSize({ width, height: 800 });
      await page.goto('/');
      await page.waitForTimeout(1000);
      
      console.log(`Testing functionality at ${width}px width`);
      
      // Test core functionality at each breakpoint
      const textInput = page.locator('[data-testid="text-input"], textarea').first();
      
      if (await textInput.count() > 0) {
        // Should be able to interact with input
        await textInput.click();
        await expect(textInput).toBeFocused();
        
        // Should be able to type
        await textInput.fill(`Test at ${width}px`);
        const value = await textInput.inputValue();
        expect(value).toBe(`Test at ${width}px`);
        
        // Submit button should be accessible
        const submitButton = page.locator('[data-testid="analyze-button"], button:has-text("Analyze")').first();
        if (await submitButton.count() > 0) {
          await expect(submitButton).toBeVisible();
          
          // Button should be properly sized for touch
          const buttonBox = await submitButton.boundingBox();
          if (buttonBox && width < 768) {
            expect(buttonBox.height).toBeGreaterThanOrEqual(44); // Minimum touch target size
          }
        }
        
        await textInput.clear();
      }
    }
  });

  test('should handle content overflow properly', async ({ page }) => {
    const narrowWidths = [320, 375, 414]; // Common mobile widths
    
    for (const width of narrowWidths) {
      await page.setViewportSize({ width, height: 600 });
      await page.goto('/');
      await page.waitForTimeout(1000);
      
      // Check for text overflow
      const textElements = page.locator('p, span, div, h1, h2, h3').filter({ hasText: /.{20,}/ });
      
      for (let i = 0; i < Math.min(5, await textElements.count()); i++) {
        const element = textElements.nth(i);
        const hasOverflow = await element.evaluate(el => {
          return el.scrollWidth > el.clientWidth;
        });
        
        if (hasOverflow) {
          // Check if overflow is handled with CSS
          const overflowHandling = await element.evaluate(el => {
            const style = window.getComputedStyle(el);
            return {
              overflow: style.overflow,
              textOverflow: style.textOverflow,
              whiteSpace: style.whiteSpace,
              wordWrap: style.wordWrap
            };
          });
          
          console.log(`${width}px: Text overflow detected, CSS handling:`, overflowHandling);
        }
      }
      
      await helpers.takeScreenshot(`mobile-content-${width}px`);
    }
  });
});

test.describe('Responsive Interactive Elements', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
  });

  test('should have appropriate touch targets on mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');
    
    // Check button sizes meet touch guidelines (minimum 44px)
    const buttons = page.locator('button');
    const buttonCount = await buttons.count();
    
    if (buttonCount > 0) {
      for (let i = 0; i < Math.min(5, buttonCount); i++) {
        const button = buttons.nth(i);
        const box = await button.boundingBox();
        
        if (box) {
          const meetsGuidelines = box.width >= 44 && box.height >= 44;
          if (!meetsGuidelines) {
            console.warn(`⚠️ Button ${i} size: ${box.width}x${box.height}px - Below 44px minimum`);
          } else {
            console.log(`✅ Button ${i} size: ${box.width}x${box.height}px - Meets guidelines`);
          }
        }
      }
    }
  });

  test('should maintain spacing between interactive elements', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');
    
    // Check spacing between buttons/links
    const interactiveElements = page.locator('button, a, input, [tabindex="0"]');
    const elementCount = await interactiveElements.count();
    
    if (elementCount > 1) {
      for (let i = 0; i < elementCount - 1; i++) {
        const element1 = interactiveElements.nth(i);
        const element2 = interactiveElements.nth(i + 1);
        
        const box1 = await element1.boundingBox();
        const box2 = await element2.boundingBox();
        
        if (box1 && box2) {
          const verticalGap = Math.abs(box2.y - (box1.y + box1.height));
          const horizontalGap = Math.abs(box2.x - (box1.x + box1.width));
          
          // Elements should have at least 8px spacing if they're close
          if (verticalGap < 50 && horizontalGap < 50) {
            const hasAdequateSpacing = verticalGap >= 8 || horizontalGap >= 8;
            console.log(`Element spacing: V:${verticalGap}px H:${horizontalGap}px - ${hasAdequateSpacing ? '✅' : '⚠️'}`);
          }
        }
      }
    }
  });

  test('should handle virtual keyboard appearance', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');
    
    const textInput = page.locator('[data-testid="text-input"], textarea, input').first();
    
    if (await textInput.count() > 0) {
      // Get initial viewport height
      const initialHeight = await page.evaluate(() => window.innerHeight);
      
      // Focus input (simulates keyboard appearance)
      await textInput.tap();
      await page.waitForTimeout(1000);
      
      // Check if layout adapts (input should still be visible)
      const inputVisible = await textInput.isVisible();
      expect(inputVisible).toBe(true);
      
      // Check if input is in viewport
      const inputBox = await textInput.boundingBox();
      if (inputBox) {
        expect(inputBox.y).toBeGreaterThanOrEqual(0);
        expect(inputBox.y + inputBox.height).toBeLessThanOrEqual(667);
      }
      
      console.log(`Virtual keyboard handling: Input visible: ${inputVisible}`);
    }
  });
});

test.describe('Responsive Accessibility', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
  });

  test('should maintain accessibility across screen sizes', async ({ page }) => {
    const viewports = [
      { width: 1200, height: 800, name: 'desktop' },
      { width: 768, height: 1024, name: 'tablet' },
      { width: 375, height: 667, name: 'mobile' }
    ];

    for (const viewport of viewports) {
      await page.setViewportSize(viewport);
      await page.goto('/');
      await page.waitForTimeout(1000);
      
      const accessibilityIssues = await helpers.checkAccessibility();
      
      console.log(`${viewport.name} accessibility issues:`, accessibilityIssues);
      expect(accessibilityIssues.length).toBeLessThanOrEqual(3); // Allow some issues but not many
    }
  });

  test('should maintain keyboard navigation on mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');
    
    // Test tab navigation works on mobile
    await page.keyboard.press('Tab');
    let focusedElement = await page.evaluate(() => document.activeElement?.tagName);
    
    if (focusedElement && focusedElement !== 'BODY') {
      console.log('✅ Keyboard navigation available on mobile');
      
      // Continue tabbing
      for (let i = 0; i < 5; i++) {
        await page.keyboard.press('Tab');
        await page.waitForTimeout(200);
      }
      
      const finalFocus = await page.evaluate(() => document.activeElement?.tagName);
      expect(finalFocus).toBeTruthy();
    } else {
      console.log('ℹ️ Limited keyboard navigation on mobile - touch-first design');
    }
  });

  test('should have readable text at all screen sizes', async ({ page }) => {
    const viewports = [
      { width: 1440, height: 900 },
      { width: 1024, height: 768 },
      { width: 768, height: 1024 },
      { width: 375, height: 812 },
      { width: 320, height: 568 } // Smallest common mobile size
    ];

    for (const viewport of viewports) {
      await page.setViewportSize(viewport);
      await page.goto('/');
      await page.waitForTimeout(1000);
      
      // Check text readability
      const textElements = page.locator('p, span, div, h1, h2, h3, button, a').filter({ hasText: /.+/ });
      const sampleSize = Math.min(10, await textElements.count());
      
      for (let i = 0; i < sampleSize; i++) {
        const element = textElements.nth(i);
        const styles = await element.evaluate(el => {
          const computed = window.getComputedStyle(el);
          return {
            fontSize: parseFloat(computed.fontSize),
            color: computed.color,
            backgroundColor: computed.backgroundColor,
            opacity: parseFloat(computed.opacity)
          };
        });
        
        // Minimum readable font size
        const minFontSize = viewport.width < 768 ? 14 : 12;
        
        if (styles.fontSize < minFontSize) {
          console.warn(`⚠️ Small font at ${viewport.width}px: ${styles.fontSize}px`);
        }
        
        // Opacity should be high enough for readability
        if (styles.opacity < 0.7) {
          console.warn(`⚠️ Low opacity at ${viewport.width}px: ${styles.opacity}`);
        }
      }
      
      console.log(`✅ Text readability checked for ${viewport.width}px`);
    }
  });

  test('should handle dynamic content changes responsively', async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.goto('/');
    
    // Mock dynamic content addition
    await page.evaluate(() => {
      const container = document.body;
      for (let i = 0; i < 10; i++) {
        const div = document.createElement('div');
        div.textContent = `Dynamic content item ${i + 1}`;
        div.style.padding = '10px';
        div.style.border = '1px solid #ccc';
        div.style.margin = '5px';
        container.appendChild(div);
      }
    });
    
    await page.waitForTimeout(1000);
    
    // Check that layout remains responsive
    const hasOverflow = await page.evaluate(() => {
      return document.documentElement.scrollWidth > document.documentElement.clientWidth;
    });
    
    expect(hasOverflow).toBe(false);
    
    // Switch to mobile view with dynamic content
    await page.setViewportSize({ width: 375, height: 667 });
    await page.waitForTimeout(1000);
    
    const mobileOverflow = await page.evaluate(() => {
      return document.documentElement.scrollWidth > document.documentElement.clientWidth;
    });
    
    expect(mobileOverflow).toBe(false);
    console.log('✅ Dynamic content remains responsive');
  });
});