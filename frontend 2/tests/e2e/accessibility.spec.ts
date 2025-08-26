import { test, expect } from '@playwright/test';
import { TestHelpers, customExpect } from '../utils/test-helpers';

/**
 * Accessibility Tests
 * 
 * Tests WCAG 2.1 compliance, keyboard navigation, screen reader compatibility,
 * and other accessibility features to ensure inclusive design.
 */

test.describe('WCAG 2.1 AA Compliance', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
    await page.goto('/');
  });

  test('should have proper heading hierarchy', async ({ page }) => {
    const headings = await page.locator('h1, h2, h3, h4, h5, h6').all();
    
    if (headings.length > 0) {
      const headingLevels: number[] = [];
      
      for (const heading of headings) {
        const tagName = await heading.evaluate(el => el.tagName);
        const level = parseInt(tagName.substring(1));
        headingLevels.push(level);
      }
      
      console.log('Heading hierarchy:', headingLevels);
      
      // Should start with h1
      expect(headingLevels[0]).toBe(1);
      
      // Should not skip levels
      for (let i = 1; i < headingLevels.length; i++) {
        const levelJump = headingLevels[i] - headingLevels[i - 1];
        expect(levelJump).toBeLessThanOrEqual(1);
      }
      
      console.log('✅ Heading hierarchy is valid');
    } else {
      console.log('⚠️ No headings found on page');
    }
  });

  test('should have accessible form labels', async ({ page }) => {
    const formInputs = page.locator('input, textarea, select');
    const inputCount = await formInputs.count();
    
    if (inputCount > 0) {
      for (let i = 0; i < inputCount; i++) {
        const input = formInputs.nth(i);
        
        // Check for associated label
        const inputId = await input.getAttribute('id');
        const ariaLabel = await input.getAttribute('aria-label');
        const ariaLabelledby = await input.getAttribute('aria-labelledby');
        
        let hasLabel = false;
        
        if (inputId) {
          const associatedLabel = page.locator(`label[for="${inputId}"]`);
          hasLabel = await associatedLabel.count() > 0;
        }
        
        if (!hasLabel && ariaLabel) {
          hasLabel = true;
        }
        
        if (!hasLabel && ariaLabelledby) {
          const labelElement = page.locator(`#${ariaLabelledby}`);
          hasLabel = await labelElement.count() > 0;
        }
        
        const inputType = await input.getAttribute('type') || 'text';
        console.log(`Input ${i} (${inputType}): ${hasLabel ? '✅' : '⚠️'} Has accessible label`);
        
        if (!hasLabel && inputType !== 'hidden') {
          console.warn(`Input without label: ${await input.getAttribute('name') || 'unnamed'}`);
        }
      }
    }
  });

  test('should have sufficient color contrast', async ({ page }) => {
    const textElements = page.locator('p, span, div, h1, h2, h3, button, a').filter({ hasText: /.+/ });
    const elementCount = await textElements.count();
    
    if (elementCount > 0) {
      const contrastIssues = [];
      
      for (let i = 0; i < Math.min(10, elementCount); i++) {
        const element = textElements.nth(i);
        
        const colors = await element.evaluate(el => {
          const computed = window.getComputedStyle(el);
          return {
            color: computed.color,
            backgroundColor: computed.backgroundColor,
            fontSize: parseFloat(computed.fontSize)
          };
        });
        
        // Parse RGB values (simplified - real contrast testing is complex)
        const rgbRegex = /rgb\((\d+),\s*(\d+),\s*(\d+)\)/;
        const textMatch = colors.color.match(rgbRegex);
        const bgMatch = colors.backgroundColor.match(rgbRegex);
        
        if (textMatch && bgMatch) {
          const textRgb = [parseInt(textMatch[1]), parseInt(textMatch[2]), parseInt(textMatch[3])];
          const bgRgb = [parseInt(bgMatch[1]), parseInt(bgMatch[2]), parseInt(bgMatch[3])];
          
          // Simple contrast calculation (luminance ratio)
          const textLum = (0.299 * textRgb[0] + 0.587 * textRgb[1] + 0.114 * textRgb[2]) / 255;
          const bgLum = (0.299 * bgRgb[0] + 0.587 * bgRgb[1] + 0.114 * bgRgb[2]) / 255;
          
          const contrast = Math.abs(textLum - bgLum);
          const minimumContrast = colors.fontSize < 18 ? 0.5 : 0.4; // Simplified threshold
          
          if (contrast < minimumContrast) {
            contrastIssues.push({
              element: i,
              contrast: contrast.toFixed(2),
              fontSize: colors.fontSize
            });
          }
        }
      }
      
      console.log(`Contrast analysis: ${contrastIssues.length} potential issues of ${Math.min(10, elementCount)} elements`);
      
      if (contrastIssues.length > 0) {
        console.warn('Low contrast elements:', contrastIssues);
      }
      
      // Should have minimal contrast issues
      expect(contrastIssues.length).toBeLessThan(elementCount * 0.3); // Allow up to 30% to have issues
    }
  });

  test('should have proper ARIA attributes', async ({ page }) => {
    const interactiveElements = page.locator('button, a, input, [tabindex], [role]');
    const elementCount = await interactiveElements.count();
    
    if (elementCount > 0) {
      const ariaIssues = [];
      
      for (let i = 0; i < Math.min(10, elementCount); i++) {
        const element = interactiveElements.nth(i);
        
        const attributes = await element.evaluate(el => ({
          tagName: el.tagName,
          role: el.getAttribute('role'),
          ariaLabel: el.getAttribute('aria-label'),
          ariaLabelledby: el.getAttribute('aria-labelledby'),
          ariaDescribedby: el.getAttribute('aria-describedby'),
          tabIndex: el.getAttribute('tabindex'),
          textContent: el.textContent?.trim()
        }));
        
        // Check for accessible name
        const hasAccessibleName = attributes.ariaLabel || 
                                 attributes.ariaLabelledby || 
                                 (attributes.textContent && attributes.textContent.length > 0);
        
        if (!hasAccessibleName && attributes.tagName === 'BUTTON') {
          ariaIssues.push(`Button ${i} missing accessible name`);
        }
        
        // Check for proper roles
        if (attributes.role === 'button' && attributes.tagName !== 'BUTTON') {
          // Custom button should have proper keyboard support
          const hasTabIndex = attributes.tabIndex !== null;
          if (!hasTabIndex) {
            ariaIssues.push(`Custom button ${i} missing tabindex`);
          }
        }
      }
      
      console.log(`ARIA analysis: ${ariaIssues.length} issues found`);
      
      if (ariaIssues.length > 0) {
        console.warn('ARIA issues:', ariaIssues);
      }
      
      expect(ariaIssues.length).toBeLessThan(3); // Should have minimal ARIA issues
    }
  });

  test('should support keyboard navigation', async ({ page }) => {
    await page.goto('/');
    
    const focusableElements = [];
    let currentElement = null;
    
    // Tab through all focusable elements
    for (let i = 0; i < 20; i++) {
      await page.keyboard.press('Tab');
      await page.waitForTimeout(100);
      
      currentElement = await page.evaluate(() => {
        const element = document.activeElement;
        return element ? {
          tagName: element.tagName,
          id: element.id,
          className: element.className,
          textContent: element.textContent?.trim().substring(0, 50),
          tabIndex: element.getAttribute('tabindex')
        } : null;
      });
      
      if (currentElement && currentElement.tagName !== 'BODY') {
        focusableElements.push(currentElement);
      }
    }
    
    console.log(`Focusable elements found: ${focusableElements.length}`);
    console.log('Focus order:', focusableElements.map(el => `${el.tagName}${el.id ? '#' + el.id : ''}`));
    
    // Should have reasonable number of focusable elements
    expect(focusableElements.length).toBeGreaterThan(0);
    expect(focusableElements.length).toBeLessThan(50); // Not too many for good UX
  });

  test('should handle Enter and Space key activation', async ({ page }) => {
    await page.goto('/');
    
    const buttons = page.locator('button, [role="button"]');
    const buttonCount = await buttons.count();
    
    if (buttonCount > 0) {
      for (let i = 0; i < Math.min(3, buttonCount); i++) {
        const button = buttons.nth(i);
        
        // Test Enter key activation
        await button.focus();
        await page.keyboard.press('Enter');
        await page.waitForTimeout(500);
        
        // Test Space key activation
        await button.focus();
        await page.keyboard.press('Space');
        await page.waitForTimeout(500);
        
        console.log(`Button ${i}: Keyboard activation tested`);
      }
      
      // Page should remain stable after keyboard interactions
      const pageStable = await page.evaluate(() => document.readyState === 'complete');
      expect(pageStable).toBe(true);
    }
  });

  test('should provide skip links for main content', async ({ page }) => {
    await page.goto('/');
    
    // Press Tab to focus first element (often skip link)
    await page.keyboard.press('Tab');
    
    const firstFocused = await page.evaluate(() => {
      const element = document.activeElement;
      return element ? {
        tagName: element.tagName,
        textContent: element.textContent?.trim(),
        href: (element as HTMLAnchorElement).href
      } : null;
    });
    
    if (firstFocused?.textContent?.toLowerCase().includes('skip')) {
      console.log('✅ Skip link detected:', firstFocused.textContent);
      
      // Test skip link functionality
      await page.keyboard.press('Enter');
      await page.waitForTimeout(500);
      
      const focusAfterSkip = await page.evaluate(() => {
        const element = document.activeElement;
        return element?.id || element?.tagName;
      });
      
      console.log('Focus after skip link:', focusAfterSkip);
    } else {
      console.log('ℹ️ No skip link detected');
    }
  });
});

test.describe('Screen Reader Compatibility', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
    await page.goto('/');
  });

  test('should have descriptive page title', async ({ page }) => {
    const title = await page.title();
    
    console.log('Page title:', title);
    
    expect(title.length).toBeGreaterThan(10);
    expect(title.length).toBeLessThan(60); // Good for search engines and screen readers
    expect(title).not.toBe(''); // Not empty
    expect(title.toLowerCase()).toMatch(/arbitration|detector|analysis/); // Relevant to content
  });

  test('should have proper landmark regions', async ({ page }) => {
    const landmarks = await page.locator('[role="main"], [role="navigation"], [role="banner"], [role="contentinfo"], main, nav, header, footer').all();
    
    const landmarkTypes = [];
    
    for (const landmark of landmarks) {
      const role = await landmark.getAttribute('role') || await landmark.evaluate(el => el.tagName.toLowerCase());
      landmarkTypes.push(role);
    }
    
    console.log('Landmark regions found:', landmarkTypes);
    
    // Should have at least main content area
    const hasMainContent = landmarkTypes.some(type => type === 'main' || type === 'main');
    expect(hasMainContent).toBe(true);
    
    if (landmarkTypes.length >= 3) {
      console.log('✅ Good landmark structure for screen readers');
    } else {
      console.log('⚠️ Limited landmark structure');
    }
  });

  test('should provide alternative text for images', async ({ page }) => {
    const images = page.locator('img');
    const imageCount = await images.count();
    
    if (imageCount > 0) {
      const imagesWithoutAlt = [];
      
      for (let i = 0; i < imageCount; i++) {
        const img = images.nth(i);
        const alt = await img.getAttribute('alt');
        const ariaLabel = await img.getAttribute('aria-label');
        const src = await img.getAttribute('src');
        
        if (!alt && !ariaLabel) {
          imagesWithoutAlt.push({ index: i, src: src?.substring(0, 50) });
        }
      }
      
      console.log(`Images: ${imageCount} total, ${imagesWithoutAlt.length} missing alt text`);
      
      if (imagesWithoutAlt.length > 0) {
        console.warn('Images without alt text:', imagesWithoutAlt);
      }
      
      // Most images should have alt text (allow some exceptions for decorative images)
      expect(imagesWithoutAlt.length).toBeLessThan(imageCount * 0.2);
    }
  });

  test('should announce dynamic content changes', async ({ page }) => {
    const textInput = page.locator('[data-testid="text-input"], textarea').first();
    
    if (await textInput.count() > 0) {
      // Look for ARIA live regions
      const liveRegions = page.locator('[aria-live], [aria-atomic], [role="status"], [role="alert"]');
      const liveRegionCount = await liveRegions.count();
      
      console.log(`ARIA live regions found: ${liveRegionCount}`);
      
      // Test dynamic content update
      await textInput.fill('Testing dynamic updates');
      
      const submitButton = page.locator('[data-testid="analyze-button"], button:has-text("Analyze")').first();
      if (await submitButton.count() > 0) {
        await submitButton.click();
        await page.waitForTimeout(3000);
        
        // Check if results area has proper ARIA attributes
        const resultsArea = page.locator('[data-testid*="result"], .results').first();
        
        if (await resultsArea.count() > 0) {
          const ariaLive = await resultsArea.getAttribute('aria-live');
          const role = await resultsArea.getAttribute('role');
          
          if (ariaLive || role === 'status') {
            console.log('✅ Results announced to screen readers');
          } else {
            console.log('⚠️ Results may not be announced to screen readers');
          }
        }
      }
    }
  });

  test('should handle high contrast mode', async ({ page }) => {
    // Enable high contrast
    await page.emulateMedia({ forcedColors: 'active' });
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Check that essential elements are still visible
    const buttons = page.locator('button');
    const buttonCount = await buttons.count();
    
    if (buttonCount > 0) {
      for (let i = 0; i < Math.min(3, buttonCount); i++) {
        const button = buttons.nth(i);
        const isVisible = await button.isVisible();
        expect(isVisible).toBe(true);
      }
      
      console.log('✅ Buttons visible in high contrast mode');
    }
    
    // Check text inputs
    const inputs = page.locator('input, textarea');
    const inputCount = await inputs.count();
    
    if (inputCount > 0) {
      const firstInput = inputs.first();
      const isVisible = await firstInput.isVisible();
      expect(isVisible).toBe(true);
      
      console.log('✅ Inputs visible in high contrast mode');
    }
  });
});

test.describe('Keyboard Navigation', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
    await page.goto('/');
  });

  test('should support full keyboard navigation', async ({ page }) => {
    const navigationPath = [];
    
    // Record tab navigation path
    for (let i = 0; i < 15; i++) {
      await page.keyboard.press('Tab');
      await page.waitForTimeout(200);
      
      const focusedElement = await page.evaluate(() => {
        const el = document.activeElement;
        return el ? {
          tag: el.tagName,
          id: el.id,
          class: el.className.split(' ')[0], // First class only
          text: el.textContent?.trim().substring(0, 20)
        } : null;
      });
      
      if (focusedElement && focusedElement.tag !== 'BODY') {
        navigationPath.push(focusedElement);
      }
    }
    
    console.log('Keyboard navigation path:', navigationPath);
    
    // Should be able to reach multiple interactive elements
    const uniqueTags = new Set(navigationPath.map(el => el.tag));
    expect(uniqueTags.size).toBeGreaterThan(1);
    
    // Should include common interactive elements
    const hasButtons = navigationPath.some(el => el.tag === 'BUTTON');
    const hasInputs = navigationPath.some(el => el.tag === 'INPUT' || el.tag === 'TEXTAREA');
    
    console.log(`Navigation includes: Buttons: ${hasButtons}, Inputs: ${hasInputs}`);
  });

  test('should trap focus in modals', async ({ page }) => {
    // Look for modal triggers
    const modalTriggers = page.locator('[data-testid*="modal"], [data-testid*="dialog"], button:has-text("Help"), button:has-text("Info")');
    
    if (await modalTriggers.count() > 0) {
      await modalTriggers.first().click();
      await page.waitForTimeout(1000);
      
      // Check if modal is open
      const modal = page.locator('[role="dialog"], .modal, .overlay').first();
      
      if (await modal.count() > 0) {
        const focusableInModal = await modal.locator('button, input, textarea, select, a, [tabindex="0"]').count();
        
        if (focusableInModal > 0) {
          // Test focus trapping
          for (let i = 0; i < focusableInModal + 2; i++) {
            await page.keyboard.press('Tab');
            await page.waitForTimeout(100);
            
            // Check if focus stays within modal
            const focusedElement = await page.evaluate(() => document.activeElement);
            const focusInModal = await modal.evaluate((modalEl, focusedEl) => {
              return modalEl.contains(focusedEl);
            }, focusedElement);
            
            if (i >= focusableInModal && !focusInModal) {
              console.log('⚠️ Focus escaped modal');
            }
          }
          
          console.log('✅ Focus trapping tested');
          
          // Close modal with Escape
          await page.keyboard.press('Escape');
          await page.waitForTimeout(500);
        }
      }
    }
  });

  test('should support Escape key for closing overlays', async ({ page }) => {
    // Test various overlay-opening interactions
    const overlayTriggers = page.locator('button, [data-testid*="menu"], [data-testid*="dropdown"]');
    const triggerCount = await overlayTriggers.count();
    
    if (triggerCount > 0) {
      for (let i = 0; i < Math.min(3, triggerCount); i++) {
        const trigger = overlayTriggers.nth(i);
        
        // Open overlay
        await trigger.click();
        await page.waitForTimeout(1000);
        
        // Check if overlay opened
        const overlay = page.locator('[role="menu"], [role="dialog"], .dropdown, .popover, .modal').first();
        
        if (await overlay.count() > 0 && await overlay.isVisible()) {
          // Press Escape to close
          await page.keyboard.press('Escape');
          await page.waitForTimeout(500);
          
          // Check if overlay closed
          const overlayClosed = !(await overlay.isVisible());
          
          if (overlayClosed) {
            console.log(`✅ Overlay ${i} closed with Escape key`);
          } else {
            console.log(`⚠️ Overlay ${i} did not close with Escape key`);
          }
        }
      }
    }
  });

  test('should provide visible focus indicators', async ({ page }) => {
    await page.goto('/');
    
    const focusableElements = page.locator('button, input, textarea, select, a, [tabindex="0"]');
    const elementCount = await focusableElements.count();
    
    if (elementCount > 0) {
      const focusIndicatorResults = [];
      
      for (let i = 0; i < Math.min(5, elementCount); i++) {
        const element = focusableElements.nth(i);
        
        // Focus element
        await element.focus();
        await page.waitForTimeout(300);
        
        // Check for visible focus indicator
        const focusStyles = await element.evaluate(el => {
          const computed = window.getComputedStyle(el);
          return {
            outline: computed.outline,
            outlineWidth: computed.outlineWidth,
            outlineStyle: computed.outlineStyle,
            outlineColor: computed.outlineColor,
            boxShadow: computed.boxShadow,
            border: computed.border
          };
        });
        
        // Check if element has visible focus indicator
        const hasFocusIndicator = 
          focusStyles.outline !== 'none' ||
          focusStyles.outlineWidth !== '0px' ||
          focusStyles.boxShadow.includes('inset') ||
          focusStyles.border.includes('blue') || // Common focus color
          focusStyles.boxShadow.includes('blue');
        
        focusIndicatorResults.push({
          element: i,
          hasFocusIndicator,
          styles: focusStyles
        });
      }
      
      const withFocusIndicators = focusIndicatorResults.filter(r => r.hasFocusIndicator).length;
      const focusIndicatorRate = withFocusIndicators / focusIndicatorResults.length;
      
      console.log(`Focus indicators: ${withFocusIndicators}/${focusIndicatorResults.length} (${(focusIndicatorRate * 100).toFixed(1)}%)`);
      
      // Most elements should have focus indicators
      expect(focusIndicatorRate).toBeGreaterThan(0.7);
    }
  });
});

test.describe('Mobile Accessibility', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
    await page.setViewportSize({ width: 375, height: 667 });
  });

  test('should have appropriate touch target sizes', async ({ page }) => {
    await page.goto('/');
    
    const touchTargets = page.locator('button, a, input, [onclick], [data-testid*="button"]');
    const targetCount = await touchTargets.count();
    
    if (targetCount > 0) {
      const smallTargets = [];
      
      for (let i = 0; i < Math.min(10, targetCount); i++) {
        const target = touchTargets.nth(i);
        const box = await target.boundingBox();
        
        if (box) {
          const meetsMinimum = box.width >= 44 && box.height >= 44;
          
          if (!meetsMinimum) {
            smallTargets.push({
              index: i,
              size: `${box.width}x${box.height}px`,
              text: await target.textContent()
            });
          }
        }
      }
      
      console.log(`Touch targets: ${smallTargets.length}/${Math.min(10, targetCount)} below 44px minimum`);
      
      if (smallTargets.length > 0) {
        console.warn('Small touch targets:', smallTargets);
      }
      
      // Most touch targets should meet minimum size
      expect(smallTargets.length).toBeLessThan(targetCount * 0.3);
    }
  });

  test('should support voice control simulation', async ({ page }) => {
    await page.goto('/');
    
    // Test voice command simulation by finding elements with voice-friendly labels
    const voiceTargets = await page.locator('button, input, textarea').all();
    
    const voiceCompatible = [];
    
    for (const target of voiceTargets) {
      const label = await target.getAttribute('aria-label') || await target.textContent();
      
      if (label && label.trim().length > 0) {
        voiceCompatible.push({
          element: await target.evaluate(el => el.tagName),
          label: label.trim(),
          hasUniqueLabel: true // Simplified check
        });
      }
    }
    
    console.log(`Voice-compatible elements: ${voiceCompatible.length}/${voiceTargets.length}`);
    
    if (voiceCompatible.length > 0) {
      console.log('Sample voice labels:', voiceCompatible.slice(0, 3).map(v => v.label));
    }
    
    // Should have reasonable voice compatibility
    expect(voiceCompatible.length).toBeGreaterThan(voiceTargets.length * 0.5);
  });

  test('should handle zoom up to 200% without horizontal scroll', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    // Simulate 200% zoom
    await page.evaluate(() => {
      document.body.style.zoom = '2';
    });
    
    await page.waitForTimeout(1000);
    
    // Check for horizontal overflow
    const hasHorizontalOverflow = await page.evaluate(() => {
      return document.documentElement.scrollWidth > window.innerWidth;
    });
    
    console.log(`200% zoom: Horizontal overflow: ${hasHorizontalOverflow}`);
    
    // Should not have horizontal overflow at 200% zoom
    expect(hasHorizontalOverflow).toBe(false);
    
    // Reset zoom
    await page.evaluate(() => {
      document.body.style.zoom = '1';
    });
  });
});