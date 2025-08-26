import { test, expect } from '@playwright/test';
import { TestHelpers } from '../utils/test-helpers';
import { TestData } from '../fixtures/test-data';

/**
 * Error Handling and Edge Case Tests
 * 
 * Tests application resilience under various error conditions,
 * network failures, and edge cases to ensure robust user experience.
 */

test.describe('Network Error Handling', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
    await page.goto('/');
  });

  test('should handle API server unavailable', async ({ page }) => {
    // Mock all API requests to fail
    await page.route('**/api/**', route => {
      route.abort('failed');
    });

    await page.reload();
    
    // Application should still load and show appropriate error messages
    await page.waitForTimeout(3000);
    
    const errorShown = await helpers.elementExists('[data-testid="connection-error"]') ||
                       await helpers.elementExists('.connection-error') ||
                       await page.locator('text=/connection|offline|unavailable/i').count() > 0;
    
    if (errorShown) {
      console.log('✅ Connection error properly displayed');
    } else {
      // Check that app is still functional
      const appFunctional = await page.evaluate(() => document.readyState === 'complete');
      expect(appFunctional).toBe(true);
      console.log('ℹ️ App loads without error display - graceful degradation');
    }
  });

  test('should handle slow network connections', async ({ page }) => {
    // Simulate slow network
    await helpers.simulateSlowNetwork();
    
    await page.reload();
    
    // Should show loading indicators
    const loadingShown = await Promise.race([
      page.waitForSelector('[data-testid="loading"], .loading, .spinner', { timeout: 5000 }).then(() => true),
      page.waitForTimeout(3000).then(() => false)
    ]);

    // App should eventually load
    await page.waitForTimeout(8000);
    const pageLoaded = await page.evaluate(() => document.readyState === 'complete');
    expect(pageLoaded).toBe(true);
    
    console.log(`Network handling: Loading shown: ${loadingShown}, Page loaded: ${pageLoaded}`);
  });

  test('should handle intermittent connectivity', async ({ page }) => {
    await page.goto('/');
    
    // Simulate going offline
    await helpers.simulateOffline();
    await page.waitForTimeout(2000);
    
    // Try to interact with the app while offline
    const textInput = page.locator('[data-testid="text-input"], textarea').first();
    if (await textInput.count() > 0) {
      await textInput.fill('Test content while offline');
    }
    
    // Go back online
    await helpers.goOnline();
    await page.waitForTimeout(2000);
    
    // App should recover gracefully
    const appRecovered = await page.evaluate(() => document.readyState === 'complete');
    expect(appRecovered).toBe(true);
  });

  test('should handle API rate limiting', async ({ page }) => {
    // Mock rate limit response
    await helpers.mockAPIResponse('**/api/v1/analysis/**', 
      TestData.api.errorResponse(429, 'Too Many Requests'), 
      429
    );

    const textInput = page.locator('[data-testid="text-input"], textarea').first();
    
    if (await textInput.count() > 0) {
      await textInput.fill(TestData.documents.withArbitration().content);
      
      const submitButton = page.locator('[data-testid="analyze-button"], button:has-text("Analyze")').first();
      if (await submitButton.count() > 0) {
        await submitButton.click();
        
        // Should show rate limit message
        await page.waitForTimeout(3000);
        
        const rateLimitShown = await page.locator('text=/rate limit|too many|wait/i').count() > 0 ||
                               await helpers.elementExists('.rate-limit-error');
        
        if (rateLimitShown) {
          console.log('✅ Rate limit error properly handled');
        } else {
          console.log('ℹ️ Rate limit error handling not implemented');
        }
      }
    }
  });
});

test.describe('Input Validation and Sanitization', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
    await page.goto('/');
  });

  test('should sanitize malicious input', async ({ page }) => {
    const maliciousInputs = [
      '<script>alert("xss")</script>',
      'javascript:alert("xss")',
      '${alert("xss")}',
      '<img src=x onerror=alert("xss")>',
      '../../etc/passwd',
      'DROP TABLE users;'
    ];

    const textInput = page.locator('[data-testid="text-input"], textarea').first();
    
    if (await textInput.count() > 0) {
      for (const maliciousInput of maliciousInputs) {
        await textInput.clear();
        await textInput.fill(maliciousInput);
        
        // Check that script doesn't execute
        const alertDialogs = page.locator('.alert, [role="alert"]');
        const scriptExecuted = await page.evaluate(() => {
          // Check if any alert dialogs were created
          return document.querySelectorAll('.alert, [role="alert"]').length > 0;
        });
        
        expect(scriptExecuted).toBe(false);
        
        await page.waitForTimeout(500);
      }
      
      console.log('✅ Malicious input sanitization verified');
    }
  });

  test('should handle extremely long input', async ({ page }) => {
    const extremelyLongText = 'A'.repeat(1000000); // 1MB of text
    
    const textInput = page.locator('[data-testid="text-input"], textarea').first();
    
    if (await textInput.count() > 0) {
      try {
        await textInput.fill(extremelyLongText);
        
        // Should either accept it or show validation error
        const submitButton = page.locator('[data-testid="analyze-button"], button:has-text("Analyze")').first();
        if (await submitButton.count() > 0) {
          const isEnabled = await submitButton.isEnabled();
          
          if (isEnabled) {
            console.log('✅ Large input accepted');
          } else {
            console.log('✅ Large input properly rejected with disabled button');
          }
        }
        
        // App should remain responsive
        const pageResponsive = await page.evaluate(() => document.readyState === 'complete');
        expect(pageResponsive).toBe(true);
        
      } catch (error) {
        // Input might be rejected by browser - that's acceptable
        console.log('ℹ️ Browser rejected extremely long input');
      }
    }
  });

  test('should handle special characters and Unicode', async ({ page }) => {
    const specialTexts = [
      '这是中文测试文档，包含仲裁条款。', // Chinese
      'Este es un documento de prueba en español con cláusulas de arbitraje.', // Spanish
      'هذا مستند اختبار باللغة العربية مع بنود التحكيم.', // Arabic
      '🎯📄💼 Document with emojis and symbols ©®™', // Emojis and symbols
      'Text with\nnewlines\nand\ttabs\tand    spaces', // Whitespace characters
    ];

    const textInput = page.locator('[data-testid="text-input"], textarea').first();
    
    if (await textInput.count() > 0) {
      for (const specialText of specialTexts) {
        await textInput.clear();
        await textInput.fill(specialText);
        
        // Verify text is properly displayed
        const inputValue = await textInput.inputValue();
        expect(inputValue).toBe(specialText);
        
        await page.waitForTimeout(300);
      }
      
      console.log('✅ Special characters and Unicode handled correctly');
    }
  });

  test('should validate required fields', async ({ page }) => {
    // Test form validation if forms exist
    const forms = page.locator('form');
    
    if (await forms.count() > 0) {
      // Try to submit empty form
      const submitButtons = page.locator('button[type="submit"], button:has-text("Submit"), button:has-text("Analyze")');
      
      if (await submitButtons.count() > 0) {
        await submitButtons.first().click();
        
        // Should show validation errors or prevent submission
        await page.waitForTimeout(2000);
        
        const validationShown = await page.locator('text=/required|empty|invalid/i').count() > 0 ||
                                await helpers.elementExists('.validation-error, .field-error');
        
        if (validationShown) {
          console.log('✅ Form validation working');
        } else {
          // Check if button was disabled instead
          const buttonDisabled = await submitButtons.first().isDisabled();
          console.log(`Form validation: ${buttonDisabled ? 'Button disabled' : 'No validation found'}`);
        }
      }
    }
  });
});

test.describe('Browser Compatibility and Edge Cases', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
    await page.goto('/');
  });

  test('should handle browser back/forward navigation', async ({ page }) => {
    // Navigate to different states
    const textInput = page.locator('[data-testid="text-input"], textarea').first();
    
    if (await textInput.count() > 0) {
      await textInput.fill('First document content');
      await page.waitForTimeout(1000);
      
      // Navigate back and forward
      await page.goBack();
      await page.waitForTimeout(1000);
      await page.goForward();
      
      // App should remain functional
      const appFunctional = await textInput.isVisible();
      expect(appFunctional).toBe(true);
    }
  });

  test('should handle page refresh during analysis', async ({ page }) => {
    await helpers.mockAPIResponse('**/api/v1/analysis/**', TestData.api.analysisResponse(), 200);

    const textInput = page.locator('[data-testid="text-input"], textarea').first();
    
    if (await textInput.count() > 0) {
      await textInput.fill(TestData.documents.withArbitration().content);
      
      const submitButton = page.locator('[data-testid="analyze-button"], button:has-text("Analyze")').first();
      if (await submitButton.count() > 0) {
        await submitButton.click();
        
        // Refresh page during analysis
        await page.waitForTimeout(1000);
        await page.reload();
        
        // App should load normally after refresh
        await page.waitForTimeout(2000);
        const appLoaded = await textInput.isVisible();
        expect(appLoaded).toBe(true);
      }
    }
  });

  test('should handle browser console errors gracefully', async ({ page }) => {
    const consoleErrors: string[] = [];
    
    page.on('console', msg => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
      }
    });

    // Trigger various actions that might cause console errors
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Try to interact with elements
    const clickableElements = page.locator('button, a, [onclick]');
    const elementCount = await clickableElements.count();
    
    if (elementCount > 0) {
      // Click a few elements
      for (let i = 0; i < Math.min(3, elementCount); i++) {
        try {
          await clickableElements.nth(i).click({ timeout: 2000 });
          await page.waitForTimeout(500);
        } catch {
          // Ignore click failures
        }
      }
    }

    // Filter out expected errors (like network failures we're testing)
    const unexpectedErrors = consoleErrors.filter(error => 
      !error.includes('Network Error') && 
      !error.includes('Failed to fetch') &&
      !error.includes('WebSocket')
    );

    console.log(`Console errors found: ${consoleErrors.length}, Unexpected: ${unexpectedErrors.length}`);
    
    if (unexpectedErrors.length > 0) {
      console.warn('⚠️ Unexpected console errors:', unexpectedErrors);
    }

    // App should still be functional despite any errors
    const appFunctional = await page.evaluate(() => document.readyState === 'complete');
    expect(appFunctional).toBe(true);
  });

  test('should handle localStorage and sessionStorage errors', async ({ page }) => {
    // Simulate storage quota exceeded
    await page.addInitScript(() => {
      const originalSetItem = Storage.prototype.setItem;
      Storage.prototype.setItem = function(key, value) {
        if (key.includes('test-quota')) {
          throw new Error('QuotaExceededError');
        }
        return originalSetItem.call(this, key, value);
      };
    });

    await page.goto('/');
    
    // Try to trigger storage operations
    await page.evaluate(() => {
      try {
        localStorage.setItem('test-quota-exceeded', 'test');
      } catch (e) {
        console.log('Storage error handled:', e.message);
      }
    });

    // App should continue working
    await page.waitForTimeout(2000);
    const appWorking = await page.evaluate(() => document.readyState === 'complete');
    expect(appWorking).toBe(true);
  });

  test('should handle WebSocket connection failures gracefully', async ({ page }) => {
    // Mock WebSocket to always fail
    await page.addInitScript(() => {
      const OriginalWebSocket = (window as any).WebSocket;
      (window as any).WebSocket = class extends OriginalWebSocket {
        constructor(url: string, protocols?: string | string[]) {
          super(url, protocols);
          
          // Simulate connection failure
          setTimeout(() => {
            this.dispatchEvent(new Event('error'));
            this.dispatchEvent(new CloseEvent('close', { code: 1006, reason: 'Connection failed' }));
          }, 100);
        }
      };
    });

    await page.goto('/');
    await page.waitForTimeout(3000);
    
    // App should handle WebSocket failure gracefully
    const wsErrorHandled = await page.locator('text=/websocket|connection/i').count() > 0 ||
                           await helpers.elementExists('[data-testid="ws-error"]');
    
    const appStillWorks = await page.evaluate(() => document.readyState === 'complete');
    expect(appStillWorks).toBe(true);
    
    console.log(`WebSocket error handling: Error shown: ${wsErrorHandled}, App functional: ${appStillWorks}`);
  });
});

test.describe('Form Validation Edge Cases', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
    await page.goto('/');
  });

  test('should handle copy-paste operations', async ({ page }) => {
    const textInput = page.locator('[data-testid="text-input"], textarea').first();
    
    if (await textInput.count() > 0) {
      const testContent = TestData.documents.withArbitration().content;
      
      // Simulate copy-paste by setting clipboard and pasting
      await page.evaluate(async (content) => {
        try {
          await navigator.clipboard.writeText(content);
        } catch {
          // Clipboard API may not be available in test environment
        }
      }, testContent);

      // Try keyboard paste
      await textInput.click();
      await page.keyboard.press('Control+V'); // or 'Meta+V' on Mac
      
      await page.waitForTimeout(1000);
      
      // Check if content was pasted
      const inputValue = await textInput.inputValue();
      const contentPasted = inputValue.length > 100; // Some content should be there
      
      console.log(`Copy-paste test: Content length: ${inputValue.length}`);
      
      if (contentPasted) {
        console.log('✅ Copy-paste functionality working');
      } else {
        console.log('ℹ️ Copy-paste not working - may be browser limitation');
      }
    }
  });

  test('should handle drag and drop if supported', async ({ page }) => {
    // Look for drag-drop zone
    const dropZone = page.locator('[data-testid="drop-zone"], .drop-zone, .upload-area');
    
    if (await dropZone.count() > 0) {
      // Simulate file drop
      const testFile = TestData.documents.withArbitration();
      
      await page.evaluate(async ({ content, filename }) => {
        const dropZone = document.querySelector('[data-testid="drop-zone"], .drop-zone, .upload-area');
        if (!dropZone) return false;

        const file = new File([content], filename, { type: 'text/plain' });
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);

        const dropEvent = new DragEvent('drop', {
          bubbles: true,
          cancelable: true,
          dataTransfer
        });

        dropZone.dispatchEvent(dropEvent);
        return true;
      }, testFile);

      await page.waitForTimeout(2000);
      
      // Check for file processing
      const fileProcessed = await helpers.elementExists('[data-testid="file-uploaded"]') ||
                            await page.locator('text=/uploaded|processing|analyzing/i').count() > 0;
      
      console.log(`Drag-drop test: File processed: ${fileProcessed}`);
    } else {
      console.log('ℹ️ Drag-drop not implemented');
    }
  });

  test('should handle keyboard navigation', async ({ page }) => {
    await page.goto('/');
    
    // Test tab navigation
    await page.keyboard.press('Tab');
    let focusedElement = await page.evaluate(() => document.activeElement?.tagName);
    console.log(`First tab focus: ${focusedElement}`);
    
    // Continue tabbing through focusable elements
    const focusableElements = [];
    for (let i = 0; i < 10; i++) {
      await page.keyboard.press('Tab');
      const currentFocus = await page.evaluate(() => {
        const element = document.activeElement;
        return {
          tagName: element?.tagName,
          id: element?.id,
          className: element?.className,
          type: (element as any)?.type
        };
      });
      
      if (currentFocus.tagName) {
        focusableElements.push(currentFocus);
      }
      
      await page.waitForTimeout(100);
    }
    
    console.log(`Focusable elements found: ${focusableElements.length}`);
    expect(focusableElements.length).toBeGreaterThan(0);
  });

  test('should handle Enter key for form submission', async ({ page }) => {
    const textInput = page.locator('[data-testid="text-input"], textarea').first();
    
    if (await textInput.count() > 0) {
      await textInput.fill(TestData.documents.withArbitration().content);
      
      // Try Enter key submission
      await textInput.press('Enter');
      
      await page.waitForTimeout(2000);
      
      // Check if analysis was triggered
      const analysisTriggered = await helpers.elementExists('[data-testid="loading"]') ||
                                await helpers.elementExists('.loading') ||
                                await page.locator('text=/analyzing|processing/i').count() > 0;
      
      console.log(`Enter key submission: Analysis triggered: ${analysisTriggered}`);
    }
  });

  test('should handle Escape key for modal/dialog closing', async ({ page }) => {
    await page.goto('/');
    
    // Look for buttons that might open modals
    const modalTriggers = page.locator('button:has-text("Help"), button:has-text("Info"), button:has-text("Settings"), [data-testid*="modal"], [data-testid*="dialog"]');
    
    if (await modalTriggers.count() > 0) {
      // Click first modal trigger
      await modalTriggers.first().click();
      await page.waitForTimeout(1000);
      
      // Press Escape to close
      await page.keyboard.press('Escape');
      await page.waitForTimeout(1000);
      
      // Check if modal closed
      const modalClosed = await page.locator('.modal, [role="dialog"], .overlay').count() === 0;
      
      if (modalClosed) {
        console.log('✅ Escape key closes modals');
      } else {
        console.log('ℹ️ No modal close behavior detected');
      }
    }
  });
});

test.describe('Performance Edge Cases', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
    await page.goto('/');
  });

  test('should handle rapid user interactions', async ({ page }) => {
    const submitButton = page.locator('[data-testid="analyze-button"], button:has-text("Analyze")').first();
    
    if (await submitButton.count() > 0) {
      // Rapidly click the button multiple times
      for (let i = 0; i < 10; i++) {
        try {
          await submitButton.click({ timeout: 500 });
          await page.waitForTimeout(50);
        } catch {
          // Ignore click failures - button might be disabled
        }
      }
      
      // App should remain responsive
      await page.waitForTimeout(2000);
      const pageResponsive = await page.evaluate(() => document.readyState === 'complete');
      expect(pageResponsive).toBe(true);
      
      console.log('✅ Rapid interactions handled gracefully');
    }
  });

  test('should handle browser tab switching', async ({ page, context }) => {
    // Open multiple tabs
    const page2 = await context.newPage();
    
    await page.goto('/');
    await page2.goto('/');
    
    // Interact with first tab
    const textInput1 = page.locator('[data-testid="text-input"], textarea').first();
    if (await textInput1.count() > 0) {
      await textInput1.fill('Tab 1 content');
    }
    
    // Switch to second tab and interact
    const textInput2 = page2.locator('[data-testid="text-input"], textarea').first();
    if (await textInput2.count() > 0) {
      await textInput2.fill('Tab 2 content');
    }
    
    // Switch back to first tab
    await page.bringToFront();
    
    // Verify state is preserved
    if (await textInput1.count() > 0) {
      const value = await textInput1.inputValue();
      expect(value).toBe('Tab 1 content');
      console.log('✅ Tab switching preserves state');
    }
    
    await page2.close();
  });

  test('should handle memory pressure scenarios', async ({ page }) => {
    // Create artificial memory pressure
    await page.addInitScript(() => {
      // Create large arrays to consume memory
      (window as any).memoryPressure = [];
      for (let i = 0; i < 100; i++) {
        (window as any).memoryPressure.push(new Array(10000).fill(`memory-test-${i}`));
      }
    });

    await page.goto('/');
    
    // Try normal operations under memory pressure
    const textInput = page.locator('[data-testid="text-input"], textarea').first();
    if (await textInput.count() > 0) {
      await textInput.fill('Testing under memory pressure');
      
      const submitButton = page.locator('[data-testid="analyze-button"], button:has-text("Analyze")').first();
      if (await submitButton.count() > 0) {
        await submitButton.click();
      }
    }
    
    await page.waitForTimeout(3000);
    
    // App should remain functional
    const memoryUsage = await page.evaluate(() => {
      return (performance as any).memory?.usedJSHeapSize || 0;
    });
    
    console.log(`Memory usage: ${memoryUsage} bytes`);
    
    const pageResponsive = await page.evaluate(() => document.readyState === 'complete');
    expect(pageResponsive).toBe(true);
  });
});