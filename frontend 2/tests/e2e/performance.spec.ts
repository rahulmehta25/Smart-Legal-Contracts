import { test, expect } from '@playwright/test';
import { TestHelpers } from '../utils/test-helpers';
import { TestData } from '../fixtures/test-data';

/**
 * Performance Tests
 * 
 * Tests application performance including page load times,
 * resource usage, and responsiveness under various conditions.
 */

test.describe('Page Load Performance', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
  });

  test('should load homepage within acceptable time', async ({ page }) => {
    const startTime = Date.now();
    
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    const loadTime = Date.now() - startTime;
    
    console.log(`Homepage load time: ${loadTime}ms`);
    
    // Should load within 5 seconds on local development
    expect(loadTime).toBeLessThan(5000);
    
    // Check Core Web Vitals
    const webVitals = await page.evaluate(() => {
      return new Promise(resolve => {
        const vitals: any = {};
        
        // First Contentful Paint
        new PerformanceObserver((list) => {
          const entries = list.getEntries();
          entries.forEach((entry: any) => {
            if (entry.name === 'first-contentful-paint') {
              vitals.fcp = entry.startTime;
            }
          });
        }).observe({ entryTypes: ['paint'] });
        
        // Largest Contentful Paint
        new PerformanceObserver((list) => {
          const entries = list.getEntries();
          const lastEntry = entries[entries.length - 1];
          vitals.lcp = lastEntry.startTime;
        }).observe({ entryTypes: ['largest-contentful-paint'] });
        
        // Cumulative Layout Shift
        let clsValue = 0;
        new PerformanceObserver((list) => {
          for (const entry of list.getEntries() as any[]) {
            if (!entry.hadRecentInput) {
              clsValue += entry.value;
            }
          }
          vitals.cls = clsValue;
        }).observe({ entryTypes: ['layout-shift'] });
        
        setTimeout(() => resolve(vitals), 3000);
      });
    });

    console.log('Web Vitals:', webVitals);
    
    if ((webVitals as any).fcp) {
      expect((webVitals as any).fcp).toBeLessThan(2500); // Good FCP
    }
    
    if ((webVitals as any).cls !== undefined) {
      expect((webVitals as any).cls).toBeLessThan(0.1); // Good CLS
    }
  });

  test('should load efficiently on slow networks', async ({ page }) => {
    // Simulate slow 3G connection
    await page.route('**/*', route => {
      setTimeout(() => route.continue(), 300); // 300ms delay per request
    });

    const startTime = Date.now();
    await page.goto('/');
    await page.waitForLoadState('domcontentloaded');
    const loadTime = Date.now() - startTime;
    
    console.log(`Slow network load time: ${loadTime}ms`);
    
    // Should still load within reasonable time on slow connection
    expect(loadTime).toBeLessThan(15000); // 15 seconds max on slow network
  });

  test('should minimize resource requests', async ({ page }) => {
    const requests: string[] = [];
    
    page.on('request', request => {
      requests.push(request.url());
    });

    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    console.log(`Total requests made: ${requests.length}`);
    
    // Analyze request types
    const jsRequests = requests.filter(url => url.includes('.js'));
    const cssRequests = requests.filter(url => url.includes('.css'));
    const imageRequests = requests.filter(url => url.match(/\.(png|jpg|jpeg|gif|svg|webp)$/i));
    const apiRequests = requests.filter(url => url.includes('/api/'));
    
    console.log('Resource breakdown:', {
      total: requests.length,
      javascript: jsRequests.length,
      css: cssRequests.length,
      images: imageRequests.length,
      api: apiRequests.length
    });
    
    // Should not have excessive requests
    expect(requests.length).toBeLessThan(50); // Reasonable limit for SPA
    expect(jsRequests.length).toBeLessThan(20); // Shouldn't load too many JS files
  });

  test('should handle large document analysis efficiently', async ({ page }) => {
    await helpers.mockAPIResponse('**/api/v1/analysis/**', 
      TestData.api.analysisResponse(), 
      200
    );

    await page.goto('/');
    
    const textInput = page.locator('[data-testid="text-input"], textarea').first();
    
    if (await textInput.count() > 0) {
      const largeDoc = TestData.documents.largeDocument();
      
      const startTime = Date.now();
      await textInput.fill(largeDoc.content);
      const inputTime = Date.now() - startTime;
      
      console.log(`Large document input time: ${inputTime}ms`);
      expect(inputTime).toBeLessThan(5000);
      
      // Test analysis submission
      const submitButton = page.locator('[data-testid="analyze-button"], button:has-text("Analyze")').first();
      if (await submitButton.count() > 0) {
        const analysisStartTime = Date.now();
        await submitButton.click();
        
        await page.waitForTimeout(3000);
        const analysisTime = Date.now() - analysisStartTime;
        
        console.log(`Analysis processing time: ${analysisTime}ms`);
        
        // UI should remain responsive during analysis
        const pageResponsive = await page.evaluate(() => document.readyState === 'complete');
        expect(pageResponsive).toBe(true);
      }
    }
  });
});

test.describe('Memory and Resource Usage', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
  });

  test('should not have memory leaks during repeated operations', async ({ page }) => {
    await page.goto('/');
    
    const textInput = page.locator('[data-testid="text-input"], textarea').first();
    
    if (await textInput.count() > 0) {
      // Perform repeated operations to test for leaks
      for (let i = 0; i < 10; i++) {
        await textInput.clear();
        await textInput.fill(`Test iteration ${i}: ${TestData.documents.withArbitration().content}`);
        await page.waitForTimeout(500);
      }
      
      // Check memory usage
      const memoryInfo = await page.evaluate(() => {
        return (performance as any).memory ? {
          usedJSHeapSize: (performance as any).memory.usedJSHeapSize,
          totalJSHeapSize: (performance as any).memory.totalJSHeapSize,
          jsHeapSizeLimit: (performance as any).memory.jsHeapSizeLimit
        } : null;
      });
      
      if (memoryInfo) {
        console.log('Memory usage:', memoryInfo);
        
        // Memory should not exceed reasonable limits (100MB)
        expect(memoryInfo.usedJSHeapSize).toBeLessThan(100 * 1024 * 1024);
      } else {
        console.log('ℹ️ Memory API not available in this browser');
      }
    }
  });

  test('should handle DOM manipulation efficiently', async ({ page }) => {
    await page.goto('/');
    
    // Add many elements to test DOM performance
    const startTime = Date.now();
    
    await page.evaluate(() => {
      const container = document.body;
      for (let i = 0; i < 1000; i++) {
        const div = document.createElement('div');
        div.textContent = `Dynamic element ${i}`;
        div.className = 'performance-test-element';
        container.appendChild(div);
      }
    });
    
    const manipulationTime = Date.now() - startTime;
    console.log(`DOM manipulation time (1000 elements): ${manipulationTime}ms`);
    
    // Should complete quickly
    expect(manipulationTime).toBeLessThan(1000);
    
    // Clean up
    await page.evaluate(() => {
      const elements = document.querySelectorAll('.performance-test-element');
      elements.forEach(el => el.remove());
    });
  });

  test('should maintain performance with multiple WebSocket connections', async ({ page }) => {
    await page.goto('/');
    
    // Create multiple WebSocket connections
    const connectionResults = await page.evaluate(async () => {
      const connections = [];
      const results = [];
      
      for (let i = 0; i < 5; i++) {
        try {
          const ws = new WebSocket('ws://localhost:8000/ws');
          connections.push(ws);
          
          await new Promise((resolve) => {
            ws.onopen = () => {
              results.push({ id: i, status: 'connected' });
              resolve(true);
            };
            ws.onerror = () => {
              results.push({ id: i, status: 'error' });
              resolve(false);
            };
            setTimeout(() => {
              results.push({ id: i, status: 'timeout' });
              resolve(false);
            }, 3000);
          });
        } catch (e) {
          results.push({ id: i, status: 'exception' });
        }
      }
      
      // Close connections
      connections.forEach(ws => {
        try { ws.close(); } catch {}
      });
      
      return results;
    });
    
    console.log('Multiple WebSocket connections:', connectionResults);
    
    const successfulConnections = connectionResults.filter(r => r.status === 'connected').length;
    console.log(`Successful connections: ${successfulConnections}/5`);
    
    // Should handle multiple connections without major issues
    expect(successfulConnections).toBeGreaterThanOrEqual(3);
  });
});

test.describe('Accessibility Performance', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
  });

  test('should maintain fast keyboard navigation', async ({ page }) => {
    await page.goto('/');
    
    // Test keyboard navigation speed
    const startTime = Date.now();
    
    for (let i = 0; i < 10; i++) {
      await page.keyboard.press('Tab');
      await page.waitForTimeout(50);
    }
    
    const navigationTime = Date.now() - startTime;
    console.log(`Keyboard navigation time (10 tabs): ${navigationTime}ms`);
    
    // Navigation should be responsive
    expect(navigationTime).toBeLessThan(2000);
  });

  test('should handle screen reader simulation efficiently', async ({ page }) => {
    await page.goto('/');
    
    // Simulate screen reader by querying accessibility tree
    const startTime = Date.now();
    
    const accessibilityTree = await page.accessibility.snapshot();
    const queryTime = Date.now() - startTime;
    
    console.log(`Accessibility tree query time: ${queryTime}ms`);
    expect(queryTime).toBeLessThan(1000);
    
    // Should have meaningful accessibility tree
    expect(accessibilityTree).toBeTruthy();
    
    if (accessibilityTree?.children) {
      console.log(`Accessibility tree has ${accessibilityTree.children.length} top-level elements`);
      expect(accessibilityTree.children.length).toBeGreaterThan(0);
    }
  });

  test('should handle focus management efficiently', async ({ page }) => {
    await page.goto('/');
    
    // Test focus trapping in modals/dialogs
    const modalTriggers = page.locator('[data-testid*="modal"], [data-testid*="dialog"], button:has-text("Help")');
    
    if (await modalTriggers.count() > 0) {
      await modalTriggers.first().click();
      await page.waitForTimeout(1000);
      
      // Test rapid focus changes
      const startTime = Date.now();
      
      for (let i = 0; i < 5; i++) {
        await page.keyboard.press('Tab');
        await page.waitForTimeout(10);
      }
      
      const focusTime = Date.now() - startTime;
      console.log(`Focus management time: ${focusTime}ms`);
      
      expect(focusTime).toBeLessThan(500);
      
      // Close modal
      await page.keyboard.press('Escape');
    }
  });
});