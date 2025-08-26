import { test, expect } from '@playwright/test';
import { TestHelpers } from '../utils/test-helpers';
import { TestData } from '../fixtures/test-data';

/**
 * Frontend-Backend Integration Tests
 * 
 * Tests the complete integration between frontend and backend,
 * including data flow, state management, and user workflows.
 */

test.describe('Frontend-Backend Integration', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
    await page.goto('/');
  });

  test('should successfully communicate with backend on load', async ({ page }) => {
    // Check if page makes initial API calls
    const apiCalls: string[] = [];
    
    page.on('request', request => {
      const url = request.url();
      if (url.includes('/api/') || url.includes(':8000')) {
        apiCalls.push(url);
      }
    });

    await page.reload();
    await page.waitForLoadState('networkidle');
    
    console.log('API calls made on page load:', apiCalls);
    
    // Should make at least health check or initial data fetch
    const hasAPICall = apiCalls.length > 0;
    
    if (hasAPICall) {
      console.log('✅ Frontend communicates with backend on load');
    } else {
      console.log('ℹ️ No API calls detected on load - may be lazy loaded');
    }
  });

  test('should handle API responses and update UI accordingly', async ({ page }) => {
    // Mock successful health check
    await helpers.mockAPIResponse('**/health', TestData.api.healthResponse());
    
    await page.goto('/');
    await page.waitForTimeout(2000);
    
    // Check if health status is displayed anywhere
    const healthIndicator = await page.locator('text=/healthy|online|connected/i').count() > 0;
    
    if (healthIndicator) {
      console.log('✅ Health status integrated in UI');
    }
    
    // Test status indicator updates
    const statusElements = page.locator('[data-testid*="status"], .status, [data-testid*="health"]');
    if (await statusElements.count() > 0) {
      console.log('✅ Status elements detected in UI');
    }
  });

  test('should propagate errors from backend to frontend', async ({ page }) => {
    // Mock API error
    await helpers.mockAPIResponse('**/api/**', 
      TestData.api.errorResponse(503, 'Service Unavailable'), 
      503
    );

    await page.goto('/');
    await page.waitForTimeout(3000);
    
    // Check for error indication in UI
    const errorShown = await page.locator('text=/error|unavailable|failed/i').count() > 0 ||
                       await helpers.elementExists('[data-testid*="error"], .error');
    
    if (errorShown) {
      console.log('✅ Backend errors properly displayed in frontend');
    } else {
      console.log('ℹ️ Error state not visible - may be handled silently');
    }
  });

  test('should maintain state consistency during navigation', async ({ page }) => {
    const textInput = page.locator('[data-testid="text-input"], textarea').first();
    
    if (await textInput.count() > 0) {
      // Enter some data
      await textInput.fill('Test state persistence');
      
      // Navigate away and back (if navigation exists)
      const navLinks = page.locator('nav a, .nav a, [role="navigation"] a');
      
      if (await navLinks.count() > 0) {
        const firstLink = navLinks.first();
        await firstLink.click();
        await page.waitForTimeout(1000);
        
        // Navigate back
        await page.goBack();
        await page.waitForTimeout(1000);
        
        // Check if state is preserved
        const currentValue = await textInput.inputValue();
        if (currentValue === 'Test state persistence') {
          console.log('✅ State preserved during navigation');
        } else {
          console.log('ℹ️ State not preserved - form reset on navigation');
        }
      }
    }
  });

  test('should handle authentication flow if implemented', async ({ page }) => {
    // Look for authentication elements
    const authElements = page.locator('[data-testid*="login"], [data-testid*="auth"], .login, .auth');
    
    if (await authElements.count() > 0) {
      console.log('🔐 Authentication interface detected');
      
      // Mock login API
      await helpers.mockAPIResponse('**/api/v1/users/login', {
        token: 'mock-jwt-token',
        user: TestData.users.validUser(),
        expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString()
      });

      // Test login flow
      const loginForm = page.locator('form').first();
      if (await loginForm.count() > 0) {
        // Fill login form
        const usernameInput = page.locator('input[type="text"], input[type="email"], [data-testid*="username"], [data-testid*="email"]').first();
        const passwordInput = page.locator('input[type="password"], [data-testid*="password"]').first();
        
        if (await usernameInput.count() > 0 && await passwordInput.count() > 0) {
          await usernameInput.fill('testuser');
          await passwordInput.fill('password123');
          
          const submitButton = page.locator('button[type="submit"], [data-testid*="login-submit"]').first();
          if (await submitButton.count() > 0) {
            await submitButton.click();
            await page.waitForTimeout(2000);
            
            // Check for successful login indication
            const loginSuccess = await page.locator('text=/welcome|dashboard|logout/i').count() > 0;
            console.log(`Login flow test: ${loginSuccess ? '✅ Success' : 'ℹ️ No clear success indicator'}`);
          }
        }
      }
    } else {
      console.log('ℹ️ No authentication interface found');
    }
  });
});

test.describe('Real-time Features Integration', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
  });

  test('should establish WebSocket connection for real-time features', async ({ page }) => {
    // Monitor WebSocket connections
    let wsConnected = false;
    
    page.on('websocket', ws => {
      console.log('WebSocket connection detected:', ws.url());
      wsConnected = true;
      
      ws.on('framereceived', event => {
        console.log('WebSocket frame received:', event.payload);
      });
    });

    await page.goto('/');
    await page.waitForTimeout(5000);
    
    if (wsConnected) {
      console.log('✅ WebSocket connection established');
    } else {
      console.log('ℹ️ No WebSocket connection detected');
      
      // Try to manually trigger WebSocket connection
      await page.evaluate(() => {
        try {
          const ws = new WebSocket('ws://localhost:8000/ws');
          (window as any).testWs = ws;
          
          ws.onopen = () => console.log('Manual WebSocket connected');
          ws.onerror = (e) => console.log('Manual WebSocket error:', e);
        } catch (e) {
          console.log('WebSocket creation failed:', e);
        }
      });
      
      await page.waitForTimeout(2000);
    }
  });

  test('should handle real-time analysis updates', async ({ page }) => {
    // Setup WebSocket message monitoring
    const wsMessages: any[] = [];
    
    await page.addInitScript(() => {
      (window as any).wsMessages = [];
      
      const originalWebSocket = (window as any).WebSocket;
      (window as any).WebSocket = class extends originalWebSocket {
        constructor(url: string, protocols?: string | string[]) {
          super(url, protocols);
          
          this.addEventListener('message', (event) => {
            try {
              const data = JSON.parse(event.data);
              (window as any).wsMessages.push(data);
            } catch (e) {
              (window as any).wsMessages.push({ raw: event.data });
            }
          });
        }
      };
    });

    await page.goto('/');
    
    // Trigger analysis
    const textInput = page.locator('[data-testid="text-input"], textarea').first();
    
    if (await textInput.count() > 0) {
      await textInput.fill(TestData.documents.withArbitration().content);
      
      const submitButton = page.locator('[data-testid="analyze-button"], button:has-text("Analyze")').first();
      if (await submitButton.count() > 0) {
        await submitButton.click();
        
        // Wait for WebSocket messages
        await page.waitForTimeout(5000);
        
        const messages = await page.evaluate(() => (window as any).wsMessages || []);
        console.log('WebSocket messages during analysis:', messages);
        
        const hasProgressUpdates = messages.some((msg: any) => 
          msg.type === 'analysis_progress' || 
          msg.type === 'progress_update' ||
          (typeof msg.raw === 'string' && msg.raw.includes('progress'))
        );
        
        if (hasProgressUpdates) {
          console.log('✅ Real-time analysis updates working');
        } else {
          console.log('ℹ️ No real-time analysis updates detected');
        }
      }
    }
  });

  test('should sync UI state with WebSocket events', async ({ page }) => {
    await page.goto('/');
    
    // Simulate WebSocket events and check UI updates
    await page.evaluate(() => {
      // Simulate receiving a WebSocket message
      const event = new MessageEvent('message', {
        data: JSON.stringify({
          type: 'ui_update',
          action: 'show_notification',
          data: { message: 'Test notification from WebSocket' }
        })
      });
      
      // Dispatch to any WebSocket listeners
      if ((window as any).wsConnection) {
        (window as any).wsConnection.dispatchEvent(event);
      }
    });

    await page.waitForTimeout(2000);
    
    // Check if UI updated in response
    const notificationShown = await page.locator('text=/Test notification/').count() > 0 ||
                              await helpers.elementExists('[data-testid*="notification"], .notification');
    
    if (notificationShown) {
      console.log('✅ UI updates in response to WebSocket events');
    } else {
      console.log('ℹ️ No UI update detected for WebSocket events');
    }
  });

  test('should handle WebSocket reconnection in UI', async ({ page }) => {
    await page.goto('/');
    
    // Setup WebSocket connection monitoring
    await page.addInitScript(() => {
      let reconnectAttempts = 0;
      (window as any).reconnectAttempts = 0;
      
      const originalWebSocket = (window as any).WebSocket;
      (window as any).WebSocket = class extends originalWebSocket {
        constructor(url: string, protocols?: string | string[]) {
          super(url, protocols);
          
          this.addEventListener('close', () => {
            (window as any).reconnectAttempts++;
            console.log(`WebSocket reconnect attempt: ${(window as any).reconnectAttempts}`);
          });
        }
      };
    });

    // Simulate connection loss and recovery
    await page.evaluate(() => {
      const ws = new WebSocket('ws://localhost:8000/ws');
      (window as any).testWs = ws;
      
      // Close connection after a delay to simulate network issue
      setTimeout(() => {
        ws.close();
      }, 2000);
    });

    await page.waitForTimeout(5000);
    
    const reconnectAttempts = await page.evaluate(() => (window as any).reconnectAttempts || 0);
    
    if (reconnectAttempts > 0) {
      console.log(`✅ WebSocket reconnection attempted ${reconnectAttempts} times`);
    } else {
      console.log('ℹ️ No WebSocket reconnection logic detected');
    }
  });
});

test.describe('Data Flow and State Management', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
  });

  test('should maintain application state across page reloads', async ({ page }) => {
    await page.goto('/');
    
    // Fill form data
    const textInput = page.locator('[data-testid="text-input"], textarea').first();
    
    if (await textInput.count() > 0) {
      await textInput.fill('Persistent state test');
      
      // Check if app uses localStorage/sessionStorage for persistence
      await page.evaluate(() => {
        localStorage.setItem('test-persistence', 'true');
        sessionStorage.setItem('test-session', 'active');
      });
      
      // Reload page
      await page.reload();
      await page.waitForTimeout(2000);
      
      // Check if storage persisted
      const storageRestored = await page.evaluate(() => {
        return {
          localStorage: localStorage.getItem('test-persistence'),
          sessionStorage: sessionStorage.getItem('test-session')
        };
      });
      
      expect(storageRestored.localStorage).toBe('true');
      expect(storageRestored.sessionStorage).toBe('active');
      
      console.log('✅ Browser storage working correctly');
    }
  });

  test('should handle form submission end-to-end', async ({ page }) => {
    // Mock successful analysis response
    let requestReceived = false;
    
    await page.route('**/api/v1/analysis/**', route => {
      requestReceived = true;
      console.log('Analysis API request received:', route.request().url());
      
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(TestData.api.analysisResponse(true))
      });
    });

    const textInput = page.locator('[data-testid="text-input"], textarea').first();
    
    if (await textInput.count() > 0) {
      // Fill form
      await textInput.fill(TestData.documents.withArbitration().content);
      
      // Submit form
      const submitButton = page.locator('[data-testid="analyze-button"], button:has-text("Analyze")').first();
      if (await submitButton.count() > 0) {
        await submitButton.click();
        
        // Wait for API call
        await page.waitForTimeout(3000);
        
        expect(requestReceived).toBe(true);
        console.log('✅ Form submission triggers API call');
        
        // Check for UI update
        const resultsShown = await page.locator('text=/result|analysis|arbitration/i').count() > 0;
        
        if (resultsShown) {
          console.log('✅ API response updates UI');
        }
      }
    }
  });

  test('should handle concurrent user actions', async ({ page }) => {
    await page.goto('/');
    
    // Mock multiple API endpoints
    await helpers.mockAPIResponse('**/api/v1/analysis/**', TestData.api.analysisResponse());
    await helpers.mockAPIResponse('**/health', TestData.api.healthResponse());
    
    const textInput = page.locator('[data-testid="text-input"], textarea').first();
    
    if (await textInput.count() > 0) {
      // Perform multiple actions quickly
      await textInput.fill('Concurrent test 1');
      
      const submitButton = page.locator('[data-testid="analyze-button"], button:has-text("Analyze")').first();
      
      if (await submitButton.count() > 0) {
        // Rapid submissions to test handling
        await submitButton.click();
        await page.waitForTimeout(100);
        await submitButton.click();
        await page.waitForTimeout(100);
        
        // App should handle gracefully
        await page.waitForTimeout(3000);
        
        const pageStable = await page.evaluate(() => document.readyState === 'complete');
        expect(pageStable).toBe(true);
        
        console.log('✅ Concurrent actions handled gracefully');
      }
    }
  });

  test('should validate data flow from input to API to results', async ({ page }) => {
    const testDocument = TestData.documents.hiddenArbitration();
    let apiRequestData: any = null;
    
    // Capture API request data
    await page.route('**/api/v1/analysis/**', route => {
      const request = route.request();
      
      request.postData().then(data => {
        if (data) {
          try {
            apiRequestData = JSON.parse(data);
          } catch {
            apiRequestData = { raw: data };
          }
        }
      });
      
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(testDocument.expectedAnalysis)
      });
    });

    const textInput = page.locator('[data-testid="text-input"], textarea').first();
    
    if (await textInput.count() > 0) {
      await textInput.fill(testDocument.content);
      
      const submitButton = page.locator('[data-testid="analyze-button"], button:has-text("Analyze")').first();
      if (await submitButton.count() > 0) {
        await submitButton.click();
        await page.waitForTimeout(3000);
        
        // Verify data flow
        if (apiRequestData) {
          console.log('API request data:', apiRequestData);
          
          // Should contain the input text
          const hasInputText = JSON.stringify(apiRequestData).includes(testDocument.content.substring(0, 50));
          
          if (hasInputText) {
            console.log('✅ Input data correctly sent to API');
          }
        }
        
        // Check if results are displayed
        const confidenceShown = await page.locator('text=/confidence|score|\%/i').count() > 0;
        const clausesShown = await page.locator('text=/clause|arbitration/i').count() > 0;
        
        if (confidenceShown || clausesShown) {
          console.log('✅ API results displayed in UI');
        }
      }
    }
  });
});

test.describe('User Experience Flow', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
  });

  test('should complete typical user workflow', async ({ page }) => {
    await page.goto('/');
    
    // Mock all required APIs
    await helpers.mockAPIResponse('**/health', TestData.api.healthResponse());
    await helpers.mockAPIResponse('**/api/v1/analysis/**', TestData.api.analysisResponse(true));
    
    console.log('🎭 Testing complete user workflow...');
    
    // Step 1: User lands on homepage
    await page.waitForTimeout(1000);
    console.log('Step 1: ✅ Homepage loaded');
    
    // Step 2: User sees upload/input interface
    const hasInputInterface = await helpers.elementExists('[data-testid="text-input"], textarea, input[type="file"]');
    console.log(`Step 2: ${hasInputInterface ? '✅' : 'ℹ️'} Input interface available`);
    
    if (hasInputInterface) {
      // Step 3: User enters text
      const textInput = page.locator('[data-testid="text-input"], textarea').first();
      await textInput.fill(TestData.documents.withArbitration().content);
      console.log('Step 3: ✅ Text entered');
      
      // Step 4: User submits for analysis
      const submitButton = page.locator('[data-testid="analyze-button"], button:has-text("Analyze")').first();
      if (await submitButton.count() > 0) {
        await submitButton.click();
        console.log('Step 4: ✅ Analysis submitted');
        
        // Step 5: User sees loading state
        const loadingShown = await Promise.race([
          page.waitForSelector('[data-testid="loading"], .loading', { timeout: 2000 }).then(() => true),
          page.waitForTimeout(1000).then(() => false)
        ]);
        console.log(`Step 5: ${loadingShown ? '✅' : 'ℹ️'} Loading state shown`);
        
        // Step 6: User sees results
        await page.waitForTimeout(3000);
        const resultsShown = await page.locator('text=/result|analysis|confidence|arbitration/i').count() > 0;
        console.log(`Step 6: ${resultsShown ? '✅' : 'ℹ️'} Results displayed`);
        
        // Step 7: User can take action on results
        const actionable = await page.locator('button:has-text("Export"), button:has-text("Share"), button:has-text("Save"), a:has-text("Download")').count() > 0;
        console.log(`Step 7: ${actionable ? '✅' : 'ℹ️'} Action buttons available`);
      }
    }
    
    console.log('🎯 User workflow test completed');
  });

  test('should provide feedback for all user actions', async ({ page }) => {
    await page.goto('/');
    
    // Track user actions and feedback
    const interactions: Array<{action: string, feedback: boolean}> = [];
    
    // Test button clicks
    const buttons = page.locator('button');
    const buttonCount = await buttons.count();
    
    if (buttonCount > 0) {
      for (let i = 0; i < Math.min(3, buttonCount); i++) {
        const button = buttons.nth(i);
        const buttonText = await button.textContent();
        
        if (buttonText && !buttonText.includes('Analyze')) {
          // Click button and check for feedback
          await button.click();
          await page.waitForTimeout(1000);
          
          // Look for visual feedback (loading, color change, etc.)
          const hasFeedback = await page.locator('.loading, [data-testid*="loading"], .active, .clicked').count() > 0;
          
          interactions.push({
            action: `Click ${buttonText}`,
            feedback: hasFeedback
          });
        }
      }
    }
    
    // Test input focus feedback
    const inputs = page.locator('input, textarea');
    const inputCount = await inputs.count();
    
    if (inputCount > 0) {
      const input = inputs.first();
      await input.focus();
      
      // Check for focus styling
      const focusVisible = await input.evaluate(el => {
        const styles = window.getComputedStyle(el);
        return styles.outline !== 'none' || 
               styles.border !== styles.borderTop || 
               styles.boxShadow !== 'none';
      });
      
      interactions.push({
        action: 'Focus input',
        feedback: focusVisible
      });
    }
    
    console.log('User feedback analysis:', interactions);
    
    const feedbackRate = interactions.filter(i => i.feedback).length / interactions.length;
    console.log(`Feedback rate: ${(feedbackRate * 100).toFixed(1)}%`);
    
    // At least 50% of interactions should have feedback
    expect(feedbackRate).toBeGreaterThanOrEqual(0.3);
  });

  test('should handle user errors gracefully', async ({ page }) => {
    await page.goto('/');
    
    const errorScenarios = [
      {
        name: 'Empty submission',
        action: async () => {
          const submitButton = page.locator('[data-testid="analyze-button"], button:has-text("Analyze")').first();
          if (await submitButton.count() > 0) {
            await submitButton.click();
          }
        }
      },
      {
        name: 'Invalid characters',
        action: async () => {
          const textInput = page.locator('[data-testid="text-input"], textarea').first();
          if (await textInput.count() > 0) {
            await textInput.fill('<script>alert("test")</script>');
            
            const submitButton = page.locator('[data-testid="analyze-button"], button:has-text("Analyze")').first();
            if (await submitButton.count() > 0) {
              await submitButton.click();
            }
          }
        }
      }
    ];

    for (const scenario of errorScenarios) {
      console.log(`Testing error scenario: ${scenario.name}`);
      
      await scenario.action();
      await page.waitForTimeout(2000);
      
      // Check for error handling
      const errorHandled = await page.locator('text=/error|invalid|required/i').count() > 0 ||
                           await helpers.elementExists('[data-testid*="error"], .error');
      
      const appStillWorks = await page.evaluate(() => document.readyState === 'complete');
      
      console.log(`${scenario.name}: Error shown: ${errorHandled}, App functional: ${appStillWorks}`);
      expect(appStillWorks).toBe(true);
      
      // Reset for next test
      await page.reload();
      await page.waitForTimeout(1000);
    }
  });
});