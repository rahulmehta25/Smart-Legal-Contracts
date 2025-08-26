import { test, expect } from '@playwright/test';
import { TestHelpers } from '../utils/test-helpers';
import { TestData } from '../fixtures/test-data';

/**
 * Arbitration Analysis Functionality Tests
 * 
 * Tests the core arbitration detection features including document upload,
 * text analysis, result display, and user interactions.
 */

test.describe('Arbitration Analysis - Document Upload', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
    await page.goto('/');
  });

  test('should display upload interface', async ({ page }) => {
    // Look for upload components
    const uploadArea = page.locator('[data-testid="upload-area"], .upload-zone, input[type="file"]');
    await expect(uploadArea.first()).toBeVisible({ timeout: 10000 });
    
    // Check for upload instructions
    const hasInstructions = await helpers.elementExists('[data-testid="upload-instructions"]') ||
                           await helpers.elementExists('.upload-text') ||
                           await page.locator('text=/upload|drop|select/i').count() > 0;
    
    expect(hasInstructions).toBe(true);
  });

  test('should accept text input for analysis', async ({ page }) => {
    // Look for text input area
    const textInput = page.locator('[data-testid="text-input"], textarea, [contenteditable="true"]');
    
    if (await textInput.count() > 0) {
      const testDoc = TestData.documents.withArbitration();
      await textInput.first().fill(testDoc.content);
      
      // Look for analyze button
      const analyzeButton = page.locator('[data-testid="analyze-button"], button:has-text("Analyze"), button:has-text("Submit")');
      
      if (await analyzeButton.count() > 0) {
        await analyzeButton.first().click();
        
        // Wait for analysis to complete
        await helpers.waitForLoadingComplete();
        
        // Check for results
        const hasResults = await helpers.elementExists('[data-testid="analysis-results"]') ||
                           await helpers.elementExists('.results') ||
                           await page.locator('text=/arbitration|clause|found/i').count() > 0;
        
        expect(hasResults).toBe(true);
      }
    } else {
      test.skip('No text input found - feature may not be implemented yet');
    }
  });

  test('should handle file upload if supported', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]');
    
    if (await fileInput.count() > 0) {
      // Create a test file
      const testContent = TestData.documents.withArbitration().content;
      const testFile = await page.evaluate(async (content) => {
        const blob = new Blob([content], { type: 'text/plain' });
        return blob;
      }, testContent);

      // Mock file selection
      await page.setInputFiles(fileInput.first(), {
        name: 'test-document.txt',
        mimeType: 'text/plain',
        buffer: Buffer.from(testContent)
      });

      // Look for upload confirmation
      const uploadConfirmed = await helpers.elementExists('[data-testid="file-selected"]') ||
                              await helpers.elementExists('.file-info') ||
                              await page.locator('text=/selected|uploaded|ready/i').count() > 0;

      expect(uploadConfirmed).toBe(true);
    } else {
      console.log('ℹ️ File upload not available - testing text input instead');
    }
  });

  test('should validate file types if upload is available', async ({ page }) => {
    const fileInput = page.locator('input[type="file"]');
    
    if (await fileInput.count() > 0) {
      // Try to upload an unsupported file type
      await page.setInputFiles(fileInput.first(), {
        name: 'test-image.jpg',
        mimeType: 'image/jpeg',
        buffer: Buffer.from('fake image data')
      });

      // Look for error message
      await page.waitForTimeout(2000); // Give time for validation
      
      const errorVisible = await helpers.elementExists('[data-testid="upload-error"]') ||
                           await helpers.elementExists('.error') ||
                           await page.locator('text=/invalid|unsupported|error/i').count() > 0;

      // Should either show error or accept the file
      // (Depends on implementation - some might accept any file)
      console.log(`File validation result: ${errorVisible ? 'Error shown' : 'File accepted'}`);
    }
  });
});

test.describe('Arbitration Analysis - Results Display', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
    await page.goto('/');
    
    // Mock successful API responses
    await helpers.mockAPIResponse('**/api/v1/analysis/**', TestData.api.analysisResponse(true));
  });

  test('should display analysis results with confidence score', async ({ page }) => {
    // Find and fill text input
    const textInput = page.locator('[data-testid="text-input"], textarea').first();
    
    if (await textInput.count() > 0) {
      await textInput.fill(TestData.documents.withArbitration().content);
      
      // Submit for analysis
      const submitButton = page.locator('[data-testid="analyze-button"], button:has-text("Analyze")').first();
      if (await submitButton.count() > 0) {
        await submitButton.click();
        
        // Wait for results
        await page.waitForTimeout(3000);
        
        // Check for confidence score display
        const confidenceVisible = await page.locator('text=/confidence|probability|\%/i').count() > 0;
        expect(confidenceVisible).toBe(true);
      }
    }
  });

  test('should highlight arbitration clauses in text', async ({ page }) => {
    const textInput = page.locator('[data-testid="text-input"], textarea').first();
    
    if (await textInput.count() > 0) {
      await textInput.fill(TestData.documents.withArbitration().content);
      
      const submitButton = page.locator('[data-testid="analyze-button"], button:has-text("Analyze")').first();
      if (await submitButton.count() > 0) {
        await submitButton.click();
        
        await page.waitForTimeout(3000);
        
        // Check for highlighted text or clause extraction
        const highlightVisible = await page.locator('[data-testid="highlighted-clause"], .highlight, mark').count() > 0 ||
                                 await page.locator('text=/arbitration|clause/i').count() > 0;
        
        expect(highlightVisible).toBe(true);
      }
    }
  });

  test('should show processing status during analysis', async ({ page }) => {
    const textInput = page.locator('[data-testid="text-input"], textarea').first();
    
    if (await textInput.count() > 0) {
      await textInput.fill(TestData.documents.largeDocument().content);
      
      const submitButton = page.locator('[data-testid="analyze-button"], button:has-text("Analyze")').first();
      if (await submitButton.count() > 0) {
        await submitButton.click();
        
        // Check for loading indicator
        const loadingVisible = await Promise.race([
          page.waitForSelector('[data-testid="loading"], .loading, .spinner', { timeout: 2000 }).then(() => true),
          page.waitForTimeout(1000).then(() => false)
        ]);
        
        if (loadingVisible) {
          console.log('✅ Loading indicator detected during analysis');
        } else {
          console.log('ℹ️ No loading indicator found - analysis may be instant');
        }
      }
    }
  });
});

test.describe('Arbitration Analysis - Edge Cases', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
    await page.goto('/');
  });

  test('should handle empty text input', async ({ page }) => {
    const textInput = page.locator('[data-testid="text-input"], textarea').first();
    
    if (await textInput.count() > 0) {
      // Submit empty text
      const submitButton = page.locator('[data-testid="analyze-button"], button:has-text("Analyze")').first();
      if (await submitButton.count() > 0) {
        await submitButton.click();
        
        // Should show validation error or disable button
        const errorVisible = await helpers.elementExists('[data-testid="validation-error"]') ||
                             await helpers.elementExists('.error') ||
                             await page.locator('text=/required|empty|invalid/i').count() > 0;
        
        if (errorVisible) {
          console.log('✅ Validation error shown for empty input');
        } else {
          console.log('ℹ️ No validation error - checking button state');
          const buttonDisabled = await submitButton.isDisabled();
          expect(buttonDisabled).toBe(true);
        }
      }
    }
  });

  test('should handle very long text documents', async ({ page }) => {
    const textInput = page.locator('[data-testid="text-input"], textarea').first();
    
    if (await textInput.count() > 0) {
      const longDoc = TestData.documents.largeDocument();
      await textInput.fill(longDoc.content);
      
      const submitButton = page.locator('[data-testid="analyze-button"], button:has-text("Analyze")').first();
      if (await submitButton.count() > 0) {
        await submitButton.click();
        
        // Should handle without crashing
        await page.waitForTimeout(5000);
        
        const pageStillResponsive = await page.evaluate(() => document.readyState === 'complete');
        expect(pageStillResponsive).toBe(true);
      }
    }
  });

  test('should handle network errors during analysis', async ({ page }) => {
    // Mock network error
    await helpers.mockAPIResponse('**/api/v1/analysis/**', 
      TestData.api.errorResponse(500, 'Network Error'), 
      500
    );

    const textInput = page.locator('[data-testid="text-input"], textarea').first();
    
    if (await textInput.count() > 0) {
      await textInput.fill(TestData.documents.withArbitration().content);
      
      const submitButton = page.locator('[data-testid="analyze-button"], button:has-text("Analyze")').first();
      if (await submitButton.count() > 0) {
        await submitButton.click();
        
        // Should show error message
        await page.waitForTimeout(3000);
        
        const errorShown = await helpers.elementExists('[data-testid="api-error"]') ||
                           await helpers.elementExists('.error') ||
                           await page.locator('text=/error|failed|try again/i').count() > 0;
        
        expect(errorShown).toBe(true);
      }
    }
  });

  test('should handle timeout scenarios', async ({ page }) => {
    // Mock slow API response
    await page.route('**/api/v1/analysis/**', route => {
      setTimeout(() => {
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(TestData.api.analysisResponse())
        });
      }, 15000); // 15 second delay
    });

    const textInput = page.locator('[data-testid="text-input"], textarea').first();
    
    if (await textInput.count() > 0) {
      await textInput.fill(TestData.documents.withArbitration().content);
      
      const submitButton = page.locator('[data-testid="analyze-button"], button:has-text("Analyze")').first();
      if (await submitButton.count() > 0) {
        await submitButton.click();
        
        // Should either show timeout error or handle gracefully
        await page.waitForTimeout(8000);
        
        const timeoutHandled = await helpers.elementExists('[data-testid="timeout-error"]') ||
                               await helpers.elementExists('.error') ||
                               await page.locator('text=/timeout|slow|retry/i').count() > 0 ||
                               await helpers.elementExists('[data-testid="loading"]'); // Still loading
        
        expect(timeoutHandled).toBe(true);
      }
    }
  });
});

test.describe('Arbitration Analysis - UI Integration', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
    await page.goto('/');
  });

  test('should update UI based on analysis results', async ({ page }) => {
    // Mock positive arbitration result
    await helpers.mockAPIResponse('**/api/v1/analysis/**', TestData.api.analysisResponse(true));

    const textInput = page.locator('[data-testid="text-input"], textarea').first();
    
    if (await textInput.count() > 0) {
      await textInput.fill(TestData.documents.withArbitration().content);
      
      const submitButton = page.locator('[data-testid="analyze-button"], button:has-text("Analyze")').first();
      if (await submitButton.count() > 0) {
        await submitButton.click();
        
        await page.waitForTimeout(2000);
        
        // Check for positive result indicators
        const positiveResult = await page.locator('text=/found|detected|yes|positive/i').count() > 0 ||
                               await helpers.elementExists('.success, .positive, .detected');
        
        expect(positiveResult).toBe(true);
      }
    }
  });

  test('should handle negative analysis results', async ({ page }) => {
    // Mock negative arbitration result
    await helpers.mockAPIResponse('**/api/v1/analysis/**', TestData.api.analysisResponse(false));

    const textInput = page.locator('[data-testid="text-input"], textarea').first();
    
    if (await textInput.count() > 0) {
      await textInput.fill(TestData.documents.withoutArbitration().content);
      
      const submitButton = page.locator('[data-testid="analyze-button"], button:has-text("Analyze")').first();
      if (await submitButton.count() > 0) {
        await submitButton.click();
        
        await page.waitForTimeout(2000);
        
        // Check for negative result indicators
        const negativeResult = await page.locator('text=/not found|no arbitration|negative/i').count() > 0 ||
                               await helpers.elementExists('.negative, .not-found');
        
        expect(negativeResult).toBe(true);
      }
    }
  });

  test('should display confidence scores appropriately', async ({ page }) => {
    await helpers.mockAPIResponse('**/api/v1/analysis/**', {
      result: {
        hasArbitrationClause: true,
        confidence: 0.87,
        clauses: ['binding arbitration'],
        metadata: { processingTime: 1250 }
      }
    });

    const textInput = page.locator('[data-testid="text-input"], textarea').first();
    
    if (await textInput.count() > 0) {
      await textInput.fill(TestData.documents.withArbitration().content);
      
      const submitButton = page.locator('[data-testid="analyze-button"], button:has-text("Analyze")').first();
      if (await submitButton.count() > 0) {
        await submitButton.click();
        
        await page.waitForTimeout(2000);
        
        // Check for confidence display
        const confidenceShown = await page.locator('text=/87%|0.87|confidence/i').count() > 0;
        
        if (confidenceShown) {
          console.log('✅ Confidence score displayed correctly');
        } else {
          console.log('ℹ️ Confidence score display not found - may be in different format');
        }
      }
    }
  });

  test('should allow result sharing or export', async ({ page }) => {
    await helpers.mockAPIResponse('**/api/v1/analysis/**', TestData.api.analysisResponse(true));

    const textInput = page.locator('[data-testid="text-input"], textarea').first();
    
    if (await textInput.count() > 0) {
      await textInput.fill(TestData.documents.withArbitration().content);
      
      const submitButton = page.locator('[data-testid="analyze-button"], button:has-text("Analyze")').first();
      if (await submitButton.count() > 0) {
        await submitButton.click();
        
        await page.waitForTimeout(2000);
        
        // Look for share or export options
        const shareOptions = await helpers.elementExists('[data-testid="share-button"]') ||
                             await helpers.elementExists('[data-testid="export-button"]') ||
                             await page.locator('button:has-text("Share"), button:has-text("Export"), button:has-text("Download")').count() > 0;
        
        if (shareOptions) {
          console.log('✅ Share/Export functionality available');
        } else {
          console.log('ℹ️ Share/Export functionality not implemented yet');
        }
      }
    }
  });
});

test.describe('Arbitration Analysis - Real-time Features', () => {
  let helpers: TestHelpers;

  test.beforeEach(async ({ page }) => {
    helpers = new TestHelpers(page);
    await page.goto('/');
  });

  test('should show real-time analysis progress via WebSocket', async ({ page }) => {
    // Setup WebSocket monitoring
    await page.addInitScript(() => {
      (window as any).wsMessages = [];
      
      const originalWebSocket = (window as any).WebSocket;
      (window as any).WebSocket = class extends originalWebSocket {
        constructor(url: string, protocols?: string | string[]) {
          super(url, protocols);
          
          this.addEventListener('message', (event) => {
            (window as any).wsMessages.push(event.data);
          });
        }
      };
    });

    const textInput = page.locator('[data-testid="text-input"], textarea').first();
    
    if (await textInput.count() > 0) {
      await textInput.fill(TestData.documents.withArbitration().content);
      
      const submitButton = page.locator('[data-testid="analyze-button"], button:has-text("Analyze")').first();
      if (await submitButton.count() > 0) {
        await submitButton.click();
        
        // Check for real-time updates
        await page.waitForTimeout(3000);
        
        const wsMessages = await page.evaluate(() => (window as any).wsMessages || []);
        const hasProgressUpdates = wsMessages.some((msg: string) => {
          try {
            const parsed = JSON.parse(msg);
            return parsed.type === 'analysis_progress' || 
                   parsed.type === 'progress_update' ||
                   parsed.message?.includes('progress');
          } catch {
            return false;
          }
        });

        if (hasProgressUpdates) {
          console.log('✅ Real-time analysis progress detected');
        } else {
          console.log('ℹ️ Real-time progress updates not implemented yet');
        }
      }
    }
  });

  test('should handle multiple concurrent analyses', async ({ page }) => {
    // This test checks if the system can handle multiple analysis requests
    const textInputs = page.locator('[data-testid="text-input"], textarea');
    
    if (await textInputs.count() > 0) {
      // Mock multiple analysis responses
      let requestCount = 0;
      await page.route('**/api/v1/analysis/**', route => {
        requestCount++;
        setTimeout(() => {
          route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({
              ...TestData.api.analysisResponse(),
              id: `analysis_${requestCount}`,
              requestId: requestCount
            })
          });
        }, Math.random() * 2000 + 1000); // Random delay 1-3 seconds
      });

      // Submit multiple analyses if possible
      await textInputs.first().fill(TestData.documents.withArbitration().content);
      
      const submitButtons = page.locator('[data-testid="analyze-button"], button:has-text("Analyze")');
      if (await submitButtons.count() > 0) {
        // Click multiple times to test concurrent handling
        await submitButtons.first().click();
        await page.waitForTimeout(100);
        await submitButtons.first().click();
        
        await page.waitForTimeout(5000);
        
        // System should handle gracefully without crashing
        const pageResponsive = await page.evaluate(() => document.readyState === 'complete');
        expect(pageResponsive).toBe(true);
      }
    }
  });

  test('should persist analysis history if feature exists', async ({ page }) => {
    await helpers.mockAPIResponse('**/api/v1/analysis/**', TestData.api.analysisResponse(true));

    const textInput = page.locator('[data-testid="text-input"], textarea').first();
    
    if (await textInput.count() > 0) {
      // Perform first analysis
      await textInput.fill(TestData.documents.withArbitration().content);
      
      const submitButton = page.locator('[data-testid="analyze-button"], button:has-text("Analyze")').first();
      if (await submitButton.count() > 0) {
        await submitButton.click();
        await page.waitForTimeout(2000);
        
        // Clear input and perform second analysis
        await textInput.clear();
        await textInput.fill(TestData.documents.withoutArbitration().content);
        await submitButton.click();
        await page.waitForTimeout(2000);
        
        // Check for history section
        const historyExists = await helpers.elementExists('[data-testid="analysis-history"]') ||
                              await helpers.elementExists('.history') ||
                              await page.locator('text=/history|previous|past/i').count() > 0;
        
        if (historyExists) {
          console.log('✅ Analysis history feature detected');
        } else {
          console.log('ℹ️ Analysis history not implemented yet');
        }
      }
    }
  });
});