import { test, expect } from '@playwright/test';
import { uploadDocument, uploadDocumentViaDragDrop } from '../utils/upload';
import { takeScreenshot, takeComparisonScreenshots } from '../utils/screenshot';
import { waitForAnalysisComplete, waitForFileUpload, smartWait } from '../utils/wait';
import { enableDemoMode } from '../utils/login';
import { SAMPLE_DOCUMENTS } from '../utils/test-data';

test.describe('Document Upload Flow Visual Tests', () => {
  test.beforeEach(async ({ page }) => {
    await enableDemoMode(page);
    await page.goto('/upload');
    await smartWait(page);
  });

  test('should display upload interface correctly @visual', async ({ page }) => {
    // Verify upload components are visible
    await expect(page.locator('[data-testid="document-uploader"]')).toBeVisible();
    await expect(page.locator('[data-testid="drop-zone"]')).toBeVisible();
    await expect(page.locator('[data-testid="file-input"]')).toBeVisible();
    
    // Take screenshot of initial upload state
    await takeScreenshot(page, 'upload-initial-state', {
      fullPage: true
    });
  });

  test('should show drag and drop zone with proper styling @visual', async ({ page }) => {
    const dropZone = page.locator('[data-testid="drop-zone"]');
    
    // Take normal state screenshot
    await takeScreenshot(page, 'dropzone-normal');
    
    // Simulate drag over state
    await dropZone.hover();
    await page.waitForTimeout(300);
    
    await takeScreenshot(page, 'dropzone-hover');
  });

  test('should handle file selection and display preview @visual @demo', async ({ page }) => {
    await takeComparisonScreenshots(
      page,
      'file-selection',
      async () => {
        // Before: Empty state
        await smartWait(page);
      },
      async () => {
        // After: File selected
        await uploadDocument(page, SAMPLE_DOCUMENTS.arbitration_clause.filename, {
          waitForAnalysis: false
        });
      }
    );
  });

  test('should show upload progress indicator @visual @demo', async ({ page }) => {
    // Start upload
    const fileInput = page.locator('[data-testid="file-input"]');
    await fileInput.setInputFiles({
      name: 'test-document.txt',
      mimeType: 'text/plain',
      buffer: Buffer.from('Test document content with arbitration clause')
    });
    
    // Click upload
    await page.click('[data-testid="upload-button"]');
    
    // Wait for and screenshot progress indicator
    await page.waitForSelector('[data-testid="upload-progress"]', { timeout: 5000 });
    await takeScreenshot(page, 'upload-progress');
    
    // Wait for upload completion
    await waitForFileUpload(page);
    await takeScreenshot(page, 'upload-complete');
  });

  test('should display analysis progress correctly @visual @demo', async ({ page }) => {
    // Upload document and track analysis progress
    await uploadDocument(page, SAMPLE_DOCUMENTS.arbitration_clause.filename, {
      waitForAnalysis: false
    });
    
    // Take screenshot when analysis starts
    await page.waitForSelector('[data-testid="analysis-status"]');
    await takeScreenshot(page, 'analysis-starting');
    
    // Take screenshot during analysis
    const progressBar = page.locator('[data-testid="analysis-progress"]');
    if (await progressBar.isVisible()) {
      await takeScreenshot(page, 'analysis-in-progress');
    }
    
    // Wait for completion and take final screenshot
    await waitForAnalysisComplete(page);
    await takeScreenshot(page, 'analysis-completed');
  });

  test('should show proper error states for invalid files @visual', async ({ page }) => {
    // Try to upload unsupported file type
    await page.setInputFiles('[data-testid="file-input"]', {
      name: 'invalid-file.exe',
      mimeType: 'application/x-executable',
      buffer: Buffer.from('Invalid file content')
    });
    
    await page.click('[data-testid="upload-button"]');
    
    // Wait for error message
    await page.waitForSelector('[data-testid="error-message"]', { timeout: 5000 });
    await takeScreenshot(page, 'upload-error-invalid-file');
  });

  test('should handle large file uploads @visual', async ({ page }) => {
    // Create large file content
    const largeContent = 'Large document content. '.repeat(10000);
    
    await page.setInputFiles('[data-testid="file-input"]', {
      name: 'large-document.txt',
      mimeType: 'text/plain',
      buffer: Buffer.from(largeContent)
    });
    
    await page.click('[data-testid="upload-button"]');
    
    // Take screenshot of large file handling
    await page.waitForSelector('[data-testid="file-size-warning"]', { timeout: 5000 });
    await takeScreenshot(page, 'upload-large-file-warning');
  });

  test('should display upload queue for multiple files @visual @demo', async ({ page }) => {
    // Upload multiple files
    const files = [
      SAMPLE_DOCUMENTS.arbitration_clause.filename,
      SAMPLE_DOCUMENTS.no_arbitration.filename
    ];
    
    for (const filename of files) {
      await uploadDocument(page, filename, { waitForAnalysis: false });
      await page.waitForTimeout(1000);
    }
    
    // Take screenshot of upload queue
    await page.waitForSelector('[data-testid="upload-queue"]');
    await takeScreenshot(page, 'upload-queue-multiple-files');
  });

  test('should show drag and drop interaction @visual @demo', async ({ page }) => {
    // Test drag and drop upload
    await uploadDocumentViaDragDrop(page, SAMPLE_DOCUMENTS.arbitration_clause.filename, {
      waitForAnalysis: false
    });
    
    // Take screenshot of drag and drop result
    await takeScreenshot(page, 'dragdrop-upload-complete');
  });

  test('should handle upload cancellation @visual', async ({ page }) => {
    // Start upload
    await uploadDocument(page, SAMPLE_DOCUMENTS.complex_arbitration.filename, {
      waitForAnalysis: false
    });
    
    // Try to cancel upload if cancel button exists
    const cancelButton = page.locator('[data-testid="cancel-upload"]');
    if (await cancelButton.isVisible()) {
      await cancelButton.click();
      await takeScreenshot(page, 'upload-cancelled');
    }
  });

  test('should display file format guidelines @visual', async ({ page }) => {
    // Look for help or info section about supported formats
    const formatInfo = page.locator('[data-testid="supported-formats"]');
    
    if (await formatInfo.isVisible()) {
      await takeScreenshot(page, 'supported-formats-info');
    } else {
      // Try to open help modal or tooltip
      const helpButton = page.locator('[data-testid="help-button"]');
      if (await helpButton.isVisible()) {
        await helpButton.click();
        await smartWait(page);
        await takeScreenshot(page, 'upload-help-modal');
      }
    }
  });

  test('should be responsive on mobile devices @visual @mobile', async ({ page }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    await page.reload();
    await smartWait(page);
    
    // Take mobile upload interface screenshot
    await takeScreenshot(page, 'upload-mobile-interface', {
      fullPage: true,
      deviceType: 'mobile'
    });
    
    // Test mobile file selection
    await uploadDocument(page, SAMPLE_DOCUMENTS.arbitration_clause.filename, {
      waitForAnalysis: false
    });
    
    await takeScreenshot(page, 'upload-mobile-file-selected', {
      deviceType: 'mobile'
    });
  });
});

test.describe('Upload Flow Integration Tests', () => {
  test.beforeEach(async ({ page }) => {
    await enableDemoMode(page);
    await page.goto('/upload');
    await smartWait(page);
  });

  test('should complete full upload and analysis workflow @demo', async ({ page }) => {
    // Complete upload workflow
    await uploadDocument(page, SAMPLE_DOCUMENTS.arbitration_clause.filename, {
      waitForAnalysis: true,
      expectedResult: 'arbitration'
    });
    
    // Verify analysis results are displayed
    await expect(page.locator('[data-testid="analysis-result"]')).toBeVisible();
    await expect(page.locator('[data-testid="confidence-score"]')).toBeVisible();
    
    // Take screenshot of final results
    await takeScreenshot(page, 'upload-complete-workflow', {
      fullPage: true
    });
  });

  test('should navigate to results page after analysis @demo', async ({ page }) => {
    // Upload and complete analysis
    await uploadDocument(page, SAMPLE_DOCUMENTS.arbitration_clause.filename, {
      waitForAnalysis: true
    });
    
    // Check if there's a "View Details" or similar button
    const viewDetailsButton = page.locator('[data-testid="view-details-button"]');
    if (await viewDetailsButton.isVisible()) {
      await viewDetailsButton.click();
      await smartWait(page);
      
      // Take screenshot of detailed results page
      await takeScreenshot(page, 'detailed-results-page', {
        fullPage: true
      });
    }
  });

  test('should save upload history @demo', async ({ page }) => {
    // Upload document
    await uploadDocument(page, SAMPLE_DOCUMENTS.arbitration_clause.filename, {
      waitForAnalysis: true
    });
    
    // Navigate to history or dashboard
    const historyButton = page.locator('[data-testid="history-button"]');
    const dashboardButton = page.locator('[data-testid="dashboard-button"]');
    
    if (await historyButton.isVisible()) {
      await historyButton.click();
    } else if (await dashboardButton.isVisible()) {
      await dashboardButton.click();
    } else {
      await page.goto('/dashboard');
    }
    
    await smartWait(page);
    
    // Take screenshot of history/dashboard showing recent upload
    await takeScreenshot(page, 'upload-history', {
      fullPage: true
    });
  });

  test('should handle multiple document types correctly @demo', async ({ page }) => {
    const testCases = [
      {
        doc: SAMPLE_DOCUMENTS.arbitration_clause,
        expected: 'arbitration'
      },
      {
        doc: SAMPLE_DOCUMENTS.no_arbitration,
        expected: 'no-arbitration'
      },
      {
        doc: SAMPLE_DOCUMENTS.uncertain_case,
        expected: 'uncertain'
      }
    ];
    
    for (const testCase of testCases) {
      // Upload document
      await uploadDocument(page, testCase.doc.filename, {
        waitForAnalysis: true,
        expectedResult: testCase.expected as any
      });
      
      // Take screenshot of results
      await takeScreenshot(page, `upload-result-${testCase.expected}`, {
        fullPage: true
      });
      
      // Reset for next test
      await page.reload();
      await smartWait(page);
    }
  });
});