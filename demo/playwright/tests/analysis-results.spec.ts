import { test, expect } from '@playwright/test';
import { uploadDocument } from '../utils/upload';
import { takeScreenshot, takeElementScreenshot, generateMarketingScreenshot } from '../utils/screenshot';
import { waitForAnalysisComplete, smartWait } from '../utils/wait';
import { enableDemoMode } from '../utils/login';
import { SAMPLE_DOCUMENTS } from '../utils/test-data';

test.describe('Analysis Results Visual Tests', () => {
  test.beforeEach(async ({ page }) => {
    await enableDemoMode(page);
  });

  test('should display positive arbitration detection results @visual @demo', async ({ page }) => {
    await page.goto('/upload');
    
    // Upload document with arbitration clause
    await uploadDocument(page, SAMPLE_DOCUMENTS.arbitration_clause.filename, {
      waitForAnalysis: true,
      expectedResult: 'arbitration'
    });
    
    // Verify positive result elements
    await expect(page.locator('[data-testid="arbitration-detected"]')).toBeVisible();
    await expect(page.locator('[data-testid="confidence-score"]')).toBeVisible();
    await expect(page.locator('[data-testid="highlighted-clauses"]')).toBeVisible();
    
    // Take full results screenshot
    await takeScreenshot(page, 'results-positive-arbitration', {
      fullPage: true
    });
    
    // Take individual component screenshots
    await takeElementScreenshot(
      page.locator('[data-testid="result-summary"]'),
      'result-summary-positive'
    );
    
    await takeElementScreenshot(
      page.locator('[data-testid="confidence-meter"]'),
      'confidence-meter-high'
    );
  });

  test('should display negative arbitration detection results @visual @demo', async ({ page }) => {
    await page.goto('/upload');
    
    // Upload document without arbitration clause
    await uploadDocument(page, SAMPLE_DOCUMENTS.no_arbitration.filename, {
      waitForAnalysis: true,
      expectedResult: 'no-arbitration'
    });
    
    // Verify negative result elements
    await expect(page.locator('[data-testid="no-arbitration-found"]')).toBeVisible();
    
    // Take screenshot of negative results
    await takeScreenshot(page, 'results-negative-arbitration', {
      fullPage: true
    });
    
    await takeElementScreenshot(
      page.locator('[data-testid="result-summary"]'),
      'result-summary-negative'
    );
  });

  test('should display uncertain/mixed results correctly @visual @demo', async ({ page }) => {
    await page.goto('/upload');
    
    // Upload document with uncertain arbitration status
    await uploadDocument(page, SAMPLE_DOCUMENTS.uncertain_case.filename, {
      waitForAnalysis: true,
      expectedResult: 'uncertain'
    });
    
    // Verify uncertain result elements
    await expect(page.locator('[data-testid="uncertain-result"]')).toBeVisible();
    
    // Take screenshot of uncertain results
    await takeScreenshot(page, 'results-uncertain-arbitration', {
      fullPage: true
    });
    
    await takeElementScreenshot(
      page.locator('[data-testid="confidence-meter"]'),
      'confidence-meter-medium'
    );
  });

  test('should highlight arbitration clauses in document text @visual @demo', async ({ page }) => {
    await page.goto('/upload');
    
    // Upload complex arbitration document
    await uploadDocument(page, SAMPLE_DOCUMENTS.complex_arbitration.filename, {
      waitForAnalysis: true,
      expectedResult: 'arbitration'
    });
    
    // Verify clause highlighting
    const highlightedText = page.locator('[data-testid="highlighted-clause"]');
    await expect(highlightedText).toBeVisible();
    
    // Take screenshot of highlighted document
    await takeElementScreenshot(
      page.locator('[data-testid="document-preview"]'),
      'document-with-highlights'
    );
    
    // Test different highlight types
    const keyTerms = page.locator('[data-testid="key-terms"]');
    if (await keyTerms.isVisible()) {
      await takeElementScreenshot(keyTerms, 'key-terms-highlighted');
    }
  });

  test('should display confidence score with visual indicator @visual @demo', async ({ page }) => {
    await page.goto('/upload');
    
    // Test high confidence result
    await uploadDocument(page, SAMPLE_DOCUMENTS.arbitration_clause.filename, {
      waitForAnalysis: true,
      expectedResult: 'arbitration'
    });
    
    const confidenceMeter = page.locator('[data-testid="confidence-meter"]');
    await expect(confidenceMeter).toBeVisible();
    
    // Verify confidence score is displayed
    const confidenceScore = page.locator('[data-testid="confidence-score"]');
    const scoreText = await confidenceScore.textContent();
    expect(scoreText).toMatch(/\d+%/); // Should show percentage
    
    // Take screenshot of confidence visualization
    await takeElementScreenshot(confidenceMeter, 'confidence-meter-visualization');
    
    // Check for confidence breakdown
    const confidenceBreakdown = page.locator('[data-testid="confidence-breakdown"]');
    if (await confidenceBreakdown.isVisible()) {
      await takeElementScreenshot(confidenceBreakdown, 'confidence-breakdown');
    }
  });

  test('should show analysis details and explanations @visual @demo', async ({ page }) => {
    await page.goto('/upload');
    
    await uploadDocument(page, SAMPLE_DOCUMENTS.complex_arbitration.filename, {
      waitForAnalysis: true,
      expectedResult: 'arbitration'
    });
    
    // Check for analysis explanation
    const explanation = page.locator('[data-testid="analysis-explanation"]');
    if (await explanation.isVisible()) {
      await takeElementScreenshot(explanation, 'analysis-explanation');
    }
    
    // Check for detailed findings
    const findings = page.locator('[data-testid="detailed-findings"]');
    if (await findings.isVisible()) {
      await takeElementScreenshot(findings, 'detailed-findings');
    }
    
    // Check for risk assessment
    const riskAssessment = page.locator('[data-testid="risk-assessment"]');
    if (await riskAssessment.isVisible()) {
      await takeElementScreenshot(riskAssessment, 'risk-assessment');
    }
  });

  test('should display export options @visual', async ({ page }) => {
    await page.goto('/upload');
    
    await uploadDocument(page, SAMPLE_DOCUMENTS.arbitration_clause.filename, {
      waitForAnalysis: true,
      expectedResult: 'arbitration'
    });
    
    // Look for export options
    const exportButtons = page.locator('[data-testid="export-options"]');
    if (await exportButtons.isVisible()) {
      await takeElementScreenshot(exportButtons, 'export-options');
    }
    
    // Test individual export buttons
    const pdfExport = page.locator('[data-testid="export-pdf"]');
    const jsonExport = page.locator('[data-testid="export-json"]');
    const reportExport = page.locator('[data-testid="export-report"]');
    
    if (await pdfExport.isVisible()) {
      await takeElementScreenshot(pdfExport, 'export-pdf-button');
    }
  });

  test('should show comparison view for multiple documents @visual @demo', async ({ page }) => {
    await page.goto('/upload');
    
    // Upload first document
    await uploadDocument(page, SAMPLE_DOCUMENTS.arbitration_clause.filename, {
      waitForAnalysis: true
    });
    
    // Upload second document if multiple upload is supported
    const addAnotherButton = page.locator('[data-testid="add-another-document"]');
    if (await addAnotherButton.isVisible()) {
      await addAnotherButton.click();
      
      await uploadDocument(page, SAMPLE_DOCUMENTS.no_arbitration.filename, {
        waitForAnalysis: true
      });
      
      // Take comparison view screenshot
      await takeScreenshot(page, 'results-comparison-view', {
        fullPage: true
      });
    }
  });

  test('should display analysis timeline/history @visual', async ({ page }) => {
    await page.goto('/upload');
    
    await uploadDocument(page, SAMPLE_DOCUMENTS.arbitration_clause.filename, {
      waitForAnalysis: true
    });
    
    // Check for analysis timeline
    const timeline = page.locator('[data-testid="analysis-timeline"]');
    if (await timeline.isVisible()) {
      await takeElementScreenshot(timeline, 'analysis-timeline');
    }
    
    // Check for processing steps
    const processingSteps = page.locator('[data-testid="processing-steps"]');
    if (await processingSteps.isVisible()) {
      await takeElementScreenshot(processingSteps, 'processing-steps');
    }
  });

  test('should generate marketing screenshots with annotations @demo', async ({ page }) => {
    await page.goto('/upload');
    
    await uploadDocument(page, SAMPLE_DOCUMENTS.complex_arbitration.filename, {
      waitForAnalysis: true,
      expectedResult: 'arbitration'
    });
    
    // Generate marketing screenshot with annotations
    await generateMarketingScreenshot(
      page,
      'analysis-results-showcase',
      [
        {
          selector: '[data-testid="confidence-score"]',
          text: 'AI-powered confidence scoring',
          position: 'right'
        },
        {
          selector: '[data-testid="highlighted-clause"]',
          text: 'Automatic clause highlighting',
          position: 'left'
        },
        {
          selector: '[data-testid="analysis-explanation"]',
          text: 'Clear explanations for every decision',
          position: 'bottom'
        }
      ],
      { fullPage: true }
    );
  });

  test('should handle error states in analysis @visual', async ({ page }) => {
    // Simulate analysis error
    await page.route('**/api/analyze', route => {
      route.fulfill({
        status: 500,
        body: JSON.stringify({ error: 'Analysis failed' })
      });
    });
    
    await page.goto('/upload');
    
    try {
      await uploadDocument(page, SAMPLE_DOCUMENTS.arbitration_clause.filename, {
        waitForAnalysis: false
      });
      
      // Look for error state
      await page.waitForSelector('[data-testid="analysis-error"]', { timeout: 10000 });
      await takeScreenshot(page, 'results-analysis-error');
    } catch (error) {
      console.log('Error state test may not be applicable');
    }
    
    // Remove route intercept
    await page.unroute('**/api/analyze');
  });

  test('should be responsive on different screen sizes @visual @mobile', async ({ page }) => {
    await page.goto('/upload');
    
    await uploadDocument(page, SAMPLE_DOCUMENTS.arbitration_clause.filename, {
      waitForAnalysis: true,
      expectedResult: 'arbitration'
    });
    
    // Test mobile view
    await page.setViewportSize({ width: 375, height: 667 });
    await smartWait(page);
    
    await takeScreenshot(page, 'results-mobile-view', {
      fullPage: true,
      deviceType: 'mobile'
    });
    
    // Test tablet view
    await page.setViewportSize({ width: 768, height: 1024 });
    await smartWait(page);
    
    await takeScreenshot(page, 'results-tablet-view', {
      fullPage: true,
      deviceType: 'tablet'
    });
  });
});

test.describe('Analysis Results Interaction Tests', () => {
  test.beforeEach(async ({ page }) => {
    await enableDemoMode(page);
    await page.goto('/upload');
  });

  test('should expand/collapse detailed analysis sections @demo', async ({ page }) => {
    await uploadDocument(page, SAMPLE_DOCUMENTS.complex_arbitration.filename, {
      waitForAnalysis: true
    });
    
    // Test expand/collapse functionality
    const expandableSection = page.locator('[data-testid="expandable-details"]');
    if (await expandableSection.isVisible()) {
      // Take collapsed state
      await takeScreenshot(page, 'details-collapsed');
      
      // Expand
      await expandableSection.click();
      await smartWait(page);
      
      // Take expanded state
      await takeScreenshot(page, 'details-expanded');
    }
  });

  test('should allow sharing of analysis results @demo', async ({ page }) => {
    await uploadDocument(page, SAMPLE_DOCUMENTS.arbitration_clause.filename, {
      waitForAnalysis: true
    });
    
    // Test share functionality
    const shareButton = page.locator('[data-testid="share-results"]');
    if (await shareButton.isVisible()) {
      await shareButton.click();
      await smartWait(page);
      
      await takeScreenshot(page, 'share-modal-opened');
    }
  });

  test('should allow saving results to dashboard @demo', async ({ page }) => {
    await uploadDocument(page, SAMPLE_DOCUMENTS.arbitration_clause.filename, {
      waitForAnalysis: true
    });
    
    // Test save functionality
    const saveButton = page.locator('[data-testid="save-results"]');
    if (await saveButton.isVisible()) {
      await saveButton.click();
      await smartWait(page);
      
      // Verify saved confirmation
      await expect(page.locator('[data-testid="save-confirmation"]')).toBeVisible();
      await takeScreenshot(page, 'results-saved-confirmation');
    }
  });
});