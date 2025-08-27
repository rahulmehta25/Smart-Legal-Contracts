import { test, expect } from '@playwright/test';
import { loginAsTestUser, enableDemoMode } from '../utils/login';
import { uploadDocument } from '../utils/upload';
import { takeScreenshot, takeElementScreenshot, takeResponsiveScreenshots } from '../utils/screenshot';
import { smartWait } from '../utils/wait';
import { SAMPLE_DOCUMENTS } from '../utils/test-data';

test.describe('Dashboard Visual Tests', () => {
  test.beforeEach(async ({ page }) => {
    await enableDemoMode(page);
    // Upload some sample data for dashboard display
    await page.goto('/upload');
    await uploadDocument(page, SAMPLE_DOCUMENTS.arbitration_clause.filename, {
      waitForAnalysis: true
    });
    await uploadDocument(page, SAMPLE_DOCUMENTS.no_arbitration.filename, {
      waitForAnalysis: true
    });
    
    // Navigate to dashboard
    await page.goto('/dashboard');
    await smartWait(page);
  });

  test('should display dashboard overview correctly @visual @demo', async ({ page }) => {
    // Verify main dashboard components
    await expect(page.locator('[data-testid="dashboard-header"]')).toBeVisible();
    await expect(page.locator('[data-testid="stats-overview"]')).toBeVisible();
    await expect(page.locator('[data-testid="recent-analyses"]')).toBeVisible();
    
    // Take full dashboard screenshot
    await takeScreenshot(page, 'dashboard-overview', {
      fullPage: true
    });
  });

  test('should show analytics and metrics correctly @visual @demo', async ({ page }) => {
    // Take screenshot of stats section
    const statsSection = page.locator('[data-testid="stats-overview"]');
    await takeElementScreenshot(statsSection, 'dashboard-stats-overview');
    
    // Verify key metrics are visible
    await expect(page.locator('[data-testid="total-documents"]')).toBeVisible();
    await expect(page.locator('[data-testid="arbitration-detected"]')).toBeVisible();
    await expect(page.locator('[data-testid="average-confidence"]')).toBeVisible();
    
    // Take individual metric screenshots
    await takeElementScreenshot(
      page.locator('[data-testid="metric-card-documents"]'),
      'metric-total-documents'
    );
    
    await takeElementScreenshot(
      page.locator('[data-testid="metric-card-accuracy"]'),
      'metric-accuracy-rate'
    );
  });

  test('should display recent analyses list @visual @demo', async ({ page }) => {
    const recentSection = page.locator('[data-testid="recent-analyses"]');
    await expect(recentSection).toBeVisible();
    
    // Take screenshot of recent analyses
    await takeElementScreenshot(recentSection, 'dashboard-recent-analyses');
    
    // Verify analysis items are displayed
    const analysisItems = page.locator('[data-testid^="analysis-item-"]');
    const count = await analysisItems.count();
    expect(count).toBeGreaterThan(0);
    
    // Take screenshot of first analysis item
    if (count > 0) {
      await takeElementScreenshot(
        analysisItems.first(),
        'dashboard-analysis-item'
      );
    }
  });

  test('should show charts and visualizations @visual @demo', async ({ page }) => {
    // Look for chart components
    const chartSection = page.locator('[data-testid="analytics-charts"]');
    if (await chartSection.isVisible()) {
      await takeElementScreenshot(chartSection, 'dashboard-charts');
    }
    
    // Test individual chart types
    const pieChart = page.locator('[data-testid="arbitration-pie-chart"]');
    if (await pieChart.isVisible()) {
      await takeElementScreenshot(pieChart, 'dashboard-pie-chart');
    }
    
    const timelineChart = page.locator('[data-testid="timeline-chart"]');
    if (await timelineChart.isVisible()) {
      await takeElementScreenshot(timelineChart, 'dashboard-timeline-chart');
    }
    
    const confidenceChart = page.locator('[data-testid="confidence-distribution"]');
    if (await confidenceChart.isVisible()) {
      await takeElementScreenshot(confidenceChart, 'dashboard-confidence-chart');
    }
  });

  test('should display document library/history @visual @demo', async ({ page }) => {
    // Navigate to document library if available
    const libraryTab = page.locator('[data-testid="library-tab"]');
    if (await libraryTab.isVisible()) {
      await libraryTab.click();
      await smartWait(page);
      
      await takeScreenshot(page, 'dashboard-document-library', {
        fullPage: true
      });
    }
    
    // Test document list view
    const documentList = page.locator('[data-testid="document-list"]');
    if (await documentList.isVisible()) {
      await takeElementScreenshot(documentList, 'dashboard-document-list');
    }
    
    // Test document grid view
    const viewToggle = page.locator('[data-testid="view-toggle-grid"]');
    if (await viewToggle.isVisible()) {
      await viewToggle.click();
      await smartWait(page);
      
      await takeElementScreenshot(
        page.locator('[data-testid="document-grid"]'),
        'dashboard-document-grid'
      );
    }
  });

  test('should show search and filter functionality @visual', async ({ page }) => {
    // Test search functionality
    const searchBox = page.locator('[data-testid="search-documents"]');
    if (await searchBox.isVisible()) {
      await searchBox.fill('arbitration');
      await page.waitForTimeout(500); // Wait for search results
      
      await takeScreenshot(page, 'dashboard-search-results');
      
      // Clear search
      await searchBox.clear();
    }
    
    // Test filter options
    const filterButton = page.locator('[data-testid="filter-button"]');
    if (await filterButton.isVisible()) {
      await filterButton.click();
      await smartWait(page);
      
      await takeElementScreenshot(
        page.locator('[data-testid="filter-panel"]'),
        'dashboard-filter-panel'
      );
    }
  });

  test('should display user profile and settings @visual', async ({ page }) => {
    // Test user menu
    const userMenu = page.locator('[data-testid="user-menu"]');
    if (await userMenu.isVisible()) {
      await userMenu.click();
      await smartWait(page);
      
      await takeElementScreenshot(
        page.locator('[data-testid="user-dropdown"]'),
        'dashboard-user-menu'
      );
      
      // Close menu
      await page.click('body');
    }
    
    // Test settings page if accessible
    const settingsLink = page.locator('[data-testid="settings-link"]');
    if (await settingsLink.isVisible()) {
      await settingsLink.click();
      await smartWait(page);
      
      await takeScreenshot(page, 'dashboard-settings-page');
    }
  });

  test('should show notification center @visual', async ({ page }) => {
    const notificationButton = page.locator('[data-testid="notifications-button"]');
    if (await notificationButton.isVisible()) {
      await notificationButton.click();
      await smartWait(page);
      
      await takeElementScreenshot(
        page.locator('[data-testid="notifications-panel"]'),
        'dashboard-notifications'
      );
    }
  });

  test('should display export and reporting options @visual', async ({ page }) => {
    // Test export functionality
    const exportButton = page.locator('[data-testid="export-dashboard"]');
    if (await exportButton.isVisible()) {
      await exportButton.click();
      await smartWait(page);
      
      await takeElementScreenshot(
        page.locator('[data-testid="export-options"]'),
        'dashboard-export-options'
      );
    }
    
    // Test report generation
    const reportButton = page.locator('[data-testid="generate-report"]');
    if (await reportButton.isVisible()) {
      await reportButton.click();
      await smartWait(page);
      
      await takeElementScreenshot(
        page.locator('[data-testid="report-modal"]'),
        'dashboard-report-modal'
      );
    }
  });

  test('should be responsive across devices @visual @mobile', async ({ page }) => {
    await takeResponsiveScreenshots(page, 'dashboard-responsive', {
      fullPage: true
    });
  });

  test('should handle empty state correctly @visual', async ({ page }) => {
    // Clear all data or navigate to fresh dashboard
    await page.evaluate(() => {
      localStorage.clear();
      sessionStorage.clear();
    });
    
    await page.reload();
    await smartWait(page);
    
    // Take screenshot of empty dashboard
    const emptyState = page.locator('[data-testid="empty-dashboard"]');
    if (await emptyState.isVisible()) {
      await takeScreenshot(page, 'dashboard-empty-state');
    }
  });

  test('should show loading states correctly @visual', async ({ page }) => {
    // Reload page to catch loading states
    await page.reload();
    
    // Try to capture loading spinner
    const loadingSpinner = page.locator('[data-testid="dashboard-loading"]');
    if (await loadingSpinner.isVisible()) {
      await takeScreenshot(page, 'dashboard-loading-state');
    }
    
    // Wait for full load
    await smartWait(page);
  });
});

test.describe('Dashboard Interaction Tests', () => {
  test.beforeEach(async ({ page }) => {
    await enableDemoMode(page);
    await page.goto('/dashboard');
    await smartWait(page);
  });

  test('should navigate to analysis details from dashboard @demo', async ({ page }) => {
    // Click on an analysis item
    const analysisItem = page.locator('[data-testid^="analysis-item-"]').first();
    if (await analysisItem.isVisible()) {
      await analysisItem.click();
      await smartWait(page);
      
      // Should navigate to detailed view
      await takeScreenshot(page, 'dashboard-to-analysis-details');
    }
  });

  test('should allow quick actions from dashboard @demo', async ({ page }) => {
    // Test quick upload from dashboard
    const quickUploadButton = page.locator('[data-testid="quick-upload"]');
    if (await quickUploadButton.isVisible()) {
      await quickUploadButton.click();
      await smartWait(page);
      
      await takeScreenshot(page, 'dashboard-quick-upload-modal');
    }
    
    // Test other quick actions
    const quickActionsMenu = page.locator('[data-testid="quick-actions"]');
    if (await quickActionsMenu.isVisible()) {
      await quickActionsMenu.click();
      await smartWait(page);
      
      await takeElementScreenshot(
        page.locator('[data-testid="quick-actions-dropdown"]'),
        'dashboard-quick-actions'
      );
    }
  });

  test('should handle dashboard customization @demo', async ({ page }) => {
    // Test widget customization
    const customizeButton = page.locator('[data-testid="customize-dashboard"]');
    if (await customizeButton.isVisible()) {
      await customizeButton.click();
      await smartWait(page);
      
      await takeScreenshot(page, 'dashboard-customization-mode');
    }
    
    // Test widget reordering
    const widget = page.locator('[data-testid^="dashboard-widget-"]').first();
    if (await widget.isVisible()) {
      // Take before screenshot
      await takeScreenshot(page, 'dashboard-before-reorder');
      
      // Simulate drag and drop if supported
      const dragHandle = widget.locator('[data-testid="drag-handle"]');
      if (await dragHandle.isVisible()) {
        // Perform drag operation
        await dragHandle.hover();
        await takeScreenshot(page, 'dashboard-drag-active');
      }
    }
  });

  test('should filter and sort dashboard data @demo', async ({ page }) => {
    // Test date range filter
    const dateFilter = page.locator('[data-testid="date-range-filter"]');
    if (await dateFilter.isVisible()) {
      await dateFilter.click();
      await smartWait(page);
      
      await takeElementScreenshot(
        page.locator('[data-testid="date-picker"]'),
        'dashboard-date-filter'
      );
      
      // Select last 7 days
      const last7Days = page.locator('[data-testid="filter-last-7-days"]');
      if (await last7Days.isVisible()) {
        await last7Days.click();
        await smartWait(page);
        
        await takeScreenshot(page, 'dashboard-filtered-7-days');
      }
    }
    
    // Test sorting options
    const sortButton = page.locator('[data-testid="sort-button"]');
    if (await sortButton.isVisible()) {
      await sortButton.click();
      await smartWait(page);
      
      await takeElementScreenshot(
        page.locator('[data-testid="sort-options"]'),
        'dashboard-sort-options'
      );
    }
  });

  test('should handle bulk operations @demo', async ({ page }) => {
    // Test bulk selection
    const selectAllCheckbox = page.locator('[data-testid="select-all-analyses"]');
    if (await selectAllCheckbox.isVisible()) {
      await selectAllCheckbox.check();
      await smartWait(page);
      
      await takeScreenshot(page, 'dashboard-bulk-selected');
      
      // Test bulk actions
      const bulkActionsBar = page.locator('[data-testid="bulk-actions-bar"]');
      if (await bulkActionsBar.isVisible()) {
        await takeElementScreenshot(bulkActionsBar, 'dashboard-bulk-actions');
      }
    }
  });
});