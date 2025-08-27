import { test, expect } from '@playwright/test';
import { takeScreenshot, takeResponsiveScreenshots } from '../utils/screenshot';
import { uploadDocument } from '../utils/upload';
import { smartWait } from '../utils/wait';
import { enableDemoMode } from '../utils/login';
import { SAMPLE_DOCUMENTS } from '../utils/test-data';

test.describe('Responsive Design Visual Tests', () => {
  const viewports = [
    { name: 'mobile-portrait', width: 375, height: 667 },
    { name: 'mobile-landscape', width: 667, height: 375 },
    { name: 'tablet-portrait', width: 768, height: 1024 },
    { name: 'tablet-landscape', width: 1024, height: 768 },
    { name: 'desktop-small', width: 1280, height: 720 },
    { name: 'desktop-large', width: 1920, height: 1080 },
    { name: 'ultrawide', width: 2560, height: 1440 }
  ];

  test.beforeEach(async ({ page }) => {
    await enableDemoMode(page);
  });

  test('should display homepage responsively across all viewports @visual @mobile', async ({ page }) => {
    for (const viewport of viewports) {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      await page.goto('/');
      await smartWait(page);
      
      await takeScreenshot(page, `homepage-${viewport.name}`, {
        fullPage: true,
        deviceType: viewport.name.includes('mobile') ? 'mobile' : 
                   viewport.name.includes('tablet') ? 'tablet' : 'desktop'
      });
    }
  });

  test('should handle navigation menu responsively @visual @mobile', async ({ page }) => {
    for (const viewport of viewports) {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      await page.goto('/');
      await smartWait(page);
      
      // Test navigation on different screen sizes
      if (viewport.width < 768) {
        // Mobile: Test hamburger menu
        const mobileMenuButton = page.locator('[data-testid="mobile-menu-button"]');
        if (await mobileMenuButton.isVisible()) {
          await takeScreenshot(page, `nav-mobile-closed-${viewport.name}`);
          
          await mobileMenuButton.click();
          await smartWait(page);
          
          await takeScreenshot(page, `nav-mobile-open-${viewport.name}`);
          
          // Close menu
          const closeButton = page.locator('[data-testid="mobile-menu-close"]');
          if (await closeButton.isVisible()) {
            await closeButton.click();
          }
        }
      } else {
        // Desktop/Tablet: Test full navigation
        const navigation = page.locator('[data-testid="main-navigation"]');
        if (await navigation.isVisible()) {
          await takeScreenshot(page, `nav-desktop-${viewport.name}`);
        }
      }
    }
  });

  test('should handle upload interface responsively @visual @mobile', async ({ page }) => {
    await page.goto('/upload');
    
    for (const viewport of viewports) {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      await smartWait(page);
      
      await takeScreenshot(page, `upload-interface-${viewport.name}`, {
        fullPage: true,
        deviceType: viewport.name.includes('mobile') ? 'mobile' : 
                   viewport.name.includes('tablet') ? 'tablet' : 'desktop'
      });
    }
  });

  test('should display analysis results responsively @visual @mobile @demo', async ({ page }) => {
    await page.goto('/upload');
    
    // Upload document first at desktop size
    await page.setViewportSize({ width: 1920, height: 1080 });
    await uploadDocument(page, SAMPLE_DOCUMENTS.arbitration_clause.filename, {
      waitForAnalysis: true
    });
    
    // Test results display at different viewports
    for (const viewport of viewports) {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      await smartWait(page);
      
      await takeScreenshot(page, `results-${viewport.name}`, {
        fullPage: true,
        deviceType: viewport.name.includes('mobile') ? 'mobile' : 
                   viewport.name.includes('tablet') ? 'tablet' : 'desktop'
      });
    }
  });

  test('should handle dashboard layout responsively @visual @mobile', async ({ page }) => {
    await page.goto('/dashboard');
    
    for (const viewport of viewports) {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      await smartWait(page);
      
      await takeScreenshot(page, `dashboard-${viewport.name}`, {
        fullPage: true,
        deviceType: viewport.name.includes('mobile') ? 'mobile' : 
                   viewport.name.includes('tablet') ? 'tablet' : 'desktop'
      });
    }
  });

  test('should adapt form layouts for mobile @visual @mobile', async ({ page }) => {
    // Test login form responsiveness
    await page.goto('/login');
    
    for (const viewport of viewports) {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      await smartWait(page);
      
      await takeScreenshot(page, `login-form-${viewport.name}`, {
        deviceType: viewport.name.includes('mobile') ? 'mobile' : 
                   viewport.name.includes('tablet') ? 'tablet' : 'desktop'
      });
    }
  });

  test('should handle modal dialogs responsively @visual @mobile', async ({ page }) => {
    await page.goto('/');
    
    for (const viewport of viewports) {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      await smartWait(page);
      
      // Try to open a modal (help, about, etc.)
      const helpButton = page.locator('[data-testid="help-button"]');
      if (await helpButton.isVisible()) {
        await helpButton.click();
        await smartWait(page);
        
        await takeScreenshot(page, `modal-${viewport.name}`, {
          deviceType: viewport.name.includes('mobile') ? 'mobile' : 
                     viewport.name.includes('tablet') ? 'tablet' : 'desktop'
        });
        
        // Close modal
        const closeButton = page.locator('[data-testid="modal-close"]');
        if (await closeButton.isVisible()) {
          await closeButton.click();
        } else {
          await page.keyboard.press('Escape');
        }
        await smartWait(page);
      }
    }
  });

  test('should handle typography scaling @visual @mobile', async ({ page }) => {
    await page.goto('/');
    
    // Test text readability at different sizes
    for (const viewport of viewports) {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      await smartWait(page);
      
      // Focus on text-heavy sections
      const heroText = page.locator('[data-testid="hero-text"]');
      if (await heroText.isVisible()) {
        await takeScreenshot(page, `typography-hero-${viewport.name}`);
      }
      
      const featuresText = page.locator('[data-testid="features-text"]');
      if (await featuresText.isVisible()) {
        await featuresText.scrollIntoViewIfNeeded();
        await takeScreenshot(page, `typography-features-${viewport.name}`);
      }
    }
  });

  test('should maintain touch targets on mobile @visual @mobile', async ({ page }) => {
    const mobileViewports = viewports.filter(v => v.name.includes('mobile'));
    
    for (const viewport of mobileViewports) {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      await page.goto('/');
      await smartWait(page);
      
      // Highlight interactive elements for touch target verification
      await page.addStyleTag({
        content: `
          button, a, input, [role="button"], [onclick] {
            outline: 2px solid red !important;
            outline-offset: 2px !important;
          }
        `
      });
      
      await takeScreenshot(page, `touch-targets-${viewport.name}`, {
        deviceType: 'mobile'
      });
    }
  });

  test('should handle data tables responsively @visual @mobile', async ({ page }) => {
    await page.goto('/dashboard');
    
    // Look for data tables or lists
    const dataTable = page.locator('[data-testid="data-table"]');
    const documentList = page.locator('[data-testid="document-list"]');
    
    for (const viewport of viewports) {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      await smartWait(page);
      
      if (await dataTable.isVisible()) {
        await takeScreenshot(page, `data-table-${viewport.name}`, {
          deviceType: viewport.name.includes('mobile') ? 'mobile' : 
                     viewport.name.includes('tablet') ? 'tablet' : 'desktop'
        });
      } else if (await documentList.isVisible()) {
        await takeScreenshot(page, `document-list-${viewport.name}`, {
          deviceType: viewport.name.includes('mobile') ? 'mobile' : 
                     viewport.name.includes('tablet') ? 'tablet' : 'desktop'
        });
      }
    }
  });

  test('should handle image and media responsively @visual @mobile', async ({ page }) => {
    await page.goto('/');
    
    for (const viewport of viewports) {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      await smartWait(page);
      
      // Check for images, videos, or media content
      const heroImage = page.locator('[data-testid="hero-image"]');
      const featureImages = page.locator('[data-testid^="feature-image-"]');
      
      if (await heroImage.isVisible()) {
        await takeScreenshot(page, `hero-image-${viewport.name}`);
      }
      
      if (await featureImages.first().isVisible()) {
        await featureImages.first().scrollIntoViewIfNeeded();
        await takeScreenshot(page, `feature-images-${viewport.name}`);
      }
    }
  });

  test('should handle orientation changes @visual @mobile', async ({ page }) => {
    const mobileViewports = [
      { name: 'mobile-portrait', width: 375, height: 667 },
      { name: 'mobile-landscape', width: 667, height: 375 },
      { name: 'tablet-portrait', width: 768, height: 1024 },
      { name: 'tablet-landscape', width: 1024, height: 768 }
    ];
    
    for (const viewport of mobileViewports) {
      await page.setViewportSize({ width: viewport.width, height: viewport.height });
      await page.goto('/upload');
      await smartWait(page);
      
      await takeScreenshot(page, `orientation-${viewport.name}`, {
        fullPage: true,
        deviceType: viewport.name.includes('mobile') ? 'mobile' : 'tablet'
      });
    }
  });
});

test.describe('Responsive Interaction Tests', () => {
  test.beforeEach(async ({ page }) => {
    await enableDemoMode(page);
  });

  test('should handle touch interactions properly @mobile @demo', async ({ page }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');
    await smartWait(page);
    
    // Test touch interactions
    const ctaButton = page.locator('[data-testid="start-demo-button"]');
    if (await ctaButton.isVisible()) {
      // Simulate touch tap
      await ctaButton.tap();
      await smartWait(page);
      
      await takeScreenshot(page, 'mobile-touch-interaction');
    }
  });

  test('should handle swipe gestures @mobile @demo', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');
    await smartWait(page);
    
    // Look for swipeable components (carousels, image galleries, etc.)
    const carousel = page.locator('[data-testid="carousel"]');
    if (await carousel.isVisible()) {
      // Simulate swipe gesture
      await carousel.hover();
      await page.mouse.down();
      await page.mouse.move(100, 0); // Swipe left
      await page.mouse.up();
      await smartWait(page);
      
      await takeScreenshot(page, 'mobile-swipe-gesture');
    }
  });

  test('should handle pinch-to-zoom @mobile @demo', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/upload');
    
    // Upload document and view results
    await uploadDocument(page, SAMPLE_DOCUMENTS.arbitration_clause.filename, {
      waitForAnalysis: true
    });
    
    // Look for zoomable content
    const documentPreview = page.locator('[data-testid="document-preview"]');
    if (await documentPreview.isVisible()) {
      // Simulate pinch zoom (limited simulation capability)
      await documentPreview.dblclick(); // Double-tap to zoom
      await smartWait(page);
      
      await takeScreenshot(page, 'mobile-pinch-zoom');
    }
  });

  test('should handle mobile form submission @mobile @demo', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/upload');
    await smartWait(page);
    
    // Test mobile file upload
    await uploadDocument(page, SAMPLE_DOCUMENTS.arbitration_clause.filename, {
      waitForAnalysis: false
    });
    
    await takeScreenshot(page, 'mobile-form-submission');
  });
});