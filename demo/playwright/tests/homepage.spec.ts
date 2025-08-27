import { test, expect } from '@playwright/test';
import { takeScreenshot, takeResponsiveScreenshots, preparePageForScreenshot } from '../utils/screenshot';
import { smartWait } from '../utils/wait';
import { enableDemoMode } from '../utils/login';

test.describe('Homepage Visual Tests', () => {
  test.beforeEach(async ({ page }) => {
    await enableDemoMode(page);
    await page.goto('/');
    await smartWait(page);
  });

  test('should display homepage hero section correctly @visual', async ({ page }) => {
    // Wait for hero section to load
    await page.waitForSelector('[data-testid="hero-section"]');
    
    // Take full page screenshot
    await takeScreenshot(page, 'homepage-hero', {
      fullPage: true,
      mask: ['[data-testid="timestamp"]'], // Hide dynamic timestamps
    });
  });

  test('should show features section with proper layout @visual', async ({ page }) => {
    // Scroll to features section
    await page.locator('[data-testid="features-section"]').scrollIntoViewIfNeeded();
    await smartWait(page);
    
    // Take screenshot of features section
    await takeScreenshot(page, 'homepage-features', {
      clip: await page.locator('[data-testid="features-section"]').boundingBox() || undefined
    });
  });

  test('should display call-to-action buttons prominently @visual', async ({ page }) => {
    const ctaSection = page.locator('[data-testid="cta-section"]');
    await ctaSection.scrollIntoViewIfNeeded();
    
    // Verify CTA buttons are visible
    await expect(page.locator('[data-testid="start-demo-button"]')).toBeVisible();
    await expect(page.locator('[data-testid="upload-document-button"]')).toBeVisible();
    
    // Take screenshot
    await takeScreenshot(page, 'homepage-cta', {
      clip: await ctaSection.boundingBox() || undefined
    });
  });

  test('should show testimonials section @visual', async ({ page }) => {
    const testimonialsSection = page.locator('[data-testid="testimonials-section"]');
    
    if (await testimonialsSection.isVisible()) {
      await testimonialsSection.scrollIntoViewIfNeeded();
      await smartWait(page);
      
      await takeScreenshot(page, 'homepage-testimonials', {
        clip: await testimonialsSection.boundingBox() || undefined
      });
    }
  });

  test('should display footer with all links @visual', async ({ page }) => {
    const footer = page.locator('[data-testid="footer"]');
    await footer.scrollIntoViewIfNeeded();
    
    // Verify footer links
    await expect(page.locator('[data-testid="footer-privacy"]')).toBeVisible();
    await expect(page.locator('[data-testid="footer-terms"]')).toBeVisible();
    await expect(page.locator('[data-testid="footer-contact"]')).toBeVisible();
    
    await takeScreenshot(page, 'homepage-footer', {
      clip: await footer.boundingBox() || undefined
    });
  });

  test('should be responsive across different viewports @visual @mobile', async ({ page }) => {
    await takeResponsiveScreenshots(page, 'homepage-responsive', {
      fullPage: true,
      mask: ['[data-testid="timestamp"]']
    });
  });

  test('should display navigation menu correctly @visual', async ({ page }) => {
    const navigation = page.locator('[data-testid="main-navigation"]');
    await expect(navigation).toBeVisible();
    
    // Take screenshot of navigation
    await takeScreenshot(page, 'homepage-navigation', {
      clip: await navigation.boundingBox() || undefined
    });
    
    // Test mobile menu if on mobile viewport
    const viewport = page.viewportSize();
    if (viewport && viewport.width < 768) {
      const mobileMenuButton = page.locator('[data-testid="mobile-menu-button"]');
      if (await mobileMenuButton.isVisible()) {
        await mobileMenuButton.click();
        await smartWait(page);
        
        await takeScreenshot(page, 'homepage-mobile-menu', {
          fullPage: true
        });
        
        // Close menu
        await page.locator('[data-testid="mobile-menu-close"]').click();
      }
    }
  });

  test('should handle dark mode correctly @visual', async ({ page }) => {
    // Take light mode screenshot
    await takeScreenshot(page, 'homepage-light-mode', {
      fullPage: true,
      mode: 'light'
    });
    
    // Switch to dark mode
    const themeToggle = page.locator('[data-testid="theme-toggle"]');
    if (await themeToggle.isVisible()) {
      await themeToggle.click();
      await smartWait(page);
      
      // Take dark mode screenshot
      await takeScreenshot(page, 'homepage-dark-mode', {
        fullPage: true,
        mode: 'dark'
      });
    }
  });

  test('should show loading states correctly @visual', async ({ page }) => {
    // Navigate to a page that shows loading
    await page.goto('/upload');
    
    // Take screenshot of loading state if visible
    const loadingSpinner = page.locator('[data-testid="loading-spinner"]');
    if (await loadingSpinner.isVisible()) {
      await takeScreenshot(page, 'homepage-loading-state');
    }
    
    // Go back to homepage
    await page.goto('/');
    await smartWait(page);
  });

  test('should display stats/metrics section @visual', async ({ page }) => {
    const statsSection = page.locator('[data-testid="stats-section"]');
    
    if (await statsSection.isVisible()) {
      await statsSection.scrollIntoViewIfNeeded();
      await smartWait(page);
      
      // Verify key metrics are displayed
      await expect(page.locator('[data-testid="documents-analyzed"]')).toBeVisible();
      await expect(page.locator('[data-testid="accuracy-rate"]')).toBeVisible();
      
      await takeScreenshot(page, 'homepage-stats', {
        clip: await statsSection.boundingBox() || undefined
      });
    }
  });

  test('should show proper hover states @visual', async ({ page }) => {
    // Test hover state on CTA button
    const ctaButton = page.locator('[data-testid="start-demo-button"]');
    await ctaButton.scrollIntoViewIfNeeded();
    
    // Take screenshot before hover
    await takeScreenshot(page, 'homepage-cta-normal');
    
    // Hover and take screenshot
    await ctaButton.hover();
    await page.waitForTimeout(300); // Wait for hover animation
    
    await takeScreenshot(page, 'homepage-cta-hover');
  });

  test('should handle error states gracefully @visual', async ({ page }) => {
    // Simulate network error
    await page.route('**/api/**', route => route.abort());
    
    // Reload page to trigger error
    await page.reload();
    await smartWait(page);
    
    // Take screenshot of error state
    const errorMessage = page.locator('[data-testid="error-message"]');
    if (await errorMessage.isVisible()) {
      await takeScreenshot(page, 'homepage-error-state');
    }
    
    // Remove route intercept
    await page.unroute('**/api/**');
  });
});

test.describe('Homepage Interaction Tests', () => {
  test.beforeEach(async ({ page }) => {
    await enableDemoMode(page);
    await page.goto('/');
    await smartWait(page);
  });

  test('should navigate to upload page from CTA @demo', async ({ page }) => {
    await page.click('[data-testid="upload-document-button"]');
    await page.waitForURL('**/upload');
    
    await takeScreenshot(page, 'navigation-to-upload');
  });

  test('should start demo from hero button @demo', async ({ page }) => {
    await page.click('[data-testid="start-demo-button"]');
    
    // Should either navigate to demo page or open demo modal
    const demoModal = page.locator('[data-testid="demo-modal"]');
    const demoPage = page.locator('[data-testid="demo-page"]');
    
    if (await demoModal.isVisible()) {
      await takeScreenshot(page, 'demo-modal-opened');
    } else if (await demoPage.isVisible()) {
      await takeScreenshot(page, 'demo-page-loaded');
    }
  });

  test('should display feature details on click @demo', async ({ page }) => {
    const featureCards = page.locator('[data-testid^="feature-card-"]');
    const count = await featureCards.count();
    
    if (count > 0) {
      // Click first feature card
      await featureCards.first().click();
      await smartWait(page);
      
      // Take screenshot of expanded feature
      await takeScreenshot(page, 'feature-detail-expanded');
    }
  });

  test('should scroll smoothly to sections @demo', async ({ page }) => {
    const featuresLink = page.locator('[data-testid="nav-features"]');
    
    if (await featuresLink.isVisible()) {
      await featuresLink.click();
      await smartWait(page);
      
      // Verify scrolled to features section
      const featuresSection = page.locator('[data-testid="features-section"]');
      await expect(featuresSection).toBeInViewport();
      
      await takeScreenshot(page, 'scrolled-to-features');
    }
  });
});