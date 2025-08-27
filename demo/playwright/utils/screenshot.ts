import { Page, Locator, expect } from '@playwright/test';
import { smartWait, waitForAnimations } from './wait';

export interface ScreenshotOptions {
  fullPage?: boolean;
  mask?: string[];
  clip?: { x: number; y: number; width: number; height: number };
  threshold?: number;
  maxDiffPixels?: number;
  animations?: 'disabled' | 'allow';
  mode?: 'light' | 'dark' | 'both';
  deviceType?: 'desktop' | 'tablet' | 'mobile';
}

/**
 * Take a screenshot with proper preparation
 */
export async function takeScreenshot(
  page: Page, 
  name: string, 
  options: ScreenshotOptions = {}
) {
  const {
    fullPage = true,
    mask = [],
    clip,
    threshold = 0.2,
    maxDiffPixels = 1000,
    animations = 'disabled',
    mode = 'light',
    deviceType = 'desktop'
  } = options;

  console.log(`📸 Taking screenshot: ${name}`);

  // Prepare page for screenshot
  await preparePageForScreenshot(page, { animations, mode });

  // Build screenshot path with device and mode info
  const screenshotName = `${name}-${deviceType}-${mode}`;

  // Take screenshot with comparison
  await expect(page).toHaveScreenshot(`${screenshotName}.png`, {
    fullPage,
    mask: mask.map(selector => page.locator(selector)),
    clip,
    threshold,
    maxDiffPixels,
    animations: animations as any,
  });

  console.log(`✅ Screenshot saved: ${screenshotName}`);
}

/**
 * Take element screenshot
 */
export async function takeElementScreenshot(
  element: Locator,
  name: string,
  options: ScreenshotOptions = {}
) {
  const {
    mask = [],
    threshold = 0.2,
    maxDiffPixels = 1000,
    animations = 'disabled',
    mode = 'light',
    deviceType = 'desktop'
  } = options;

  console.log(`📸 Taking element screenshot: ${name}`);

  const page = element.page();
  await preparePageForScreenshot(page, { animations, mode });

  const screenshotName = `${name}-${deviceType}-${mode}`;

  await expect(element).toHaveScreenshot(`${screenshotName}.png`, {
    mask: mask.map(selector => page.locator(selector)),
    threshold,
    maxDiffPixels,
    animations: animations as any,
  });

  console.log(`✅ Element screenshot saved: ${screenshotName}`);
}

/**
 * Prepare page for consistent screenshots
 */
export async function preparePageForScreenshot(
  page: Page,
  options: { animations?: string; mode?: string } = {}
) {
  const { animations = 'disabled', mode = 'light' } = options;

  // Set theme mode
  await page.evaluate((theme) => {
    document.documentElement.setAttribute('data-theme', theme);
    document.body.className = document.body.className.replace(/theme-\w+/g, '');
    document.body.classList.add(`theme-${theme}`);
  }, mode);

  // Disable animations if requested
  if (animations === 'disabled') {
    await page.addStyleTag({
      content: `
        *, *::before, *::after {
          animation-duration: 0s !important;
          animation-delay: 0s !important;
          transition-duration: 0s !important;
          transition-delay: 0s !important;
          scroll-behavior: auto !important;
        }
      `
    });
  }

  // Hide dynamic content that might cause flakiness
  await page.addStyleTag({
    content: `
      [data-testid*="timestamp"],
      [data-testid*="time"],
      .cursor-blink,
      .loading-dot {
        visibility: hidden !important;
      }
    `
  });

  // Wait for everything to settle
  await smartWait(page);
  await waitForAnimations(page);

  // Scroll to top to ensure consistent positioning
  await page.evaluate(() => window.scrollTo(0, 0));
  
  // Small delay for final settling
  await page.waitForTimeout(500);
}

/**
 * Take responsive screenshots across different viewports
 */
export async function takeResponsiveScreenshots(
  page: Page,
  name: string,
  options: ScreenshotOptions = {}
) {
  const viewports = [
    { name: 'desktop', width: 1920, height: 1080 },
    { name: 'tablet', width: 1024, height: 768 },
    { name: 'mobile', width: 375, height: 667 }
  ];

  for (const viewport of viewports) {
    await page.setViewportSize(viewport);
    await smartWait(page);
    
    await takeScreenshot(page, `${name}-${viewport.name}`, {
      ...options,
      deviceType: viewport.name as any
    });
  }
}

/**
 * Take comparison screenshots (before/after)
 */
export async function takeComparisonScreenshots(
  page: Page,
  name: string,
  beforeAction: () => Promise<void>,
  afterAction: () => Promise<void>,
  options: ScreenshotOptions = {}
) {
  // Take before screenshot
  await beforeAction();
  await takeScreenshot(page, `${name}-before`, options);
  
  // Perform action
  await afterAction();
  
  // Take after screenshot
  await takeScreenshot(page, `${name}-after`, options);
}

/**
 * Generate marketing screenshots with annotations
 */
export async function generateMarketingScreenshot(
  page: Page,
  name: string,
  annotations: Array<{
    selector: string;
    text: string;
    position: 'top' | 'bottom' | 'left' | 'right';
  }> = [],
  options: ScreenshotOptions = {}
) {
  console.log(`🎨 Generating marketing screenshot: ${name}`);

  // Prepare page
  await preparePageForScreenshot(page, options);

  // Add annotations
  for (const annotation of annotations) {
    await addAnnotation(page, annotation);
  }

  // Take screenshot
  await takeScreenshot(page, `marketing-${name}`, {
    ...options,
    fullPage: true
  });

  // Remove annotations
  await page.evaluate(() => {
    document.querySelectorAll('.playwright-annotation').forEach(el => el.remove());
  });
}

/**
 * Add annotation to page element
 */
async function addAnnotation(
  page: Page,
  annotation: {
    selector: string;
    text: string;
    position: 'top' | 'bottom' | 'left' | 'right';
  }
) {
  await page.evaluate((ann) => {
    const element = document.querySelector(ann.selector);
    if (!element) return;

    const tooltip = document.createElement('div');
    tooltip.className = 'playwright-annotation';
    tooltip.style.cssText = `
      position: absolute;
      background: #007acc;
      color: white;
      padding: 8px 12px;
      border-radius: 4px;
      font-size: 14px;
      font-weight: 500;
      z-index: 10000;
      pointer-events: none;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
      max-width: 200px;
      word-wrap: break-word;
    `;
    tooltip.textContent = ann.text;

    const rect = element.getBoundingClientRect();
    const tooltipRect = { width: 200, height: 40 }; // Estimated

    switch (ann.position) {
      case 'top':
        tooltip.style.left = `${rect.left + rect.width / 2 - tooltipRect.width / 2}px`;
        tooltip.style.top = `${rect.top - tooltipRect.height - 10}px`;
        break;
      case 'bottom':
        tooltip.style.left = `${rect.left + rect.width / 2 - tooltipRect.width / 2}px`;
        tooltip.style.top = `${rect.bottom + 10}px`;
        break;
      case 'left':
        tooltip.style.left = `${rect.left - tooltipRect.width - 10}px`;
        tooltip.style.top = `${rect.top + rect.height / 2 - tooltipRect.height / 2}px`;
        break;
      case 'right':
        tooltip.style.left = `${rect.right + 10}px`;
        tooltip.style.top = `${rect.top + rect.height / 2 - tooltipRect.height / 2}px`;
        break;
    }

    document.body.appendChild(tooltip);
  }, annotation);
}

/**
 * Create screenshot gallery for documentation
 */
export async function createScreenshotGallery(
  page: Page,
  scenarios: Array<{
    name: string;
    description: string;
    setup: () => Promise<void>;
    annotations?: Array<{
      selector: string;
      text: string;
      position: 'top' | 'bottom' | 'left' | 'right';
    }>;
  }>,
  options: ScreenshotOptions = {}
) {
  console.log('📚 Creating screenshot gallery...');

  for (const scenario of scenarios) {
    console.log(`📸 Processing scenario: ${scenario.name}`);
    
    // Setup scenario
    await scenario.setup();
    
    // Take screenshot with annotations
    if (scenario.annotations) {
      await generateMarketingScreenshot(
        page,
        scenario.name,
        scenario.annotations,
        options
      );
    } else {
      await takeScreenshot(page, scenario.name, options);
    }
  }

  console.log('✅ Screenshot gallery created');
}