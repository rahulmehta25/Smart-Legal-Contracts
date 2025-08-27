import { Page, Locator, expect } from '@playwright/test';

/**
 * Wait for element to be visible and stable
 */
export async function waitForStableElement(
  page: Page, 
  selector: string, 
  options: { timeout?: number; stabilityTimeout?: number } = {}
) {
  const { timeout = 10000, stabilityTimeout = 1000 } = options;
  
  console.log(`⏳ Waiting for stable element: ${selector}`);
  
  // Wait for element to be visible
  await page.waitForSelector(selector, { timeout, state: 'visible' });
  
  // Wait for element to be stable (no position/size changes)
  const element = page.locator(selector);
  let lastBox = await element.boundingBox();
  let stableCount = 0;
  const requiredStableChecks = Math.ceil(stabilityTimeout / 100);
  
  while (stableCount < requiredStableChecks) {
    await page.waitForTimeout(100);
    const currentBox = await element.boundingBox();
    
    if (JSON.stringify(lastBox) === JSON.stringify(currentBox)) {
      stableCount++;
    } else {
      stableCount = 0;
      lastBox = currentBox;
    }
  }
  
  console.log(`✅ Element stable: ${selector}`);
  return element;
}

/**
 * Wait for page to be fully loaded (no pending network requests)
 */
export async function waitForPageLoad(page: Page, timeout = 30000) {
  console.log('⏳ Waiting for page to load completely...');
  
  await page.waitForLoadState('networkidle', { timeout });
  
  // Wait for any custom loading indicators to disappear
  try {
    await page.waitForSelector('[data-testid="loading-spinner"]', { 
      state: 'hidden', 
      timeout: 5000 
    });
  } catch {
    // Loading spinner might not exist, continue
  }
  
  console.log('✅ Page loaded completely');
}

/**
 * Wait for analysis to complete
 */
export async function waitForAnalysisComplete(page: Page, timeout = 60000) {
  console.log('⏳ Waiting for analysis to complete...');
  
  // Wait for analysis to start
  await page.waitForSelector('[data-testid="analysis-status"]', { timeout: 10000 });
  
  // Wait for progress indicators
  const progressBar = page.locator('[data-testid="analysis-progress"]');
  if (await progressBar.isVisible()) {
    console.log('📊 Monitoring analysis progress...');
    
    // Wait for progress to reach 100% or complete
    await page.waitForFunction(() => {
      const progress = document.querySelector('[data-testid="analysis-progress"]');
      if (!progress) return true; // Progress bar disappeared
      
      const value = progress.getAttribute('value') || progress.getAttribute('aria-valuenow');
      return value === '100' || !document.querySelector('[data-testid="analysis-status"]');
    }, { timeout });
  }
  
  // Wait for final results
  await page.waitForSelector('[data-testid="analysis-complete"]', { timeout });
  
  // Ensure results are stable
  await waitForStableElement(page, '[data-testid="analysis-result"]');
  
  console.log('✅ Analysis completed');
}

/**
 * Wait for notification to appear and optionally dismiss it
 */
export async function waitForNotification(
  page: Page, 
  type: 'success' | 'error' | 'warning' | 'info' = 'success',
  dismiss = true
) {
  console.log(`🔔 Waiting for ${type} notification...`);
  
  const notification = page.locator(`[data-testid="notification-${type}"]`);
  await expect(notification).toBeVisible({ timeout: 10000 });
  
  if (dismiss) {
    // Auto-dismiss after reading the message
    await page.waitForTimeout(1000);
    
    const dismissButton = notification.locator('[data-testid="dismiss-notification"]');
    if (await dismissButton.isVisible()) {
      await dismissButton.click();
    }
  }
  
  console.log(`✅ ${type} notification handled`);
  return notification;
}

/**
 * Wait for modal to open and be interactive
 */
export async function waitForModal(page: Page, modalId: string, timeout = 10000) {
  console.log(`🪟 Waiting for modal: ${modalId}`);
  
  const modal = page.locator(`[data-testid="${modalId}"]`);
  
  // Wait for modal to be visible
  await expect(modal).toBeVisible({ timeout });
  
  // Wait for modal animation to complete
  await page.waitForFunction(
    (id) => {
      const modal = document.querySelector(`[data-testid="${id}"]`);
      if (!modal) return false;
      
      const styles = window.getComputedStyle(modal);
      return styles.opacity === '1' && styles.transform === 'none';
    },
    modalId,
    { timeout: 5000 }
  );
  
  // Wait for any form elements to be ready
  await page.waitForTimeout(500);
  
  console.log(`✅ Modal ready: ${modalId}`);
  return modal;
}

/**
 * Wait for form validation to complete
 */
export async function waitForFormValidation(page: Page, formSelector: string) {
  console.log(`📝 Waiting for form validation: ${formSelector}`);
  
  const form = page.locator(formSelector);
  
  // Trigger validation by attempting to submit
  await form.press('Enter');
  
  // Wait for validation messages to appear
  await page.waitForTimeout(500);
  
  // Check if form is valid (no error messages)
  const errorMessages = form.locator('[data-testid*="error"]');
  const hasErrors = await errorMessages.count() > 0;
  
  console.log(`✅ Form validation complete: ${hasErrors ? 'has errors' : 'valid'}`);
  return !hasErrors;
}

/**
 * Wait for file upload to complete
 */
export async function waitForFileUpload(page: Page, timeout = 30000) {
  console.log('📤 Waiting for file upload to complete...');
  
  // Wait for upload to start
  await page.waitForSelector('[data-testid="upload-progress"]', { timeout: 5000 });
  
  // Monitor upload progress
  await page.waitForFunction(() => {
    const progress = document.querySelector('[data-testid="upload-progress"]');
    if (!progress) return true; // Upload completed or failed
    
    const value = parseInt(progress.getAttribute('value') || '0');
    return value >= 100;
  }, { timeout });
  
  // Wait for upload completion indicator
  await page.waitForSelector('[data-testid="upload-complete"]', { timeout: 5000 });
  
  console.log('✅ File upload completed');
}

/**
 * Wait for animations to finish
 */
export async function waitForAnimations(page: Page) {
  console.log('🎬 Waiting for animations to finish...');
  
  await page.waitForFunction(() => {
    // Check for CSS animations and transitions
    const animatedElements = document.querySelectorAll('*');
    
    for (const element of animatedElements) {
      const styles = window.getComputedStyle(element);
      
      // Check for ongoing animations
      if (styles.animationPlayState === 'running') {
        return false;
      }
      
      // Check for ongoing transitions
      if (styles.transitionProperty !== 'none' && 
          styles.transitionDuration !== '0s') {
        return false;
      }
    }
    
    return true;
  });
  
  console.log('✅ Animations finished');
}

/**
 * Wait for network requests to complete
 */
export async function waitForNetworkIdle(page: Page, timeout = 10000) {
  console.log('🌐 Waiting for network to be idle...');
  
  await page.waitForLoadState('networkidle', { timeout });
  
  console.log('✅ Network is idle');
}

/**
 * Smart wait that combines multiple waiting strategies
 */
export async function smartWait(page: Page, options: {
  selector?: string;
  networkIdle?: boolean;
  animations?: boolean;
  timeout?: number;
} = {}) {
  const { 
    selector, 
    networkIdle = true, 
    animations = true, 
    timeout = 10000 
  } = options;
  
  console.log('🧠 Smart wait starting...');
  
  // Wait for specific element if provided
  if (selector) {
    await waitForStableElement(page, selector, { timeout });
  }
  
  // Wait for network to be idle
  if (networkIdle) {
    await waitForNetworkIdle(page, timeout);
  }
  
  // Wait for animations to finish
  if (animations) {
    await waitForAnimations(page);
  }
  
  console.log('✅ Smart wait completed');
}