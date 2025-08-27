import { chromium, FullConfig } from '@playwright/test';
import { uploadTestDocument, createTestUser } from './utils/test-data';

async function globalSetup(config: FullConfig) {
  console.log('🚀 Starting global setup...');
  
  // Launch browser for setup
  const browser = await chromium.launch();
  const context = await browser.newContext();
  const page = await context.newPage();

  try {
    // Wait for the application to be ready
    console.log('⏳ Waiting for application to be ready...');
    await page.goto(config.projects[0].use?.baseURL || 'http://localhost:3000');
    await page.waitForSelector('[data-testid="app-ready"]', { timeout: 60000 });
    
    // Create test user for authenticated tests
    console.log('👤 Creating test user...');
    await createTestUser(page);
    
    // Upload test documents for analysis tests
    console.log('📄 Uploading test documents...');
    await uploadTestDocument(page, 'sample_tos_with_arbitration.txt');
    await uploadTestDocument(page, 'sample_contract_without_arbitration.txt');
    
    // Clear any modal dialogs or notifications
    await page.evaluate(() => {
      // Clear any existing notifications
      const notifications = document.querySelectorAll('[data-testid*="notification"]');
      notifications.forEach(notification => notification.remove());
      
      // Clear any modal overlays
      const modals = document.querySelectorAll('[data-testid*="modal"]');
      modals.forEach(modal => modal.remove());
    });
    
    console.log('✅ Global setup completed successfully');
    
  } catch (error) {
    console.error('❌ Global setup failed:', error);
    throw error;
  } finally {
    await context.close();
    await browser.close();
  }
}

export default globalSetup;