import { chromium, FullConfig } from '@playwright/test';

async function globalTeardown(config: FullConfig) {
  console.log('🧹 Starting global teardown...');
  
  const browser = await chromium.launch();
  const context = await browser.newContext();
  const page = await context.newPage();

  try {
    // Clean up test data
    console.log('🗑️ Cleaning up test data...');
    await page.goto(config.projects[0].use?.baseURL || 'http://localhost:3000');
    
    // Clear browser storage
    await page.evaluate(() => {
      localStorage.clear();
      sessionStorage.clear();
      
      // Clear any test artifacts
      if ('serviceWorker' in navigator) {
        navigator.serviceWorker.getRegistrations().then(registrations => {
          registrations.forEach(registration => registration.unregister());
        });
      }
    });
    
    // Clear cookies
    await context.clearCookies();
    
    console.log('✅ Global teardown completed successfully');
    
  } catch (error) {
    console.error('❌ Global teardown failed:', error);
    // Don't throw error in teardown to avoid masking test failures
  } finally {
    await context.close();
    await browser.close();
  }
}

export default globalTeardown;