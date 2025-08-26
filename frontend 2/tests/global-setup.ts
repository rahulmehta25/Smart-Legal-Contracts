import { chromium, FullConfig } from '@playwright/test';

async function globalSetup(config: FullConfig) {
  console.log('🚀 Global setup: Starting test environment...');
  
  // Wait for backend to be ready
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  let retries = 0;
  const maxRetries = 30; // 30 seconds total wait
  
  while (retries < maxRetries) {
    try {
      const response = await page.goto('http://localhost:8000/health', { 
        timeout: 2000,
        waitUntil: 'networkidle'
      });
      
      if (response?.status() === 200) {
        const body = await response.json();
        if (body.status === 'healthy') {
          console.log('✅ Backend health check passed');
          break;
        }
      }
    } catch (error) {
      console.log(`⏳ Waiting for backend... (${retries + 1}/${maxRetries})`);
    }
    
    retries++;
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  
  if (retries >= maxRetries) {
    console.error('❌ Backend failed to start within timeout');
    throw new Error('Backend not ready');
  }

  // Wait for frontend to be ready  
  retries = 0;
  while (retries < maxRetries) {
    try {
      const response = await page.goto('http://localhost:5173', { 
        timeout: 2000,
        waitUntil: 'networkidle'
      });
      
      if (response?.status() === 200) {
        console.log('✅ Frontend health check passed');
        break;
      }
    } catch (error) {
      console.log(`⏳ Waiting for frontend... (${retries + 1}/${maxRetries})`);
    }
    
    retries++;
    await new Promise(resolve => setTimeout(resolve, 1000));
  }
  
  if (retries >= maxRetries) {
    console.error('❌ Frontend failed to start within timeout');
    throw new Error('Frontend not ready');
  }

  await browser.close();
  console.log('🎯 Global setup complete: Test environment ready');
}

export default globalSetup;