import { test as setup, expect } from '@playwright/test';
import { TestHelpers } from './utils/test-helpers';

const authFile = 'tests/.auth/user.json';

setup('authenticate', async ({ page }) => {
  const helpers = new TestHelpers(page);

  console.log('🔐 Setting up authentication for tests...');

  // Navigate to login page
  await page.goto('/');
  
  // Check if login is required or if we can access the app directly
  const isLoginRequired = await helpers.elementExists('[data-testid="login-form"]');
  
  if (!isLoginRequired) {
    console.log('🚀 No authentication required, proceeding with tests');
    return;
  }

  // Mock successful authentication if login form exists
  await helpers.mockAPIResponse('**/api/v1/users/login', {
    token: 'mock-jwt-token',
    user: {
      id: 1,
      username: 'testuser',
      email: 'test@example.com',
      firstName: 'Test',
      lastName: 'User'
    },
    expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString()
  });

  // Fill login form
  await page.fill('[data-testid="username-input"]', 'testuser');
  await page.fill('[data-testid="password-input"]', 'password123');
  
  // Submit login
  await page.click('[data-testid="login-submit"]');
  
  // Wait for successful login
  await expect(page).toHaveURL(/.*dashboard.*|.*home.*/);
  
  // Save authentication state
  await page.context().storageState({ path: authFile });
  
  console.log('✅ Authentication setup complete');
});