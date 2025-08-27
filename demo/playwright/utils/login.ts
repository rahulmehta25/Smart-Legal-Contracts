import { Page, expect } from '@playwright/test';

export interface LoginCredentials {
  email: string;
  password: string;
}

export interface TestUser extends LoginCredentials {
  name: string;
  role: 'admin' | 'user' | 'demo';
}

/**
 * Default test users for different scenarios
 */
export const TEST_USERS: Record<string, TestUser> = {
  admin: {
    name: 'Test Admin',
    email: 'admin@test.com',
    password: 'admin123',
    role: 'admin'
  },
  user: {
    name: 'Test User',
    email: 'user@test.com', 
    password: 'user123',
    role: 'user'
  },
  demo: {
    name: 'Demo User',
    email: 'demo@test.com',
    password: 'demo123',
    role: 'demo'
  }
};

/**
 * Login with credentials
 */
export async function login(page: Page, credentials: LoginCredentials) {
  console.log(`🔐 Logging in as: ${credentials.email}`);
  
  // Navigate to login page
  await page.goto('/login');
  
  // Wait for login form
  await page.waitForSelector('[data-testid="login-form"]');
  
  // Fill in credentials
  await page.fill('[data-testid="email-input"]', credentials.email);
  await page.fill('[data-testid="password-input"]', credentials.password);
  
  // Submit form
  await page.click('[data-testid="login-button"]');
  
  // Wait for successful login (redirect to dashboard or home)
  await page.waitForURL(/\/(dashboard|home)/, { timeout: 10000 });
  
  // Verify user is logged in
  await expect(page.locator('[data-testid="user-menu"]')).toBeVisible();
  
  console.log('✅ Login successful');
}

/**
 * Login as specific test user
 */
export async function loginAsTestUser(page: Page, userType: keyof typeof TEST_USERS) {
  const user = TEST_USERS[userType];
  if (!user) {
    throw new Error(`Unknown test user type: ${userType}`);
  }
  
  await login(page, user);
  return user;
}

/**
 * Logout current user
 */
export async function logout(page: Page) {
  console.log('🚪 Logging out...');
  
  // Click user menu
  await page.click('[data-testid="user-menu"]');
  
  // Click logout
  await page.click('[data-testid="logout-button"]');
  
  // Wait for redirect to login or home page
  await page.waitForURL(/\/(login|home|\/)/, { timeout: 5000 });
  
  // Verify logout
  await expect(page.locator('[data-testid="login-button"]')).toBeVisible();
  
  console.log('✅ Logout successful');
}

/**
 * Check if user is logged in
 */
export async function isLoggedIn(page: Page): Promise<boolean> {
  try {
    await page.waitForSelector('[data-testid="user-menu"]', { timeout: 1000 });
    return true;
  } catch {
    return false;
  }
}

/**
 * Login with demo account (for demo scenarios)
 */
export async function loginForDemo(page: Page) {
  // Check if already logged in
  if (await isLoggedIn(page)) {
    console.log('ℹ️ Already logged in, skipping login');
    return;
  }
  
  // Try to login as demo user, fallback to creating account
  try {
    await loginAsTestUser(page, 'demo');
  } catch (error) {
    console.log('⚠️ Demo user not found, creating demo account...');
    await createDemoAccount(page);
  }
}

/**
 * Create demo account for testing
 */
export async function createDemoAccount(page: Page) {
  console.log('👤 Creating demo account...');
  
  // Navigate to signup page
  await page.goto('/signup');
  
  // Wait for signup form
  await page.waitForSelector('[data-testid="signup-form"]');
  
  const demoUser = TEST_USERS.demo;
  
  // Fill in signup form
  await page.fill('[data-testid="name-input"]', demoUser.name);
  await page.fill('[data-testid="email-input"]', demoUser.email);
  await page.fill('[data-testid="password-input"]', demoUser.password);
  await page.fill('[data-testid="confirm-password-input"]', demoUser.password);
  
  // Accept terms
  await page.check('[data-testid="terms-checkbox"]');
  
  // Submit form
  await page.click('[data-testid="signup-button"]');
  
  // Wait for successful signup
  await page.waitForURL(/\/(dashboard|home|welcome)/, { timeout: 10000 });
  
  console.log('✅ Demo account created successfully');
}

/**
 * Create test user for global setup
 */
export async function createTestUser(page: Page) {
  for (const [userType, user] of Object.entries(TEST_USERS)) {
    try {
      console.log(`👤 Creating test user: ${user.email}`);
      
      // Navigate to signup
      await page.goto('/signup');
      await page.waitForSelector('[data-testid="signup-form"]', { timeout: 5000 });
      
      // Fill form
      await page.fill('[data-testid="name-input"]', user.name);
      await page.fill('[data-testid="email-input"]', user.email);
      await page.fill('[data-testid="password-input"]', user.password);
      await page.fill('[data-testid="confirm-password-input"]', user.password);
      
      // Accept terms
      await page.check('[data-testid="terms-checkbox"]');
      
      // Submit
      await page.click('[data-testid="signup-button"]');
      
      // Wait for completion
      await page.waitForURL(/\/(dashboard|home|welcome)/, { timeout: 10000 });
      
      // Logout to create next user
      await logout(page);
      
    } catch (error) {
      console.warn(`⚠️ Failed to create test user ${user.email}:`, error);
    }
  }
}

/**
 * Skip authentication for demo mode
 */
export async function enableDemoMode(page: Page) {
  // Add demo mode token to localStorage
  await page.evaluate(() => {
    localStorage.setItem('demo-mode', 'true');
    localStorage.setItem('demo-user', JSON.stringify({
      name: 'Demo User',
      email: 'demo@example.com',
      role: 'demo'
    }));
  });
  
  console.log('🎭 Demo mode enabled');
}