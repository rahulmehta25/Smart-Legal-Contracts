import { defineConfig, devices } from '@playwright/test';

/**
 * See https://playwright.dev/docs/test-configuration.
 */
export default defineConfig({
  testDir: './tests/e2e',
  /* Run tests in files in parallel */
  fullyParallel: true,
  /* Fail the build on CI if you accidentally left test.only in the source code. */
  forbidOnly: !!process.env.CI,
  /* Retry on CI only */
  retries: process.env.CI ? 2 : 0,
  /* Opt out of parallel tests on CI. */
  workers: process.env.CI ? 1 : undefined,
  /* Reporter to use. See https://playwright.dev/docs/test-reporters */
  reporter: [
    ['html', { outputFolder: 'playwright-report' }],
    ['json', { outputFile: 'test-results/results.json' }],
    ['junit', { outputFile: 'test-results/junit.xml' }],
    ['github'] // GitHub Actions integration
  ],
  /* Shared settings for all the projects below. See https://playwright.dev/docs/api/class-testoptions. */
  use: {
    /* Base URL to use in actions like `await page.goto('/')`. */
    baseURL: process.env.FRONTEND_URL || 'http://localhost:5173',
    /* Collect trace when retrying the failed test. See https://playwright.dev/docs/trace-viewer */
    trace: 'retain-on-failure',
    /* Take screenshot on failure */
    screenshot: 'only-on-failure',
    /* Record video on failure */
    video: 'retain-on-failure',
  },

  /* Configure projects for major browsers and environments */
  projects: [
    // Setup project for authentication and shared state
    {
      name: 'setup',
      testMatch: /.*\.setup\.ts/,
    },

    // Desktop browsers - Chrome
    {
      name: 'chromium',
      use: { 
        ...devices['Desktop Chrome'],
        channel: 'chrome',
        // Enable WebSocket testing
        contextOptions: {
          permissions: ['notifications'],
        }
      },
      dependencies: ['setup'],
    },

    // Desktop browsers - Firefox
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
      dependencies: ['setup'],
    },

    // Desktop browsers - Safari
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
      dependencies: ['setup'],
    },

    /* Test against mobile viewports. */
    {
      name: 'Mobile Chrome',
      use: { ...devices['Pixel 5'] },
      dependencies: ['setup'],
    },
    {
      name: 'Mobile Safari',
      use: { ...devices['iPhone 12'] },
      dependencies: ['setup'],
    },

    /* Test against tablet viewports. */
    {
      name: 'Tablet',
      use: { ...devices['iPad Pro'] },
      dependencies: ['setup'],
    },

    // API testing project - no browser needed
    {
      name: 'api',
      testMatch: /.*\.api\.spec\.ts/,
      use: {
        // API tests don't need a browser
        baseURL: process.env.API_URL || 'http://localhost:8000',
      },
    },

    // WebSocket testing project
    {
      name: 'websocket',
      testMatch: /.*\.websocket\.spec\.ts/,
      use: {
        baseURL: process.env.API_URL || 'http://localhost:8000',
      },
    },

    // Visual regression testing
    {
      name: 'visual',
      testMatch: /.*\.visual\.spec\.ts/,
      use: { 
        ...devices['Desktop Chrome'],
        channel: 'chrome'
      },
      dependencies: ['setup'],
    }
  ],

  /* Run your local dev server before starting the tests */
  webServer: [
    {
      command: 'npm run dev',
      url: 'http://localhost:5173',
      reuseExistingServer: !process.env.CI,
      timeout: 120 * 1000,
      env: {
        NODE_ENV: 'test',
        VITE_API_URL: 'http://localhost:8000'
      }
    },
    {
      command: 'cd ../backend && python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000',
      url: 'http://localhost:8000/health',
      reuseExistingServer: !process.env.CI,
      timeout: 120 * 1000,
      cwd: '../backend'
    }
  ],

  /* Global setup and teardown */
  globalSetup: require.resolve('./tests/global-setup.ts'),
  globalTeardown: require.resolve('./tests/global-teardown.ts'),

  /* Expect options */
  expect: {
    // Visual comparison threshold
    threshold: 0.2,
    // Timeout for assertions
    timeout: 10000,
  },

  /* Test timeout */
  timeout: 30000,

  /* Output directories */
  outputDir: 'test-results',
  
  /* Environment-specific configurations */
  ...(process.env.NODE_ENV === 'production' && {
    use: {
      baseURL: process.env.FRONTEND_URL || 'https://arbitration-detector.com',
    },
    webServer: undefined, // Don't start local servers in production
  }),
});