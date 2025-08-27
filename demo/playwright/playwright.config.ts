import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests',
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
    ['html', { outputFolder: 'reports/html' }],
    ['json', { outputFile: 'reports/results.json' }],
    ['junit', { outputFile: 'reports/results.xml' }],
    ['line']
  ],
  /* Shared settings for all the projects below. See https://playwright.dev/docs/api/class-testoptions. */
  use: {
    /* Base URL to use in actions like `await page.goto('/')`. */
    baseURL: process.env.BASE_URL || 'http://localhost:3000',
    
    /* Collect trace when retrying the failed test. See https://playwright.dev/docs/trace-viewer */
    trace: 'on-first-retry',
    
    /* Take screenshot on failure */
    screenshot: 'only-on-failure',
    
    /* Record video on failure */
    video: 'retain-on-failure',
    
    /* Maximum time each action such as `click()` can take. */
    actionTimeout: 10000,
    
    /* Maximum time each navigation can take. */
    navigationTimeout: 30000,
    
    /* Global test timeout */
    testIdAttribute: 'data-testid',
  },

  /* Global setup and teardown */
  globalSetup: require.resolve('./global-setup'),
  globalTeardown: require.resolve('./global-teardown'),

  /* Configure projects for major browsers */
  projects: [
    {
      name: 'setup',
      testMatch: /.*\.setup\.ts/,
    },
    
    // Desktop browsers
    {
      name: 'chromium-desktop',
      use: { 
        ...devices['Desktop Chrome'],
        viewport: { width: 1920, height: 1080 }
      },
      dependencies: ['setup'],
    },
    {
      name: 'firefox-desktop',
      use: { 
        ...devices['Desktop Firefox'],
        viewport: { width: 1920, height: 1080 }
      },
      dependencies: ['setup'],
    },
    {
      name: 'webkit-desktop',
      use: { 
        ...devices['Desktop Safari'],
        viewport: { width: 1920, height: 1080 }
      },
      dependencies: ['setup'],
    },

    // Tablet devices
    {
      name: 'tablet-portrait',
      use: { 
        ...devices['iPad Pro'],
        viewport: { width: 1024, height: 1366 }
      },
      dependencies: ['setup'],
    },
    {
      name: 'tablet-landscape',
      use: { 
        ...devices['iPad Pro landscape'],
        viewport: { width: 1366, height: 1024 }
      },
      dependencies: ['setup'],
    },

    // Mobile devices
    {
      name: 'mobile-chrome',
      use: { 
        ...devices['iPhone 13 Pro'],
        viewport: { width: 390, height: 844 }
      },
      dependencies: ['setup'],
    },
    {
      name: 'mobile-safari',
      use: { 
        ...devices['iPhone 13 Pro Safari'],
        viewport: { width: 390, height: 844 }
      },
      dependencies: ['setup'],
    },
    {
      name: 'android-mobile',
      use: { 
        ...devices['Pixel 5'],
        viewport: { width: 393, height: 851 }
      },
      dependencies: ['setup'],
    },

    // High-resolution displays
    {
      name: 'high-dpi',
      use: {
        ...devices['Desktop Chrome HiDPI'],
        viewport: { width: 1920, height: 1080 },
        deviceScaleFactor: 2
      },
      dependencies: ['setup'],
    },

    // Accessibility testing
    {
      name: 'accessibility',
      use: {
        ...devices['Desktop Chrome'],
        viewport: { width: 1280, height: 720 },
        // Enable screen reader simulation
        extraHTTPHeaders: {
          'User-Agent': 'Playwright-Accessibility-Test'
        }
      },
      dependencies: ['setup'],
    }
  ],

  /* Expect settings */
  expect: {
    /* Maximum time expect() should wait for the condition to be met. */
    timeout: 5000,
    /* Threshold for pixel comparison */
    threshold: 0.2,
    /* Animation handling for visual comparisons */
    animations: 'disabled',
  },

  /* Configure the development server */
  webServer: process.env.CI ? undefined : {
    command: 'cd ../.. && npm run dev',
    port: 3000,
    reuseExistingServer: !process.env.CI,
    timeout: 120000,
  },

  /* Output directory for test results */
  outputDir: 'test-results/',
  
  /* Directory for screenshots */
  snapshotDir: 'screenshots/',
  
  /* Directory for test artifacts */
  artifactsDir: 'artifacts/',
});