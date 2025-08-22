import { defineConfig } from 'cypress'

export default defineConfig({
  e2e: {
    baseUrl: 'http://localhost:3000',
    specPattern: 'cypress/e2e/**/*.cy.{js,jsx,ts,tsx}',
    supportFile: 'cypress/support/e2e.ts',
    fixturesFolder: 'cypress/fixtures',
    screenshotsFolder: 'cypress/screenshots',
    videosFolder: 'cypress/videos',
    
    // Viewport settings
    viewportWidth: 1280,
    viewportHeight: 720,
    
    // Test settings
    defaultCommandTimeout: 10000,
    requestTimeout: 10000,
    responseTimeout: 10000,
    pageLoadTimeout: 30000,
    
    // Video and screenshot settings
    video: true,
    videoUploadOnPasses: false,
    screenshotOnRunFailure: true,
    
    // Retry settings
    retries: {
      runMode: 2,
      openMode: 0,
    },
    
    // Environment variables
    env: {
      API_BASE_URL: 'http://localhost:8000/api/v1',
      TEST_USER_EMAIL: 'test@example.com',
      TEST_USER_PASSWORD: 'testpassword123',
    },
    
    setupNodeEvents(on, config) {
      // Task definitions
      on('task', {
        // Database seeding tasks
        seedDatabase() {
          // Seed test database with sample data
          return null
        },
        
        cleanDatabase() {
          // Clean test database
          return null
        },
        
        // File system tasks
        createTestFile(filename: string) {
          const fs = require('fs')
          const path = require('path')
          
          const filePath = path.join('cypress/fixtures/uploads', filename)
          const content = `Test document content for ${filename}`
          
          fs.writeFileSync(filePath, content)
          return filePath
        },
        
        // Log tasks
        log(message: string) {
          console.log(message)
          return null
        },
      })
      
      // Browser event handlers
      on('before:browser:launch', (browser, launchOptions) => {
        if (browser.name === 'chrome' && browser.isHeadless) {
          launchOptions.args.push('--disable-gpu')
          launchOptions.args.push('--no-sandbox')
          launchOptions.args.push('--disable-dev-shm-usage')
        }
        
        return launchOptions
      })
      
      // Test file preprocessing
      on('file:preprocessor', require('@cypress/webpack-preprocessor')({
        webpackOptions: {
          resolve: {
            extensions: ['.ts', '.tsx', '.js', '.jsx'],
          },
          module: {
            rules: [
              {
                test: /\.tsx?$/,
                loader: 'ts-loader',
                options: {
                  transpileOnly: true,
                },
              },
            ],
          },
        },
      }))
      
      return config
    },
  },
  
  component: {
    devServer: {
      framework: 'next',
      bundler: 'webpack',
    },
    specPattern: 'src/**/*.cy.{js,jsx,ts,tsx}',
    supportFile: 'cypress/support/component.ts',
  },
  
  // Global configuration
  watchForFileChanges: true,
  chromeWebSecurity: false,
  
  // Experimental features
  experimentalStudio: true,
  experimentalWebKitSupport: true,
})