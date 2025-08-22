/**
 * Cypress E2E support file for arbitration detection application.
 * 
 * Contains custom commands, global configuration, and test utilities.
 */

// Import Cypress commands
import './commands'

// Global before hooks
before(() => {
  // Set up test environment
  cy.task('log', 'Starting E2E test suite')
  
  // Clear browser state
  cy.clearCookies()
  cy.clearLocalStorage()
  cy.clearAllSessionStorage()
})

beforeEach(() => {
  // Reset API intercepts
  cy.intercept('GET', '/api/v1/documents*', { fixture: 'documents/empty-list.json' }).as('getDocuments')
  cy.intercept('GET', '/api/v1/analyses/recent*', { fixture: 'analyses/empty-list.json' }).as('getRecentAnalyses')
  cy.intercept('GET', '/api/v1/user/stats*', { fixture: 'user/stats.json' }).as('getUserStats')
  
  // Mock authentication if needed
  cy.window().then((win) => {
    win.localStorage.setItem('auth_token', 'mock-jwt-token')
  })
})

afterEach(() => {
  // Clean up after each test
  cy.clearLocalStorage()
  
  // Log test completion
  cy.task('log', `Completed test: ${Cypress.currentTest.title}`)
})

// Global error handling
Cypress.on('uncaught:exception', (err, runnable) => {
  // Don't fail tests on uncaught exceptions that we expect
  const expectedErrors = [
    'ResizeObserver loop limit exceeded',
    'Non-Error promise rejection captured',
    'Loading chunk',
  ]
  
  return !expectedErrors.some(expectedError => 
    err.message.includes(expectedError)
  )
})

// Custom configuration
Cypress.config('scrollBehavior', 'center')

// Add custom types
declare global {
  namespace Cypress {
    interface Chainable {
      /**
       * Custom command to login a user
       */
      login(email?: string, password?: string): Chainable<void>
      
      /**
       * Custom command to upload a file
       */
      uploadFile(filePath: string, fileName?: string, mimeType?: string): Chainable<void>
      
      /**
       * Custom command to wait for analysis completion
       */
      waitForAnalysis(documentId: string, timeout?: number): Chainable<void>
      
      /**
       * Custom command to mock API responses
       */
      mockApiResponse(method: string, url: string, response: any): Chainable<void>
      
      /**
       * Custom command to check arbitration result
       */
      checkArbitrationResult(expected: {
        hasArbitration: boolean
        confidence: number
        clauseType?: string
      }): Chainable<void>
      
      /**
       * Custom command to navigate to dashboard
       */
      visitDashboard(): Chainable<void>
      
      /**
       * Custom command to seed test data
       */
      seedTestData(scenario: string): Chainable<void>
    }
  }
}