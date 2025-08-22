/**
 * Custom Cypress commands for arbitration detection E2E tests.
 */

// Login command
Cypress.Commands.add('login', (email = 'test@example.com', password = 'testpassword123') => {
  cy.session([email, password], () => {
    cy.visit('/login')
    cy.get('[data-testid="email-input"]').type(email)
    cy.get('[data-testid="password-input"]').type(password)
    cy.get('[data-testid="login-button"]').click()
    cy.url().should('include', '/dashboard')
    cy.get('[data-testid="user-menu"]').should('be.visible')
  })
})

// File upload command
Cypress.Commands.add('uploadFile', (filePath: string, fileName?: string, mimeType = 'application/pdf') => {
  const actualFileName = fileName || filePath.split('/').pop() || 'test-file.pdf'
  
  cy.fixture(filePath, 'base64').then(fileContent => {
    const blob = Cypress.Blob.base64StringToBlob(fileContent, mimeType)
    const file = new File([blob], actualFileName, { type: mimeType })
    
    cy.get('[data-testid="file-input"]').then(input => {
      const dataTransfer = new DataTransfer()
      dataTransfer.items.add(file)
      input[0].files = dataTransfer.files
      
      // Trigger change event
      cy.wrap(input).trigger('change', { force: true })
    })
  })
})

// Wait for analysis completion
Cypress.Commands.add('waitForAnalysis', (documentId: string, timeout = 30000) => {
  cy.intercept('GET', `/api/v1/documents/${documentId}/analysis`).as('getAnalysis')
  
  const checkAnalysis = () => {
    cy.request({
      url: `/api/v1/documents/${documentId}/analysis`,
      failOnStatusCode: false,
    }).then(response => {
      if (response.status === 200 && response.body.status === 'completed') {
        return response.body
      } else {
        cy.wait(1000)
        checkAnalysis()
      }
    })
  }
  
  checkAnalysis()
})

// Mock API response
Cypress.Commands.add('mockApiResponse', (method: string, url: string, response: any) => {
  cy.intercept(method, url, response).as(`mock${method}${url.replace(/[^a-zA-Z0-9]/g, '')}`)
})

// Check arbitration result
Cypress.Commands.add('checkArbitrationResult', (expected) => {
  cy.get('[data-testid="arbitration-result"]').should('be.visible')
  
  if (expected.hasArbitration) {
    cy.get('[data-testid="positive-result-icon"]').should('be.visible')
    cy.contains(/arbitration clause detected/i).should('be.visible')
  } else {
    cy.get('[data-testid="negative-result-icon"]').should('be.visible')
    cy.contains(/no arbitration clause/i).should('be.visible')
  }
  
  cy.get('[data-testid="confidence-score"]').should('contain', `${Math.round(expected.confidence * 100)}%`)
  
  if (expected.clauseType) {
    cy.get('[data-testid="clause-type-badge"]').should('contain', expected.clauseType)
  }
})

// Visit dashboard with authentication
Cypress.Commands.add('visitDashboard', () => {
  cy.login()
  cy.visit('/dashboard')
  cy.get('[data-testid="dashboard-header"]').should('be.visible')
})

// Seed test data
Cypress.Commands.add('seedTestData', (scenario: string) => {
  cy.task('seedDatabase')
  
  switch (scenario) {
    case 'empty':
      cy.mockApiResponse('GET', '/api/v1/documents*', { fixture: 'documents/empty-list.json' })
      break
      
    case 'with-documents':
      cy.mockApiResponse('GET', '/api/v1/documents*', { fixture: 'documents/sample-documents.json' })
      cy.mockApiResponse('GET', '/api/v1/analyses/recent*', { fixture: 'analyses/recent-analyses.json' })
      break
      
    case 'with-pending-analysis':
      cy.mockApiResponse('GET', '/api/v1/documents*', { fixture: 'documents/pending-documents.json' })
      break
      
    default:
      throw new Error(`Unknown test scenario: ${scenario}`)
  }
})

// Drag and drop file upload
Cypress.Commands.add('dragAndDropFile', (filePath: string, dropSelector: string) => {
  cy.fixture(filePath, 'base64').then(fileContent => {
    const blob = Cypress.Blob.base64StringToBlob(fileContent, 'application/pdf')
    const file = new File([blob], filePath.split('/').pop() || 'test.pdf', { type: 'application/pdf' })
    
    cy.get(dropSelector).trigger('dragenter')
    cy.get(dropSelector).trigger('dragover')
    
    cy.window().then(win => {
      const dataTransfer = new win.DataTransfer()
      dataTransfer.items.add(file)
      
      cy.get(dropSelector).trigger('drop', {
        dataTransfer,
      })
    })
  })
})

// Wait for element to be stable (not moving/changing)
Cypress.Commands.add('waitForStable', (selector: string, timeout = 5000) => {
  let lastPosition: DOMRect
  let stableCount = 0
  const requiredStableChecks = 5
  
  const checkStability = () => {
    cy.get(selector).then($el => {
      const currentPosition = $el[0].getBoundingClientRect()
      
      if (lastPosition && 
          currentPosition.top === lastPosition.top && 
          currentPosition.left === lastPosition.left &&
          currentPosition.width === lastPosition.width &&
          currentPosition.height === lastPosition.height) {
        stableCount++
      } else {
        stableCount = 0
      }
      
      lastPosition = currentPosition
      
      if (stableCount < requiredStableChecks) {
        cy.wait(100)
        checkStability()
      }
    })
  }
  
  checkStability()
})

// Assert toast notification
Cypress.Commands.add('checkToast', (message: string, type = 'success') => {
  cy.get(`[data-testid="toast-${type}"]`).should('be.visible')
  cy.get(`[data-testid="toast-${type}"]`).should('contain', message)
})

// Clear all notifications
Cypress.Commands.add('clearNotifications', () => {
  cy.get('[data-testid*="toast-"]').each($toast => {
    cy.wrap($toast).within(() => {
      cy.get('[data-testid="close-toast"]').click({ force: true })
    })
  })
})

// Add type definitions
declare global {
  namespace Cypress {
    interface Chainable {
      dragAndDropFile(filePath: string, dropSelector: string): Chainable<void>
      waitForStable(selector: string, timeout?: number): Chainable<void>
      checkToast(message: string, type?: string): Chainable<void>
      clearNotifications(): Chainable<void>
    }
  }
}