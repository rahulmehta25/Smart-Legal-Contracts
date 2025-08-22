/**
 * E2E tests for document upload and analysis workflow.
 * 
 * Tests the complete user journey from uploading documents
 * to receiving arbitration analysis results.
 */

describe('Document Upload and Analysis Flow', () => {
  beforeEach(() => {
    cy.seedTestData('empty')
    cy.visitDashboard()
  })

  describe('File Upload Interface', () => {
    it('displays upload area and instructions', () => {
      cy.get('[data-testid="upload-area"]').should('be.visible')
      cy.contains('Drag and drop your documents here').should('be.visible')
      cy.contains('or browse files').should('be.visible')
      cy.contains('Supported formats: PDF, DOCX, DOC').should('be.visible')
      cy.contains('Max size: 20 MB').should('be.visible')
    })

    it('shows file type restrictions', () => {
      cy.get('[data-testid="file-restrictions"]').should('contain', 'PDF')
      cy.get('[data-testid="file-restrictions"]').should('contain', 'DOCX')
      cy.get('[data-testid="file-restrictions"]').should('contain', 'DOC')
    })
  })

  describe('Successful File Upload', () => {
    it('uploads PDF file via file input', () => {
      // Mock successful upload
      cy.intercept('POST', '/api/v1/documents/upload', {
        statusCode: 200,
        body: {
          document_id: 'doc-123',
          filename: 'contract.pdf',
          status: 'uploaded'
        }
      }).as('uploadFile')

      // Upload file
      cy.uploadFile('documents/sample-contract.pdf', 'contract.pdf')

      // Verify upload request
      cy.wait('@uploadFile').then((interception) => {
        expect(interception.request.body).to.include('contract.pdf')
      })

      // Check success feedback
      cy.checkToast('File uploaded successfully')
      cy.get('[data-testid="upload-progress"]').should('not.exist')
      cy.get('[data-testid="upload-success"]').should('be.visible')
    })

    it('uploads file via drag and drop', () => {
      // Mock successful upload
      cy.intercept('POST', '/api/v1/documents/upload', {
        statusCode: 200,
        body: {
          document_id: 'doc-456',
          filename: 'agreement.pdf',
          status: 'uploaded'
        }
      }).as('uploadFileDragDrop')

      // Drag and drop file
      cy.dragAndDropFile('documents/sample-agreement.pdf', '[data-testid="upload-area"]')

      // Verify upload
      cy.wait('@uploadFileDragDrop')
      cy.checkToast('File uploaded successfully')
    })

    it('shows upload progress during file upload', () => {
      // Mock slow upload with progress
      cy.intercept('POST', '/api/v1/documents/upload', (req) => {
        req.reply({
          delay: 2000,
          statusCode: 200,
          body: {
            document_id: 'doc-789',
            filename: 'contract.pdf',
            status: 'uploaded'
          }
        })
      }).as('slowUpload')

      cy.uploadFile('documents/large-contract.pdf')

      // Check progress indicators
      cy.get('[data-testid="upload-progress"]').should('be.visible')
      cy.get('[data-testid="progress-bar"]').should('be.visible')
      cy.contains('Uploading...').should('be.visible')

      cy.wait('@slowUpload')
      cy.get('[data-testid="upload-progress"]').should('not.exist')
    })
  })

  describe('Automatic Analysis After Upload', () => {
    it('triggers analysis automatically after successful upload', () => {
      // Mock upload
      cy.intercept('POST', '/api/v1/documents/upload', {
        statusCode: 200,
        body: {
          document_id: 'doc-auto-123',
          filename: 'contract.pdf',
          status: 'uploaded'
        }
      }).as('uploadForAnalysis')

      // Mock analysis
      cy.intercept('POST', '/api/v1/arbitration/detect-by-id', {
        statusCode: 200,
        body: {
          document_id: 'doc-auto-123',
          has_arbitration: true,
          confidence: 0.92,
          clause_type: 'mandatory_binding',
          explanation: 'The document contains a clear arbitration clause.',
          keywords: ['arbitration', 'binding', 'mandatory'],
          processing_time: 1.5
        }
      }).as('autoAnalysis')

      cy.uploadFile('documents/arbitration-contract.pdf')

      // Wait for upload completion
      cy.wait('@uploadForAnalysis')

      // Verify analysis is triggered
      cy.wait('@autoAnalysis')

      // Check analysis results display
      cy.get('[data-testid="analysis-in-progress"]').should('be.visible')
      cy.contains('Analyzing document...').should('be.visible')

      // Wait for results
      cy.get('[data-testid="arbitration-result"]', { timeout: 10000 }).should('be.visible')
      cy.checkArbitrationResult({
        hasArbitration: true,
        confidence: 0.92,
        clauseType: 'mandatory_binding'
      })
    })

    it('handles analysis that finds no arbitration clause', () => {
      cy.intercept('POST', '/api/v1/documents/upload', {
        statusCode: 200,
        body: {
          document_id: 'doc-no-arb',
          filename: 'simple-contract.pdf',
          status: 'uploaded'
        }
      }).as('uploadNoArb')

      cy.intercept('POST', '/api/v1/arbitration/detect-by-id', {
        statusCode: 200,
        body: {
          document_id: 'doc-no-arb',
          has_arbitration: false,
          confidence: 0.15,
          clause_type: null,
          explanation: 'No arbitration clause was found in this document.',
          keywords: [],
          processing_time: 0.8
        }
      }).as('noArbAnalysis')

      cy.uploadFile('documents/no-arbitration-contract.pdf')
      cy.wait('@uploadNoArb')
      cy.wait('@noArbAnalysis')

      cy.checkArbitrationResult({
        hasArbitration: false,
        confidence: 0.15
      })

      cy.contains('No arbitration clause found').should('be.visible')
    })
  })

  describe('Multiple File Upload', () => {
    it('handles multiple files uploaded simultaneously', () => {
      // Mock multiple file upload
      cy.intercept('POST', '/api/v1/documents/upload-multiple', {
        statusCode: 200,
        body: {
          uploaded_documents: [
            { document_id: 'doc-multi-1', filename: 'contract1.pdf', status: 'uploaded' },
            { document_id: 'doc-multi-2', filename: 'contract2.pdf', status: 'uploaded' }
          ]
        }
      }).as('multiUpload')

      // Enable multiple file selection
      cy.get('[data-testid="multiple-files-toggle"]').click()

      // Upload multiple files
      cy.fixture('documents/contract1.pdf', 'base64').then(file1 => {
        cy.fixture('documents/contract2.pdf', 'base64').then(file2 => {
          const blob1 = Cypress.Blob.base64StringToBlob(file1, 'application/pdf')
          const blob2 = Cypress.Blob.base64StringToBlob(file2, 'application/pdf')
          
          const files = [
            new File([blob1], 'contract1.pdf', { type: 'application/pdf' }),
            new File([blob2], 'contract2.pdf', { type: 'application/pdf' })
          ]

          cy.get('[data-testid="file-input"]').then(input => {
            const dataTransfer = new DataTransfer()
            files.forEach(file => dataTransfer.items.add(file))
            input[0].files = dataTransfer.files
            
            cy.wrap(input).trigger('change', { force: true })
          })
        })
      })

      cy.wait('@multiUpload')
      cy.checkToast('2 files uploaded successfully')
    })
  })

  describe('Error Handling', () => {
    it('handles invalid file types', () => {
      // Try to upload invalid file type
      cy.fixture('documents/invalid-file.txt', 'base64').then(fileContent => {
        const blob = Cypress.Blob.base64StringToBlob(fileContent, 'text/plain')
        const file = new File([blob], 'invalid-file.txt', { type: 'text/plain' })
        
        cy.get('[data-testid="file-input"]').then(input => {
          const dataTransfer = new DataTransfer()
          dataTransfer.items.add(file)
          input[0].files = dataTransfer.files
          
          cy.wrap(input).trigger('change', { force: true })
        })
      })

      // Check error message
      cy.checkToast('Invalid file type. Please upload PDF, DOCX, or DOC files.', 'error')
      cy.get('[data-testid="file-type-error"]').should('be.visible')
    })

    it('handles files that exceed size limit', () => {
      // Mock file too large error
      cy.intercept('POST', '/api/v1/documents/upload', {
        statusCode: 413,
        body: {
          error: 'File too large. Maximum size is 20 MB.'
        }
      }).as('fileTooLarge')

      cy.uploadFile('documents/oversized-file.pdf')

      cy.wait('@fileTooLarge')
      cy.checkToast('File too large. Maximum size is 20 MB.', 'error')
    })

    it('handles upload server errors', () => {
      cy.intercept('POST', '/api/v1/documents/upload', {
        statusCode: 500,
        body: { error: 'Internal server error' }
      }).as('serverError')

      cy.uploadFile('documents/sample-contract.pdf')

      cy.wait('@serverError')
      cy.checkToast('Upload failed. Please try again.', 'error')
      cy.get('[data-testid="retry-upload-button"]').should('be.visible')
    })

    it('handles analysis failures', () => {
      // Mock successful upload
      cy.intercept('POST', '/api/v1/documents/upload', {
        statusCode: 200,
        body: {
          document_id: 'doc-fail',
          filename: 'contract.pdf',
          status: 'uploaded'
        }
      }).as('uploadSuccess')

      // Mock analysis failure
      cy.intercept('POST', '/api/v1/arbitration/detect-by-id', {
        statusCode: 500,
        body: { error: 'Analysis failed' }
      }).as('analysisFailure')

      cy.uploadFile('documents/sample-contract.pdf')
      cy.wait('@uploadSuccess')
      cy.wait('@analysisFailure')

      cy.checkToast('Analysis failed. You can retry from the documents list.', 'error')
      cy.get('[data-testid="analysis-error"]').should('be.visible')
    })

    it('allows retry after upload failure', () => {
      // First attempt fails
      cy.intercept('POST', '/api/v1/documents/upload', {
        statusCode: 500,
        body: { error: 'Upload failed' }
      }).as('firstAttempt')

      cy.uploadFile('documents/sample-contract.pdf')
      cy.wait('@firstAttempt')

      // Second attempt succeeds
      cy.intercept('POST', '/api/v1/documents/upload', {
        statusCode: 200,
        body: {
          document_id: 'doc-retry',
          filename: 'contract.pdf',
          status: 'uploaded'
        }
      }).as('retrySuccess')

      // Click retry button
      cy.get('[data-testid="retry-upload-button"]').click()
      cy.wait('@retrySuccess')
      cy.checkToast('File uploaded successfully')
    })
  })

  describe('File Preview', () => {
    it('shows file preview after successful upload', () => {
      cy.intercept('POST', '/api/v1/documents/upload', {
        statusCode: 200,
        body: {
          document_id: 'doc-preview',
          filename: 'contract-with-preview.pdf',
          status: 'uploaded',
          preview_url: '/api/v1/documents/doc-preview/preview'
        }
      }).as('uploadWithPreview')

      cy.uploadFile('documents/sample-contract.pdf', 'contract-with-preview.pdf')
      cy.wait('@uploadWithPreview')

      cy.get('[data-testid="file-preview"]').should('be.visible')
      cy.get('[data-testid="preview-filename"]').should('contain', 'contract-with-preview.pdf')
      cy.get('[data-testid="pdf-icon"]').should('be.visible')
    })

    it('allows removing uploaded file from preview', () => {
      cy.intercept('POST', '/api/v1/documents/upload', {
        statusCode: 200,
        body: {
          document_id: 'doc-remove',
          filename: 'removable-contract.pdf',
          status: 'uploaded'
        }
      }).as('uploadRemovable')

      cy.uploadFile('documents/sample-contract.pdf', 'removable-contract.pdf')
      cy.wait('@uploadRemovable')

      cy.get('[data-testid="file-preview"]').should('be.visible')
      
      // Remove file
      cy.get('[data-testid="remove-file-button"]').click()
      cy.get('[data-testid="file-preview"]').should('not.exist')
      
      // Upload area should be available again
      cy.get('[data-testid="upload-area"]').should('be.visible')
    })
  })

  describe('Accessibility', () => {
    it('supports keyboard navigation', () => {
      // Tab to upload area
      cy.get('body').tab()
      cy.get('[data-testid="upload-area"]').should('have.focus')

      // Activate with Enter
      cy.get('[data-testid="upload-area"]').type('{enter}')
      cy.get('[data-testid="file-input"]').should('have.focus')
    })

    it('provides proper ARIA labels', () => {
      cy.get('[data-testid="upload-area"]')
        .should('have.attr', 'role', 'button')
        .should('have.attr', 'aria-label')
        .should('have.attr', 'tabindex', '0')

      cy.get('[data-testid="file-input"]')
        .should('have.attr', 'aria-describedby')
    })

    it('announces upload status to screen readers', () => {
      cy.intercept('POST', '/api/v1/documents/upload', {
        statusCode: 200,
        body: {
          document_id: 'doc-a11y',
          filename: 'accessible-contract.pdf',
          status: 'uploaded'
        }
      }).as('a11yUpload')

      cy.uploadFile('documents/sample-contract.pdf', 'accessible-contract.pdf')
      cy.wait('@a11yUpload')

      cy.get('[data-testid="upload-status"]')
        .should('have.attr', 'role', 'status')
        .should('have.attr', 'aria-live', 'polite')
        .should('contain', 'Upload completed successfully')
    })
  })
})

describe('Upload Edge Cases', () => {
  beforeEach(() => {
    cy.seedTestData('empty')
    cy.visitDashboard()
  })

  it('handles network timeouts during upload', () => {
    cy.intercept('POST', '/api/v1/documents/upload', (req) => {
      req.reply({
        forceNetworkError: true
      })
    }).as('networkError')

    cy.uploadFile('documents/sample-contract.pdf')
    cy.wait('@networkError')

    cy.checkToast('Network error. Please check your connection and try again.', 'error')
  })

  it('handles corrupted files gracefully', () => {
    cy.intercept('POST', '/api/v1/documents/upload', {
      statusCode: 400,
      body: {
        error: 'File appears to be corrupted or unreadable'
      }
    }).as('corruptedFile')

    cy.uploadFile('documents/corrupted-file.pdf')
    cy.wait('@corruptedFile')

    cy.checkToast('File appears to be corrupted. Please try uploading a different file.', 'error')
  })

  it('prevents duplicate file uploads', () => {
    // First upload
    cy.intercept('POST', '/api/v1/documents/upload', {
      statusCode: 200,
      body: {
        document_id: 'doc-dup-1',
        filename: 'duplicate-contract.pdf',
        status: 'uploaded'
      }
    }).as('firstUpload')

    cy.uploadFile('documents/sample-contract.pdf', 'duplicate-contract.pdf')
    cy.wait('@firstUpload')

    // Second upload of same file
    cy.intercept('POST', '/api/v1/documents/upload', {
      statusCode: 409,
      body: {
        error: 'A file with this name already exists'
      }
    }).as('duplicateUpload')

    cy.uploadFile('documents/sample-contract.pdf', 'duplicate-contract.pdf')
    cy.wait('@duplicateUpload')

    cy.checkToast('A file with this name already exists. Please rename the file or upload a different one.', 'warning')
  })
})