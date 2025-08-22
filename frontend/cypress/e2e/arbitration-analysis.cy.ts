/**
 * E2E tests for arbitration clause analysis functionality.
 * 
 * Tests the arbitration detection workflow including:
 * - Analysis triggering and processing
 * - Result display and interpretation
 * - Different types of arbitration clauses
 * - Analysis accuracy and confidence scoring
 */

describe('Arbitration Analysis Workflow', () => {
  beforeEach(() => {
    cy.seedTestData('with-documents')
    cy.visitDashboard()
  })

  describe('Analysis Triggering', () => {
    it('analyzes document from documents list', () => {
      // Mock analysis request
      cy.intercept('POST', '/api/v1/arbitration/detect-by-id', {
        statusCode: 200,
        body: {
          document_id: 'doc-pending-1',
          has_arbitration: true,
          confidence: 0.87,
          clause_type: 'mandatory_binding',
          explanation: 'The document contains a binding arbitration clause in Section 15.',
          keywords: ['binding arbitration', 'Section 15', 'disputes'],
          processing_time: 2.1,
          arbitration_details: {
            arbitration_body: 'American Arbitration Association',
            location: 'New York',
            number_of_arbitrators: 1
          }
        }
      }).as('analyzeDocument')

      // Find and click analyze button for pending document
      cy.get('[data-testid="document-doc-pending-1"]').within(() => {
        cy.get('[data-testid="analyze-button"]').click()
      })

      // Verify analysis request
      cy.wait('@analyzeDocument')

      // Check loading state
      cy.get('[data-testid="analysis-loading"]').should('be.visible')
      cy.contains('Analyzing document...').should('be.visible')

      // Wait for results
      cy.get('[data-testid="analysis-result"]', { timeout: 10000 }).should('be.visible')
    })

    it('shows analysis progress with estimated time', () => {
      cy.intercept('POST', '/api/v1/arbitration/detect-by-id', (req) => {
        req.reply({
          delay: 3000,
          statusCode: 200,
          body: {
            document_id: 'doc-slow',
            has_arbitration: true,
            confidence: 0.92
          }
        })
      }).as('slowAnalysis')

      cy.get('[data-testid="analyze-button-doc-slow"]').click()

      // Check progress indicators
      cy.get('[data-testid="analysis-progress"]').should('be.visible')
      cy.get('[data-testid="progress-spinner"]').should('be.visible')
      cy.contains('Estimated time: 2-5 seconds').should('be.visible')

      cy.wait('@slowAnalysis')
    })

    it('supports bulk analysis of multiple documents', () => {
      // Mock bulk analysis
      cy.intercept('POST', '/api/v1/arbitration/detect-batch', {
        statusCode: 200,
        body: {
          results: [
            {
              document_id: 'doc-bulk-1',
              has_arbitration: true,
              confidence: 0.89,
              clause_type: 'mandatory_binding'
            },
            {
              document_id: 'doc-bulk-2',
              has_arbitration: false,
              confidence: 0.12,
              clause_type: null
            }
          ]
        }
      }).as('bulkAnalysis')

      // Select multiple documents
      cy.get('[data-testid="select-document-doc-bulk-1"]').check()
      cy.get('[data-testid="select-document-doc-bulk-2"]').check()

      // Trigger bulk analysis
      cy.get('[data-testid="bulk-analyze-button"]').click()
      cy.get('[data-testid="confirm-bulk-analysis"]').click()

      cy.wait('@bulkAnalysis')
      cy.checkToast('Bulk analysis completed for 2 documents')
    })
  })

  describe('Analysis Results Display', () => {
    it('displays positive arbitration detection results', () => {
      cy.intercept('POST', '/api/v1/arbitration/detect-by-id', {
        fixture: 'analyses/positive-arbitration-result.json'
      }).as('positiveResult')

      cy.get('[data-testid="analyze-button-doc-1"]').click()
      cy.wait('@positiveResult')

      // Check main result display
      cy.get('[data-testid="arbitration-result"]').should('be.visible')
      cy.get('[data-testid="positive-result-icon"]').should('be.visible')
      cy.contains('Arbitration Clause Detected').should('be.visible')

      // Check confidence score
      cy.get('[data-testid="confidence-score"]').should('contain', '94%')
      cy.get('[data-testid="confidence-bar"]').should('have.class', 'very-high-confidence')

      // Check clause type
      cy.get('[data-testid="clause-type-badge"]')
        .should('contain', 'Mandatory Binding')
        .should('have.class', 'mandatory')

      // Check explanation
      cy.contains('The document contains a comprehensive arbitration clause').should('be.visible')

      // Check keywords
      cy.get('[data-testid="keyword-tag"]').should('have.length.at.least', 3)
      cy.contains('[data-testid="keyword-tag"]', 'arbitration').should('be.visible')
      cy.contains('[data-testid="keyword-tag"]', 'binding').should('be.visible')
    })

    it('displays negative arbitration detection results', () => {
      cy.intercept('POST', '/api/v1/arbitration/detect-by-id', {
        fixture: 'analyses/negative-arbitration-result.json'
      }).as('negativeResult')

      cy.get('[data-testid="analyze-button-doc-no-arb"]').click()
      cy.wait('@negativeResult')

      // Check negative result display
      cy.get('[data-testid="negative-result-icon"]').should('be.visible')
      cy.contains('No Arbitration Clause Detected').should('be.visible')

      // Check low confidence score
      cy.get('[data-testid="confidence-score"]').should('contain', '8%')
      cy.get('[data-testid="confidence-bar"]').should('have.class', 'very-low-confidence')

      // Check absence of clause type
      cy.get('[data-testid="clause-type-badge"]').should('not.exist')

      // Check explanation
      cy.contains('No arbitration clause was found').should('be.visible')
    })

    it('displays complex arbitration clause details', () => {
      cy.intercept('POST', '/api/v1/arbitration/detect-by-id', {
        fixture: 'analyses/complex-arbitration-result.json'
      }).as('complexResult')

      cy.get('[data-testid="analyze-button-doc-complex"]').click()
      cy.wait('@complexResult')

      // Check complex clause details
      cy.get('[data-testid="arbitration-details"]').should('be.visible')
      cy.contains('ICC International Court of Arbitration').should('be.visible')
      cy.contains('Geneva, Switzerland').should('be.visible')
      cy.contains('3 arbitrators').should('be.visible')
      cy.contains('English language').should('be.visible')

      // Check multi-tier process
      cy.get('[data-testid="multi-tier-indicator"]').should('be.visible')
      cy.contains('Mediation → Arbitration').should('be.visible')

      // Check procedural details
      cy.get('[data-testid="procedure-details"]').within(() => {
        cy.contains('Step 1: Direct negotiation').should('be.visible')
        cy.contains('Step 2: Mediation').should('be.visible')
        cy.contains('Step 3: Binding arbitration').should('be.visible')
      })
    })

    it('handles ambiguous results with moderate confidence', () => {
      cy.intercept('POST', '/api/v1/arbitration/detect-by-id', {
        statusCode: 200,
        body: {
          document_id: 'doc-ambiguous',
          has_arbitration: false,
          confidence: 0.45,
          clause_type: 'optional',
          explanation: 'The document mentions arbitration as one of several dispute resolution options.',
          keywords: ['arbitration', 'optional', 'dispute resolution'],
          ambiguity_flag: true
        }
      }).as('ambiguousResult')

      cy.get('[data-testid="analyze-button-doc-ambiguous"]').click()
      cy.wait('@ambiguousResult')

      // Check ambiguity indicators
      cy.get('[data-testid="ambiguity-warning"]').should('be.visible')
      cy.contains('Ambiguous Language Detected').should('be.visible')
      cy.get('[data-testid="confidence-bar"]').should('have.class', 'medium-confidence')

      // Check recommendation
      cy.contains('Consider legal review for clarification').should('be.visible')
    })
  })

  describe('Analysis Actions', () => {
    beforeEach(() => {
      // Set up successful analysis result
      cy.intercept('POST', '/api/v1/arbitration/detect-by-id', {
        fixture: 'analyses/positive-arbitration-result.json'
      }).as('analysisResult')

      cy.get('[data-testid="analyze-button-doc-1"]').click()
      cy.wait('@analysisResult')
    })

    it('exports analysis report', () => {
      cy.intercept('GET', '/api/v1/documents/doc-1/report', {
        statusCode: 200,
        body: 'PDF report content'
      }).as('exportReport')

      cy.get('[data-testid="export-report-button"]').click()
      cy.wait('@exportReport')

      // Check download was triggered
      cy.checkToast('Report exported successfully')
    })

    it('shares analysis results', () => {
      cy.get('[data-testid="share-results-button"]').click()

      // Check share modal
      cy.get('[data-testid="share-modal"]').should('be.visible')
      cy.get('[data-testid="share-link"]').should('contain', 'https://')
      
      // Copy link
      cy.get('[data-testid="copy-link-button"]').click()
      cy.checkToast('Link copied to clipboard')
    })

    it('saves analysis to favorites', () => {
      cy.intercept('POST', '/api/v1/analyses/doc-1/favorite', {
        statusCode: 200,
        body: { success: true }
      }).as('saveFavorite')

      cy.get('[data-testid="save-favorite-button"]').click()
      cy.wait('@saveFavorite')

      cy.checkToast('Analysis saved to favorites')
      cy.get('[data-testid="favorite-icon"]').should('have.class', 'favorited')
    })

    it('opens detailed analysis view', () => {
      cy.get('[data-testid="view-details-button"]').click()

      // Check detailed modal
      cy.get('[data-testid="analysis-details-modal"]').should('be.visible')
      cy.get('[data-testid="detailed-explanation"]').should('be.visible')
      cy.get('[data-testid="source-text-highlights"]').should('be.visible')
      cy.get('[data-testid="confidence-breakdown"]').should('be.visible')
    })
  })

  describe('RAG-Enhanced Analysis', () => {
    it('uses RAG context for improved analysis', () => {
      cy.intercept('POST', '/api/v1/arbitration/detect-rag', {
        statusCode: 200,
        body: {
          document_id: 'doc-rag',
          has_arbitration: true,
          confidence: 0.96,
          clause_type: 'mandatory_binding',
          explanation: 'Analysis enhanced with similar clause patterns from legal database.',
          keywords: ['arbitration', 'binding', 'JAMS'],
          rag_context: [
            'Similar arbitration clause from Contract XYZ',
            'Related precedent from Case ABC'
          ],
          context_sources: ['contract_db_001.pdf', 'case_law_456.pdf'],
          rag_confidence_boost: 0.12
        }
      }).as('ragAnalysis')

      // Enable RAG analysis
      cy.get('[data-testid="rag-analysis-toggle"]').click()
      cy.get('[data-testid="analyze-button-doc-rag"]').click()

      cy.wait('@ragAnalysis')

      // Check RAG indicators
      cy.get('[data-testid="rag-enhanced-badge"]').should('be.visible')
      cy.contains('Enhanced with Legal Context').should('be.visible')

      // Check context sources
      cy.get('[data-testid="view-context-button"]').click()
      cy.get('[data-testid="context-modal"]').should('be.visible')
      cy.contains('contract_db_001.pdf').should('be.visible')
      cy.contains('case_law_456.pdf').should('be.visible')

      // Check confidence boost indicator
      cy.get('[data-testid="confidence-boost"]').should('contain', '+12%')
    })

    it('shows when RAG context is not available', () => {
      cy.intercept('POST', '/api/v1/arbitration/detect-rag', {
        statusCode: 200,
        body: {
          document_id: 'doc-no-context',
          has_arbitration: true,
          confidence: 0.78,
          rag_context: [],
          context_sources: [],
          rag_available: false
        }
      }).as('noContextAnalysis')

      cy.get('[data-testid="rag-analysis-toggle"]').click()
      cy.get('[data-testid="analyze-button-doc-no-context"]').click()

      cy.wait('@noContextAnalysis')

      cy.get('[data-testid="no-context-warning"]').should('be.visible')
      cy.contains('No similar clauses found in database').should('be.visible')
    })
  })

  describe('Error Handling', () => {
    it('handles analysis failures gracefully', () => {
      cy.intercept('POST', '/api/v1/arbitration/detect-by-id', {
        statusCode: 500,
        body: { error: 'Analysis service temporarily unavailable' }
      }).as('analysisError')

      cy.get('[data-testid="analyze-button-doc-error"]').click()
      cy.wait('@analysisError')

      cy.get('[data-testid="analysis-error"]').should('be.visible')
      cy.checkToast('Analysis failed. Please try again later.', 'error')
      cy.get('[data-testid="retry-analysis-button"]').should('be.visible')
    })

    it('handles timeout during analysis', () => {
      cy.intercept('POST', '/api/v1/arbitration/detect-by-id', (req) => {
        // Simulate timeout
        req.reply({
          delay: 35000,
          statusCode: 408,
          body: { error: 'Analysis timeout' }
        })
      }).as('analysisTimeout')

      cy.get('[data-testid="analyze-button-doc-timeout"]').click()
      cy.wait('@analysisTimeout')

      cy.contains('Analysis is taking longer than expected').should('be.visible')
      cy.get('[data-testid="cancel-analysis-button"]').should('be.visible')
    })

    it('allows retry after analysis failure', () => {
      // First attempt fails
      cy.intercept('POST', '/api/v1/arbitration/detect-by-id', {
        statusCode: 500,
        body: { error: 'Temporary failure' }
      }).as('firstFailure')

      cy.get('[data-testid="analyze-button-doc-retry"]').click()
      cy.wait('@firstFailure')

      // Second attempt succeeds
      cy.intercept('POST', '/api/v1/arbitration/detect-by-id', {
        fixture: 'analyses/positive-arbitration-result.json'
      }).as('retrySuccess')

      cy.get('[data-testid="retry-analysis-button"]').click()
      cy.wait('@retrySuccess')

      cy.get('[data-testid="arbitration-result"]').should('be.visible')
      cy.checkToast('Analysis completed successfully')
    })

    it('handles corrupted document errors', () => {
      cy.intercept('POST', '/api/v1/arbitration/detect-by-id', {
        statusCode: 400,
        body: { 
          error: 'Document format not supported or file is corrupted',
          error_code: 'CORRUPTED_DOCUMENT'
        }
      }).as('corruptedDoc')

      cy.get('[data-testid="analyze-button-doc-corrupted"]').click()
      cy.wait('@corruptedDoc')

      cy.get('[data-testid="corrupted-file-error"]').should('be.visible')
      cy.contains('The document appears to be corrupted').should('be.visible')
      cy.get('[data-testid="reupload-suggestion"]').should('be.visible')
    })
  })

  describe('Performance Monitoring', () => {
    it('displays processing time metrics', () => {
      cy.intercept('POST', '/api/v1/arbitration/detect-by-id', {
        statusCode: 200,
        body: {
          document_id: 'doc-metrics',
          has_arbitration: true,
          confidence: 0.89,
          processing_time: 3.24,
          performance_metrics: {
            text_extraction_time: 0.45,
            vector_search_time: 1.12,
            llm_processing_time: 1.67
          }
        }
      }).as('analysisWithMetrics')

      cy.get('[data-testid="analyze-button-doc-metrics"]').click()
      cy.wait('@analysisWithMetrics')

      // Check performance display
      cy.get('[data-testid="processing-time"]').should('contain', '3.24 seconds')
      
      // Check detailed metrics
      cy.get('[data-testid="show-metrics-button"]').click()
      cy.get('[data-testid="metrics-breakdown"]').should('be.visible')
      cy.contains('Text extraction: 0.45s').should('be.visible')
      cy.contains('Vector search: 1.12s').should('be.visible')
      cy.contains('LLM processing: 1.67s').should('be.visible')
    })

    it('warns about slow analysis performance', () => {
      cy.intercept('POST', '/api/v1/arbitration/detect-by-id', {
        statusCode: 200,
        body: {
          document_id: 'doc-slow-perf',
          has_arbitration: true,
          confidence: 0.85,
          processing_time: 12.5,
          performance_warning: 'Analysis took longer than usual due to document complexity'
        }
      }).as('slowAnalysis')

      cy.get('[data-testid="analyze-button-doc-slow-perf"]').click()
      cy.wait('@slowAnalysis')

      cy.get('[data-testid="performance-warning"]').should('be.visible')
      cy.contains('Analysis took longer than usual').should('be.visible')
    })
  })

  describe('Accessibility', () => {
    it('supports keyboard navigation through results', () => {
      cy.intercept('POST', '/api/v1/arbitration/detect-by-id', {
        fixture: 'analyses/positive-arbitration-result.json'
      }).as('keyboardResult')

      cy.get('[data-testid="analyze-button-doc-1"]').click()
      cy.wait('@keyboardResult')

      // Navigate through result actions
      cy.get('[data-testid="export-report-button"]').focus()
      cy.get('[data-testid="export-report-button"]').should('have.focus')

      cy.tab()
      cy.get('[data-testid="share-results-button"]').should('have.focus')

      cy.tab()
      cy.get('[data-testid="view-details-button"]').should('have.focus')
    })

    it('provides screen reader announcements', () => {
      cy.intercept('POST', '/api/v1/arbitration/detect-by-id', {
        fixture: 'analyses/positive-arbitration-result.json'
      }).as('screenReaderResult')

      cy.get('[data-testid="analyze-button-doc-1"]').click()
      cy.wait('@screenReaderResult')

      cy.get('[data-testid="result-announcement"]')
        .should('have.attr', 'aria-live', 'polite')
        .should('contain', 'Analysis complete: Arbitration clause detected with 94% confidence')
    })
  })
})