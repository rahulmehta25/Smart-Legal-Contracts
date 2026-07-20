/**
 * Unit tests for ArbitrationResult component.
 * 
 * Tests arbitration detection result display including:
 * - Result rendering for different detection outcomes
 * - Confidence score visualization
 * - Explanation text formatting
 * - Keyword highlighting
 * - Action buttons and interactions
 * - Loading and error states
 */

import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import ArbitrationResult from '@/components/ArbitrationResult'
import { ArbitrationDetectionResult } from '@/types/arbitration'

describe('ArbitrationResult Component', () => {
  const baseResult: ArbitrationDetectionResult = {
    has_arbitration: true,
    confidence: 0.85,
    clause_type: 'mandatory_binding',
    explanation: 'The document contains a clear arbitration clause requiring binding arbitration.',
    keywords: ['arbitration', 'binding', 'mandatory'],
    processing_time: 1.2,
    document_id: 'doc-123'
  }

  const defaultProps = {
    result: baseResult,
    onExportReport: jest.fn(),
    onViewDetails: jest.fn(),
    loading: false,
    error: null
  }

  beforeEach(() => {
    jest.clearAllMocks()
  })

  describe('Positive Detection Results', () => {
    it('renders positive arbitration detection result', () => {
      render(<ArbitrationResult {...defaultProps} />)
      
      expect(screen.getByTestId('arbitration-result')).toBeInTheDocument()
      expect(screen.getByText(/arbitration clause detected/i)).toBeInTheDocument()
      expect(screen.getByTestId('positive-result-icon')).toBeInTheDocument()
      expect(screen.getByText('85%')).toBeInTheDocument() // Confidence score
    })

    it('displays confidence score with appropriate styling', () => {
      render(<ArbitrationResult {...defaultProps} />)
      
      const confidenceBar = screen.getByTestId('confidence-bar')
      expect(confidenceBar).toBeInTheDocument()
      expect(confidenceBar).toHaveStyle('width: 85%')
      expect(confidenceBar).toHaveClass('high-confidence')
    })

    it('shows clause type information', () => {
      render(<ArbitrationResult {...defaultProps} />)
      
      expect(screen.getByText(/mandatory binding/i)).toBeInTheDocument()
      expect(screen.getByTestId('clause-type-badge')).toHaveClass('mandatory')
    })

    it('displays explanation text', () => {
      render(<ArbitrationResult {...defaultProps} />)
      
      expect(screen.getByText(baseResult.explanation)).toBeInTheDocument()
    })

    it('highlights keywords in explanation', () => {
      render(<ArbitrationResult {...defaultProps} />)
      
      const keywords = screen.getAllByTestId('highlighted-keyword')
      expect(keywords).toHaveLength(3)
      expect(keywords[0]).toHaveTextContent('arbitration')
      expect(keywords[1]).toHaveTextContent('binding')
      expect(keywords[2]).toHaveTextContent('mandatory')
    })

    it('shows processing time', () => {
      render(<ArbitrationResult {...defaultProps} />)
      
      expect(screen.getByText(/processed in 1.2 seconds/i)).toBeInTheDocument()
    })
  })

  describe('Negative Detection Results', () => {
    const negativeResult: ArbitrationDetectionResult = {
      has_arbitration: false,
      confidence: 0.15,
      clause_type: null,
      explanation: 'No arbitration clause was found in the document.',
      keywords: [],
      processing_time: 0.8,
      document_id: 'doc-456'
    }

    it('renders negative arbitration detection result', () => {
      render(<ArbitrationResult {...defaultProps} result={negativeResult} />)
      
      expect(screen.getByText(/no arbitration clause detected/i)).toBeInTheDocument()
      expect(screen.getByTestId('negative-result-icon')).toBeInTheDocument()
      expect(screen.getByText('15%')).toBeInTheDocument()
    })

    it('shows low confidence styling', () => {
      render(<ArbitrationResult {...defaultProps} result={negativeResult} />)
      
      const confidenceBar = screen.getByTestId('confidence-bar')
      expect(confidenceBar).toHaveStyle('width: 15%')
      expect(confidenceBar).toHaveClass('low-confidence')
    })

    it('does not show clause type for negative results', () => {
      render(<ArbitrationResult {...defaultProps} result={negativeResult} />)
      
      expect(screen.queryByTestId('clause-type-badge')).not.toBeInTheDocument()
    })

    it('shows empty keywords message', () => {
      render(<ArbitrationResult {...defaultProps} result={negativeResult} />)
      
      expect(screen.getByText(/no arbitration-related keywords found/i)).toBeInTheDocument()
    })
  })

  describe('Confidence Score Visualization', () => {
    const confidenceTestCases = [
      { confidence: 0.95, level: 'very-high', color: 'green' },
      { confidence: 0.75, level: 'high', color: 'green' },
      { confidence: 0.55, level: 'medium', color: 'yellow' },
      { confidence: 0.35, level: 'low', color: 'orange' },
      { confidence: 0.15, level: 'very-low', color: 'red' }
    ]

    confidenceTestCases.forEach(({ confidence, level, color }) => {
      it(`displays ${level} confidence (${confidence * 100}%) with ${color} styling`, () => {
        const result = { ...baseResult, confidence }
        render(<ArbitrationResult {...defaultProps} result={result} />)
        
        const confidenceBar = screen.getByTestId('confidence-bar')
        expect(confidenceBar).toHaveClass(level)
        expect(confidenceBar).toHaveClass(`bg-${color}`)
        expect(screen.getByText(`${Math.round(confidence * 100)}%`)).toBeInTheDocument()
      })
    })
  })

  describe('Keywords Display', () => {
    it('displays all detected keywords', () => {
      render(<ArbitrationResult {...defaultProps} />)
      
      const keywordTags = screen.getAllByTestId('keyword-tag')
      expect(keywordTags).toHaveLength(3)
      
      expect(keywordTags[0]).toHaveTextContent('arbitration')
      expect(keywordTags[1]).toHaveTextContent('binding')
      expect(keywordTags[2]).toHaveTextContent('mandatory')
    })

    it('handles many keywords with truncation', () => {
      const manyKeywords = Array.from({ length: 15 }, (_, i) => `keyword${i}`)
      const result = { ...baseResult, keywords: manyKeywords }
      
      render(<ArbitrationResult {...defaultProps} result={result} />)
      
      const keywordTags = screen.getAllByTestId('keyword-tag')
      expect(keywordTags).toHaveLength(10) // Should truncate to first 10
      
      expect(screen.getByText(/\+5 more keywords/i)).toBeInTheDocument()
    })

    it('allows expanding truncated keywords', async () => {
      const user = userEvent.setup()
      const manyKeywords = Array.from({ length: 15 }, (_, i) => `keyword${i}`)
      const result = { ...baseResult, keywords: manyKeywords }
      
      render(<ArbitrationResult {...defaultProps} result={result} />)
      
      const showMoreButton = screen.getByTestId('show-more-keywords')
      await user.click(showMoreButton)
      
      const keywordTags = screen.getAllByTestId('keyword-tag')
      expect(keywordTags).toHaveLength(15) // Should show all keywords
    })
  })

  describe('Action Buttons', () => {
    it('renders export report button', () => {
      render(<ArbitrationResult {...defaultProps} />)
      
      const exportButton = screen.getByTestId('export-report-button')
      expect(exportButton).toBeInTheDocument()
      expect(exportButton).toHaveTextContent(/export report/i)
    })

    it('calls onExportReport when export button is clicked', async () => {
      const user = userEvent.setup()
      
      render(<ArbitrationResult {...defaultProps} />)
      
      const exportButton = screen.getByTestId('export-report-button')
      await user.click(exportButton)
      
      expect(defaultProps.onExportReport).toHaveBeenCalledWith(baseResult)
    })

    it('renders view details button', () => {
      render(<ArbitrationResult {...defaultProps} />)
      
      const detailsButton = screen.getByTestId('view-details-button')
      expect(detailsButton).toBeInTheDocument()
      expect(detailsButton).toHaveTextContent(/view details/i)
    })

    it('calls onViewDetails when details button is clicked', async () => {
      const user = userEvent.setup()
      
      render(<ArbitrationResult {...defaultProps} />)
      
      const detailsButton = screen.getByTestId('view-details-button')
      await user.click(detailsButton)
      
      expect(defaultProps.onViewDetails).toHaveBeenCalledWith(baseResult)
    })

    it('disables buttons when loading', () => {
      render(<ArbitrationResult {...defaultProps} loading={true} />)
      
      const exportButton = screen.getByTestId('export-report-button')
      const detailsButton = screen.getByTestId('view-details-button')
      
      expect(exportButton).toBeDisabled()
      expect(detailsButton).toBeDisabled()
    })
  })

  describe('Loading State', () => {
    it('shows loading spinner when loading', () => {
      render(<ArbitrationResult {...defaultProps} loading={true} result={null} />)
      
      expect(screen.getByTestId('loading-spinner')).toBeInTheDocument()
      expect(screen.getByText(/analyzing document/i)).toBeInTheDocument()
    })

    it('shows processing message during loading', () => {
      render(<ArbitrationResult {...defaultProps} loading={true} result={null} />)
      
      expect(screen.getByText(/detecting arbitration clauses/i)).toBeInTheDocument()
    })

    it('hides content when loading', () => {
      render(<ArbitrationResult {...defaultProps} loading={true} result={null} />)
      
      expect(screen.queryByTestId('arbitration-result')).not.toBeInTheDocument()
    })
  })

  describe('Error State', () => {
    const errorMessage = 'Failed to analyze document'

    it('shows error message when error occurs', () => {
      render(<ArbitrationResult {...defaultProps} error={errorMessage} result={null} />)
      
      expect(screen.getByTestId('error-message')).toBeInTheDocument()
      expect(screen.getByText(errorMessage)).toBeInTheDocument()
      expect(screen.getByTestId('error-icon')).toBeInTheDocument()
    })

    it('shows retry button on error', () => {
      render(<ArbitrationResult {...defaultProps} error={errorMessage} result={null} onRetry={jest.fn()} />)
      
      expect(screen.getByTestId('retry-button')).toBeInTheDocument()
    })

    it('calls onRetry when retry button is clicked', async () => {
      const user = userEvent.setup()
      const onRetry = jest.fn()
      
      render(<ArbitrationResult {...defaultProps} error={errorMessage} result={null} onRetry={onRetry} />)
      
      const retryButton = screen.getByTestId('retry-button')
      await user.click(retryButton)
      
      expect(onRetry).toHaveBeenCalled()
    })

    it('hides content when error occurs', () => {
      render(<ArbitrationResult {...defaultProps} error={errorMessage} result={null} />)
      
      expect(screen.queryByTestId('arbitration-result')).not.toBeInTheDocument()
    })
  })

  describe('Complex Arbitration Results', () => {
    const complexResult: ArbitrationDetectionResult = {
      has_arbitration: true,
      confidence: 0.92,
      clause_type: 'multi_tier_mandatory',
      explanation: 'The document contains a comprehensive multi-tier dispute resolution process including mediation followed by binding arbitration.',
      keywords: ['mediation', 'arbitration', 'binding', 'ICC', 'three arbitrators'],
      processing_time: 2.1,
      document_id: 'doc-complex',
      arbitration_details: {
        arbitration_body: 'ICC International Court of Arbitration',
        location: 'Geneva, Switzerland',
        number_of_arbitrators: 3,
        language: 'English',
        expedited: false,
        multi_tier: true
      }
    }

    it('displays complex arbitration details', () => {
      render(<ArbitrationResult {...defaultProps} result={complexResult} />)
      
      expect(screen.getByText(/multi-tier mandatory/i)).toBeInTheDocument()
      expect(screen.getByText(/ICC International Court/i)).toBeInTheDocument()
      expect(screen.getByText(/Geneva, Switzerland/i)).toBeInTheDocument()
      expect(screen.getByText(/3 arbitrators/i)).toBeInTheDocument()
    })

    it('shows multi-tier process indicator', () => {
      render(<ArbitrationResult {...defaultProps} result={complexResult} />)
      
      expect(screen.getByTestId('multi-tier-indicator')).toBeInTheDocument()
      expect(screen.getByText(/mediation → arbitration/i)).toBeInTheDocument()
    })
  })

  describe('Accessibility', () => {
    it('has proper ARIA labels and roles', () => {
      render(<ArbitrationResult {...defaultProps} />)
      
      const result = screen.getByTestId('arbitration-result')
      expect(result).toHaveAttribute('role', 'region')
      expect(result).toHaveAttribute('aria-label', 'Arbitration analysis result')
      
      const confidenceBar = screen.getByTestId('confidence-bar')
      expect(confidenceBar).toHaveAttribute('role', 'progressbar')
      expect(confidenceBar).toHaveAttribute('aria-valuenow', '85')
    })

    it('provides screen reader friendly content', () => {
      render(<ArbitrationResult {...defaultProps} />)
      
      expect(screen.getByLabelText(/confidence score: 85 percent/i)).toBeInTheDocument()
      expect(screen.getByLabelText(/clause type: mandatory binding/i)).toBeInTheDocument()
    })

    it('supports keyboard navigation', async () => {
      const user = userEvent.setup()
      
      render(<ArbitrationResult {...defaultProps} />)
      
      // Tab through interactive elements
      await user.tab()
      expect(screen.getByTestId('export-report-button')).toHaveFocus()
      
      await user.tab()
      expect(screen.getByTestId('view-details-button')).toHaveFocus()
    })

    it('announces result changes to screen readers', () => {
      const { rerender } = render(<ArbitrationResult {...defaultProps} loading={true} result={null} />)
      
      rerender(<ArbitrationResult {...defaultProps} loading={false} />)
      
      const liveRegion = screen.getByTestId('result-announcement')
      expect(liveRegion).toHaveAttribute('aria-live', 'polite')
      expect(liveRegion).toHaveTextContent(/arbitration clause detected/i)
    })
  })

  describe('Mobile Responsiveness', () => {
    it('adapts layout for small screens', () => {
      // Mock small screen
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 375,
      })
      
      render(<ArbitrationResult {...defaultProps} />)
      
      const result = screen.getByTestId('arbitration-result')
      expect(result).toHaveClass('mobile-layout')
    })

    it('stacks action buttons vertically on mobile', () => {
      Object.defineProperty(window, 'innerWidth', {
        writable: true,
        configurable: true,
        value: 375,
      })
      
      render(<ArbitrationResult {...defaultProps} />)
      
      const buttonContainer = screen.getByTestId('action-buttons')
      expect(buttonContainer).toHaveClass('flex-col')
    })
  })

  describe('Data Export', () => {
    it('includes all relevant data in export', async () => {
      const user = userEvent.setup()
      
      render(<ArbitrationResult {...defaultProps} />)
      
      const exportButton = screen.getByTestId('export-report-button')
      await user.click(exportButton)
      
      expect(defaultProps.onExportReport).toHaveBeenCalledWith(
        expect.objectContaining({
          has_arbitration: true,
          confidence: 0.85,
          clause_type: 'mandatory_binding',
          explanation: expect.any(String),
          keywords: expect.any(Array),
          processing_time: 1.2,
          document_id: 'doc-123'
        })
      )
    })
  })
})