/**
 * Integration tests for Dashboard page component.
 * 
 * Tests the main dashboard functionality including:
 * - Page rendering and layout
 * - Document management operations
 * - Arbitration analysis workflow
 * - Data fetching and state management
 * - User interactions and navigation
 */

import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from 'react-query'
import Dashboard from '@/pages/dashboard'
import * as api from '@/lib/api'

// Mock API calls
jest.mock('@/lib/api')
const mockApi = api as jest.Mocked<typeof api>

// Mock Next.js router
jest.mock('next/router', () => ({
  useRouter: () => ({
    push: jest.fn(),
    query: {},
    pathname: '/dashboard',
  }),
}))

// Test wrapper with QueryClient
const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  })
  
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  )
}

describe('Dashboard Page', () => {
  beforeEach(() => {
    jest.clearAllMocks()
    
    // Default API mocks
    mockApi.getDocuments.mockResolvedValue({
      documents: [],
      total_count: 0,
      page: 1,
      total_pages: 1,
    })
    
    mockApi.getRecentAnalyses.mockResolvedValue([])
  })

  describe('Page Rendering', () => {
    it('renders dashboard with all main sections', async () => {
      render(<Dashboard />, { wrapper: createWrapper() })
      
      await waitFor(() => {
        expect(screen.getByTestId('dashboard-header')).toBeInTheDocument()
        expect(screen.getByTestId('upload-section')).toBeInTheDocument()
        expect(screen.getByTestId('documents-section')).toBeInTheDocument()
        expect(screen.getByTestId('recent-analyses-section')).toBeInTheDocument()
      })
    })

    it('displays welcome message and user stats', async () => {
      mockApi.getUserStats.mockResolvedValue({
        total_documents: 15,
        total_analyses: 42,
        arbitration_found: 28,
        accuracy_rate: 0.94,
      })

      render(<Dashboard />, { wrapper: createWrapper() })
      
      await waitFor(() => {
        expect(screen.getByText(/welcome back/i)).toBeInTheDocument()
        expect(screen.getByText('15')).toBeInTheDocument() // total documents
        expect(screen.getByText('42')).toBeInTheDocument() // total analyses
        expect(screen.getByText('94%')).toBeInTheDocument() // accuracy rate
      })
    })

    it('shows loading state initially', () => {
      render(<Dashboard />, { wrapper: createWrapper() })
      
      expect(screen.getByTestId('dashboard-loading')).toBeInTheDocument()
    })

    it('handles empty state when no documents', async () => {
      render(<Dashboard />, { wrapper: createWrapper() })
      
      await waitFor(() => {
        expect(screen.getByTestId('empty-documents-state')).toBeInTheDocument()
        expect(screen.getByText(/no documents uploaded yet/i)).toBeInTheDocument()
        expect(screen.getByText(/upload your first document/i)).toBeInTheDocument()
      })
    })
  })

  describe('Document Upload Workflow', () => {
    it('handles successful document upload', async () => {
      const user = userEvent.setup()
      
      mockApi.uploadFile.mockResolvedValue({
        document_id: 'doc-123',
        filename: 'contract.pdf',
        status: 'uploaded',
      })

      render(<Dashboard />, { wrapper: createWrapper() })
      
      await waitFor(() => {
        expect(screen.getByTestId('upload-section')).toBeInTheDocument()
      })

      // Upload file
      const fileInput = screen.getByTestId('file-input')
      const testFile = testUtils.createMockFile('contract.pdf', 'PDF content', 'application/pdf')

      await user.upload(fileInput, testFile)

      await waitFor(() => {
        expect(mockApi.uploadFile).toHaveBeenCalledWith(testFile)
        expect(screen.getByText(/upload successful/i)).toBeInTheDocument()
      })
    })

    it('triggers automatic analysis after upload', async () => {
      const user = userEvent.setup()
      
      mockApi.uploadFile.mockResolvedValue({
        document_id: 'doc-123',
        filename: 'contract.pdf',
        status: 'uploaded',
      })

      mockApi.analyzeDocument.mockResolvedValue({
        has_arbitration: true,
        confidence: 0.88,
        clause_type: 'mandatory_binding',
        explanation: 'Arbitration clause detected',
        keywords: ['arbitration', 'binding'],
        processing_time: 1.5,
      })

      render(<Dashboard />, { wrapper: createWrapper() })
      
      await waitFor(() => {
        expect(screen.getByTestId('upload-section')).toBeInTheDocument()
      })

      const fileInput = screen.getByTestId('file-input')
      const testFile = testUtils.createMockFile('contract.pdf', 'PDF content', 'application/pdf')

      await user.upload(fileInput, testFile)

      await waitFor(() => {
        expect(mockApi.analyzeDocument).toHaveBeenCalledWith('doc-123')
        expect(screen.getByText(/arbitration clause detected/i)).toBeInTheDocument()
      })
    })

    it('handles upload errors gracefully', async () => {
      const user = userEvent.setup()
      
      mockApi.uploadFile.mockRejectedValue(new Error('Upload failed'))

      render(<Dashboard />, { wrapper: createWrapper() })
      
      await waitFor(() => {
        expect(screen.getByTestId('upload-section')).toBeInTheDocument()
      })

      const fileInput = screen.getByTestId('file-input')
      const testFile = testUtils.createMockFile('contract.pdf', 'PDF content', 'application/pdf')

      await user.upload(fileInput, testFile)

      await waitFor(() => {
        expect(screen.getByText(/upload failed/i)).toBeInTheDocument()
        expect(screen.getByTestId('error-message')).toBeInTheDocument()
      })
    })
  })

  describe('Document Management', () => {
    const mockDocuments = [
      {
        document_id: 'doc-1',
        filename: 'contract1.pdf',
        upload_date: '2024-01-15T10:30:00Z',
        status: 'analyzed',
        last_analysis: {
          has_arbitration: true,
          confidence: 0.92,
          clause_type: 'mandatory_binding',
        },
      },
      {
        document_id: 'doc-2',
        filename: 'agreement2.docx',
        upload_date: '2024-01-14T15:45:00Z',
        status: 'pending',
        last_analysis: null,
      },
    ]

    it('displays uploaded documents list', async () => {
      mockApi.getDocuments.mockResolvedValue({
        documents: mockDocuments,
        total_count: 2,
        page: 1,
        total_pages: 1,
      })

      render(<Dashboard />, { wrapper: createWrapper() })
      
      await waitFor(() => {
        expect(screen.getByText('contract1.pdf')).toBeInTheDocument()
        expect(screen.getByText('agreement2.docx')).toBeInTheDocument()
        expect(screen.getByTestId('document-doc-1')).toBeInTheDocument()
        expect(screen.getByTestId('document-doc-2')).toBeInTheDocument()
      })
    })

    it('shows document analysis status', async () => {
      mockApi.getDocuments.mockResolvedValue({
        documents: mockDocuments,
        total_count: 2,
        page: 1,
        total_pages: 1,
      })

      render(<Dashboard />, { wrapper: createWrapper() })
      
      await waitFor(() => {
        // Analyzed document
        const doc1 = screen.getByTestId('document-doc-1')
        expect(doc1).toHaveTextContent(/arbitration detected/i)
        expect(doc1).toHaveTextContent('92%')
        
        // Pending document
        const doc2 = screen.getByTestId('document-doc-2')
        expect(doc2).toHaveTextContent(/pending analysis/i)
      })
    })

    it('allows analyzing pending documents', async () => {
      const user = userEvent.setup()
      
      mockApi.getDocuments.mockResolvedValue({
        documents: mockDocuments,
        total_count: 2,
        page: 1,
        total_pages: 1,
      })

      mockApi.analyzeDocument.mockResolvedValue({
        has_arbitration: false,
        confidence: 0.15,
        clause_type: null,
        explanation: 'No arbitration clause found',
        keywords: [],
        processing_time: 0.8,
      })

      render(<Dashboard />, { wrapper: createWrapper() })
      
      await waitFor(() => {
        expect(screen.getByTestId('document-doc-2')).toBeInTheDocument()
      })

      const analyzeButton = screen.getByTestId('analyze-button-doc-2')
      await user.click(analyzeButton)

      await waitFor(() => {
        expect(mockApi.analyzeDocument).toHaveBeenCalledWith('doc-2')
        expect(screen.getByText(/no arbitration clause found/i)).toBeInTheDocument()
      })
    })

    it('supports document deletion', async () => {
      const user = userEvent.setup()
      
      mockApi.getDocuments.mockResolvedValue({
        documents: mockDocuments,
        total_count: 2,
        page: 1,
        total_pages: 1,
      })

      mockApi.deleteDocument.mockResolvedValue({ success: true })

      render(<Dashboard />, { wrapper: createWrapper() })
      
      await waitFor(() => {
        expect(screen.getByTestId('document-doc-1')).toBeInTheDocument()
      })

      const deleteButton = screen.getByTestId('delete-button-doc-1')
      await user.click(deleteButton)

      // Confirm deletion
      const confirmButton = screen.getByTestId('confirm-delete-button')
      await user.click(confirmButton)

      await waitFor(() => {
        expect(mockApi.deleteDocument).toHaveBeenCalledWith('doc-1')
      })
    })

    it('handles pagination for many documents', async () => {
      const user = userEvent.setup()
      
      const manyDocuments = Array.from({ length: 25 }, (_, i) => ({
        document_id: `doc-${i}`,
        filename: `document${i}.pdf`,
        upload_date: '2024-01-15T10:30:00Z',
        status: 'analyzed',
        last_analysis: { has_arbitration: i % 2 === 0, confidence: 0.8 },
      }))

      mockApi.getDocuments.mockResolvedValue({
        documents: manyDocuments.slice(0, 10),
        total_count: 25,
        page: 1,
        total_pages: 3,
      })

      render(<Dashboard />, { wrapper: createWrapper() })
      
      await waitFor(() => {
        expect(screen.getByTestId('pagination')).toBeInTheDocument()
        expect(screen.getByText('Page 1 of 3')).toBeInTheDocument()
      })

      const nextButton = screen.getByTestId('next-page-button')
      await user.click(nextButton)

      expect(mockApi.getDocuments).toHaveBeenCalledWith({ page: 2, limit: 10 })
    })
  })

  describe('Recent Analyses Section', () => {
    const mockAnalyses = [
      {
        id: 'analysis-1',
        document_id: 'doc-1',
        filename: 'contract1.pdf',
        timestamp: '2024-01-15T14:30:00Z',
        has_arbitration: true,
        confidence: 0.94,
        clause_type: 'mandatory_binding',
      },
      {
        id: 'analysis-2',
        document_id: 'doc-2',
        filename: 'agreement2.docx',
        timestamp: '2024-01-15T13:15:00Z',
        has_arbitration: false,
        confidence: 0.12,
        clause_type: null,
      },
    ]

    it('displays recent analyses', async () => {
      mockApi.getRecentAnalyses.mockResolvedValue(mockAnalyses)

      render(<Dashboard />, { wrapper: createWrapper() })
      
      await waitFor(() => {
        expect(screen.getByText('contract1.pdf')).toBeInTheDocument()
        expect(screen.getByText('agreement2.docx')).toBeInTheDocument()
        expect(screen.getByText('94%')).toBeInTheDocument()
        expect(screen.getByText('12%')).toBeInTheDocument()
      })
    })

    it('shows analysis timestamps', async () => {
      mockApi.getRecentAnalyses.mockResolvedValue(mockAnalyses)

      render(<Dashboard />, { wrapper: createWrapper() })
      
      await waitFor(() => {
        expect(screen.getByText(/2:30 PM/)).toBeInTheDocument()
        expect(screen.getByText(/1:15 PM/)).toBeInTheDocument()
      })
    })

    it('allows viewing detailed analysis results', async () => {
      const user = userEvent.setup()
      
      mockApi.getRecentAnalyses.mockResolvedValue(mockAnalyses)

      render(<Dashboard />, { wrapper: createWrapper() })
      
      await waitFor(() => {
        expect(screen.getByTestId('analysis-analysis-1')).toBeInTheDocument()
      })

      const viewButton = screen.getByTestId('view-analysis-analysis-1')
      await user.click(viewButton)

      expect(screen.getByTestId('analysis-details-modal')).toBeInTheDocument()
    })
  })

  describe('Search and Filtering', () => {
    it('supports document search', async () => {
      const user = userEvent.setup()
      
      mockApi.getDocuments.mockResolvedValue({
        documents: [],
        total_count: 0,
        page: 1,
        total_pages: 1,
      })

      render(<Dashboard />, { wrapper: createWrapper() })
      
      await waitFor(() => {
        expect(screen.getByTestId('search-input')).toBeInTheDocument()
      })

      const searchInput = screen.getByTestId('search-input')
      await user.type(searchInput, 'contract')

      await waitFor(() => {
        expect(mockApi.getDocuments).toHaveBeenCalledWith({
          search: 'contract',
          page: 1,
          limit: 10,
        })
      })
    })

    it('supports filtering by analysis status', async () => {
      const user = userEvent.setup()

      render(<Dashboard />, { wrapper: createWrapper() })
      
      await waitFor(() => {
        expect(screen.getByTestId('status-filter')).toBeInTheDocument()
      })

      const statusFilter = screen.getByTestId('status-filter')
      await user.selectOptions(statusFilter, 'analyzed')

      await waitFor(() => {
        expect(mockApi.getDocuments).toHaveBeenCalledWith({
          status: 'analyzed',
          page: 1,
          limit: 10,
        })
      })
    })

    it('supports filtering by arbitration result', async () => {
      const user = userEvent.setup()

      render(<Dashboard />, { wrapper: createWrapper() })
      
      await waitFor(() => {
        expect(screen.getByTestId('arbitration-filter')).toBeInTheDocument()
      })

      const arbitrationFilter = screen.getByTestId('arbitration-filter')
      await user.selectOptions(arbitrationFilter, 'true')

      await waitFor(() => {
        expect(mockApi.getDocuments).toHaveBeenCalledWith({
          has_arbitration: true,
          page: 1,
          limit: 10,
        })
      })
    })
  })

  describe('Error Handling', () => {
    it('handles API errors gracefully', async () => {
      mockApi.getDocuments.mockRejectedValue(new Error('API Error'))

      render(<Dashboard />, { wrapper: createWrapper() })
      
      await waitFor(() => {
        expect(screen.getByTestId('error-message')).toBeInTheDocument()
        expect(screen.getByText(/failed to load documents/i)).toBeInTheDocument()
      })
    })

    it('shows retry option on error', async () => {
      const user = userEvent.setup()
      
      mockApi.getDocuments.mockRejectedValueOnce(new Error('API Error'))
      mockApi.getDocuments.mockResolvedValueOnce({
        documents: [],
        total_count: 0,
        page: 1,
        total_pages: 1,
      })

      render(<Dashboard />, { wrapper: createWrapper() })
      
      await waitFor(() => {
        expect(screen.getByTestId('retry-button')).toBeInTheDocument()
      })

      const retryButton = screen.getByTestId('retry-button')
      await user.click(retryButton)

      await waitFor(() => {
        expect(mockApi.getDocuments).toHaveBeenCalledTimes(2)
      })
    })

    it('handles network timeouts', async () => {
      mockApi.getDocuments.mockImplementation(() => 
        new Promise((_, reject) => 
          setTimeout(() => reject(new Error('Timeout')), 100)
        )
      )

      render(<Dashboard />, { wrapper: createWrapper() })
      
      await waitFor(() => {
        expect(screen.getByText(/request timed out/i)).toBeInTheDocument()
      }, { timeout: 2000 })
    })
  })

  describe('Keyboard Navigation', () => {
    it('supports keyboard navigation through documents', async () => {
      const user = userEvent.setup()
      
      const mockDocuments = [
        { document_id: 'doc-1', filename: 'contract1.pdf', status: 'analyzed' },
        { document_id: 'doc-2', filename: 'contract2.pdf', status: 'analyzed' },
      ]

      mockApi.getDocuments.mockResolvedValue({
        documents: mockDocuments,
        total_count: 2,
        page: 1,
        total_pages: 1,
      })

      render(<Dashboard />, { wrapper: createWrapper() })
      
      await waitFor(() => {
        expect(screen.getByTestId('document-doc-1')).toBeInTheDocument()
      })

      // Navigate with Tab key
      await user.tab()
      expect(screen.getByTestId('analyze-button-doc-1')).toHaveFocus()

      await user.tab()
      expect(screen.getByTestId('delete-button-doc-1')).toHaveFocus()
    })
  })

  describe('Real-time Updates', () => {
    it('updates document status in real-time', async () => {
      const { rerender } = render(<Dashboard />, { wrapper: createWrapper() })
      
      // Initial state - document pending
      mockApi.getDocuments.mockResolvedValue({
        documents: [{
          document_id: 'doc-1',
          filename: 'contract.pdf',
          status: 'pending',
          last_analysis: null,
        }],
        total_count: 1,
        page: 1,
        total_pages: 1,
      })

      await waitFor(() => {
        expect(screen.getByText(/pending analysis/i)).toBeInTheDocument()
      })

      // Updated state - document analyzed
      mockApi.getDocuments.mockResolvedValue({
        documents: [{
          document_id: 'doc-1',
          filename: 'contract.pdf',
          status: 'analyzed',
          last_analysis: {
            has_arbitration: true,
            confidence: 0.89,
            clause_type: 'mandatory_binding',
          },
        }],
        total_count: 1,
        page: 1,
        total_pages: 1,
      })

      rerender(<Dashboard />)

      await waitFor(() => {
        expect(screen.getByText(/arbitration detected/i)).toBeInTheDocument()
        expect(screen.getByText('89%')).toBeInTheDocument()
      })
    })
  })
})