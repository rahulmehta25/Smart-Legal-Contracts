/**
 * Unit tests for DocumentUploader component.
 * 
 * Tests file upload functionality including:
 * - Single and multiple file uploads
 * - File type validation
 * - File size validation
 * - Upload progress tracking
 * - Error handling
 * - Drag and drop functionality
 */

import React from 'react'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { act } from 'react-dom/test-utils'
import DocumentUploader from '@/components/DocumentUploader'

// Mock API calls
const mockUploadFile = jest.fn()
jest.mock('@/lib/api', () => ({
  uploadFile: (...args: any[]) => mockUploadFile(...args),
}))

describe('DocumentUploader Component', () => {
  const defaultProps = {
    onUploadSuccess: jest.fn(),
    onUploadError: jest.fn(),
    maxFileSize: 10 * 1024 * 1024, // 10MB
    acceptedFileTypes: ['.pdf', '.docx', '.doc'],
    multiple: false,
  }

  beforeEach(() => {
    jest.clearAllMocks()
  })

  describe('Rendering', () => {
    it('renders upload area with correct elements', () => {
      render(<DocumentUploader {...defaultProps} />)
      
      expect(screen.getByTestId('upload-area')).toBeInTheDocument()
      expect(screen.getByText(/drag and drop/i)).toBeInTheDocument()
      expect(screen.getByText(/browse files/i)).toBeInTheDocument()
      expect(screen.getByText(/supported formats/i)).toBeInTheDocument()
    })

    it('displays accepted file types', () => {
      render(<DocumentUploader {...defaultProps} />)
      
      expect(screen.getByText(/PDF, DOCX, DOC/i)).toBeInTheDocument()
    })

    it('shows file size limit', () => {
      render(<DocumentUploader {...defaultProps} />)
      
      expect(screen.getByText(/max size: 10 MB/i)).toBeInTheDocument()
    })

    it('renders with multiple file selection when enabled', () => {
      render(<DocumentUploader {...defaultProps} multiple={true} />)
      
      const fileInput = screen.getByTestId('file-input')
      expect(fileInput).toHaveAttribute('multiple')
    })
  })

  describe('File Selection via Browse', () => {
    it('handles single file selection', async () => {
      const user = userEvent.setup()
      mockUploadFile.mockResolvedValue({ 
        document_id: 'doc-123', 
        status: 'uploaded',
        filename: 'test.pdf' 
      })

      render(<DocumentUploader {...defaultProps} />)
      
      const fileInput = screen.getByTestId('file-input')
      const testFile = testUtils.createMockFile('test.pdf', 'PDF content', 'application/pdf')

      await act(async () => {
        await user.upload(fileInput, testFile)
      })

      await waitFor(() => {
        expect(mockUploadFile).toHaveBeenCalledWith(testFile)
        expect(defaultProps.onUploadSuccess).toHaveBeenCalledWith({
          document_id: 'doc-123',
          status: 'uploaded',
          filename: 'test.pdf'
        })
      })
    })

    it('handles multiple file selection', async () => {
      const user = userEvent.setup()
      mockUploadFile.mockResolvedValue({ 
        uploaded_documents: [
          { document_id: 'doc-1', filename: 'file1.pdf' },
          { document_id: 'doc-2', filename: 'file2.pdf' }
        ]
      })

      render(<DocumentUploader {...defaultProps} multiple={true} />)
      
      const fileInput = screen.getByTestId('file-input')
      const testFiles = [
        testUtils.createMockFile('file1.pdf', 'PDF 1', 'application/pdf'),
        testUtils.createMockFile('file2.pdf', 'PDF 2', 'application/pdf')
      ]

      await act(async () => {
        await user.upload(fileInput, testFiles)
      })

      await waitFor(() => {
        expect(mockUploadFile).toHaveBeenCalledWith(testFiles)
        expect(defaultProps.onUploadSuccess).toHaveBeenCalled()
      })
    })

    it('validates file types and rejects invalid files', async () => {
      const user = userEvent.setup()
      
      render(<DocumentUploader {...defaultProps} />)
      
      const fileInput = screen.getByTestId('file-input')
      const invalidFile = testUtils.createMockFile('test.txt', 'Text content', 'text/plain')

      await act(async () => {
        await user.upload(fileInput, invalidFile)
      })

      await waitFor(() => {
        expect(screen.getByText(/invalid file type/i)).toBeInTheDocument()
        expect(mockUploadFile).not.toHaveBeenCalled()
        expect(defaultProps.onUploadError).toHaveBeenCalledWith(
          expect.objectContaining({
            type: 'invalid_file_type',
            message: expect.stringContaining('Invalid file type')
          })
        )
      })
    })

    it('validates file size and rejects oversized files', async () => {
      const user = userEvent.setup()
      
      render(<DocumentUploader {...defaultProps} maxFileSize={1024} />)
      
      const fileInput = screen.getByTestId('file-input')
      const largeFile = testUtils.createMockFile(
        'large.pdf', 
        'x'.repeat(2048), // 2KB file, exceeds 1KB limit
        'application/pdf'
      )

      await act(async () => {
        await user.upload(fileInput, largeFile)
      })

      await waitFor(() => {
        expect(screen.getByText(/file too large/i)).toBeInTheDocument()
        expect(mockUploadFile).not.toHaveBeenCalled()
        expect(defaultProps.onUploadError).toHaveBeenCalledWith(
          expect.objectContaining({
            type: 'file_too_large',
            message: expect.stringContaining('File too large')
          })
        )
      })
    })
  })

  describe('Drag and Drop Functionality', () => {
    it('handles drag enter and leave events', () => {
      render(<DocumentUploader {...defaultProps} />)
      
      const uploadArea = screen.getByTestId('upload-area')
      
      // Drag enter
      fireEvent.dragEnter(uploadArea)
      expect(uploadArea).toHaveClass('drag-over')
      
      // Drag leave
      fireEvent.dragLeave(uploadArea)
      expect(uploadArea).not.toHaveClass('drag-over')
    })

    it('handles drag over event', () => {
      render(<DocumentUploader {...defaultProps} />)
      
      const uploadArea = screen.getByTestId('upload-area')
      const dragEvent = new Event('dragover', { bubbles: true })
      
      fireEvent.dragOver(uploadArea, dragEvent)
      expect(uploadArea).toHaveClass('drag-over')
    })

    it('handles file drop', async () => {
      mockUploadFile.mockResolvedValue({ 
        document_id: 'doc-123', 
        status: 'uploaded' 
      })

      render(<DocumentUploader {...defaultProps} />)
      
      const uploadArea = screen.getByTestId('upload-area')
      const testFile = testUtils.createMockFile('dropped.pdf', 'PDF content', 'application/pdf')
      
      const dropEvent = new Event('drop', { bubbles: true }) as any
      dropEvent.dataTransfer = {
        files: [testFile],
        types: ['Files']
      }

      await act(async () => {
        fireEvent.drop(uploadArea, dropEvent)
      })

      await waitFor(() => {
        expect(mockUploadFile).toHaveBeenCalledWith(testFile)
        expect(defaultProps.onUploadSuccess).toHaveBeenCalled()
        expect(uploadArea).not.toHaveClass('drag-over')
      })
    })

    it('validates dropped files', async () => {
      render(<DocumentUploader {...defaultProps} />)
      
      const uploadArea = screen.getByTestId('upload-area')
      const invalidFile = testUtils.createMockFile('invalid.txt', 'Text', 'text/plain')
      
      const dropEvent = new Event('drop', { bubbles: true }) as any
      dropEvent.dataTransfer = {
        files: [invalidFile],
        types: ['Files']
      }

      await act(async () => {
        fireEvent.drop(uploadArea, dropEvent)
      })

      await waitFor(() => {
        expect(screen.getByText(/invalid file type/i)).toBeInTheDocument()
        expect(mockUploadFile).not.toHaveBeenCalled()
      })
    })
  })

  describe('Upload Progress', () => {
    it('shows upload progress during file upload', async () => {
      const user = userEvent.setup()
      
      // Mock upload with progress
      mockUploadFile.mockImplementation(() => {
        return new Promise((resolve) => {
          setTimeout(() => {
            resolve({ document_id: 'doc-123', status: 'uploaded' })
          }, 1000)
        })
      })

      render(<DocumentUploader {...defaultProps} />)
      
      const fileInput = screen.getByTestId('file-input')
      const testFile = testUtils.createMockFile('test.pdf', 'PDF content', 'application/pdf')

      await act(async () => {
        await user.upload(fileInput, testFile)
      })

      // Check for progress indicator
      expect(screen.getByTestId('upload-progress')).toBeInTheDocument()
      expect(screen.getByText(/uploading/i)).toBeInTheDocument()

      // Wait for upload completion
      await waitFor(() => {
        expect(screen.queryByTestId('upload-progress')).not.toBeInTheDocument()
      }, { timeout: 2000 })
    })

    it('shows upload success state', async () => {
      const user = userEvent.setup()
      mockUploadFile.mockResolvedValue({ 
        document_id: 'doc-123', 
        status: 'uploaded',
        filename: 'test.pdf'
      })

      render(<DocumentUploader {...defaultProps} />)
      
      const fileInput = screen.getByTestId('file-input')
      const testFile = testUtils.createMockFile('test.pdf', 'PDF content', 'application/pdf')

      await act(async () => {
        await user.upload(fileInput, testFile)
      })

      await waitFor(() => {
        expect(screen.getByText(/upload successful/i)).toBeInTheDocument()
        expect(screen.getByText('test.pdf')).toBeInTheDocument()
        expect(screen.getByTestId('success-icon')).toBeInTheDocument()
      })
    })

    it('shows upload error state', async () => {
      const user = userEvent.setup()
      mockUploadFile.mockRejectedValue(new Error('Upload failed'))

      render(<DocumentUploader {...defaultProps} />)
      
      const fileInput = screen.getByTestId('file-input')
      const testFile = testUtils.createMockFile('test.pdf', 'PDF content', 'application/pdf')

      await act(async () => {
        await user.upload(fileInput, testFile)
      })

      await waitFor(() => {
        expect(screen.getByText(/upload failed/i)).toBeInTheDocument()
        expect(screen.getByTestId('error-icon')).toBeInTheDocument()
        expect(defaultProps.onUploadError).toHaveBeenCalledWith(
          expect.objectContaining({
            type: 'upload_error',
            message: 'Upload failed'
          })
        )
      })
    })
  })

  describe('File Preview', () => {
    it('shows file preview for uploaded files', async () => {
      const user = userEvent.setup()
      mockUploadFile.mockResolvedValue({ 
        document_id: 'doc-123', 
        status: 'uploaded',
        filename: 'contract.pdf',
        preview_url: 'blob:preview-url'
      })

      render(<DocumentUploader {...defaultProps} showPreview={true} />)
      
      const fileInput = screen.getByTestId('file-input')
      const testFile = testUtils.createMockFile('contract.pdf', 'PDF content', 'application/pdf')

      await act(async () => {
        await user.upload(fileInput, testFile)
      })

      await waitFor(() => {
        expect(screen.getByTestId('file-preview')).toBeInTheDocument()
        expect(screen.getByText('contract.pdf')).toBeInTheDocument()
        expect(screen.getByTestId('pdf-icon')).toBeInTheDocument()
      })
    })

    it('allows file removal from preview', async () => {
      const user = userEvent.setup()
      mockUploadFile.mockResolvedValue({ 
        document_id: 'doc-123', 
        status: 'uploaded',
        filename: 'contract.pdf'
      })

      render(<DocumentUploader {...defaultProps} showPreview={true} />)
      
      const fileInput = screen.getByTestId('file-input')
      const testFile = testUtils.createMockFile('contract.pdf', 'PDF content', 'application/pdf')

      await act(async () => {
        await user.upload(fileInput, testFile)
      })

      await waitFor(() => {
        expect(screen.getByTestId('file-preview')).toBeInTheDocument()
      })

      // Remove file
      const removeButton = screen.getByTestId('remove-file-button')
      await user.click(removeButton)

      expect(screen.queryByTestId('file-preview')).not.toBeInTheDocument()
    })
  })

  describe('Accessibility', () => {
    it('has proper ARIA labels and roles', () => {
      render(<DocumentUploader {...defaultProps} />)
      
      const uploadArea = screen.getByTestId('upload-area')
      const fileInput = screen.getByTestId('file-input')
      
      expect(uploadArea).toHaveAttribute('role', 'button')
      expect(uploadArea).toHaveAttribute('aria-label', expect.stringContaining('upload'))
      expect(fileInput).toHaveAttribute('aria-describedby')
    })

    it('supports keyboard navigation', async () => {
      const user = userEvent.setup()
      
      render(<DocumentUploader {...defaultProps} />)
      
      const uploadArea = screen.getByTestId('upload-area')
      
      // Focus on upload area
      await user.tab()
      expect(uploadArea).toHaveFocus()
      
      // Activate with Enter or Space
      await user.keyboard('{Enter}')
      expect(screen.getByTestId('file-input')).toHaveFocus()
    })

    it('announces upload status to screen readers', async () => {
      const user = userEvent.setup()
      mockUploadFile.mockResolvedValue({ 
        document_id: 'doc-123', 
        status: 'uploaded' 
      })

      render(<DocumentUploader {...defaultProps} />)
      
      const fileInput = screen.getByTestId('file-input')
      const testFile = testUtils.createMockFile('test.pdf', 'PDF content', 'application/pdf')

      await act(async () => {
        await user.upload(fileInput, testFile)
      })

      await waitFor(() => {
        const statusRegion = screen.getByTestId('upload-status')
        expect(statusRegion).toHaveAttribute('role', 'status')
        expect(statusRegion).toHaveAttribute('aria-live', 'polite')
      })
    })
  })

  describe('Error Recovery', () => {
    it('allows retry after upload failure', async () => {
      const user = userEvent.setup()
      
      // First upload fails
      mockUploadFile.mockRejectedValueOnce(new Error('Network error'))
      // Second upload succeeds
      mockUploadFile.mockResolvedValueOnce({ 
        document_id: 'doc-123', 
        status: 'uploaded' 
      })

      render(<DocumentUploader {...defaultProps} />)
      
      const fileInput = screen.getByTestId('file-input')
      const testFile = testUtils.createMockFile('test.pdf', 'PDF content', 'application/pdf')

      // First upload attempt
      await act(async () => {
        await user.upload(fileInput, testFile)
      })

      await waitFor(() => {
        expect(screen.getByText(/upload failed/i)).toBeInTheDocument()
      })

      // Retry upload
      const retryButton = screen.getByTestId('retry-upload-button')
      await user.click(retryButton)

      await waitFor(() => {
        expect(screen.getByText(/upload successful/i)).toBeInTheDocument()
        expect(defaultProps.onUploadSuccess).toHaveBeenCalled()
      })
    })

    it('clears error state when new file is selected', async () => {
      const user = userEvent.setup()
      
      // First upload fails
      mockUploadFile.mockRejectedValueOnce(new Error('Upload error'))
      // Second upload succeeds
      mockUploadFile.mockResolvedValueOnce({ 
        document_id: 'doc-123', 
        status: 'uploaded' 
      })

      render(<DocumentUploader {...defaultProps} />)
      
      const fileInput = screen.getByTestId('file-input')
      const testFile1 = testUtils.createMockFile('test1.pdf', 'PDF content', 'application/pdf')
      const testFile2 = testUtils.createMockFile('test2.pdf', 'PDF content', 'application/pdf')

      // First upload (fails)
      await act(async () => {
        await user.upload(fileInput, testFile1)
      })

      await waitFor(() => {
        expect(screen.getByText(/upload failed/i)).toBeInTheDocument()
      })

      // Second upload (succeeds)
      await act(async () => {
        await user.upload(fileInput, testFile2)
      })

      await waitFor(() => {
        expect(screen.queryByText(/upload failed/i)).not.toBeInTheDocument()
        expect(screen.getByText(/upload successful/i)).toBeInTheDocument()
      })
    })
  })
})