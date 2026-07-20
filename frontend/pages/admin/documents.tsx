import React, { useState, useEffect } from 'react';
import { MagnifyingGlassIcon, FunnelIcon, ArrowDownTrayIcon, DocumentTextIcon, EyeIcon, TrashIcon, ArchiveBoxIcon } from '@heroicons/react/24/outline';
import AdminLayout from '../../src/components/admin/AdminLayout';
import DocumentTable from '../../src/components/admin/DocumentTable';
import DocumentPreviewModal from '../../src/components/admin/DocumentPreviewModal';
import BulkActionsBar from '../../src/components/admin/BulkActionsBar';

interface Document {
  id: string;
  name: string;
  type: 'pdf' | 'docx' | 'txt';
  size: number;
  uploadedBy: string;
  uploadedAt: string;
  analysisStatus: 'pending' | 'processing' | 'completed' | 'failed';
  arbitrationDetected: boolean | null;
  confidenceScore: number | null;
  tags: string[];
  category: 'contract' | 'agreement' | 'policy' | 'other';
}

const DocumentsPage: React.FC = () => {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);
  const [isPreviewModalOpen, setIsPreviewModalOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [selectedStatus, setSelectedStatus] = useState<string>('all');
  const [isLoading, setIsLoading] = useState(true);
  const [selectedDocuments, setSelectedDocuments] = useState<string[]>([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalDocuments, setTotalDocuments] = useState(0);
  const documentsPerPage = 15;

  useEffect(() => {
    fetchDocuments();
  }, [currentPage, searchQuery, selectedCategory, selectedStatus]);

  const fetchDocuments = async () => {
    setIsLoading(true);
    // Mock API call - replace with actual implementation
    setTimeout(() => {
      const mockDocuments: Document[] = Array.from({ length: 150 }, (_, i) => ({
        id: `doc-${i + 1}`,
        name: `Document ${i + 1}.pdf`,
        type: ['pdf', 'docx', 'txt'][i % 3] as Document['type'],
        size: Math.floor(Math.random() * 5000000) + 10000, // 10KB to 5MB
        uploadedBy: `user${Math.floor(i / 10) + 1}@example.com`,
        uploadedAt: new Date(Date.now() - Math.random() * 90 * 24 * 60 * 60 * 1000).toISOString(),
        analysisStatus: ['pending', 'processing', 'completed', 'failed'][i % 4] as Document['analysisStatus'],
        arbitrationDetected: i % 4 === 2 ? Math.random() > 0.5 : null,
        confidenceScore: i % 4 === 2 ? Math.random() * 100 : null,
        tags: [`tag-${i % 5 + 1}`, `category-${i % 3 + 1}`],
        category: ['contract', 'agreement', 'policy', 'other'][i % 4] as Document['category']
      }));

      // Apply filters
      let filteredDocuments = mockDocuments.filter(doc => {
        const matchesSearch = doc.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                             doc.uploadedBy.toLowerCase().includes(searchQuery.toLowerCase());
        const matchesCategory = selectedCategory === 'all' || doc.category === selectedCategory;
        const matchesStatus = selectedStatus === 'all' || doc.analysisStatus === selectedStatus;
        
        return matchesSearch && matchesCategory && matchesStatus;
      });

      const startIndex = (currentPage - 1) * documentsPerPage;
      const paginatedDocuments = filteredDocuments.slice(startIndex, startIndex + documentsPerPage);

      setDocuments(paginatedDocuments);
      setTotalDocuments(filteredDocuments.length);
      setIsLoading(false);
    }, 500);
  };

  const handleDocumentAction = (action: 'view' | 'download' | 'delete' | 'archive', document: Document) => {
    switch (action) {
      case 'view':
        setSelectedDocument(document);
        setIsPreviewModalOpen(true);
        break;
      case 'download':
        // Handle download
        console.log('Downloading document:', document.id);
        break;
      case 'delete':
        if (confirm('Are you sure you want to delete this document?')) {
          console.log('Deleting document:', document.id);
          fetchDocuments();
        }
        break;
      case 'archive':
        console.log('Archiving document:', document.id);
        fetchDocuments();
        break;
    }
  };

  const handleBulkAction = (action: string) => {
    if (selectedDocuments.length === 0) return;

    switch (action) {
      case 'delete':
        if (confirm(`Delete ${selectedDocuments.length} documents?`)) {
          console.log('Bulk deleting:', selectedDocuments);
          setSelectedDocuments([]);
          fetchDocuments();
        }
        break;
      case 'archive':
        console.log('Bulk archiving:', selectedDocuments);
        setSelectedDocuments([]);
        fetchDocuments();
        break;
      case 'export':
        console.log('Bulk exporting:', selectedDocuments);
        break;
      case 'reanalyze':
        console.log('Bulk reanalyzing:', selectedDocuments);
        setSelectedDocuments([]);
        fetchDocuments();
        break;
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const totalPages = Math.ceil(totalDocuments / documentsPerPage);

  return (
    <AdminLayout>
      <div className="space-y-6" id="documents-page-container">
        {/* Header */}
        <div className="sm:flex sm:items-center sm:justify-between" id="documents-header">
          <div id="documents-title-section">
            <h1 className="text-2xl font-semibold text-gray-900" id="documents-title">
              Document Management
            </h1>
            <p className="mt-1 text-sm text-gray-500" id="documents-subtitle">
              View, manage, and analyze uploaded documents
            </p>
          </div>
          <div className="mt-4 flex space-x-3 sm:mt-0" id="documents-actions">
            <button
              onClick={() => handleBulkAction('export')}
              className="inline-flex items-center px-4 py-2 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50"
              id="export-documents-btn"
            >
              <ArrowDownTrayIcon className="h-4 w-4 mr-2" id="export-icon" />
              Export Data
            </button>
          </div>
        </div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4" id="document-stats">
          <div className="bg-white rounded-lg border border-gray-200 p-6" id="total-documents-stat">
            <div className="flex items-center" id="total-documents-content">
              <DocumentTextIcon className="h-8 w-8 text-blue-600" id="total-documents-icon" />
              <div className="ml-4" id="total-documents-info">
                <div className="text-2xl font-semibold text-gray-900" id="total-documents-value">
                  {totalDocuments.toLocaleString()}
                </div>
                <div className="text-sm text-gray-500" id="total-documents-label">
                  Total Documents
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg border border-gray-200 p-6" id="processing-stat">
            <div className="flex items-center" id="processing-content">
              <div className="w-8 h-8 rounded-full bg-yellow-100 flex items-center justify-center" id="processing-icon-wrapper">
                <div className="w-4 h-4 rounded-full bg-yellow-500 animate-pulse" id="processing-icon"></div>
              </div>
              <div className="ml-4" id="processing-info">
                <div className="text-2xl font-semibold text-gray-900" id="processing-value">
                  {documents.filter(d => d.analysisStatus === 'processing').length}
                </div>
                <div className="text-sm text-gray-500" id="processing-label">
                  Processing
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg border border-gray-200 p-6" id="arbitration-detected-stat">
            <div className="flex items-center" id="arbitration-detected-content">
              <div className="w-8 h-8 rounded-full bg-red-100 flex items-center justify-center" id="arbitration-detected-icon-wrapper">
                <div className="w-4 h-4 rounded-full bg-red-500" id="arbitration-detected-icon"></div>
              </div>
              <div className="ml-4" id="arbitration-detected-info">
                <div className="text-2xl font-semibold text-gray-900" id="arbitration-detected-value">
                  {documents.filter(d => d.arbitrationDetected === true).length}
                </div>
                <div className="text-sm text-gray-500" id="arbitration-detected-label">
                  Arbitration Detected
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg border border-gray-200 p-6" id="avg-confidence-stat">
            <div className="flex items-center" id="avg-confidence-content">
              <div className="w-8 h-8 rounded-full bg-green-100 flex items-center justify-center" id="avg-confidence-icon-wrapper">
                <div className="text-green-600 font-bold text-sm" id="avg-confidence-icon">%</div>
              </div>
              <div className="ml-4" id="avg-confidence-info">
                <div className="text-2xl font-semibold text-gray-900" id="avg-confidence-value">
                  {Math.round(documents.filter(d => d.confidenceScore).reduce((acc, d) => acc + (d.confidenceScore || 0), 0) / documents.filter(d => d.confidenceScore).length || 0)}%
                </div>
                <div className="text-sm text-gray-500" id="avg-confidence-label">
                  Avg. Confidence
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Search and Filters */}
        <div className="bg-white rounded-lg border border-gray-200 p-6" id="search-filters-section">
          <div className="flex flex-col sm:flex-row gap-4" id="search-filters-content">
            {/* Search Bar */}
            <div className="flex-1" id="search-bar-container">
              <div className="relative" id="search-input-wrapper">
                <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" id="search-icon" />
                <input
                  type="text"
                  placeholder="Search documents by name or uploader..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10 pr-4 py-2 w-full border border-gray-300 rounded-md focus:ring-blue-500 focus:border-blue-500"
                  id="search-input"
                />
              </div>
            </div>

            {/* Quick Filters */}
            <div className="flex space-x-3" id="quick-filters">
              <select
                value={selectedCategory}
                onChange={(e) => setSelectedCategory(e.target.value)}
                className="border border-gray-300 rounded-md px-3 py-2 text-sm focus:ring-blue-500 focus:border-blue-500"
                id="category-filter"
              >
                <option value="all">All Categories</option>
                <option value="contract">Contract</option>
                <option value="agreement">Agreement</option>
                <option value="policy">Policy</option>
                <option value="other">Other</option>
              </select>

              <select
                value={selectedStatus}
                onChange={(e) => setSelectedStatus(e.target.value)}
                className="border border-gray-300 rounded-md px-3 py-2 text-sm focus:ring-blue-500 focus:border-blue-500"
                id="status-filter"
              >
                <option value="all">All Status</option>
                <option value="pending">Pending</option>
                <option value="processing">Processing</option>
                <option value="completed">Completed</option>
                <option value="failed">Failed</option>
              </select>
            </div>
          </div>
        </div>

        {/* Bulk Actions Bar */}
        {selectedDocuments.length > 0 && (
          <BulkActionsBar
            selectedCount={selectedDocuments.length}
            onBulkAction={handleBulkAction}
            onClearSelection={() => setSelectedDocuments([])}
          />
        )}

        {/* Documents Table */}
        <div className="bg-white rounded-lg border border-gray-200" id="documents-table-container">
          <DocumentTable
            documents={documents}
            isLoading={isLoading}
            selectedDocuments={selectedDocuments}
            onDocumentAction={handleDocumentAction}
            onSelectionChange={setSelectedDocuments}
          />

          {/* Pagination */}
          <div className="px-6 py-4 border-t border-gray-200" id="pagination-container">
            <div className="flex items-center justify-between" id="pagination-content">
              <div className="text-sm text-gray-700" id="pagination-info">
                Showing {((currentPage - 1) * documentsPerPage) + 1} to {Math.min(currentPage * documentsPerPage, totalDocuments)} of {totalDocuments} documents
              </div>
              <div className="flex space-x-2" id="pagination-buttons">
                <button
                  onClick={() => setCurrentPage(Math.max(1, currentPage - 1))}
                  disabled={currentPage === 1}
                  className="px-3 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                  id="prev-page-btn"
                >
                  Previous
                </button>
                
                {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                  const pageNum = currentPage <= 3 ? i + 1 : currentPage - 2 + i;
                  if (pageNum > totalPages) return null;
                  
                  return (
                    <button
                      key={pageNum}
                      onClick={() => setCurrentPage(pageNum)}
                      className={`px-3 py-2 rounded-md text-sm font-medium ${
                        pageNum === currentPage
                          ? 'bg-blue-600 text-white'
                          : 'border border-gray-300 text-gray-700 bg-white hover:bg-gray-50'
                      }`}
                      id={`page-${pageNum}-btn`}
                    >
                      {pageNum}
                    </button>
                  );
                })}

                <button
                  onClick={() => setCurrentPage(Math.min(totalPages, currentPage + 1))}
                  disabled={currentPage === totalPages}
                  className="px-3 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
                  id="next-page-btn"
                >
                  Next
                </button>
              </div>
            </div>
          </div>
        </div>

        {/* Document Preview Modal */}
        {isPreviewModalOpen && selectedDocument && (
          <DocumentPreviewModal
            document={selectedDocument}
            isOpen={isPreviewModalOpen}
            onClose={() => {
              setIsPreviewModalOpen(false);
              setSelectedDocument(null);
            }}
            onAction={(action) => {
              handleDocumentAction(action, selectedDocument);
              if (action !== 'view') {
                setIsPreviewModalOpen(false);
                setSelectedDocument(null);
              }
            }}
          />
        )}
      </div>
    </AdminLayout>
  );
};

export default DocumentsPage;