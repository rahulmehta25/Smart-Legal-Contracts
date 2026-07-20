import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { FileText, TrendingUp, Clock, Download, Eye, Trash2, Filter, Search, BarChart3, PieChart, Calendar, Users, AlertCircle, CheckCircle } from 'lucide-react';
import { format, subDays, isWithinInterval } from 'date-fns';
import { toast } from 'react-hot-toast';
import { useAuth } from '../src/contexts/AuthContext';
import { useTheme } from '../src/contexts/ThemeContext';
import StatisticsChart from '../src/components/StatisticsChart';
import NotificationBell from '../src/components/NotificationBell';

// Types
interface AnalysisRecord {
  id: string;
  fileName: string;
  uploadDate: string;
  status: 'completed' | 'processing' | 'failed';
  clausesFound: number;
  avgConfidence: number;
  fileSize: number;
  processingTime: number;
  results?: any;
}

interface DashboardStats {
  totalDocuments: number;
  totalClauses: number;
  avgConfidence: number;
  processingTime: number;
  documentsThisMonth: number;
  monthlyGrowth: number;
}

export default function Dashboard() {
  const router = useRouter();
  const { user, logout } = useAuth();
  const { isDark } = useTheme();
  
  const [documents, setDocuments] = useState<AnalysisRecord[]>([]);
  const [stats, setStats] = useState<DashboardStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [dateFilter, setDateFilter] = useState<string>('all');
  const [selectedDocs, setSelectedDocs] = useState<string[]>([]);
  const [showExportModal, setShowExportModal] = useState(false);
  const [sortBy, setSortBy] = useState<'date' | 'name' | 'confidence' | 'clauses'>('date');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  // Load dashboard data
  useEffect(() => {
    const loadDashboardData = async () => {
      try {
        // Mock API call - replace with actual data fetching
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        const mockDocuments: AnalysisRecord[] = [
          {
            id: '1',
            fileName: 'Service Agreement.pdf',
            uploadDate: new Date().toISOString(),
            status: 'completed',
            clausesFound: 3,
            avgConfidence: 87,
            fileSize: 256000,
            processingTime: 2.3
          },
          {
            id: '2',
            fileName: 'Employment Contract.docx',
            uploadDate: subDays(new Date(), 2).toISOString(),
            status: 'completed',
            clausesFound: 5,
            avgConfidence: 92,
            fileSize: 180000,
            processingTime: 1.8
          },
          {
            id: '3',
            fileName: 'Terms of Service.pdf',
            uploadDate: subDays(new Date(), 5).toISOString(),
            status: 'processing',
            clausesFound: 0,
            avgConfidence: 0,
            fileSize: 340000,
            processingTime: 0
          },
          {
            id: '4',
            fileName: 'Partnership Agreement.pdf',
            uploadDate: subDays(new Date(), 7).toISOString(),
            status: 'completed',
            clausesFound: 8,
            avgConfidence: 85,
            fileSize: 420000,
            processingTime: 3.1
          },
          {
            id: '5',
            fileName: 'Vendor Contract.docx',
            uploadDate: subDays(new Date(), 10).toISOString(),
            status: 'failed',
            clausesFound: 0,
            avgConfidence: 0,
            fileSize: 95000,
            processingTime: 0
          }
        ];
        
        const mockStats: DashboardStats = {
          totalDocuments: mockDocuments.length,
          totalClauses: mockDocuments.reduce((sum, doc) => sum + doc.clausesFound, 0),
          avgConfidence: Math.round(
            mockDocuments
              .filter(doc => doc.status === 'completed')
              .reduce((sum, doc) => sum + doc.avgConfidence, 0) / 
            mockDocuments.filter(doc => doc.status === 'completed').length
          ),
          processingTime: mockDocuments
            .filter(doc => doc.status === 'completed')
            .reduce((sum, doc) => sum + doc.processingTime, 0) / 
            mockDocuments.filter(doc => doc.status === 'completed').length,
          documentsThisMonth: mockDocuments.filter(doc => 
            isWithinInterval(new Date(doc.uploadDate), {
              start: subDays(new Date(), 30),
              end: new Date()
            })
          ).length,
          monthlyGrowth: 23.5
        };
        
        setDocuments(mockDocuments);
        setStats(mockStats);
      } catch (error) {
        toast.error('Failed to load dashboard data');
        console.error('Dashboard load error:', error);
      } finally {
        setLoading(false);
      }
    };

    if (user) {
      loadDashboardData();
    }
  }, [user]);

  // Filter and sort documents
  const filteredDocuments = documents
    .filter(doc => {
      const matchesSearch = doc.fileName.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesStatus = statusFilter === 'all' || doc.status === statusFilter;
      
      let matchesDate = true;
      if (dateFilter !== 'all') {
        const docDate = new Date(doc.uploadDate);
        const now = new Date();
        switch (dateFilter) {
          case 'today':
            matchesDate = docDate.toDateString() === now.toDateString();
            break;
          case 'week':
            matchesDate = isWithinInterval(docDate, { start: subDays(now, 7), end: now });
            break;
          case 'month':
            matchesDate = isWithinInterval(docDate, { start: subDays(now, 30), end: now });
            break;
        }
      }
      
      return matchesSearch && matchesStatus && matchesDate;
    })
    .sort((a, b) => {
      let comparison = 0;
      switch (sortBy) {
        case 'date':
          comparison = new Date(a.uploadDate).getTime() - new Date(b.uploadDate).getTime();
          break;
        case 'name':
          comparison = a.fileName.localeCompare(b.fileName);
          break;
        case 'confidence':
          comparison = a.avgConfidence - b.avgConfidence;
          break;
        case 'clauses':
          comparison = a.clausesFound - b.clausesFound;
          break;
      }
      return sortOrder === 'desc' ? -comparison : comparison;
    });

  const handleDocumentSelect = (docId: string) => {
    setSelectedDocs(prev => 
      prev.includes(docId) 
        ? prev.filter(id => id !== docId)
        : [...prev, docId]
    );
  };

  const handleSelectAll = () => {
    if (selectedDocs.length === filteredDocuments.length) {
      setSelectedDocs([]);
    } else {
      setSelectedDocs(filteredDocuments.map(doc => doc.id));
    }
  };

  const handleDeleteSelected = async () => {
    if (selectedDocs.length === 0) return;
    
    try {
      // Mock API call - replace with actual deletion
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setDocuments(prev => prev.filter(doc => !selectedDocs.includes(doc.id)));
      setSelectedDocs([]);
      toast.success(`Deleted ${selectedDocs.length} document(s)`);
    } catch (error) {
      toast.error('Failed to delete documents');
    }
  };

  const handleExportData = async (format: 'csv' | 'json' | 'pdf') => {
    try {
      // Mock export - replace with actual export logic
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const dataToExport = selectedDocs.length > 0 
        ? documents.filter(doc => selectedDocs.includes(doc.id))
        : documents;
      
      toast.success(`Exported ${dataToExport.length} documents as ${format.toUpperCase()}`);
      setShowExportModal(false);
    } catch (error) {
      toast.error('Export failed');
    }
  };

  const formatFileSize = (bytes: number) => {
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Bytes';
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case 'processing':
        return <Clock className="h-4 w-4 text-yellow-500 animate-spin" />;
      case 'failed':
        return <AlertCircle className="h-4 w-4 text-red-500" />;
      default:
        return null;
    }
  };

  const getStatusBadge = (status: string) => {
    const baseClasses = "px-2 py-1 text-xs font-medium rounded-full";
    switch (status) {
      case 'completed':
        return `${baseClasses} bg-green-100 text-green-800`;
      case 'processing':
        return `${baseClasses} bg-yellow-100 text-yellow-800`;
      case 'failed':
        return `${baseClasses} bg-red-100 text-red-800`;
      default:
        return `${baseClasses} bg-gray-100 text-gray-800`;
    }
  };

  if (!user) {
    router.push('/login');
    return null;
  }

  if (loading) {
    return (
      <div id="dashboard-loading" className="min-h-screen flex items-center justify-center">
        <div id="dashboard-loading-spinner" className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  return (
    <div id="dashboard-container" className={`min-h-screen ${isDark ? 'bg-gray-900' : 'bg-gray-50'}`}>
      {/* Header */}
      <header id="dashboard-header" className={`${isDark ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} shadow-sm border-b`}>
        <div id="header-content" className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div id="header-inner" className="flex items-center justify-between h-16">
            <div id="header-left" className="flex items-center space-x-4">
              <button
                id="back-button"
                onClick={() => router.push('/')}
                className={`p-2 rounded-lg transition-colors ${isDark ? 'text-gray-300 hover:bg-gray-700' : 'text-gray-600 hover:bg-gray-100'}`}
              >
                ←
              </button>
              <h1 id="dashboard-title" className={`text-2xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                Dashboard
              </h1>
            </div>
            
            <div id="header-right" className="flex items-center space-x-4">
              <NotificationBell />
              <button
                id="new-analysis-button"
                onClick={() => router.push('/')}
                className="bg-primary-600 text-white px-4 py-2 rounded-lg hover:bg-primary-700 transition-colors text-sm font-medium"
              >
                New Analysis
              </button>
            </div>
          </div>
        </div>
      </header>

      <div id="dashboard-content" className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Stats Cards */}
        {stats && (
          <div id="stats-section" className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div id="total-documents-card" className={`p-6 rounded-lg shadow ${isDark ? 'bg-gray-800' : 'bg-white'}`}>
              <div id="total-documents-header" className="flex items-center justify-between">
                <div>
                  <p id="total-documents-label" className={`text-sm font-medium ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                    Total Documents
                  </p>
                  <p id="total-documents-value" className={`text-2xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                    {stats.totalDocuments}
                  </p>
                </div>
                <FileText className="h-8 w-8 text-primary-600" />
              </div>
            </div>

            <div id="total-clauses-card" className={`p-6 rounded-lg shadow ${isDark ? 'bg-gray-800' : 'bg-white'}`}>
              <div id="total-clauses-header" className="flex items-center justify-between">
                <div>
                  <p id="total-clauses-label" className={`text-sm font-medium ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                    Clauses Found
                  </p>
                  <p id="total-clauses-value" className={`text-2xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                    {stats.totalClauses}
                  </p>
                </div>
                <Search className="h-8 w-8 text-green-600" />
              </div>
            </div>

            <div id="avg-confidence-card" className={`p-6 rounded-lg shadow ${isDark ? 'bg-gray-800' : 'bg-white'}`}>
              <div id="avg-confidence-header" className="flex items-center justify-between">
                <div>
                  <p id="avg-confidence-label" className={`text-sm font-medium ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                    Avg Confidence
                  </p>
                  <p id="avg-confidence-value" className={`text-2xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                    {stats.avgConfidence}%
                  </p>
                </div>
                <TrendingUp className="h-8 w-8 text-blue-600" />
              </div>
            </div>

            <div id="processing-time-card" className={`p-6 rounded-lg shadow ${isDark ? 'bg-gray-800' : 'bg-white'}`}>
              <div id="processing-time-header" className="flex items-center justify-between">
                <div>
                  <p id="processing-time-label" className={`text-sm font-medium ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                    Avg Processing
                  </p>
                  <p id="processing-time-value" className={`text-2xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                    {stats.processingTime.toFixed(1)}s
                  </p>
                </div>
                <Clock className="h-8 w-8 text-orange-600" />
              </div>
            </div>
          </div>
        )}

        {/* Charts Section */}
        <div id="charts-section" className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <StatisticsChart 
            type="line"
            title="Analysis Trends"
            data={documents.filter(doc => doc.status === 'completed')}
          />
          <StatisticsChart 
            type="doughnut"
            title="Document Status"
            data={documents}
          />
        </div>

        {/* Documents Table */}
        <div id="documents-section" className={`rounded-lg shadow ${isDark ? 'bg-gray-800' : 'bg-white'}`}>
          <div id="documents-header" className="p-6 border-b border-gray-200">
            <div id="documents-title-row" className="flex items-center justify-between mb-4">
              <h2 id="documents-title" className={`text-lg font-semibold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                Document History
              </h2>
              <div id="documents-actions" className="flex items-center space-x-2">
                {selectedDocs.length > 0 && (
                  <>
                    <button
                      id="export-button"
                      onClick={() => setShowExportModal(true)}
                      className="flex items-center space-x-2 px-3 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors text-sm"
                    >
                      <Download className="h-4 w-4" />
                      <span>Export</span>
                    </button>
                    <button
                      id="delete-selected-button"
                      onClick={handleDeleteSelected}
                      className="flex items-center space-x-2 px-3 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors text-sm"
                    >
                      <Trash2 className="h-4 w-4" />
                      <span>Delete</span>
                    </button>
                  </>
                )}
              </div>
            </div>

            {/* Filters */}
            <div id="documents-filters" className="flex flex-wrap gap-4">
              <div id="search-filter" className="flex-1 min-w-64">
                <input
                  id="search-input"
                  type="text"
                  placeholder="Search documents..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className={`w-full px-3 py-2 rounded-lg border transition-colors ${
                    isDark 
                      ? 'bg-gray-700 border-gray-600 text-white placeholder-gray-400' 
                      : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500'
                  }`}
                />
              </div>
              
              <select
                id="status-filter"
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className={`px-3 py-2 rounded-lg border transition-colors ${
                  isDark 
                    ? 'bg-gray-700 border-gray-600 text-white' 
                    : 'bg-white border-gray-300 text-gray-900'
                }`}
              >
                <option value="all">All Status</option>
                <option value="completed">Completed</option>
                <option value="processing">Processing</option>
                <option value="failed">Failed</option>
              </select>

              <select
                id="date-filter"
                value={dateFilter}
                onChange={(e) => setDateFilter(e.target.value)}
                className={`px-3 py-2 rounded-lg border transition-colors ${
                  isDark 
                    ? 'bg-gray-700 border-gray-600 text-white' 
                    : 'bg-white border-gray-300 text-gray-900'
                }`}
              >
                <option value="all">All Time</option>
                <option value="today">Today</option>
                <option value="week">This Week</option>
                <option value="month">This Month</option>
              </select>

              <select
                id="sort-filter"
                value={`${sortBy}-${sortOrder}`}
                onChange={(e) => {
                  const [by, order] = e.target.value.split('-');
                  setSortBy(by as any);
                  setSortOrder(order as any);
                }}
                className={`px-3 py-2 rounded-lg border transition-colors ${
                  isDark 
                    ? 'bg-gray-700 border-gray-600 text-white' 
                    : 'bg-white border-gray-300 text-gray-900'
                }`}
              >
                <option value="date-desc">Newest First</option>
                <option value="date-asc">Oldest First</option>
                <option value="name-asc">Name A-Z</option>
                <option value="name-desc">Name Z-A</option>
                <option value="confidence-desc">High Confidence</option>
                <option value="confidence-asc">Low Confidence</option>
                <option value="clauses-desc">Most Clauses</option>
                <option value="clauses-asc">Fewest Clauses</option>
              </select>
            </div>
          </div>

          {/* Table */}
          <div id="documents-table-container" className="overflow-x-auto">
            <table id="documents-table" className="w-full">
              <thead id="documents-table-head" className={isDark ? 'bg-gray-700' : 'bg-gray-50'}>
                <tr>
                  <th id="select-header" className="px-6 py-3 text-left">
                    <input
                      id="select-all-checkbox"
                      type="checkbox"
                      checked={selectedDocs.length === filteredDocuments.length && filteredDocuments.length > 0}
                      onChange={handleSelectAll}
                      className="rounded border-gray-300"
                    />
                  </th>
                  <th id="status-header" className={`px-6 py-3 text-left text-xs font-medium uppercase tracking-wider ${isDark ? 'text-gray-300' : 'text-gray-500'}`}>
                    Status
                  </th>
                  <th id="document-header" className={`px-6 py-3 text-left text-xs font-medium uppercase tracking-wider ${isDark ? 'text-gray-300' : 'text-gray-500'}`}>
                    Document
                  </th>
                  <th id="uploaded-header" className={`px-6 py-3 text-left text-xs font-medium uppercase tracking-wider ${isDark ? 'text-gray-300' : 'text-gray-500'}`}>
                    Uploaded
                  </th>
                  <th id="clauses-header" className={`px-6 py-3 text-left text-xs font-medium uppercase tracking-wider ${isDark ? 'text-gray-300' : 'text-gray-500'}`}>
                    Clauses
                  </th>
                  <th id="confidence-header" className={`px-6 py-3 text-left text-xs font-medium uppercase tracking-wider ${isDark ? 'text-gray-300' : 'text-gray-500'}`}>
                    Confidence
                  </th>
                  <th id="actions-header" className={`px-6 py-3 text-left text-xs font-medium uppercase tracking-wider ${isDark ? 'text-gray-300' : 'text-gray-500'}`}>
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody id="documents-table-body" className={`divide-y ${isDark ? 'divide-gray-700' : 'divide-gray-200'}`}>
                {filteredDocuments.map((doc) => (
                  <tr key={doc.id} id={`document-row-${doc.id}`} className={isDark ? 'bg-gray-800' : 'bg-white'}>
                    <td id={`select-cell-${doc.id}`} className="px-6 py-4">
                      <input
                        type="checkbox"
                        checked={selectedDocs.includes(doc.id)}
                        onChange={() => handleDocumentSelect(doc.id)}
                        className="rounded border-gray-300"
                      />
                    </td>
                    <td id={`status-cell-${doc.id}`} className="px-6 py-4">
                      <div className="flex items-center space-x-2">
                        {getStatusIcon(doc.status)}
                        <span className={getStatusBadge(doc.status)}>
                          {doc.status}
                        </span>
                      </div>
                    </td>
                    <td id={`document-cell-${doc.id}`} className="px-6 py-4">
                      <div>
                        <div id={`filename-${doc.id}`} className={`text-sm font-medium ${isDark ? 'text-white' : 'text-gray-900'}`}>
                          {doc.fileName}
                        </div>
                        <div id={`filesize-${doc.id}`} className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-500'}`}>
                          {formatFileSize(doc.fileSize)}
                        </div>
                      </div>
                    </td>
                    <td id={`uploaded-cell-${doc.id}`} className={`px-6 py-4 text-sm ${isDark ? 'text-gray-300' : 'text-gray-500'}`}>
                      {format(new Date(doc.uploadDate), 'MMM dd, yyyy')}
                    </td>
                    <td id={`clauses-cell-${doc.id}`} className={`px-6 py-4 text-sm ${isDark ? 'text-white' : 'text-gray-900'}`}>
                      {doc.clausesFound}
                    </td>
                    <td id={`confidence-cell-${doc.id}`} className="px-6 py-4">
                      {doc.status === 'completed' ? (
                        <div className="flex items-center">
                          <div className="flex-1 bg-gray-200 rounded-full h-2 mr-2">
                            <div 
                              className="bg-primary-600 h-2 rounded-full" 
                              style={{ width: `${doc.avgConfidence}%` }}
                            ></div>
                          </div>
                          <span className={`text-sm font-medium ${isDark ? 'text-white' : 'text-gray-900'}`}>
                            {doc.avgConfidence}%
                          </span>
                        </div>
                      ) : (
                        <span className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-500'}`}>
                          —
                        </span>
                      )}
                    </td>
                    <td id={`actions-cell-${doc.id}`} className="px-6 py-4">
                      <div className="flex items-center space-x-2">
                        <button
                          id={`view-button-${doc.id}`}
                          onClick={() => router.push(`/analysis/${doc.id}`)}
                          className={`p-1 rounded transition-colors ${isDark ? 'text-gray-400 hover:text-gray-300' : 'text-gray-400 hover:text-gray-600'}`}
                        >
                          <Eye className="h-4 w-4" />
                        </button>
                        <button
                          id={`download-button-${doc.id}`}
                          onClick={() => handleExportData('pdf')}
                          className={`p-1 rounded transition-colors ${isDark ? 'text-gray-400 hover:text-gray-300' : 'text-gray-400 hover:text-gray-600'}`}
                        >
                          <Download className="h-4 w-4" />
                        </button>
                        <button
                          id={`delete-button-${doc.id}`}
                          onClick={() => {
                            setSelectedDocs([doc.id]);
                            handleDeleteSelected();
                          }}
                          className="p-1 rounded text-red-400 hover:text-red-600 transition-colors"
                        >
                          <Trash2 className="h-4 w-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>

            {filteredDocuments.length === 0 && (
              <div id="no-documents" className="text-center py-12">
                <FileText className={`h-12 w-12 mx-auto mb-4 ${isDark ? 'text-gray-600' : 'text-gray-400'}`} />
                <p id="no-documents-text" className={`text-lg font-medium ${isDark ? 'text-gray-400' : 'text-gray-500'}`}>
                  No documents found
                </p>
                <p id="no-documents-subtext" className={`mt-1 ${isDark ? 'text-gray-500' : 'text-gray-400'}`}>
                  Try adjusting your filters or upload a new document.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Export Modal */}
      {showExportModal && (
        <div id="export-modal-overlay" className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div id="export-modal" className={`max-w-md w-full mx-4 p-6 rounded-lg ${isDark ? 'bg-gray-800' : 'bg-white'}`}>
            <h3 id="export-modal-title" className={`text-lg font-semibold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>
              Export Documents
            </h3>
            <p id="export-modal-description" className={`mb-6 ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>
              Choose the format to export {selectedDocs.length || documents.length} document(s).
            </p>
            <div id="export-format-options" className="space-y-3 mb-6">
              <button
                id="export-csv-button"
                onClick={() => handleExportData('csv')}
                className="w-full p-3 text-left rounded-lg border transition-colors hover:bg-gray-50"
              >
                <strong>CSV</strong> - Spreadsheet format with document metadata
              </button>
              <button
                id="export-json-button"
                onClick={() => handleExportData('json')}
                className="w-full p-3 text-left rounded-lg border transition-colors hover:bg-gray-50"
              >
                <strong>JSON</strong> - Complete data including analysis results
              </button>
              <button
                id="export-pdf-button"
                onClick={() => handleExportData('pdf')}
                className="w-full p-3 text-left rounded-lg border transition-colors hover:bg-gray-50"
              >
                <strong>PDF</strong> - Formatted report with charts and summaries
              </button>
            </div>
            <div id="export-modal-actions" className="flex space-x-3">
              <button
                id="export-cancel-button"
                onClick={() => setShowExportModal(false)}
                className={`flex-1 px-4 py-2 rounded-lg border transition-colors ${
                  isDark 
                    ? 'border-gray-600 text-gray-300 hover:bg-gray-700' 
                    : 'border-gray-300 text-gray-700 hover:bg-gray-50'
                }`}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}