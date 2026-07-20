import React, { useState, useCallback } from 'react';
import { useRouter } from 'next/router';
import { FileText, Zap, Shield, Search, ArrowRight, CheckCircle, Clock, AlertTriangle, Menu, Bell, User, Settings, LogOut } from 'lucide-react';
import { toast, Toaster } from 'react-hot-toast';
import { useAuth } from '../src/contexts/AuthContext';
import { useTheme } from '../src/contexts/ThemeContext';
import FileUploader from '../src/components/FileUploader';
import PDFViewer from '../src/components/PDFViewer';
import AnalysisResults from '../src/components/AnalysisResults';
import NotificationBell from '../src/components/NotificationBell';
import LoadingSkeleton from '../src/components/LoadingSkeleton';

export default function HomePage() {
  const router = useRouter();
  const { user, logout } = useAuth();
  const { isDark, toggleTheme } = useTheme();
  
  const [currentStep, setCurrentStep] = useState('upload'); // upload, processing, results
  const [uploadedFile, setUploadedFile] = useState(null);
  const [documentText, setDocumentText] = useState('');
  const [analysisResults, setAnalysisResults] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedClause, setSelectedClause] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [viewMode, setViewMode] = useState('split'); // split, document, results
  const [showUserMenu, setShowUserMenu] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  // Mock analysis function - replace with actual API call
  const analyzeDocument = useCallback(async (text) => {
    try {
      // Simulate API processing time with progress updates
      for (let i = 0; i <= 100; i += 10) {
        setUploadProgress(i);
        await new Promise(resolve => setTimeout(resolve, 200));
      }

      // Mock detected clauses - replace with actual analysis results
      const mockClauses = [
        {
          id: 'clause-1',
          type: 'mandatory',
          confidence: 87,
          startIndex: 245,
          endIndex: 425,
          text: 'Any dispute arising out of or relating to this contract shall be resolved through binding arbitration.',
          description: 'Standard mandatory arbitration clause with binding language',
          severity: 'high',
          recommendation: 'Consider adding specific arbitration rules reference'
        },
        {
          id: 'clause-2',
          type: 'optional',
          confidence: 72,
          startIndex: 567,
          endIndex: 689,
          text: 'Parties may elect to resolve disputes through arbitration as an alternative to litigation.',
          description: 'Optional arbitration clause allowing party choice',
          severity: 'medium',
          recommendation: 'Clarify the jurisdiction and governing law'
        },
        {
          id: 'clause-3',
          type: 'binding',
          confidence: 94,
          startIndex: 890,
          endIndex: 1120,
          text: 'The arbitrator\'s decision shall be final and binding upon all parties, with no right of appeal.',
          description: 'Strong binding language with explicit appeal waiver',
          severity: 'high',
          recommendation: 'Consider including carve-outs for certain types of disputes'
        }
      ];

      return {
        summary: {
          totalClauses: mockClauses.length,
          highConfidenceClauses: mockClauses.filter(c => c.confidence >= 80).length,
          avgConfidence: Math.round(mockClauses.reduce((sum, c) => sum + c.confidence, 0) / mockClauses.length),
          documentComplexity: 'Medium',
          processingTime: '2.3s',
          documentLength: text.length
        },
        clauses: mockClauses,
        analysis: {
          enforceability: 82,
          clarity: 78,
          completeness: 85,
          compliance: 91
        },
        recommendations: [
          'Consider adding specific arbitration rules reference (e.g., AAA, JAMS)',
          'Clarify the jurisdiction and governing law for arbitration proceedings',
          'Add cost allocation provisions for arbitration expenses',
          'Consider including carve-outs for certain types of disputes (e.g., injunctive relief)'
        ],
        riskFactors: [
          'Broad arbitration scope may limit legal remedies',
          'No appeal rights specified for arbitration decisions',
          'Missing cost allocation provisions'
        ]
      };
    } catch (error) {
      toast.error('Analysis failed. Please try again.');
      throw error;
    }
  }, []);

  const handleFileUpload = useCallback(async (file) => {
    setUploadedFile(file);
    setCurrentStep('processing');
    setIsProcessing(true);
    setUploadProgress(0);

    try {
      toast.loading('Processing document...', { id: 'processing' });
      
      // Mock document text extraction - replace with actual file parsing
      const mockText = `
        SAMPLE LEGAL DOCUMENT

        This agreement (the "Agreement") is entered into between the parties for the purpose of establishing terms and conditions for services.

        DISPUTE RESOLUTION CLAUSE:
        Any dispute arising out of or relating to this contract shall be resolved through binding arbitration. The arbitration shall be conducted under the rules of the American Arbitration Association.

        ALTERNATIVE RESOLUTION:
        Parties may elect to resolve disputes through arbitration as an alternative to litigation. Such election must be made in writing within thirty (30) days of the dispute arising.

        BINDING NATURE:
        The arbitrator's decision shall be final and binding upon all parties, with no right of appeal. The parties hereby waive their right to a jury trial for any disputes covered by this arbitration clause.

        ADDITIONAL TERMS:
        This agreement shall be governed by the laws of the State of California. Any arbitration proceedings shall take place in Los Angeles County, California.
      `;

      setDocumentText(mockText);
      
      // Analyze the document
      const results = await analyzeDocument(mockText);
      setAnalysisResults(results);
      setCurrentStep('results');
      
      toast.success('Analysis complete!', { id: 'processing' });
      
      // Save to user's document history (mock)
      if (user) {
        const analysisRecord = {
          id: Date.now().toString(),
          fileName: file.name,
          uploadDate: new Date().toISOString(),
          results: results,
          status: 'completed'
        };
        // Would save to backend here
      }
      
    } catch (error) {
      console.error('Error processing document:', error);
      toast.error('Error processing document. Please try again.', { id: 'processing' });
      setCurrentStep('upload');
    } finally {
      setIsProcessing(false);
      setUploadProgress(0);
    }
  }, [analyzeDocument, user]);

  const handleClauseSelect = useCallback((clause) => {
    setSelectedClause(clause);
  }, []);

  const handleReset = () => {
    setCurrentStep('upload');
    setUploadedFile(null);
    setDocumentText('');
    setAnalysisResults(null);
    setSelectedClause(null);
    setSearchTerm('');
    setIsProcessing(false);
    setUploadProgress(0);
  };

  const handleLogout = async () => {
    try {
      await logout();
      toast.success('Logged out successfully');
      router.push('/login');
    } catch (error) {
      toast.error('Failed to logout');
    }
  };

  const getStepIcon = (step) => {
    switch (step) {
      case 'upload':
        return currentStep === 'upload' ? Clock : CheckCircle;
      case 'processing':
        return currentStep === 'processing' ? Clock : (currentStep === 'results' ? CheckCircle : Clock);
      case 'results':
        return currentStep === 'results' ? CheckCircle : Clock;
      default:
        return Clock;
    }
  };

  const getStepStatus = (step) => {
    if (currentStep === step) return 'current';
    if ((step === 'processing' && currentStep === 'results') || 
        (step === 'upload' && (currentStep === 'processing' || currentStep === 'results'))) {
      return 'completed';
    }
    return 'upcoming';
  };

  if (!user) {
    return (
      <div id="auth-required" className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div id="auth-card" className="bg-white p-8 rounded-lg shadow-lg text-center">
          <Shield id="auth-icon" className="h-12 w-12 text-primary-600 mx-auto mb-4" />
          <h2 id="auth-title" className="text-2xl font-bold text-gray-900 mb-4">
            Authentication Required
          </h2>
          <p id="auth-description" className="text-gray-600 mb-6">
            Please log in to access the document analysis features.
          </p>
          <button
            id="auth-login-button"
            onClick={() => router.push('/login')}
            className="bg-primary-600 text-white px-6 py-2 rounded-lg hover:bg-primary-700 transition-colors"
          >
            Go to Login
          </button>
        </div>
      </div>
    );
  }

  return (
    <div id="app-container" className={`min-h-screen ${isDark ? 'bg-gray-900' : 'bg-gray-50'}`}>
      <Toaster 
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: isDark ? '#374151' : '#ffffff',
            color: isDark ? '#ffffff' : '#000000',
          },
        }}
      />
      
      {/* Header */}
      <header id="app-header" className={`${isDark ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} shadow-sm border-b`}>
        <div id="header-content" className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div id="header-inner" className="flex items-center justify-between h-16">
            <div id="header-title-section" className="flex items-center space-x-3">
              <div id="header-icon-container" className="p-2 bg-primary-100 rounded-lg">
                <Shield id="header-icon" className="h-6 w-6 text-primary-600" />
              </div>
              <div id="header-title-text">
                <h1 id="app-title" className={`text-xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                  ArbiScan Pro
                </h1>
                <p id="app-subtitle" className={`text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
                  AI-powered legal document analysis
                </p>
              </div>
            </div>
            
            <div id="header-actions" className="flex items-center space-x-4">
              <button
                id="theme-toggle"
                onClick={toggleTheme}
                className={`p-2 rounded-lg transition-colors ${isDark ? 'bg-gray-700 text-gray-300 hover:bg-gray-600' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'}`}
              >
                {isDark ? '☀️' : '🌙'}
              </button>
              
              <NotificationBell />
              
              <button
                id="dashboard-button"
                onClick={() => router.push('/dashboard')}
                className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors text-sm font-medium"
              >
                Dashboard
              </button>
              
              {currentStep === 'results' && (
                <button
                  id="reset-analysis-button"
                  onClick={handleReset}
                  className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors text-sm font-medium"
                >
                  New Analysis
                </button>
              )}
              
              {/* User Menu */}
              <div id="user-menu-container" className="relative">
                <button
                  id="user-menu-button"
                  onClick={() => setShowUserMenu(!showUserMenu)}
                  className={`flex items-center space-x-2 p-2 rounded-lg transition-colors ${isDark ? 'bg-gray-700 text-gray-300 hover:bg-gray-600' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'}`}
                >
                  <User id="user-icon" className="h-5 w-5" />
                  <span id="user-name" className="text-sm font-medium">{user.name}</span>
                </button>
                
                {showUserMenu && (
                  <div id="user-menu-dropdown" className={`absolute right-0 mt-2 w-48 ${isDark ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} border rounded-lg shadow-lg z-50`}>
                    <div id="user-menu-items" className="py-1">
                      <button
                        id="settings-menu-item"
                        onClick={() => router.push('/settings')}
                        className={`flex items-center space-x-2 w-full px-4 py-2 text-sm transition-colors ${isDark ? 'text-gray-300 hover:bg-gray-700' : 'text-gray-700 hover:bg-gray-100'}`}
                      >
                        <Settings className="h-4 w-4" />
                        <span>Settings</span>
                      </button>
                      <button
                        id="logout-menu-item"
                        onClick={handleLogout}
                        className={`flex items-center space-x-2 w-full px-4 py-2 text-sm transition-colors ${isDark ? 'text-gray-300 hover:bg-gray-700' : 'text-gray-700 hover:bg-gray-100'}`}
                      >
                        <LogOut className="h-4 w-4" />
                        <span>Logout</span>
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Progress Indicator */}
      <div id="progress-container" className={`${isDark ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} border-b`}>
        <div id="progress-content" className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div id="progress-steps" className="flex items-center justify-center space-x-8">
            {[
              { id: 'upload', label: 'Upload Document', icon: FileText },
              { id: 'processing', label: 'AI Analysis', icon: Zap },
              { id: 'results', label: 'View Results', icon: Search }
            ].map((step, index) => {
              const Icon = getStepIcon(step.id);
              const status = getStepStatus(step.id);
              
              return (
                <div key={step.id} id={`progress-step-${step.id}`} className="flex items-center">
                  <div id={`step-indicator-${step.id}`} className="flex items-center space-x-2">
                    <div 
                      id={`step-icon-container-${step.id}`}
                      className={`
                        p-2 rounded-full border-2 transition-colors
                        ${status === 'completed' ? 'bg-success-500 border-success-500 text-white' : 
                          status === 'current' ? 'bg-primary-500 border-primary-500 text-white' : 
                          'bg-gray-100 border-gray-300 text-gray-400'}
                      `}
                    >
                      <Icon id={`step-icon-${step.id}`} className="h-4 w-4" />
                    </div>
                    <span 
                      id={`step-label-${step.id}`}
                      className={`
                        text-sm font-medium
                        ${status === 'completed' ? 'text-success-700' : 
                          status === 'current' ? 'text-primary-700' : 
                          isDark ? 'text-gray-400' : 'text-gray-500'}
                      `}
                    >
                      {step.label}
                    </span>
                  </div>
                  {index < 2 && (
                    <ArrowRight 
                      id={`progress-arrow-${index}`} 
                      className={`h-4 w-4 mx-4 ${isDark ? 'text-gray-600' : 'text-gray-400'}`} 
                    />
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <main id="main-content" className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {currentStep === 'upload' && (
          <div id="upload-step-content" className="max-w-4xl mx-auto">
            <div id="upload-intro" className="text-center mb-8">
              <h2 id="upload-title" className={`text-3xl font-bold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>
                Upload Your Legal Document
              </h2>
              <p id="upload-description" className={`text-lg ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>
                Our AI system will analyze your document to detect and classify arbitration clauses with high accuracy.
              </p>
            </div>
            
            <FileUploader 
              onFileUpload={handleFileUpload}
              isProcessing={isProcessing}
              progress={uploadProgress}
            />
            
            {/* Features */}
            <div id="features-section" className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6">
              <div id="feature-ai-powered" className={`text-center p-6 rounded-lg shadow-sm border ${isDark ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}`}>
                <div id="feature-ai-icon-container" className="inline-flex items-center justify-center w-12 h-12 bg-primary-100 rounded-lg mb-4">
                  <Zap id="feature-ai-icon" className="h-6 w-6 text-primary-600" />
                </div>
                <h3 id="feature-ai-title" className={`text-lg font-semibold mb-2 ${isDark ? 'text-white' : 'text-gray-900'}`}>
                  AI-Powered Analysis
                </h3>
                <p id="feature-ai-description" className={isDark ? 'text-gray-300' : 'text-gray-600'}>
                  Advanced natural language processing to identify arbitration clauses with precision.
                </p>
              </div>
              <div id="feature-comprehensive" className={`text-center p-6 rounded-lg shadow-sm border ${isDark ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}`}>
                <div id="feature-comprehensive-icon-container" className="inline-flex items-center justify-center w-12 h-12 bg-success-100 rounded-lg mb-4">
                  <FileText id="feature-comprehensive-icon" className="h-6 w-6 text-success-600" />
                </div>
                <h3 id="feature-comprehensive-title" className={`text-lg font-semibold mb-2 ${isDark ? 'text-white' : 'text-gray-900'}`}>
                  Comprehensive Detection
                </h3>
                <p id="feature-comprehensive-description" className={isDark ? 'text-gray-300' : 'text-gray-600'}>
                  Detects mandatory, optional, binding, and non-binding arbitration clauses.
                </p>
              </div>
              <div id="feature-export" className={`text-center p-6 rounded-lg shadow-sm border ${isDark ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}`}>
                <div id="feature-export-icon-container" className="inline-flex items-center justify-center w-12 h-12 bg-warning-100 rounded-lg mb-4">
                  <Search id="feature-export-icon" className="h-6 w-6 text-warning-600" />
                </div>
                <h3 id="feature-export-title" className={`text-lg font-semibold mb-2 ${isDark ? 'text-white' : 'text-gray-900'}`}>
                  Detailed Analysis
                </h3>
                <p id="feature-export-description" className={isDark ? 'text-gray-300' : 'text-gray-600'}>
                  Get confidence scores, clause classifications, and legal recommendations.
                </p>
              </div>
            </div>
          </div>
        )}

        {currentStep === 'processing' && (
          <div id="processing-step-content" className="max-w-2xl mx-auto text-center">
            <LoadingSkeleton type="analysis" />
            <div id="processing-spinner-container" className="mb-6">
              <div id="processing-spinner" className="animate-spin rounded-full h-16 w-16 border-b-2 border-primary-600 mx-auto"></div>
            </div>
            <h2 id="processing-title" className={`text-2xl font-bold mb-4 ${isDark ? 'text-white' : 'text-gray-900'}`}>
              Analyzing Your Document
            </h2>
            <p id="processing-description" className={`mb-8 ${isDark ? 'text-gray-300' : 'text-gray-600'}`}>
              Our AI is carefully examining your document for arbitration clauses. This may take a few moments.
            </p>
            
            {uploadProgress > 0 && (
              <div id="progress-bar-container" className="w-full bg-gray-200 rounded-full h-2 mb-4">
                <div 
                  id="progress-bar-fill"
                  className="bg-primary-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                ></div>
              </div>
            )}
            
            <div id="processing-status" className="bg-primary-50 border border-primary-200 rounded-lg p-4">
              <div id="processing-status-content" className="flex items-center justify-center space-x-2">
                <div id="processing-status-spinner" className="animate-pulse w-2 h-2 bg-primary-600 rounded-full"></div>
                <span id="processing-status-text" className="text-primary-700 font-medium">
                  Processing {uploadedFile?.name}... ({uploadProgress}%)
                </span>
              </div>
            </div>
          </div>
        )}

        {currentStep === 'results' && (
          <div id="results-step-content" className="space-y-6">
            {/* View Mode Controls */}
            <div id="view-mode-controls" className="flex items-center justify-between">
              <h2 id="results-page-title" className={`text-2xl font-bold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                Analysis Complete
              </h2>
              <div id="view-mode-buttons" className="flex items-center space-x-2">
                {['split', 'document', 'results'].map((mode) => (
                  <button
                    key={mode}
                    id={`${mode}-view-button`}
                    onClick={() => setViewMode(mode)}
                    className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                      viewMode === mode 
                        ? 'bg-primary-600 text-white' 
                        : isDark ? 'bg-gray-700 text-gray-300 hover:bg-gray-600' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    {mode.charAt(0).toUpperCase() + mode.slice(1)} View
                  </button>
                ))}
              </div>
            </div>

            {/* Results Content */}
            <div id="results-layout" className={`grid gap-6 ${viewMode === 'split' ? 'lg:grid-cols-2' : 'grid-cols-1'}`}>
              {(viewMode === 'split' || viewMode === 'document') && (
                <div id="document-viewer-section" className={`rounded-lg shadow-lg overflow-hidden ${isDark ? 'bg-gray-800' : 'bg-white'}`}>
                  <PDFViewer
                    documentText={documentText}
                    detectedClauses={analysisResults?.clauses || []}
                    onClauseSelect={handleClauseSelect}
                    searchTerm={searchTerm}
                    onSearchTermChange={setSearchTerm}
                  />
                </div>
              )}
              
              {(viewMode === 'split' || viewMode === 'results') && (
                <div id="results-viewer-section">
                  <AnalysisResults
                    results={analysisResults}
                    documentName={uploadedFile?.name}
                    onClauseSelect={handleClauseSelect}
                    isLoading={isProcessing}
                  />
                </div>
              )}
            </div>

            {/* Selected Clause Details */}
            {selectedClause && (
              <div id="selected-clause-details" className={`rounded-lg shadow-lg p-6 border-l-4 border-primary-500 ${isDark ? 'bg-gray-800' : 'bg-white'}`}>
                <div id="selected-clause-header" className="flex items-start justify-between mb-4">
                  <div id="selected-clause-info">
                    <h3 id="selected-clause-title" className={`text-lg font-semibold ${isDark ? 'text-white' : 'text-gray-900'}`}>
                      Selected Clause Details
                    </h3>
                    <div id="selected-clause-type-badge" className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-primary-100 text-primary-800 mt-2">
                      {selectedClause.type}
                    </div>
                  </div>
                  <button
                    id="close-clause-details"
                    onClick={() => setSelectedClause(null)}
                    className={`transition-colors ${isDark ? 'text-gray-400 hover:text-gray-300' : 'text-gray-400 hover:text-gray-600'}`}
                  >
                    <AlertTriangle className="h-5 w-5" />
                  </button>
                </div>
                <div id="selected-clause-content" className="space-y-4">
                  <div id="selected-clause-text-section">
                    <h4 id="selected-clause-text-label" className={`font-medium mb-2 ${isDark ? 'text-white' : 'text-gray-900'}`}>Clause Text:</h4>
                    <p id="selected-clause-text-content" className={`p-4 rounded-lg border-l-4 border-primary-400 ${isDark ? 'text-gray-300 bg-gray-700' : 'text-gray-700 bg-gray-50'}`}>
                      "{selectedClause.text}"
                    </p>
                  </div>
                  <div id="selected-clause-metadata" className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div id="selected-clause-confidence-section">
                      <h4 id="selected-clause-confidence-label" className={`font-medium mb-2 ${isDark ? 'text-white' : 'text-gray-900'}`}>Confidence Score:</h4>
                      <p id="selected-clause-confidence-value" className="text-2xl font-bold text-primary-600">
                        {selectedClause.confidence}%
                      </p>
                    </div>
                    <div id="selected-clause-position-section">
                      <h4 id="selected-clause-position-label" className={`font-medium mb-2 ${isDark ? 'text-white' : 'text-gray-900'}`}>Position:</h4>
                      <p id="selected-clause-position-value" className={isDark ? 'text-gray-300' : 'text-gray-700'}>
                        Characters {selectedClause.startIndex} - {selectedClause.endIndex}
                      </p>
                    </div>
                  </div>
                  {selectedClause.description && (
                    <div id="selected-clause-description-section">
                      <h4 id="selected-clause-description-label" className={`font-medium mb-2 ${isDark ? 'text-white' : 'text-gray-900'}`}>Analysis:</h4>
                      <p id="selected-clause-description-content" className={isDark ? 'text-gray-300' : 'text-gray-700'}>
                        {selectedClause.description}
                      </p>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer id="app-footer" className={`border-t mt-12 ${isDark ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}`}>
        <div id="footer-content" className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div id="footer-inner" className={`text-center text-sm ${isDark ? 'text-gray-400' : 'text-gray-600'}`}>
            <p id="footer-text">
              ArbiScan Pro - AI-powered legal document analysis for arbitration clause detection and classification.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}