import React, { useState, useCallback } from 'react';
import { FileText, Zap, Shield, Search, ArrowRight, CheckCircle, Clock, AlertTriangle } from 'lucide-react';
import DocumentUploader from './components/DocumentUploader';
import ResultsDisplay from './components/ResultsDisplay';
import ClauseHighlighter from './components/ClauseHighlighter';

function App() {
  const [currentStep, setCurrentStep] = useState('upload'); // upload, processing, results
  const [uploadedFile, setUploadedFile] = useState(null);
  const [documentText, setDocumentText] = useState('');
  const [analysisResults, setAnalysisResults] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [selectedClause, setSelectedClause] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [viewMode, setViewMode] = useState('split'); // split, document, results

  // Mock analysis function - replace with actual API call
  const analyzeDocument = useCallback(async (text) => {
    // Simulate API processing time
    await new Promise(resolve => setTimeout(resolve, 3000));

    // Mock detected clauses - replace with actual analysis results
    const mockClauses = [
      {
        id: 'clause-1',
        type: 'mandatory',
        confidence: 87,
        startIndex: 245,
        endIndex: 425,
        text: 'Any dispute arising out of or relating to this contract shall be resolved through binding arbitration.',
        description: 'Standard mandatory arbitration clause with binding language'
      },
      {
        id: 'clause-2',
        type: 'optional',
        confidence: 72,
        startIndex: 567,
        endIndex: 689,
        text: 'Parties may elect to resolve disputes through arbitration as an alternative to litigation.',
        description: 'Optional arbitration clause allowing party choice'
      },
      {
        id: 'clause-3',
        type: 'binding',
        confidence: 94,
        startIndex: 890,
        endIndex: 1120,
        text: 'The arbitrator\'s decision shall be final and binding upon all parties, with no right of appeal.',
        description: 'Strong binding language with explicit appeal waiver'
      }
    ];

    return {
      summary: {
        totalClauses: mockClauses.length,
        highConfidenceClauses: mockClauses.filter(c => c.confidence >= 80).length,
        avgConfidence: Math.round(mockClauses.reduce((sum, c) => sum + c.confidence, 0) / mockClauses.length),
        documentComplexity: 'Medium'
      },
      clauses: mockClauses,
      analysis: {
        enforceability: 82,
        clarity: 78,
        completeness: 85
      },
      recommendations: [
        'Consider adding specific arbitration rules reference (e.g., AAA, JAMS)',
        'Clarify the jurisdiction and governing law for arbitration proceedings',
        'Add cost allocation provisions for arbitration expenses',
        'Consider including carve-outs for certain types of disputes (e.g., injunctive relief)'
      ]
    };
  }, []);

  const handleFileUpload = useCallback(async (file) => {
    setUploadedFile(file);
    setCurrentStep('processing');
    setIsProcessing(true);

    try {
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
    } catch (error) {
      console.error('Error processing document:', error);
      alert('Error processing document. Please try again.');
      setCurrentStep('upload');
    } finally {
      setIsProcessing(false);
    }
  }, [analyzeDocument]);

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

  return (
    <div id="app-container" className="min-h-screen bg-gray-50">
      {/* Header */}
      <header id="app-header" className="bg-white shadow-sm border-b border-gray-200">
        <div id="header-content" className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div id="header-inner" className="flex items-center justify-between h-16">
            <div id="header-title-section" className="flex items-center space-x-3">
              <div id="header-icon-container" className="p-2 bg-primary-100 rounded-lg">
                <Shield id="header-icon" className="h-6 w-6 text-primary-600" />
              </div>
              <div id="header-title-text">
                <h1 id="app-title" className="text-xl font-bold text-gray-900">
                  Arbitration Clause Detector
                </h1>
                <p id="app-subtitle" className="text-sm text-gray-600">
                  AI-powered legal document analysis
                </p>
              </div>
            </div>
            <div id="header-actions">
              {currentStep === 'results' && (
                <button
                  id="reset-analysis-button"
                  onClick={handleReset}
                  className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors text-sm font-medium"
                >
                  New Analysis
                </button>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Progress Indicator */}
      <div id="progress-container" className="bg-white border-b border-gray-200">
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
                          'text-gray-500'}
                      `}
                    >
                      {step.label}
                    </span>
                  </div>
                  {index < 2 && (
                    <ArrowRight 
                      id={`progress-arrow-${index}`} 
                      className="h-4 w-4 text-gray-400 mx-4" 
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
              <h2 id="upload-title" className="text-3xl font-bold text-gray-900 mb-4">
                Upload Your Legal Document
              </h2>
              <p id="upload-description" className="text-lg text-gray-600">
                Our AI system will analyze your document to detect and classify arbitration clauses with high accuracy.
              </p>
            </div>
            <DocumentUploader 
              onFileUpload={handleFileUpload}
              isProcessing={isProcessing}
            />
            
            {/* Features */}
            <div id="features-section" className="mt-12 grid grid-cols-1 md:grid-cols-3 gap-6">
              <div id="feature-ai-powered" className="text-center p-6 bg-white rounded-lg shadow-sm border border-gray-200">
                <div id="feature-ai-icon-container" className="inline-flex items-center justify-center w-12 h-12 bg-primary-100 rounded-lg mb-4">
                  <Zap id="feature-ai-icon" className="h-6 w-6 text-primary-600" />
                </div>
                <h3 id="feature-ai-title" className="text-lg font-semibold text-gray-900 mb-2">
                  AI-Powered Analysis
                </h3>
                <p id="feature-ai-description" className="text-gray-600">
                  Advanced natural language processing to identify arbitration clauses with precision.
                </p>
              </div>
              <div id="feature-comprehensive" className="text-center p-6 bg-white rounded-lg shadow-sm border border-gray-200">
                <div id="feature-comprehensive-icon-container" className="inline-flex items-center justify-center w-12 h-12 bg-success-100 rounded-lg mb-4">
                  <FileText id="feature-comprehensive-icon" className="h-6 w-6 text-success-600" />
                </div>
                <h3 id="feature-comprehensive-title" className="text-lg font-semibold text-gray-900 mb-2">
                  Comprehensive Detection
                </h3>
                <p id="feature-comprehensive-description" className="text-gray-600">
                  Detects mandatory, optional, binding, and non-binding arbitration clauses.
                </p>
              </div>
              <div id="feature-export" className="text-center p-6 bg-white rounded-lg shadow-sm border border-gray-200">
                <div id="feature-export-icon-container" className="inline-flex items-center justify-center w-12 h-12 bg-warning-100 rounded-lg mb-4">
                  <Search id="feature-export-icon" className="h-6 w-6 text-warning-600" />
                </div>
                <h3 id="feature-export-title" className="text-lg font-semibold text-gray-900 mb-2">
                  Detailed Analysis
                </h3>
                <p id="feature-export-description" className="text-gray-600">
                  Get confidence scores, clause classifications, and legal recommendations.
                </p>
              </div>
            </div>
          </div>
        )}

        {currentStep === 'processing' && (
          <div id="processing-step-content" className="max-w-2xl mx-auto text-center">
            <div id="processing-spinner-container" className="mb-6">
              <div id="processing-spinner" className="animate-spin rounded-full h-16 w-16 border-b-2 border-primary-600 mx-auto"></div>
            </div>
            <h2 id="processing-title" className="text-2xl font-bold text-gray-900 mb-4">
              Analyzing Your Document
            </h2>
            <p id="processing-description" className="text-gray-600 mb-8">
              Our AI is carefully examining your document for arbitration clauses. This may take a few moments.
            </p>
            <div id="processing-status" className="bg-primary-50 border border-primary-200 rounded-lg p-4">
              <div id="processing-status-content" className="flex items-center justify-center space-x-2">
                <div id="processing-status-spinner" className="animate-pulse w-2 h-2 bg-primary-600 rounded-full"></div>
                <span id="processing-status-text" className="text-primary-700 font-medium">
                  Processing {uploadedFile?.name}...
                </span>
              </div>
            </div>
          </div>
        )}

        {currentStep === 'results' && (
          <div id="results-step-content" className="space-y-6">
            {/* View Mode Controls */}
            <div id="view-mode-controls" className="flex items-center justify-between">
              <h2 id="results-page-title" className="text-2xl font-bold text-gray-900">
                Analysis Complete
              </h2>
              <div id="view-mode-buttons" className="flex items-center space-x-2">
                <button
                  id="split-view-button"
                  onClick={() => setViewMode('split')}
                  className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                    viewMode === 'split' 
                      ? 'bg-primary-600 text-white' 
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  Split View
                </button>
                <button
                  id="document-view-button"
                  onClick={() => setViewMode('document')}
                  className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                    viewMode === 'document' 
                      ? 'bg-primary-600 text-white' 
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  Document
                </button>
                <button
                  id="results-view-button"
                  onClick={() => setViewMode('results')}
                  className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                    viewMode === 'results' 
                      ? 'bg-primary-600 text-white' 
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  Results
                </button>
              </div>
            </div>

            {/* Results Content */}
            <div id="results-layout" className={`grid gap-6 ${viewMode === 'split' ? 'lg:grid-cols-2' : 'grid-cols-1'}`}>
              {(viewMode === 'split' || viewMode === 'document') && (
                <div id="document-viewer-section" className="bg-white rounded-lg shadow-lg overflow-hidden">
                  <ClauseHighlighter
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
                  <ResultsDisplay
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
              <div id="selected-clause-details" className="bg-white rounded-lg shadow-lg p-6 border-l-4 border-primary-500">
                <div id="selected-clause-header" className="flex items-start justify-between mb-4">
                  <div id="selected-clause-info">
                    <h3 id="selected-clause-title" className="text-lg font-semibold text-gray-900">
                      Selected Clause Details
                    </h3>
                    <div id="selected-clause-type-badge" className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-primary-100 text-primary-800 mt-2">
                      {selectedClause.type}
                    </div>
                  </div>
                  <button
                    id="close-clause-details"
                    onClick={() => setSelectedClause(null)}
                    className="text-gray-400 hover:text-gray-600 transition-colors"
                  >
                    <AlertTriangle className="h-5 w-5" />
                  </button>
                </div>
                <div id="selected-clause-content" className="space-y-4">
                  <div id="selected-clause-text-section">
                    <h4 id="selected-clause-text-label" className="font-medium text-gray-900 mb-2">Clause Text:</h4>
                    <p id="selected-clause-text-content" className="text-gray-700 bg-gray-50 p-4 rounded-lg border-l-4 border-primary-400">
                      "{selectedClause.text}"
                    </p>
                  </div>
                  <div id="selected-clause-metadata" className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div id="selected-clause-confidence-section">
                      <h4 id="selected-clause-confidence-label" className="font-medium text-gray-900 mb-2">Confidence Score:</h4>
                      <p id="selected-clause-confidence-value" className="text-2xl font-bold text-primary-600">
                        {selectedClause.confidence}%
                      </p>
                    </div>
                    <div id="selected-clause-position-section">
                      <h4 id="selected-clause-position-label" className="font-medium text-gray-900 mb-2">Position:</h4>
                      <p id="selected-clause-position-value" className="text-gray-700">
                        Characters {selectedClause.startIndex} - {selectedClause.endIndex}
                      </p>
                    </div>
                  </div>
                  {selectedClause.description && (
                    <div id="selected-clause-description-section">
                      <h4 id="selected-clause-description-label" className="font-medium text-gray-900 mb-2">Analysis:</h4>
                      <p id="selected-clause-description-content" className="text-gray-700">
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
      <footer id="app-footer" className="bg-white border-t border-gray-200 mt-12">
        <div id="footer-content" className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div id="footer-inner" className="text-center text-sm text-gray-600">
            <p id="footer-text">
              Arbitration Clause Detector - AI-powered legal document analysis for arbitration clause detection and classification.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;