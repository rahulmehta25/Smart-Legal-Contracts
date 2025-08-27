/**
 * Demo Analysis Workflow Component
 * Interactive demonstration of document analysis process
 */

import React, { useState, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Upload,
  FileText,
  Brain,
  Search,
  CheckCircle,
  AlertTriangle,
  Download,
  Eye,
  Share2,
  MessageSquare,
  BarChart3,
  Globe,
  Mic
} from 'lucide-react';

interface AnalysisStep {
  id: string;
  title: string;
  description: string;
  icon: React.ComponentType<any>;
  status: 'pending' | 'active' | 'completed' | 'error';
  duration?: number;
  progress?: number;
}

interface DetectedClause {
  id: string;
  type: 'mandatory' | 'optional' | 'binding' | 'non-binding';
  confidence: number;
  text: string;
  startPosition: number;
  endPosition: number;
  severity: 'high' | 'medium' | 'low';
  recommendations: string[];
}

interface AnalysisResult {
  documentId: string;
  documentName: string;
  language: string;
  processingTime: number;
  clauses: DetectedClause[];
  summary: {
    totalClauses: number;
    highConfidenceClauses: number;
    averageConfidence: number;
    riskLevel: 'high' | 'medium' | 'low';
  };
  recommendations: string[];
}

const initialSteps: AnalysisStep[] = [
  {
    id: 'upload',
    title: 'Document Upload',
    description: 'Upload your legal document for analysis',
    icon: Upload,
    status: 'pending'
  },
  {
    id: 'extraction',
    title: 'Text Extraction',
    description: 'Extract and process text from document',
    icon: FileText,
    status: 'pending',
    duration: 2000
  },
  {
    id: 'analysis',
    title: 'AI Analysis',
    description: 'Apply machine learning models to detect clauses',
    icon: Brain,
    status: 'pending',
    duration: 3000
  },
  {
    id: 'classification',
    title: 'Clause Classification',
    description: 'Classify and score detected arbitration clauses',
    icon: Search,
    status: 'pending',
    duration: 1500
  },
  {
    id: 'validation',
    title: 'Result Validation',
    description: 'Validate results and generate recommendations',
    icon: CheckCircle,
    status: 'pending',
    duration: 1000
  }
];

const sampleDocuments = [
  {
    id: 'tos-mandatory',
    name: 'Terms of Service - Mandatory Arbitration.pdf',
    type: 'Terms of Service',
    size: '247 KB',
    language: 'English',
    expectedClauses: 5,
    difficulty: 'Easy'
  },
  {
    id: 'license-complex',
    name: 'Enterprise License Agreement.docx',
    type: 'Software License',
    size: '456 KB',
    language: 'English',
    expectedClauses: 3,
    difficulty: 'Hard'
  },
  {
    id: 'german-contract',
    name: 'Software Lizenzvertrag.pdf',
    type: 'Software License',
    size: '312 KB',
    language: 'German',
    expectedClauses: 2,
    difficulty: 'Medium'
  },
  {
    id: 'privacy-policy',
    name: 'Privacy Policy - No Arbitration.pdf',
    type: 'Privacy Policy',
    size: '189 KB',
    language: 'English',
    expectedClauses: 0,
    difficulty: 'Easy'
  }
];

const mockAnalysisResult: AnalysisResult = {
  documentId: 'tos-mandatory',
  documentName: 'Terms of Service - Mandatory Arbitration.pdf',
  language: 'English',
  processingTime: 7500,
  clauses: [
    {
      id: 'clause-1',
      type: 'mandatory',
      confidence: 94,
      text: 'You and the Company agree that any dispute, claim, or controversy arising out of or relating to these Terms will be settled by binding arbitration.',
      startPosition: 245,
      endPosition: 425,
      severity: 'high',
      recommendations: [
        'This is a strong mandatory arbitration clause',
        'Consider reviewing AAA arbitration rules referenced',
        'Note the broad scope covering all disputes'
      ]
    },
    {
      id: 'clause-2',
      type: 'binding',
      confidence: 97,
      text: 'The arbitrator\'s decision shall be final and binding upon all parties, with no right of appeal.',
      startPosition: 890,
      endPosition: 1020,
      severity: 'high',
      recommendations: [
        'Waives right to appeal arbitration decisions',
        'Creates finality in dispute resolution',
        'May limit judicial review options'
      ]
    },
    {
      id: 'clause-3',
      type: 'mandatory',
      confidence: 89,
      text: 'YOU AND THE COMPANY HEREBY WAIVE ANY CONSTITUTIONAL AND STATUTORY RIGHTS TO SUE IN COURT AND HAVE A TRIAL IN FRONT OF A JUDGE OR A JURY.',
      startPosition: 1250,
      endPosition: 1420,
      severity: 'high',
      recommendations: [
        'Explicit jury trial waiver',
        'Constitutional rights waiver language',
        'Consider legal implications carefully'
      ]
    }
  ],
  summary: {
    totalClauses: 3,
    highConfidenceClauses: 3,
    averageConfidence: 93.3,
    riskLevel: 'high'
  },
  recommendations: [
    'This document contains strong mandatory arbitration provisions',
    'Users waive significant legal rights including jury trials',
    'Consider providing opt-out mechanisms where legally required',
    'Ensure compliance with local jurisdiction requirements'
  ]
};

interface DemoAnalysisWorkflowProps {
  onComplete?: (result: AnalysisResult) => void;
  className?: string;
}

export const DemoAnalysisWorkflow: React.FC<DemoAnalysisWorkflowProps> = ({
  onComplete,
  className = ''
}) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [steps, setSteps] = useState<AnalysisStep[]>(initialSteps);
  const [selectedDocument, setSelectedDocument] = useState(sampleDocuments[0]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isCompleted, setIsCompleted] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [showResults, setShowResults] = useState(false);

  // Start the analysis process
  const startAnalysis = useCallback(async () => {
    setIsProcessing(true);
    setCurrentStep(0);
    setShowResults(false);
    
    // Reset all steps
    setSteps(initialSteps.map(step => ({ ...step, status: 'pending', progress: 0 })));

    // Process each step
    for (let i = 0; i < initialSteps.length; i++) {
      const step = initialSteps[i];
      
      // Mark current step as active
      setSteps(prev => prev.map((s, index) => ({
        ...s,
        status: index === i ? 'active' : index < i ? 'completed' : 'pending'
      })));
      
      setCurrentStep(i);

      // Simulate processing time with progress
      if (step.duration) {
        const progressSteps = 20;
        const progressInterval = step.duration / progressSteps;
        
        for (let p = 0; p <= progressSteps; p++) {
          await new Promise(resolve => setTimeout(resolve, progressInterval));
          const progress = (p / progressSteps) * 100;
          
          setSteps(prev => prev.map((s, index) => 
            index === i ? { ...s, progress } : s
          ));
        }
      } else {
        await new Promise(resolve => setTimeout(resolve, 500));
      }

      // Mark step as completed
      setSteps(prev => prev.map((s, index) => 
        index === i ? { ...s, status: 'completed', progress: 100 } : s
      ));
    }

    // Show results
    setAnalysisResult(mockAnalysisResult);
    setIsCompleted(true);
    setIsProcessing(false);
    setShowResults(true);
    
    onComplete?.(mockAnalysisResult);
  }, [onComplete]);

  const resetDemo = () => {
    setCurrentStep(0);
    setSteps(initialSteps.map(step => ({ ...step, status: 'pending', progress: 0 })));
    setIsProcessing(false);
    setIsCompleted(false);
    setAnalysisResult(null);
    setShowResults(false);
  };

  const getStepColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-green-600 bg-green-100 border-green-200';
      case 'active': return 'text-blue-600 bg-blue-100 border-blue-200';
      case 'error': return 'text-red-600 bg-red-100 border-red-200';
      default: return 'text-gray-400 bg-gray-50 border-gray-200';
    }
  };

  return (
    <div id="demo-analysis-workflow" className={`max-w-6xl mx-auto p-6 ${className}`}>
      {/* Document Selection */}
      {!isProcessing && !showResults && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          id="document-selection"
          className="bg-white rounded-lg shadow-lg p-6 mb-8"
        >
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            Choose a Sample Document
          </h2>
          <p className="text-gray-600 mb-6">
            Select a document to analyze, or upload your own to see the AI in action.
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            {sampleDocuments.map(doc => (
              <div
                key={doc.id}
                id={`sample-doc-${doc.id}`}
                className={`border-2 rounded-lg p-4 cursor-pointer transition-all duration-200 ${
                  selectedDocument.id === doc.id
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
                onClick={() => setSelectedDocument(doc)}
              >
                <div className="flex items-start justify-between mb-2">
                  <h3 className="font-semibold text-gray-900">{doc.name}</h3>
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    doc.difficulty === 'Easy' ? 'bg-green-100 text-green-600' :
                    doc.difficulty === 'Medium' ? 'bg-yellow-100 text-yellow-600' :
                    'bg-red-100 text-red-600'
                  }`}>
                    {doc.difficulty}
                  </span>
                </div>
                <div className="text-sm text-gray-600 space-y-1">
                  <div>Type: {doc.type}</div>
                  <div>Size: {doc.size}</div>
                  <div>Language: {doc.language}</div>
                  <div>Expected Clauses: {doc.expectedClauses}</div>
                </div>
              </div>
            ))}
          </div>

          <div className="flex justify-between items-center">
            <button
              id="upload-custom-button"
              className="flex items-center px-4 py-2 border-2 border-dashed border-gray-300 rounded-lg text-gray-600 hover:border-gray-400 hover:text-gray-700 transition-colors"
            >
              <Upload className="h-4 w-4 mr-2" />
              Upload Custom Document
            </button>
            
            <button
              id="start-analysis-button"
              onClick={startAnalysis}
              className="flex items-center px-6 py-3 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 transition-colors"
            >
              <Brain className="h-5 w-5 mr-2" />
              Start AI Analysis
            </button>
          </div>
        </motion.div>
      )}

      {/* Processing Steps */}
      {(isProcessing || isCompleted) && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          id="processing-steps"
          className="bg-white rounded-lg shadow-lg p-6 mb-8"
        >
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-gray-900">
              Analyzing: {selectedDocument.name}
            </h2>
            {isCompleted && (
              <button
                id="reset-demo-button"
                onClick={resetDemo}
                className="px-4 py-2 text-blue-600 border border-blue-600 rounded-lg hover:bg-blue-50 transition-colors"
              >
                Try Another Document
              </button>
            )}
          </div>

          <div className="space-y-4">
            {steps.map((step, index) => {
              const Icon = step.icon;
              const isActive = step.status === 'active';
              const isCompleted = step.status === 'completed';
              
              return (
                <div
                  key={step.id}
                  id={`step-${step.id}`}
                  className={`flex items-center p-4 rounded-lg border-2 transition-all duration-300 ${getStepColor(step.status)}`}
                >
                  <div className="flex-shrink-0">
                    <div className={`w-12 h-12 rounded-full flex items-center justify-center ${
                      isCompleted ? 'bg-green-500 text-white' :
                      isActive ? 'bg-blue-500 text-white' :
                      'bg-gray-300 text-gray-600'
                    }`}>
                      {isCompleted ? (
                        <CheckCircle className="h-6 w-6" />
                      ) : (
                        <Icon className="h-6 w-6" />
                      )}
                    </div>
                  </div>
                  
                  <div className="flex-1 ml-4">
                    <h3 className="font-semibold">{step.title}</h3>
                    <p className="text-sm opacity-75">{step.description}</p>
                    
                    {isActive && step.progress !== undefined && (
                      <div className="mt-2">
                        <div className="bg-white/30 rounded-full h-2">
                          <div
                            className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                            style={{ width: `${step.progress}%` }}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                  
                  {isActive && (
                    <div className="flex-shrink-0 ml-4">
                      <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-current" />
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </motion.div>
      )}

      {/* Analysis Results */}
      <AnimatePresence>
        {showResults && analysisResult && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            id="analysis-results"
            className="space-y-6"
          >
            {/* Summary Card */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-2xl font-bold text-gray-900">Analysis Complete</h2>
                <div className="flex items-center space-x-2">
                  <span className="text-sm text-gray-500">
                    Processed in {(analysisResult.processingTime / 1000).toFixed(1)}s
                  </span>
                  <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                    analysisResult.summary.riskLevel === 'high' ? 'bg-red-100 text-red-600' :
                    analysisResult.summary.riskLevel === 'medium' ? 'bg-yellow-100 text-yellow-600' :
                    'bg-green-100 text-green-600'
                  }`}>
                    {analysisResult.summary.riskLevel.toUpperCase()} RISK
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
                <div className="bg-blue-50 p-4 rounded-lg text-center">
                  <div className="text-2xl font-bold text-blue-600">
                    {analysisResult.summary.totalClauses}
                  </div>
                  <div className="text-sm text-blue-700">Clauses Found</div>
                </div>
                <div className="bg-green-50 p-4 rounded-lg text-center">
                  <div className="text-2xl font-bold text-green-600">
                    {analysisResult.summary.highConfidenceClauses}
                  </div>
                  <div className="text-sm text-green-700">High Confidence</div>
                </div>
                <div className="bg-purple-50 p-4 rounded-lg text-center">
                  <div className="text-2xl font-bold text-purple-600">
                    {analysisResult.summary.averageConfidence.toFixed(1)}%
                  </div>
                  <div className="text-sm text-purple-700">Avg Confidence</div>
                </div>
                <div className="bg-orange-50 p-4 rounded-lg text-center">
                  <div className="text-2xl font-bold text-orange-600">
                    {analysisResult.language}
                  </div>
                  <div className="text-sm text-orange-700">Language</div>
                </div>
              </div>
            </div>

            {/* Detected Clauses */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-xl font-bold text-gray-900 mb-4">Detected Arbitration Clauses</h3>
              <div className="space-y-4">
                {analysisResult.clauses.map((clause, index) => (
                  <div
                    key={clause.id}
                    id={`detected-clause-${clause.id}`}
                    className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow"
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex items-center space-x-3">
                        <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                          clause.type === 'mandatory' ? 'bg-red-100 text-red-600' :
                          clause.type === 'binding' ? 'bg-orange-100 text-orange-600' :
                          clause.type === 'optional' ? 'bg-blue-100 text-blue-600' :
                          'bg-gray-100 text-gray-600'
                        }`}>
                          {clause.type.toUpperCase()}
                        </span>
                        <span className={`px-2 py-1 rounded text-xs font-medium ${
                          clause.severity === 'high' ? 'bg-red-100 text-red-600' :
                          clause.severity === 'medium' ? 'bg-yellow-100 text-yellow-600' :
                          'bg-green-100 text-green-600'
                        }`}>
                          {clause.severity} severity
                        </span>
                      </div>
                      <div className="text-right">
                        <div className="text-lg font-bold text-gray-900">{clause.confidence}%</div>
                        <div className="text-sm text-gray-500">confidence</div>
                      </div>
                    </div>
                    
                    <div className="bg-gray-50 p-3 rounded border-l-4 border-blue-400 mb-3">
                      <p className="text-gray-800 italic">"{clause.text}"</p>
                      <div className="text-xs text-gray-500 mt-2">
                        Position: {clause.startPosition} - {clause.endPosition}
                      </div>
                    </div>
                    
                    <div>
                      <h4 className="font-medium text-gray-900 mb-2">Recommendations:</h4>
                      <ul className="text-sm text-gray-600 space-y-1">
                        {clause.recommendations.map((rec, recIndex) => (
                          <li key={recIndex} className="flex items-start">
                            <span className="text-blue-500 mr-2">•</span>
                            {rec}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Action Buttons */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-xl font-bold text-gray-900 mb-4">Next Steps</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <button className="flex items-center justify-center px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
                  <Download className="h-5 w-5 mr-2" />
                  Export Report
                </button>
                <button className="flex items-center justify-center px-4 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors">
                  <Share2 className="h-5 w-5 mr-2" />
                  Share Results
                </button>
                <button className="flex items-center justify-center px-4 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors">
                  <BarChart3 className="h-5 w-5 mr-2" />
                  View Analytics
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default DemoAnalysisWorkflow;