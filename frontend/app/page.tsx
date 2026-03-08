'use client';

import { useState, useEffect } from 'react';
import { cn } from '@/lib/utils';
import {
  FileText,
  Upload,
  Shield,
  Zap,
  CheckCircle,
  AlertTriangle,
  Search,
  BarChart3,
  Brain,
  Database,
  ChevronDown,
  ChevronUp,
  ArrowRight,
  Clock,
  Eye,
  TrendingUp,
  Scale,
  FileSearch,
  Layers,
  Target,
  Activity,
  Lock,
} from 'lucide-react';

// Mock analyzed documents
const MOCK_DOCUMENTS = [
  {
    id: 'doc-1',
    name: 'SaaS Master Service Agreement.pdf',
    pages: 47,
    uploadedAt: '2 min ago',
    status: 'analyzed' as const,
    clausesFound: 3,
    riskLevel: 'high' as const,
    confidence: 0.94,
    clauses: [
      {
        id: 'c1',
        text: 'Any dispute, controversy, or claim arising out of or relating to this Agreement, or the breach, termination, or invalidity thereof, shall be settled by binding arbitration administered by the American Arbitration Association ("AAA") in accordance with its Commercial Arbitration Rules.',
        type: 'Mandatory Binding Arbitration',
        severity: 'high' as const,
        confidence: 0.96,
        section: 'Section 14.2 - Dispute Resolution',
        impact: 'Waives right to jury trial; limits discovery; binding decision with limited appeal rights',
      },
      {
        id: 'c2',
        text: 'THE PARTIES HEREBY WAIVE ANY RIGHT TO A TRIAL BY JURY in any action or proceeding arising out of or related to this Agreement. Each party irrevocably waives any and all right to trial by jury in any legal proceeding.',
        type: 'Jury Trial Waiver',
        severity: 'high' as const,
        confidence: 0.98,
        section: 'Section 14.5 - Jury Waiver',
        impact: 'Explicit waiver of constitutional right to jury trial',
      },
      {
        id: 'c3',
        text: 'Any arbitration under this Agreement shall be conducted on an individual basis and not as a class, consolidated, or representative action. The arbitrator may not consolidate more than one person\'s claims.',
        type: 'Class Action Waiver',
        severity: 'medium' as const,
        confidence: 0.91,
        section: 'Section 14.3 - Class Action Waiver',
        impact: 'Prevents joining class action lawsuits; must pursue claims individually',
      },
    ],
  },
  {
    id: 'doc-2',
    name: 'Enterprise License Agreement - Q4.pdf',
    pages: 23,
    uploadedAt: '15 min ago',
    status: 'analyzed' as const,
    clausesFound: 1,
    riskLevel: 'medium' as const,
    confidence: 0.87,
    clauses: [
      {
        id: 'c4',
        text: 'In the event of any dispute arising under this Agreement, the parties agree to first attempt mediation before a mutually agreed upon mediator. If mediation is unsuccessful within 60 days, either party may initiate binding arbitration under JAMS rules.',
        type: 'Escalation Arbitration Clause',
        severity: 'medium' as const,
        confidence: 0.89,
        section: 'Section 9.1 - Dispute Resolution',
        impact: 'Requires mediation first, then binding arbitration; more favorable than direct arbitration',
      },
    ],
  },
  {
    id: 'doc-3',
    name: 'Vendor Supply Contract 2026.docx',
    pages: 31,
    uploadedAt: '1 hour ago',
    status: 'analyzed' as const,
    clausesFound: 0,
    riskLevel: 'low' as const,
    confidence: 0.92,
    clauses: [],
  },
];

const PIPELINE_STEPS = [
  { name: 'Document Ingestion', detail: 'PDF/DOCX parsing, OCR', icon: Upload, status: 'complete', time: '0.3s' },
  { name: 'Text Chunking', detail: '512-token sliding window', icon: Layers, status: 'complete', time: '0.1s' },
  { name: 'Embedding Generation', detail: 'BERT legal-base model', icon: Brain, status: 'complete', time: '0.8s' },
  { name: 'Vector Retrieval', detail: 'ChromaDB similarity search', icon: Database, status: 'complete', time: '0.2s' },
  { name: 'Clause Classification', detail: 'Fine-tuned transformer', icon: Target, status: 'complete', time: '0.4s' },
  { name: 'Confidence Scoring', detail: 'Ensemble validation', icon: Shield, status: 'complete', time: '0.1s' },
];

export default function HomePage() {
  const [selectedDoc, setSelectedDoc] = useState(MOCK_DOCUMENTS[0]);
  const [expandedClause, setExpandedClause] = useState<string | null>('c1');
  const [showUpload, setShowUpload] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [pipelineStep, setPipelineStep] = useState(6);

  const handleDemoAnalysis = () => {
    setAnalyzing(true);
    setPipelineStep(0);
    const interval = setInterval(() => {
      setPipelineStep(prev => {
        if (prev >= 5) {
          clearInterval(interval);
          setTimeout(() => setAnalyzing(false), 500);
          return 6;
        }
        return prev + 1;
      });
    }, 600);
  };

  const severityColors = {
    high: 'text-red-400 bg-red-500/10 border-red-500/20',
    medium: 'text-amber-400 bg-amber-500/10 border-amber-500/20',
    low: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
  };

  const riskColors = {
    high: 'text-red-400',
    medium: 'text-amber-400',
    low: 'text-emerald-400',
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
      {/* Hero Section */}
      <div className="relative overflow-hidden bg-gradient-to-br from-slate-900 via-violet-950/50 to-slate-900 rounded-2xl p-8 md:p-12 border border-slate-800">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_50%,rgba(139,92,246,0.1),transparent_50%)]" />
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_70%_50%,rgba(99,102,241,0.08),transparent_50%)]" />
        <div className="relative">
          <div className="flex items-center gap-2 mb-4">
            <div className="flex items-center gap-2 text-violet-400 text-sm font-semibold tracking-wide uppercase bg-violet-500/10 px-3 py-1 rounded-full border border-violet-500/20">
              <Zap className="w-3.5 h-3.5" />
              Live Platform
            </div>
          </div>
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-4 tracking-tight">
            AI-Powered{' '}
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-violet-400 to-indigo-400">
              Arbitration Clause
            </span>
            <br />Detection
          </h2>
          <p className="text-slate-400 text-lg max-w-2xl mb-8">
            Upload legal documents and instantly identify arbitration clauses, jury waivers, and
            dispute resolution provisions using our RAG-powered NLP pipeline with 85%+ accuracy.
          </p>
          <div className="flex flex-wrap gap-6 text-sm">
            <div className="flex items-center gap-2 text-slate-300">
              <CheckCircle className="w-4 h-4 text-emerald-400" />
              85%+ Detection Accuracy
            </div>
            <div className="flex items-center gap-2 text-slate-300">
              <CheckCircle className="w-4 h-4 text-emerald-400" />
              Sub-2s Analysis Time
            </div>
            <div className="flex items-center gap-2 text-slate-300">
              <CheckCircle className="w-4 h-4 text-emerald-400" />
              Multi-format Support
            </div>
            <div className="flex items-center gap-2 text-slate-300">
              <CheckCircle className="w-4 h-4 text-emerald-400" />
              BERT + ChromaDB RAG
            </div>
          </div>
        </div>
      </div>

      {/* Metrics Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: 'Documents Analyzed', value: '12,847', change: '+342 this week', icon: FileText, color: 'violet' },
          { label: 'Clauses Detected', value: '4,291', change: '85.3% accuracy', icon: Search, color: 'indigo' },
          { label: 'Avg. Analysis Time', value: '1.9s', change: '-0.3s vs last month', icon: Clock, color: 'emerald' },
          { label: 'Risk Alerts', value: '847', change: '23% high severity', icon: AlertTriangle, color: 'amber' },
        ].map((metric) => {
          const Icon = metric.icon;
          const colorMap: Record<string, string> = {
            violet: 'bg-violet-500/10 text-violet-400',
            indigo: 'bg-indigo-500/10 text-indigo-400',
            emerald: 'bg-emerald-500/10 text-emerald-400',
            amber: 'bg-amber-500/10 text-amber-400',
          };
          return (
            <div key={metric.label} className="bg-slate-900 rounded-xl p-5 border border-slate-800 hover:border-slate-700 transition-colors">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs text-slate-400">{metric.label}</span>
                <div className={cn('p-2 rounded-lg', colorMap[metric.color])}>
                  <Icon className="w-4 h-4" />
                </div>
              </div>
              <p className="text-2xl font-bold text-white">{metric.value}</p>
              <p className="text-xs text-slate-500 mt-1">{metric.change}</p>
            </div>
          );
        })}
      </div>

      {/* Main Content: Document Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Document List */}
        <div className="bg-slate-900 rounded-xl border border-slate-800 overflow-hidden">
          <div className="p-4 border-b border-slate-800 flex items-center justify-between">
            <h3 className="font-semibold text-white flex items-center gap-2">
              <FileSearch className="w-4 h-4 text-violet-400" />
              Analyzed Documents
            </h3>
            <button
              onClick={() => setShowUpload(!showUpload)}
              className="text-xs px-3 py-1.5 bg-violet-500/10 text-violet-400 rounded-lg border border-violet-500/20 hover:bg-violet-500/20 transition-colors"
            >
              <Upload className="w-3 h-3 inline mr-1" />
              Upload
            </button>
          </div>

          {showUpload && (
            <div className="p-4 border-b border-slate-800 bg-slate-800/30">
              <div className="border-2 border-dashed border-slate-700 rounded-lg p-6 text-center hover:border-violet-500/50 transition-colors cursor-pointer">
                <Upload className="w-8 h-8 text-slate-500 mx-auto mb-2" />
                <p className="text-sm text-slate-400">Drop PDF, DOCX, or TXT files here</p>
                <p className="text-xs text-slate-500 mt-1">Max 50MB per file</p>
                <button
                  onClick={handleDemoAnalysis}
                  className="mt-3 text-xs px-4 py-2 bg-violet-600 text-white rounded-lg hover:bg-violet-500 transition-colors"
                >
                  Run Demo Analysis
                </button>
              </div>
            </div>
          )}

          <div className="divide-y divide-slate-800">
            {MOCK_DOCUMENTS.map((doc) => (
              <button
                key={doc.id}
                onClick={() => setSelectedDoc(doc)}
                className={cn(
                  'w-full p-4 text-left hover:bg-slate-800/50 transition-colors',
                  selectedDoc.id === doc.id && 'bg-violet-500/5 border-l-2 border-violet-500'
                )}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-white truncate">{doc.name}</p>
                    <div className="flex items-center gap-3 mt-1">
                      <span className="text-xs text-slate-500">{doc.pages} pages</span>
                      <span className="text-xs text-slate-500">{doc.uploadedAt}</span>
                    </div>
                  </div>
                  <div className="ml-3 flex flex-col items-end gap-1">
                    {doc.clausesFound > 0 ? (
                      <span className={cn('text-xs px-2 py-0.5 rounded-full border', severityColors[doc.riskLevel])}>
                        {doc.clausesFound} clause{doc.clausesFound !== 1 ? 's' : ''}
                      </span>
                    ) : (
                      <span className="text-xs px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                        Clear
                      </span>
                    )}
                    <span className="text-[10px] text-slate-500">{(doc.confidence * 100).toFixed(0)}% conf.</span>
                  </div>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Right: Analysis Results */}
        <div className="lg:col-span-2 space-y-6">
          {/* Selected Document Header */}
          <div className="bg-slate-900 rounded-xl p-5 border border-slate-800">
            <div className="flex items-start justify-between mb-4">
              <div>
                <h3 className="text-lg font-semibold text-white">{selectedDoc.name}</h3>
                <div className="flex items-center gap-4 mt-1 text-sm text-slate-400">
                  <span>{selectedDoc.pages} pages</span>
                  <span>Uploaded {selectedDoc.uploadedAt}</span>
                  <span className="flex items-center gap-1">
                    <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />
                    Analysis complete
                  </span>
                </div>
              </div>
              <div className="text-right">
                <div className="text-sm text-slate-400">Overall Confidence</div>
                <div className="text-2xl font-bold text-white">{(selectedDoc.confidence * 100).toFixed(1)}%</div>
              </div>
            </div>

            {/* Risk Assessment Bar */}
            <div className="flex items-center gap-4 p-3 bg-slate-800/50 rounded-lg">
              <div className="flex items-center gap-2">
                <Shield className={cn('w-5 h-5', riskColors[selectedDoc.riskLevel])} />
                <span className="text-sm font-medium text-white">Risk Level:</span>
                <span className={cn('text-sm font-semibold capitalize', riskColors[selectedDoc.riskLevel])}>
                  {selectedDoc.riskLevel}
                </span>
              </div>
              <div className="flex-1" />
              <div className="flex items-center gap-2 text-xs text-slate-400">
                <span>{selectedDoc.clausesFound} arbitration clause{selectedDoc.clausesFound !== 1 ? 's' : ''} detected</span>
              </div>
            </div>
          </div>

          {/* Detected Clauses */}
          {selectedDoc.clauses.length > 0 ? (
            <div className="space-y-3">
              <h4 className="text-sm font-semibold text-slate-400 uppercase tracking-wider px-1">Detected Clauses</h4>
              {selectedDoc.clauses.map((clause) => (
                <div
                  key={clause.id}
                  className={cn(
                    'bg-slate-900 rounded-xl border transition-colors',
                    expandedClause === clause.id ? 'border-violet-500/30' : 'border-slate-800'
                  )}
                >
                  <button
                    onClick={() => setExpandedClause(expandedClause === clause.id ? null : clause.id)}
                    className="w-full p-4 text-left"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className={cn('px-2 py-0.5 rounded text-xs font-medium border', severityColors[clause.severity])}>
                          {clause.severity.toUpperCase()}
                        </div>
                        <span className="text-sm font-medium text-white">{clause.type}</span>
                      </div>
                      <div className="flex items-center gap-3">
                        <span className="text-xs text-slate-400">{(clause.confidence * 100).toFixed(0)}% confidence</span>
                        {expandedClause === clause.id ? (
                          <ChevronUp className="w-4 h-4 text-slate-400" />
                        ) : (
                          <ChevronDown className="w-4 h-4 text-slate-400" />
                        )}
                      </div>
                    </div>
                    <p className="text-xs text-slate-500 mt-1">{clause.section}</p>
                  </button>

                  {expandedClause === clause.id && (
                    <div className="px-4 pb-4 space-y-4">
                      {/* Highlighted Clause Text */}
                      <div className="relative">
                        <div className="absolute left-0 top-0 bottom-0 w-1 bg-gradient-to-b from-violet-500 to-indigo-500 rounded-full" />
                        <div className="pl-4 py-3 bg-violet-500/5 rounded-r-lg border border-violet-500/10">
                          <p className="text-sm text-slate-300 leading-relaxed italic">
                            &ldquo;{clause.text}&rdquo;
                          </p>
                        </div>
                      </div>

                      {/* Impact Assessment */}
                      <div className="flex items-start gap-3 p-3 bg-slate-800/30 rounded-lg">
                        <AlertTriangle className={cn('w-4 h-4 mt-0.5 flex-shrink-0', riskColors[clause.severity])} />
                        <div>
                          <p className="text-xs font-semibold text-slate-300 mb-1">Impact Assessment</p>
                          <p className="text-xs text-slate-400">{clause.impact}</p>
                        </div>
                      </div>

                      {/* Confidence Breakdown */}
                      <div className="grid grid-cols-3 gap-3">
                        <div className="p-2.5 bg-slate-800/30 rounded-lg text-center">
                          <p className="text-lg font-bold text-violet-400">{(clause.confidence * 100).toFixed(0)}%</p>
                          <p className="text-[10px] text-slate-500 uppercase tracking-wider">Semantic Match</p>
                        </div>
                        <div className="p-2.5 bg-slate-800/30 rounded-lg text-center">
                          <p className="text-lg font-bold text-indigo-400">{(clause.confidence * 97).toFixed(0)}%</p>
                          <p className="text-[10px] text-slate-500 uppercase tracking-wider">Pattern Score</p>
                        </div>
                        <div className="p-2.5 bg-slate-800/30 rounded-lg text-center">
                          <p className="text-lg font-bold text-emerald-400">{(clause.confidence * 99).toFixed(0)}%</p>
                          <p className="text-[10px] text-slate-500 uppercase tracking-wider">Ensemble Avg</p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="bg-slate-900 rounded-xl border border-slate-800 p-8 text-center">
              <CheckCircle className="w-12 h-12 text-emerald-400 mx-auto mb-3" />
              <h4 className="text-lg font-semibold text-white mb-2">No Arbitration Clauses Found</h4>
              <p className="text-sm text-slate-400 max-w-md mx-auto">
                This document was analyzed with {(selectedDoc.confidence * 100).toFixed(1)}% confidence and no
                arbitration clauses, jury waivers, or dispute resolution provisions were detected.
              </p>
            </div>
          )}
        </div>
      </div>

      {/* RAG Pipeline Visualization */}
      <div className="bg-slate-900 rounded-xl p-6 border border-slate-800">
        <div className="flex items-center justify-between mb-6">
          <h3 className="text-lg font-semibold text-white flex items-center gap-2">
            <Activity className="w-5 h-5 text-violet-400" />
            RAG Pipeline Architecture
          </h3>
          <button
            onClick={handleDemoAnalysis}
            disabled={analyzing}
            className="text-xs px-4 py-2 bg-violet-600 text-white rounded-lg hover:bg-violet-500 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {analyzing ? 'Processing...' : 'Run Pipeline Demo'}
          </button>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
          {PIPELINE_STEPS.map((step, index) => {
            const Icon = step.icon;
            const isActive = analyzing && pipelineStep === index;
            const isComplete = !analyzing ? true : pipelineStep > index;
            return (
              <div
                key={step.name}
                className={cn(
                  'relative p-4 rounded-xl border text-center transition-all duration-300',
                  isActive
                    ? 'bg-violet-500/10 border-violet-500/30 scale-105 shadow-lg shadow-violet-500/10'
                    : isComplete
                    ? 'bg-slate-800/30 border-emerald-500/20'
                    : 'bg-slate-800/10 border-slate-800 opacity-40'
                )}
              >
                <div className={cn(
                  'w-10 h-10 rounded-lg mx-auto mb-2 flex items-center justify-center transition-colors',
                  isActive ? 'bg-violet-500/20 text-violet-400' :
                  isComplete ? 'bg-emerald-500/10 text-emerald-400' : 'bg-slate-800 text-slate-500'
                )}>
                  {isComplete && !isActive ? (
                    <CheckCircle className="w-5 h-5" />
                  ) : (
                    <Icon className={cn('w-5 h-5', isActive && 'animate-pulse')} />
                  )}
                </div>
                <p className="text-xs font-medium text-white mb-0.5">{step.name}</p>
                <p className="text-[10px] text-slate-500">{step.detail}</p>
                {isComplete && !analyzing && (
                  <p className="text-[10px] text-emerald-400 mt-1">{step.time}</p>
                )}
                {index < PIPELINE_STEPS.length - 1 && (
                  <div className="hidden lg:block absolute -right-2 top-1/2 -translate-y-1/2 z-10">
                    <ArrowRight className={cn('w-3 h-3', isComplete ? 'text-emerald-500/50' : 'text-slate-700')} />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Technology Stack & Architecture */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-slate-900 rounded-xl p-6 border border-slate-800">
          <div className="w-10 h-10 bg-violet-500/10 rounded-lg flex items-center justify-center mb-4">
            <Brain className="w-5 h-5 text-violet-400" />
          </div>
          <h4 className="font-semibold text-white mb-2">NLP Engine</h4>
          <p className="text-sm text-slate-400 mb-4">
            Fine-tuned BERT model trained on 50K+ legal documents with legal-domain embeddings for
            precise clause boundary detection.
          </p>
          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <span className="text-slate-500">Model</span>
              <span className="text-slate-300">legal-bert-base</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-slate-500">F1 Score</span>
              <span className="text-emerald-400">0.873</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-slate-500">Precision</span>
              <span className="text-emerald-400">0.891</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-slate-500">Recall</span>
              <span className="text-emerald-400">0.856</span>
            </div>
          </div>
        </div>

        <div className="bg-slate-900 rounded-xl p-6 border border-slate-800">
          <div className="w-10 h-10 bg-indigo-500/10 rounded-lg flex items-center justify-center mb-4">
            <Database className="w-5 h-5 text-indigo-400" />
          </div>
          <h4 className="font-semibold text-white mb-2">RAG Pipeline</h4>
          <p className="text-sm text-slate-400 mb-4">
            Retrieval-Augmented Generation using ChromaDB vector store with 512-token sliding window
            chunking and semantic similarity search.
          </p>
          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <span className="text-slate-500">Vector Store</span>
              <span className="text-slate-300">ChromaDB</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-slate-500">Chunk Size</span>
              <span className="text-slate-300">512 tokens</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-slate-500">Embedding Dim</span>
              <span className="text-slate-300">768</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-slate-500">Retrieval Top-K</span>
              <span className="text-slate-300">5</span>
            </div>
          </div>
        </div>

        <div className="bg-slate-900 rounded-xl p-6 border border-slate-800">
          <div className="w-10 h-10 bg-emerald-500/10 rounded-lg flex items-center justify-center mb-4">
            <Lock className="w-5 h-5 text-emerald-400" />
          </div>
          <h4 className="font-semibold text-white mb-2">Enterprise Security</h4>
          <p className="text-sm text-slate-400 mb-4">
            SOC 2 compliant infrastructure with end-to-end encryption, RBAC, and complete audit
            logging for legal document handling.
          </p>
          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <span className="text-slate-500">Encryption</span>
              <span className="text-slate-300">AES-256 at rest</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-slate-500">Auth</span>
              <span className="text-slate-300">JWT + RBAC</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-slate-500">Compliance</span>
              <span className="text-emerald-400">SOC 2 Type II</span>
            </div>
            <div className="flex justify-between text-xs">
              <span className="text-slate-500">Audit Log</span>
              <span className="text-emerald-400">Enabled</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
