import React, { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { 
  HelpCircle, 
  Info, 
  BookOpen, 
  Video, 
  MessageCircle, 
  ChevronRight,
  Search,
  X,
  Lightbulb,
  FileText,
  Shield,
  Zap
} from 'lucide-react';

interface HelpTopic {
  id: string;
  title: string;
  description: string;
  category: 'getting-started' | 'features' | 'troubleshooting' | 'advanced';
  icon: React.ReactNode;
  content: {
    text: string;
    steps?: string[];
    tips?: string[];
    relatedTopics?: string[];
  };
}

interface InteractiveTooltipProps {
  title: string;
  content: string;
  trigger: React.ReactNode;
  side?: 'top' | 'bottom' | 'left' | 'right';
  className?: string;
}

interface HelpSystemProps {
  className?: string;
}

// Interactive Tooltip Component
export const InteractiveTooltip: React.FC<InteractiveTooltipProps> = ({
  title,
  content,
  trigger,
  side = 'top',
  className = ''
}) => {
  return (
    <TooltipProvider delayDuration={300}>
      <Tooltip>
        <TooltipTrigger asChild className={className}>
          {trigger}
        </TooltipTrigger>
        <TooltipContent side={side} className="max-w-xs">
          <div className="space-y-2">
            {title && <div className="font-medium text-sm">{title}</div>}
            <div className="text-xs text-muted-foreground">{content}</div>
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
};

// Main Help System Component
export const HelpSystem: React.FC<HelpSystemProps> = ({ className = '' }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [selectedTopic, setSelectedTopic] = useState<HelpTopic | null>(null);
  const [searchQuery, setSearchQuery] = useState('');

  const helpTopics: HelpTopic[] = [
    {
      id: 'getting-started',
      title: 'Getting Started',
      description: 'Learn the basics of document analysis',
      category: 'getting-started',
      icon: <BookOpen className="w-4 h-4" />,
      content: {
        text: 'Welcome to the Smart Legal Contracts platform. This guide will help you get started with analyzing legal documents.',
        steps: [
          'Upload your legal document using the drag-and-drop interface',
          'Wait for the AI analysis to complete',
          'Review the results and identified clauses',
          'Download or share your analysis report'
        ],
        tips: [
          'Supported formats include PDF, DOC, DOCX, and TXT files',
          'For best results, ensure documents are text-based (not scanned images)',
          'Analysis typically takes 30-60 seconds depending on document size'
        ],
        relatedTopics: ['document-upload', 'analysis-results']
      }
    },
    {
      id: 'document-upload',
      title: 'Document Upload',
      description: 'How to upload and prepare documents',
      category: 'features',
      icon: <FileText className="w-4 h-4" />,
      content: {
        text: 'Our advanced document upload system supports multiple file formats and provides real-time feedback.',
        steps: [
          'Drag and drop files onto the upload zone',
          'Or click to browse and select files',
          'Monitor upload progress in real-time',
          'Review file validation results'
        ],
        tips: [
          'Maximum file size is 10MB per document',
          'You can upload up to 5 documents simultaneously',
          'Documents are automatically validated for format and content'
        ]
      }
    },
    {
      id: 'arbitration-clauses',
      title: 'Understanding Arbitration Clauses',
      description: 'Learn about arbitration clause detection',
      category: 'features',
      icon: <Shield className="w-4 h-4" />,
      content: {
        text: 'Our AI system specializes in detecting and analyzing arbitration clauses in legal documents.',
        steps: [
          'Upload a legal contract or agreement',
          'The AI scans for arbitration-related language',
          'Clauses are categorized by type and risk level',
          'Confidence scores indicate detection accuracy'
        ],
        tips: [
          'High confidence scores (90%+) indicate very reliable detections',
          'Medium risk clauses may require legal review',
          'Cross-reference with jurisdiction-specific laws'
        ]
      }
    },
    {
      id: 'analysis-results',
      title: 'Understanding Analysis Results',
      description: 'How to interpret AI analysis outputs',
      category: 'features',
      icon: <Zap className="w-4 h-4" />,
      content: {
        text: 'Analysis results provide comprehensive insights into your legal documents with visual charts and detailed breakdowns.',
        steps: [
          'Review the overall risk assessment score',
          'Examine individual clause detections',
          'Check confidence levels for each finding',
          'Use the visualization tabs for deeper insights'
        ],
        tips: [
          'Focus on high-risk clauses first',
          'Use the radar chart to understand overall document health',
          'Export results for team collaboration'
        ]
      }
    }
  ];

  const filteredTopics = helpTopics.filter(topic =>
    topic.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
    topic.description.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'getting-started': return 'bg-green-500/10 text-green-600 border-green-200';
      case 'features': return 'bg-blue-500/10 text-blue-600 border-blue-200';
      case 'troubleshooting': return 'bg-yellow-500/10 text-yellow-600 border-yellow-200';
      case 'advanced': return 'bg-purple-500/10 text-purple-600 border-purple-200';
      default: return 'bg-gray-500/10 text-gray-600 border-gray-200';
    }
  };

  return (
    <div className={className}>
      <Dialog open={isOpen} onOpenChange={setIsOpen}>
        <DialogTrigger asChild>
          <Button
            variant="ghost"
            size="sm"
            className="focus-ring"
            aria-label="Open help system"
          >
            <HelpCircle className="w-4 h-4" />
          </Button>
        </DialogTrigger>
        
        <DialogContent className="max-w-4xl max-h-[80vh] overflow-hidden">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-primary" />
              Help & Documentation
            </DialogTitle>
            <DialogDescription>
              Find answers to common questions and learn how to use the platform effectively.
            </DialogDescription>
          </DialogHeader>

          <div className="grid md:grid-cols-3 gap-6 h-[60vh]">
            {/* Topics Sidebar */}
            <div className="space-y-4">
              {/* Search */}
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <input
                  type="text"
                  placeholder="Search help topics..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full pl-10 pr-4 py-2 bg-muted rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-ring"
                />
              </div>

              {/* Topic List */}
              <div className="space-y-2 overflow-y-auto custom-scrollbar">
                {filteredTopics.map((topic) => (
                  <button
                    key={topic.id}
                    onClick={() => setSelectedTopic(topic)}
                    className={`
                      w-full text-left p-3 rounded-lg transition-colors
                      ${selectedTopic?.id === topic.id 
                        ? 'bg-primary/10 border-l-2 border-primary' 
                        : 'hover:bg-muted/50'
                      }
                    `}
                  >
                    <div className="flex items-start gap-3">
                      <div className="text-muted-foreground mt-0.5">
                        {topic.icon}
                      </div>
                      <div className="flex-1">
                        <div className="font-medium text-sm">{topic.title}</div>
                        <div className="text-xs text-muted-foreground mt-1">
                          {topic.description}
                        </div>
                        <Badge 
                          variant="outline" 
                          className={`text-xs mt-2 ${getCategoryColor(topic.category)}`}
                        >
                          {topic.category.replace('-', ' ')}
                        </Badge>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* Content Area */}
            <div className="md:col-span-2 overflow-y-auto custom-scrollbar">
              {selectedTopic ? (
                <div className="space-y-6">
                  <div className="flex items-center gap-3">
                    {selectedTopic.icon}
                    <h3 className="text-xl font-semibold">{selectedTopic.title}</h3>
                  </div>

                  <p className="text-muted-foreground">{selectedTopic.content.text}</p>

                  {selectedTopic.content.steps && (
                    <div>
                      <h4 className="font-medium mb-3 flex items-center gap-2">
                        <ChevronRight className="w-4 h-4" />
                        Step-by-step guide
                      </h4>
                      <ol className="list-decimal list-inside space-y-2 text-sm">
                        {selectedTopic.content.steps.map((step, index) => (
                          <li key={index} className="text-muted-foreground">{step}</li>
                        ))}
                      </ol>
                    </div>
                  )}

                  {selectedTopic.content.tips && (
                    <div>
                      <h4 className="font-medium mb-3 flex items-center gap-2">
                        <Lightbulb className="w-4 h-4" />
                        Pro Tips
                      </h4>
                      <ul className="space-y-2 text-sm">
                        {selectedTopic.content.tips.map((tip, index) => (
                          <li key={index} className="flex items-start gap-2 text-muted-foreground">
                            <div className="w-1.5 h-1.5 bg-primary rounded-full mt-2 flex-shrink-0" />
                            {tip}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Quick Actions */}
                  <div className="flex gap-3 pt-4 border-t border-border">
                    <Button variant="outline" size="sm">
                      <Video className="w-4 h-4 mr-2" />
                      Watch Tutorial
                    </Button>
                    <Button variant="outline" size="sm">
                      <MessageCircle className="w-4 h-4 mr-2" />
                      Contact Support
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-full text-center space-y-4">
                  <div className="p-4 bg-muted/50 rounded-full">
                    <BookOpen className="w-8 h-8 text-muted-foreground" />
                  </div>
                  <div>
                    <h3 className="font-medium mb-2">Select a help topic</h3>
                    <p className="text-muted-foreground text-sm">
                      Choose a topic from the sidebar to view detailed information and guidance.
                    </p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
};

// Quick Help Tooltips for specific UI elements
export const QuickHelp = {
  DocumentUpload: ({ children }: { children: React.ReactNode }) => (
    <InteractiveTooltip
      title="Document Upload"
      content="Drag and drop legal documents here. Supported formats: PDF, DOC, DOCX, TXT. Maximum size: 10MB."
      trigger={children}
      side="top"
    />
  ),

  AnalysisResults: ({ children }: { children: React.ReactNode }) => (
    <InteractiveTooltip
      title="Analysis Results"
      content="View detailed analysis results including detected clauses, confidence scores, and risk assessments."
      trigger={children}
      side="bottom"
    />
  ),

  ConfidenceScore: ({ children }: { children: React.ReactNode }) => (
    <InteractiveTooltip
      title="Confidence Score"
      content="Indicates how certain our AI is about the detection. Scores above 90% are highly reliable."
      trigger={children}
      side="right"
    />
  ),

  RiskLevel: ({ children }: { children: React.ReactNode }) => (
    <InteractiveTooltip
      title="Risk Assessment"
      content="High risk clauses may limit your legal options. Medium risk requires review. Low risk is generally acceptable."
      trigger={children}
      side="left"
    />
  )
};

export default HelpSystem;