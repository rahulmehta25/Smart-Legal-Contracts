import React, { useState, useEffect } from 'react';
import { Progress } from '@/components/ui/progress';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { 
  Brain, 
  FileText, 
  Search, 
  CheckCircle, 
  Clock, 
  AlertTriangle,
  Loader2,
  Zap,
  Eye,
  Database
} from 'lucide-react';

interface AnalysisStep {
  id: string;
  name: string;
  description: string;
  status: 'pending' | 'in_progress' | 'completed' | 'error';
  progress: number;
  duration?: number;
  icon: React.ReactNode;
}

interface AnalysisProgressIndicatorProps {
  isActive?: boolean;
  documentName?: string;
  onComplete?: () => void;
  className?: string;
}

export const AnalysisProgressIndicator: React.FC<AnalysisProgressIndicatorProps> = ({
  isActive = false,
  documentName = "Legal Document",
  onComplete,
  className = ""
}) => {
  const [steps, setSteps] = useState<AnalysisStep[]>([
    {
      id: 'upload',
      name: 'Document Upload',
      description: 'Receiving and validating document',
      status: 'pending',
      progress: 0,
      icon: <FileText className="w-4 h-4" />
    },
    {
      id: 'preprocessing',
      name: 'Text Preprocessing',
      description: 'Extracting and cleaning text content',
      status: 'pending',
      progress: 0,
      icon: <Database className="w-4 h-4" />
    },
    {
      id: 'neural_analysis',
      name: 'Neural Analysis',
      description: 'AI analyzing document structure and content',
      status: 'pending',
      progress: 0,
      icon: <Brain className="w-4 h-4" />
    },
    {
      id: 'clause_detection',
      name: 'Clause Detection',
      description: 'Identifying arbitration clauses and legal terms',
      status: 'pending',
      progress: 0,
      icon: <Search className="w-4 h-4" />
    },
    {
      id: 'confidence_scoring',
      name: 'Confidence Scoring',
      description: 'Calculating accuracy and confidence metrics',
      status: 'pending',
      progress: 0,
      icon: <Zap className="w-4 h-4" />
    },
    {
      id: 'final_review',
      name: 'Final Review',
      description: 'Generating comprehensive analysis report',
      status: 'pending',
      progress: 0,
      icon: <Eye className="w-4 h-4" />
    }
  ]);

  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [startTime, setStartTime] = useState<Date | null>(null);
  const [overallProgress, setOverallProgress] = useState(0);

  useEffect(() => {
    if (isActive && !startTime) {
      setStartTime(new Date());
      simulateAnalysis();
    }
  }, [isActive, startTime]);

  const simulateAnalysis = async () => {
    for (let i = 0; i < steps.length; i++) {
      // Mark current step as in progress
      setCurrentStepIndex(i);
      setSteps(prev => prev.map(step => 
        step.id === steps[i].id 
          ? { ...step, status: 'in_progress' }
          : step
      ));

      // Simulate step progress
      const stepDuration = Math.random() * 2000 + 1000; // 1-3 seconds
      const progressSteps = 20;
      const progressIncrement = 100 / progressSteps;

      for (let progress = 0; progress <= 100; progress += progressIncrement) {
        await new Promise(resolve => setTimeout(resolve, stepDuration / progressSteps));
        
        setSteps(prev => prev.map(step => 
          step.id === steps[i].id 
            ? { ...step, progress: Math.min(progress, 100) }
            : step
        ));

        // Update overall progress
        const completedSteps = i;
        const currentStepProgress = progress / 100;
        const totalProgress = ((completedSteps + currentStepProgress) / steps.length) * 100;
        setOverallProgress(totalProgress);
      }

      // Mark step as completed
      setSteps(prev => prev.map(step => 
        step.id === steps[i].id 
          ? { ...step, status: 'completed', progress: 100 }
          : step
      ));

      // Small delay between steps
      await new Promise(resolve => setTimeout(resolve, 500));
    }

    // Analysis complete
    setOverallProgress(100);
    if (onComplete) {
      setTimeout(onComplete, 1000);
    }
  };

  const getStepIcon = (step: AnalysisStep) => {
    switch (step.status) {
      case 'in_progress':
        return <Loader2 className="w-4 h-4 animate-spin text-primary" />;
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-success" />;
      case 'error':
        return <AlertTriangle className="w-4 h-4 text-destructive" />;
      default:
        return <Clock className="w-4 h-4 text-muted-foreground" />;
    }
  };

  const getStepStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-success/10 text-success border-success/20';
      case 'in_progress':
        return 'bg-primary/10 text-primary border-primary/20';
      case 'error':
        return 'bg-destructive/10 text-destructive border-destructive/20';
      default:
        return 'bg-muted/10 text-muted-foreground border-muted/20';
    }
  };

  const formatElapsedTime = () => {
    if (!startTime) return '0:00';
    const elapsed = Date.now() - startTime.getTime();
    const seconds = Math.floor(elapsed / 1000);
    const minutes = Math.floor(seconds / 60);
    return `${minutes}:${(seconds % 60).toString().padStart(2, '0')}`;
  };

  if (!isActive && overallProgress === 0) {
    return null;
  }

  return (
    <Card id="analysis-progress-indicator" className={`w-full animate-slide-up ${className}`}>
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Brain className="w-5 h-5 text-primary" />
            Analyzing {documentName}
          </CardTitle>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Clock className="w-4 h-4" />
            {formatElapsedTime()}
          </div>
        </div>
        
        {/* Overall Progress */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-sm">
            <span>Overall Progress</span>
            <span className="font-medium">{Math.round(overallProgress)}%</span>
          </div>
          <Progress value={overallProgress} className="h-2" />
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Analysis Steps */}
        <div className="space-y-3">
          {steps.map((step, index) => (
            <div
              key={step.id}
              className={`
                flex items-center gap-4 p-3 rounded-lg border transition-all duration-300
                ${step.status === 'in_progress' ? 'bg-primary/5 border-primary/20' : 'bg-muted/5 border-muted/10'}
                ${index === currentStepIndex ? 'animate-pulse' : ''}
              `}
            >
              {/* Step Icon */}
              <div className="flex-shrink-0">
                {getStepIcon(step)}
              </div>

              {/* Step Content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between mb-1">
                  <h4 className="text-sm font-medium">{step.name}</h4>
                  <Badge
                    variant="outline"
                    className={`text-xs ${getStepStatusColor(step.status)}`}
                  >
                    {step.status === 'in_progress' ? 'Processing' : 
                     step.status === 'completed' ? 'Done' :
                     step.status === 'error' ? 'Error' : 'Pending'}
                  </Badge>
                </div>
                <p className="text-xs text-muted-foreground">{step.description}</p>
                
                {/* Step Progress Bar */}
                {(step.status === 'in_progress' || step.status === 'completed') && (
                  <div className="mt-2">
                    <Progress value={step.progress} className="h-1" />
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Analysis Statistics */}
        <div className="grid grid-cols-3 gap-4 pt-4 border-t border-border/50">
          <div className="text-center">
            <div className="text-lg font-semibold text-primary">
              {steps.filter(s => s.status === 'completed').length}
            </div>
            <div className="text-xs text-muted-foreground">Steps Complete</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-semibold text-accent">
              {steps.filter(s => s.status === 'in_progress').length || 
               (overallProgress === 100 ? 0 : 1)}
            </div>
            <div className="text-xs text-muted-foreground">Processing</div>
          </div>
          <div className="text-center">
            <div className="text-lg font-semibold text-neural-purple">
              99.7%
            </div>
            <div className="text-xs text-muted-foreground">Accuracy</div>
          </div>
        </div>

        {/* Completion Message */}
        {overallProgress === 100 && (
          <div className="flex items-center justify-center gap-2 p-4 bg-success/10 border border-success/20 rounded-lg animate-fade-in">
            <CheckCircle className="w-5 h-5 text-success" />
            <span className="text-success font-medium">Analysis Complete!</span>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default AnalysisProgressIndicator;