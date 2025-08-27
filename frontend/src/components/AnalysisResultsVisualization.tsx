import React, { useState, useMemo } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  Area,
  AreaChart,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar
} from 'recharts';
import { 
  Scale, 
  AlertTriangle, 
  CheckCircle, 
  FileText, 
  TrendingUp,
  Eye,
  Download,
  Share,
  MoreHorizontal
} from 'lucide-react';

interface ClauseDetection {
  id: string;
  type: string;
  text: string;
  confidence: number;
  pageNumber: number;
  riskLevel: 'low' | 'medium' | 'high';
  category: string;
}

interface AnalysisResults {
  documentName: string;
  analysisDate: string;
  overallScore: number;
  riskAssessment: 'low' | 'medium' | 'high';
  clausesDetected: ClauseDetection[];
  metrics: {
    totalClauses: number;
    arbitrationClauses: number;
    jurisdictionClauses: number;
    liabilityClauses: number;
    confidentialityScore: number;
    enforceabilityScore: number;
  };
}

interface AnalysisResultsVisualizationProps {
  results: AnalysisResults;
  className?: string;
}

export const AnalysisResultsVisualization: React.FC<AnalysisResultsVisualizationProps> = ({
  results,
  className = ""
}) => {
  const [selectedTab, setSelectedTab] = useState('overview');
  const [selectedClause, setSelectedClause] = useState<ClauseDetection | null>(null);

  // Chart data preparation
  const riskDistribution = useMemo(() => {
    const distribution = { low: 0, medium: 0, high: 0 };
    results.clausesDetected.forEach(clause => {
      distribution[clause.riskLevel]++;
    });
    
    return [
      { name: 'Low Risk', value: distribution.low, color: '#10b981' },
      { name: 'Medium Risk', value: distribution.medium, color: '#f59e0b' },
      { name: 'High Risk', value: distribution.high, color: '#ef4444' }
    ];
  }, [results.clausesDetected]);

  const clauseTypeDistribution = useMemo(() => {
    const types = new Map<string, number>();
    results.clausesDetected.forEach(clause => {
      types.set(clause.type, (types.get(clause.type) || 0) + 1);
    });
    
    return Array.from(types, ([name, value]) => ({ name, value }));
  }, [results.clausesDetected]);

  const confidenceData = useMemo(() => {
    const confidenceRanges = { '90-100%': 0, '80-89%': 0, '70-79%': 0, '60-69%': 0, '<60%': 0 };
    
    results.clausesDetected.forEach(clause => {
      const conf = clause.confidence;
      if (conf >= 90) confidenceRanges['90-100%']++;
      else if (conf >= 80) confidenceRanges['80-89%']++;
      else if (conf >= 70) confidenceRanges['70-79%']++;
      else if (conf >= 60) confidenceRanges['60-69%']++;
      else confidenceRanges['<60%']++;
    });

    return Object.entries(confidenceRanges).map(([range, count]) => ({
      range,
      count,
      percentage: (count / results.clausesDetected.length) * 100
    }));
  }, [results.clausesDetected]);

  const performanceMetrics = useMemo(() => [
    { metric: 'Detection Accuracy', value: results.overallScore, max: 100 },
    { metric: 'Enforceability', value: results.metrics.enforceabilityScore, max: 100 },
    { metric: 'Confidentiality', value: results.metrics.confidentialityScore, max: 100 },
    { metric: 'Risk Assessment', value: 85, max: 100 },
    { metric: 'Compliance Check', value: 92, max: 100 },
  ], [results]);

  const getRiskBadgeColor = (risk: string) => {
    switch (risk) {
      case 'high': return 'bg-red-500/10 text-red-600 border-red-200';
      case 'medium': return 'bg-yellow-500/10 text-yellow-600 border-yellow-200';
      case 'low': return 'bg-green-500/10 text-green-600 border-green-200';
      default: return 'bg-gray-500/10 text-gray-600 border-gray-200';
    }
  };

  const getOverallRiskColor = (score: number) => {
    if (score >= 80) return 'text-green-600';
    if (score >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div id="analysis-results-visualization" className={`space-y-6 ${className}`}>
      {/* Header */}
      <Card className="professional-card">
        <CardHeader className="pb-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-primary/10">
                <Scale className="w-6 h-6 text-primary" />
              </div>
              <div>
                <CardTitle className="text-xl">Analysis Results</CardTitle>
                <p className="text-muted-foreground">{results.documentName}</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Button variant="outline" size="sm">
                <Download className="w-4 h-4 mr-2" />
                Export
              </Button>
              <Button variant="outline" size="sm">
                <Share className="w-4 h-4 mr-2" />
                Share
              </Button>
              <Button variant="ghost" size="sm">
                <MoreHorizontal className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </CardHeader>

        <CardContent>
          {/* Key Metrics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            <div className="text-center">
              <div className={`text-3xl font-bold ${getOverallRiskColor(results.overallScore)}`}>
                {results.overallScore}%
              </div>
              <div className="text-sm text-muted-foreground">Overall Score</div>
            </div>
            
            <div className="text-center">
              <div className="text-3xl font-bold text-primary">
                {results.metrics.totalClauses}
              </div>
              <div className="text-sm text-muted-foreground">Total Clauses</div>
            </div>
            
            <div className="text-center">
              <div className="text-3xl font-bold text-accent">
                {results.metrics.arbitrationClauses}
              </div>
              <div className="text-sm text-muted-foreground">Arbitration Clauses</div>
            </div>
            
            <div className="text-center">
              <Badge className={getRiskBadgeColor(results.riskAssessment)}>
                {results.riskAssessment.toUpperCase()} RISK
              </Badge>
              <div className="text-sm text-muted-foreground mt-1">Risk Level</div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Detailed Analysis Tabs */}
      <Tabs value={selectedTab} onValueChange={setSelectedTab}>
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="clauses">Clauses</TabsTrigger>
          <TabsTrigger value="risk">Risk Analysis</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          <div className="grid md:grid-cols-2 gap-6">
            {/* Risk Distribution Pie Chart */}
            <Card className="professional-card">
              <CardHeader>
                <CardTitle>Risk Distribution</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={riskDistribution}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={100}
                      paddingAngle={5}
                      dataKey="value"
                    >
                      {riskDistribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
                <div className="flex justify-center gap-4 mt-4">
                  {riskDistribution.map((item, index) => (
                    <div key={index} className="flex items-center gap-2">
                      <div 
                        className="w-3 h-3 rounded-full" 
                        style={{ backgroundColor: item.color }}
                      />
                      <span className="text-sm">{item.name}</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Clause Types Bar Chart */}
            <Card className="professional-card">
              <CardHeader>
                <CardTitle>Clause Types</CardTitle>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={clauseTypeDistribution}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="name" 
                      angle={-45}
                      textAnchor="end"
                      height={100}
                    />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="value" fill="hsl(var(--primary))" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          {/* Confidence Distribution */}
          <Card className="professional-card">
            <CardHeader>
              <CardTitle>Confidence Distribution</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={250}>
                <AreaChart data={confidenceData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="range" />
                  <YAxis />
                  <Tooltip />
                  <Area 
                    type="monotone" 
                    dataKey="count" 
                    stroke="hsl(var(--primary))" 
                    fill="hsl(var(--primary))" 
                    fillOpacity={0.3}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Clauses Tab */}
        <TabsContent value="clauses" className="space-y-6">
          <Card className="professional-card">
            <CardHeader>
              <CardTitle>Detected Clauses</CardTitle>
              <p className="text-muted-foreground">
                {results.clausesDetected.length} clauses identified with AI analysis
              </p>
            </CardHeader>
            <CardContent>
              <div className="space-y-4 max-h-96 overflow-y-auto custom-scrollbar">
                {results.clausesDetected.map((clause, index) => (
                  <div
                    key={clause.id}
                    className={`
                      p-4 rounded-lg border cursor-pointer transition-all duration-200
                      ${selectedClause?.id === clause.id 
                        ? 'border-primary bg-primary/5' 
                        : 'border-border hover:border-primary/50'
                      }
                    `}
                    onClick={() => setSelectedClause(
                      selectedClause?.id === clause.id ? null : clause
                    )}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-2">
                          <Badge variant="outline">{clause.type}</Badge>
                          <Badge className={getRiskBadgeColor(clause.riskLevel)}>
                            {clause.riskLevel}
                          </Badge>
                          <span className="text-sm text-muted-foreground">
                            Page {clause.pageNumber}
                          </span>
                        </div>
                        <p className="text-sm text-foreground mb-2">
                          {clause.text.length > 150 
                            ? `${clause.text.substring(0, 150)}...`
                            : clause.text
                          }
                        </p>
                        <div className="flex items-center gap-4 text-xs text-muted-foreground">
                          <span>Confidence: {clause.confidence}%</span>
                          <span>Category: {clause.category}</span>
                        </div>
                      </div>
                      {clause.confidence >= 90 ? (
                        <CheckCircle className="w-5 h-5 text-success flex-shrink-0 mt-1" />
                      ) : (
                        <AlertTriangle className="w-5 h-5 text-warning flex-shrink-0 mt-1" />
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Risk Analysis Tab */}
        <TabsContent value="risk" className="space-y-6">
          <div className="grid md:grid-cols-2 gap-6">
            <Card className="professional-card">
              <CardHeader>
                <CardTitle>Risk Assessment Matrix</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-3 gap-4">
                  <div className="text-center p-4 bg-red-500/10 rounded-lg border border-red-200">
                    <div className="text-2xl font-bold text-red-600">
                      {riskDistribution.find(r => r.name === 'High Risk')?.value || 0}
                    </div>
                    <div className="text-sm text-red-600">High Risk</div>
                  </div>
                  <div className="text-center p-4 bg-yellow-500/10 rounded-lg border border-yellow-200">
                    <div className="text-2xl font-bold text-yellow-600">
                      {riskDistribution.find(r => r.name === 'Medium Risk')?.value || 0}
                    </div>
                    <div className="text-sm text-yellow-600">Medium Risk</div>
                  </div>
                  <div className="text-center p-4 bg-green-500/10 rounded-lg border border-green-200">
                    <div className="text-2xl font-bold text-green-600">
                      {riskDistribution.find(r => r.name === 'Low Risk')?.value || 0}
                    </div>
                    <div className="text-sm text-green-600">Low Risk</div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="professional-card">
              <CardHeader>
                <CardTitle>Risk Recommendations</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="w-5 h-5 text-warning flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="text-sm font-medium">Review arbitration clauses</p>
                      <p className="text-xs text-muted-foreground">
                        Some clauses may limit legal recourse options
                      </p>
                    </div>
                  </div>
                  <div className="flex items-start gap-3">
                    <CheckCircle className="w-5 h-5 text-success flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="text-sm font-medium">Confidentiality terms look good</p>
                      <p className="text-xs text-muted-foreground">
                        Well-structured privacy protection clauses
                      </p>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Performance Tab */}
        <TabsContent value="performance" className="space-y-6">
          <Card className="professional-card">
            <CardHeader>
              <CardTitle>Analysis Performance Metrics</CardTitle>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <RadarChart data={performanceMetrics}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="metric" />
                  <PolarRadiusAxis />
                  <Radar
                    name="Performance"
                    dataKey="value"
                    stroke="hsl(var(--primary))"
                    fill="hsl(var(--primary))"
                    fillOpacity={0.3}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default AnalysisResultsVisualization;