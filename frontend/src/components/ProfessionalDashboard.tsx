import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import { 
  FileText, 
  TrendingUp, 
  AlertTriangle, 
  CheckCircle, 
  Clock,
  Users,
  Scale,
  Brain,
  Zap,
  Shield,
  Download,
  RefreshCw,
  Calendar,
  ArrowUpRight,
  ArrowDownRight
} from 'lucide-react';
import ThemeToggle from './ThemeToggle';
import HelpSystem from './InteractiveHelp';

interface DashboardMetric {
  id: string;
  title: string;
  value: string | number;
  change: number;
  trend: 'up' | 'down' | 'neutral';
  icon: React.ReactNode;
  description: string;
}

interface RecentAnalysis {
  id: string;
  documentName: string;
  date: string;
  status: 'completed' | 'processing' | 'error';
  score: number;
  riskLevel: 'low' | 'medium' | 'high';
  clausesFound: number;
}

interface ProfessionalDashboardProps {
  className?: string;
}

export const ProfessionalDashboard: React.FC<ProfessionalDashboardProps> = ({
  className = ""
}) => {
  const [timeRange, setTimeRange] = useState<'7d' | '30d' | '90d'>('30d');
  const [isLoading, setIsLoading] = useState(false);

  // Dashboard metrics
  const metrics: DashboardMetric[] = [
    {
      id: 'total-documents',
      title: 'Documents Analyzed',
      value: 1247,
      change: 12.5,
      trend: 'up',
      icon: <FileText className="w-5 h-5" />,
      description: 'Total documents processed this month'
    },
    {
      id: 'accuracy-rate',
      title: 'Detection Accuracy',
      value: '99.7%',
      change: 0.3,
      trend: 'up',
      icon: <Brain className="w-5 h-5" />,
      description: 'AI model accuracy for clause detection'
    },
    {
      id: 'high-risk-clauses',
      title: 'High-Risk Clauses',
      value: 89,
      change: -8.2,
      trend: 'down',
      icon: <AlertTriangle className="w-5 h-5" />,
      description: 'High-risk arbitration clauses detected'
    },
    {
      id: 'processing-time',
      title: 'Avg Processing Time',
      value: '45s',
      change: -15.4,
      trend: 'down',
      icon: <Zap className="w-5 h-5" />,
      description: 'Average time per document analysis'
    },
    {
      id: 'success-rate',
      title: 'Success Rate',
      value: '98.9%',
      change: 1.2,
      trend: 'up',
      icon: <CheckCircle className="w-5 h-5" />,
      description: 'Successful analysis completion rate'
    },
    {
      id: 'active-users',
      title: 'Active Users',
      value: 342,
      change: 23.8,
      trend: 'up',
      icon: <Users className="w-5 h-5" />,
      description: 'Active legal professionals this month'
    }
  ];

  // Chart data
  const analysisData = [
    { date: '2024-01-01', documents: 45, accuracy: 99.2 },
    { date: '2024-01-02', documents: 52, accuracy: 99.4 },
    { date: '2024-01-03', documents: 38, accuracy: 99.1 },
    { date: '2024-01-04', documents: 61, accuracy: 99.6 },
    { date: '2024-01-05', documents: 48, accuracy: 99.8 },
    { date: '2024-01-06', documents: 55, accuracy: 99.7 },
    { date: '2024-01-07', documents: 67, accuracy: 99.9 }
  ];

  const riskDistribution = [
    { name: 'Low Risk', value: 65, color: '#10b981' },
    { name: 'Medium Risk', value: 25, color: '#f59e0b' },
    { name: 'High Risk', value: 10, color: '#ef4444' }
  ];

  const clauseTypes = [
    { type: 'Arbitration', count: 156 },
    { type: 'Jurisdiction', count: 134 },
    { type: 'Liability', count: 89 },
    { type: 'Confidentiality', count: 67 },
    { type: 'Termination', count: 45 }
  ];

  // Recent analyses
  const recentAnalyses: RecentAnalysis[] = [
    {
      id: '1',
      documentName: 'Service Agreement - TechCorp',
      date: '2024-01-07 14:30',
      status: 'completed',
      score: 94,
      riskLevel: 'medium',
      clausesFound: 12
    },
    {
      id: '2',
      documentName: 'Employment Contract - Legal Dept',
      date: '2024-01-07 13:45',
      status: 'completed',
      score: 89,
      riskLevel: 'low',
      clausesFound: 8
    },
    {
      id: '3',
      documentName: 'Partnership Agreement - StartupXYZ',
      date: '2024-01-07 12:15',
      status: 'processing',
      score: 0,
      riskLevel: 'medium',
      clausesFound: 0
    },
    {
      id: '4',
      documentName: 'NDA - Confidential Project',
      date: '2024-01-07 11:20',
      status: 'completed',
      score: 96,
      riskLevel: 'low',
      clausesFound: 5
    }
  ];

  const refreshData = async () => {
    setIsLoading(true);
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1500));
    setIsLoading(false);
  };

  const getTrendIcon = (trend: 'up' | 'down' | 'neutral') => {
    switch (trend) {
      case 'up':
        return <ArrowUpRight className="w-4 h-4 text-green-600" />;
      case 'down':
        return <ArrowDownRight className="w-4 h-4 text-red-600" />;
      default:
        return <div className="w-4 h-4" />;
    }
  };

  const getTrendColor = (trend: 'up' | 'down' | 'neutral') => {
    switch (trend) {
      case 'up':
        return 'text-green-600';
      case 'down':
        return 'text-red-600';
      default:
        return 'text-muted-foreground';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-500/10 text-green-600 border-green-200';
      case 'processing':
        return 'bg-blue-500/10 text-blue-600 border-blue-200';
      case 'error':
        return 'bg-red-500/10 text-red-600 border-red-200';
      default:
        return 'bg-gray-500/10 text-gray-600 border-gray-200';
    }
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'high':
        return 'bg-red-500/10 text-red-600 border-red-200';
      case 'medium':
        return 'bg-yellow-500/10 text-yellow-600 border-yellow-200';
      case 'low':
        return 'bg-green-500/10 text-green-600 border-green-200';
      default:
        return 'bg-gray-500/10 text-gray-600 border-gray-200';
    }
  };

  return (
    <div id="professional-dashboard" className={`space-y-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Legal AI Dashboard</h1>
          <p className="text-muted-foreground">
            Monitor your document analysis performance and insights
          </p>
        </div>
        
        <div className="flex items-center gap-3">
          <Button
            variant="outline"
            size="sm"
            onClick={refreshData}
            disabled={isLoading}
          >
            <RefreshCw className={`w-4 h-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
            {isLoading ? 'Refreshing...' : 'Refresh'}
          </Button>
          <ThemeToggle variant="dropdown" />
          <HelpSystem />
        </div>
      </div>

      {/* Time Range Selector */}
      <div className="flex items-center gap-2">
        <Calendar className="w-4 h-4 text-muted-foreground" />
        <span className="text-sm text-muted-foreground">Time Range:</span>
        <div className="flex bg-muted rounded-lg p-1">
          {(['7d', '30d', '90d'] as const).map((range) => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              className={`px-3 py-1 text-sm rounded-md transition-colors ${
                timeRange === range
                  ? 'bg-background shadow-sm'
                  : 'hover:bg-background/50'
              }`}
            >
              {range === '7d' ? '7 days' : range === '30d' ? '30 days' : '90 days'}
            </button>
          ))}
        </div>
      </div>

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {metrics.map((metric) => (
          <Card key={metric.id} className="professional-card animate-slide-up">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-primary/10 rounded-lg">
                    {metric.icon}
                  </div>
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">
                      {metric.title}
                    </p>
                    <p className="text-2xl font-bold">{metric.value}</p>
                  </div>
                </div>
                <div className="flex items-center gap-1">
                  {getTrendIcon(metric.trend)}
                  <span className={`text-sm font-medium ${getTrendColor(metric.trend)}`}>
                    {metric.change > 0 ? '+' : ''}{metric.change}%
                  </span>
                </div>
              </div>
              <p className="text-xs text-muted-foreground mt-3">
                {metric.description}
              </p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Charts Row */}
      <div className="grid lg:grid-cols-2 gap-6">
        {/* Analysis Trends */}
        <Card className="professional-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="w-5 h-5" />
              Analysis Trends
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={analysisData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="date" 
                  tickFormatter={(value) => new Date(value).toLocaleDateString()}
                />
                <YAxis />
                <Tooltip 
                  labelFormatter={(value) => new Date(value).toLocaleDateString()}
                />
                <Area
                  type="monotone"
                  dataKey="documents"
                  stroke="hsl(var(--primary))"
                  fill="hsl(var(--primary))"
                  fillOpacity={0.3}
                />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Risk Distribution */}
        <Card className="professional-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Shield className="w-5 h-5" />
              Risk Distribution
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={riskDistribution}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={120}
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
            <div className="flex justify-center gap-6 mt-4">
              {riskDistribution.map((item, index) => (
                <div key={index} className="flex items-center gap-2">
                  <div 
                    className="w-3 h-3 rounded-full" 
                    style={{ backgroundColor: item.color }}
                  />
                  <span className="text-sm">
                    {item.name}: {item.value}%
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Bottom Row */}
      <div className="grid lg:grid-cols-3 gap-6">
        {/* Clause Types */}
        <Card className="professional-card">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Scale className="w-5 h-5" />
              Clause Types
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={clauseTypes} layout="horizontal">
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" />
                <YAxis dataKey="type" type="category" width={80} />
                <Tooltip />
                <Bar dataKey="count" fill="hsl(var(--accent))" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Recent Analyses */}
        <Card className="lg:col-span-2 professional-card">
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>Recent Analyses</CardTitle>
            <Button variant="outline" size="sm">
              <Download className="w-4 h-4 mr-2" />
              Export All
            </Button>
          </CardHeader>
          <CardContent>
            <div className="space-y-4 max-h-80 overflow-y-auto custom-scrollbar">
              {recentAnalyses.map((analysis) => (
                <div
                  key={analysis.id}
                  className="flex items-center justify-between p-4 border border-border rounded-lg hover:bg-muted/50 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-primary/10 rounded-lg">
                      <FileText className="w-4 h-4" />
                    </div>
                    <div>
                      <p className="font-medium text-sm">{analysis.documentName}</p>
                      <div className="flex items-center gap-3 mt-1">
                        <span className="text-xs text-muted-foreground">
                          {analysis.date}
                        </span>
                        {analysis.status === 'completed' && (
                          <>
                            <Badge className={getRiskColor(analysis.riskLevel)}>
                              {analysis.riskLevel} risk
                            </Badge>
                            <span className="text-xs text-muted-foreground">
                              {analysis.clausesFound} clauses
                            </span>
                          </>
                        )}
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-3">
                    {analysis.status === 'completed' && (
                      <div className="text-right">
                        <div className="text-sm font-medium">{analysis.score}%</div>
                        <Progress value={analysis.score} className="w-20 h-1 mt-1" />
                      </div>
                    )}
                    <Badge className={getStatusColor(analysis.status)}>
                      {analysis.status === 'processing' ? (
                        <>
                          <Clock className="w-3 h-3 mr-1" />
                          Processing
                        </>
                      ) : (
                        analysis.status
                      )}
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default ProfessionalDashboard;