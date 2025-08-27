import React, { useState, useCallback } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { 
  Scale, 
  Upload, 
  BarChart3,
  Brain,
  FileText,
  Zap,
  Shield,
  Users,
  Star,
  CheckCircle,
  ArrowRight,
  Menu,
  X
} from 'lucide-react';

// Import our new components
import DocumentUploadZone from '@/components/DocumentUploadZone';
import AnalysisProgressIndicator from '@/components/AnalysisProgressIndicator';
import AnalysisResultsVisualization from '@/components/AnalysisResultsVisualization';
import ProfessionalDashboard from '@/components/ProfessionalDashboard';
import ThemeToggle from '@/components/ThemeToggle';
import HelpSystem, { QuickHelp } from '@/components/InteractiveHelp';

const EnhancedIndex: React.FC = () => {
  const [activeTab, setActiveTab] = useState('upload');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Mock analysis results
  const mockResults = {
    documentName: "Legal Service Agreement",
    analysisDate: new Date().toISOString(),
    overallScore: 87,
    riskAssessment: 'medium' as const,
    clausesDetected: [
      {
        id: '1',
        type: 'Arbitration Clause',
        text: 'Any dispute arising out of or relating to this Agreement shall be resolved through binding arbitration.',
        confidence: 94,
        pageNumber: 3,
        riskLevel: 'medium' as const,
        category: 'Dispute Resolution'
      },
      {
        id: '2',
        type: 'Jurisdiction Clause',
        text: 'This Agreement shall be governed by the laws of California.',
        confidence: 98,
        pageNumber: 8,
        riskLevel: 'low' as const,
        category: 'Legal Framework'
      },
      {
        id: '3',
        type: 'Liability Limitation',
        text: 'In no event shall the Company be liable for indirect or consequential damages.',
        confidence: 91,
        pageNumber: 5,
        riskLevel: 'high' as const,
        category: 'Risk Management'
      }
    ],
    metrics: {
      totalClauses: 12,
      arbitrationClauses: 2,
      jurisdictionClauses: 1,
      liabilityClauses: 3,
      confidentialityScore: 85,
      enforceabilityScore: 92
    }
  };

  const handleFileUpload = useCallback((files: File[]) => {
    setUploadedFiles(files);
    setIsAnalyzing(true);
    setActiveTab('analysis');
  }, []);

  const handleAnalysisComplete = useCallback(() => {
    setIsAnalyzing(false);
    setAnalysisResults(mockResults);
    setActiveTab('results');
  }, []);

  const features = [
    {
      icon: <Brain className="w-8 h-8" />,
      title: "AI-Powered Analysis",
      description: "Advanced machine learning algorithms analyze legal documents with 99.7% accuracy.",
      color: "text-primary"
    },
    {
      icon: <Zap className="w-8 h-8" />,
      title: "Real-Time Processing",
      description: "Get analysis results in under 60 seconds with live progress tracking.",
      color: "text-accent"
    },
    {
      icon: <Shield className="w-8 h-8" />,
      title: "Enterprise Security",
      description: "Bank-grade encryption and compliance with legal industry standards.",
      color: "text-success"
    },
    {
      icon: <BarChart3 className="w-8 h-8" />,
      title: "Advanced Analytics",
      description: "Comprehensive dashboards with interactive visualizations and insights.",
      color: "text-warning"
    }
  ];

  const testimonials = [
    {
      name: "Sarah Johnson",
      role: "Senior Legal Counsel, TechCorp",
      content: "This platform has revolutionized our contract review process. What used to take hours now takes minutes.",
      rating: 5
    },
    {
      name: "Michael Chen",
      role: "Partner, Chen & Associates",
      content: "The accuracy and speed of arbitration clause detection is remarkable. Essential tool for our practice.",
      rating: 5
    },
    {
      name: "Emma Rodriguez",
      role: "Legal Operations Manager",
      content: "Outstanding user experience and powerful analytics. Our team's productivity has increased by 300%.",
      rating: 5
    }
  ];

  return (
    <div className="min-h-screen bg-background">
      {/* Navigation */}
      <nav 
        className="sticky top-0 z-50 border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60"
        role="banner"
      >
        <div className="container-responsive">
          <div className="flex h-16 items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 rounded-lg bg-gradient-gold">
                <Scale className="w-6 h-6 text-background" />
              </div>
              <div>
                <h1 className="text-xl font-bold gradient-text">LegalAI</h1>
                <p className="text-xs text-muted-foreground hidden sm:block">
                  Smart Legal Contracts Platform
                </p>
              </div>
            </div>
            
            {/* Desktop Navigation */}
            <div className="hidden md:flex items-center space-x-6">
              <nav className="flex items-center space-x-6" role="navigation">
                <a href="#features" className="text-sm font-medium text-muted-foreground hover:text-primary transition-colors">
                  Features
                </a>
                <a href="#dashboard" className="text-sm font-medium text-muted-foreground hover:text-primary transition-colors">
                  Dashboard
                </a>
                <a href="#testimonials" className="text-sm font-medium text-muted-foreground hover:text-primary transition-colors">
                  Reviews
                </a>
              </nav>
              
              <div className="flex items-center space-x-2">
                <ThemeToggle variant="dropdown" />
                <HelpSystem />
                <Button size="sm">
                  Get Started
                </Button>
              </div>
            </div>

            {/* Mobile Navigation Button */}
            <div className="flex md:hidden items-center space-x-2">
              <ThemeToggle variant="button" />
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                aria-label="Toggle mobile menu"
              >
                {mobileMenuOpen ? <X className="w-4 h-4" /> : <Menu className="w-4 h-4" />}
              </Button>
            </div>
          </div>

          {/* Mobile Navigation Menu */}
          {mobileMenuOpen && (
            <div className="md:hidden border-t border-border py-4 animate-slide-down">
              <nav className="flex flex-col space-y-3" role="navigation">
                <a href="#features" className="text-sm font-medium text-muted-foreground hover:text-primary px-4 py-2">
                  Features
                </a>
                <a href="#dashboard" className="text-sm font-medium text-muted-foreground hover:text-primary px-4 py-2">
                  Dashboard
                </a>
                <a href="#testimonials" className="text-sm font-medium text-muted-foreground hover:text-primary px-4 py-2">
                  Reviews
                </a>
                <div className="px-4 pt-2 border-t border-border">
                  <Button size="sm" className="w-full">
                    Get Started
                  </Button>
                </div>
              </nav>
            </div>
          )}
        </div>
      </nav>

      {/* Main Content */}
      <main id="main-content" className="container-responsive section-spacing">
        {/* Hero Section */}
        <section className="text-center space-y-6 mb-16">
          <div className="space-y-4">
            <Badge variant="outline" className="text-xs px-3 py-1">
              ✨ Now with Advanced AI Analysis
            </Badge>
            <h1 className="text-4xl md:text-6xl font-bold tracking-tight gradient-text">
              Smart Legal Contract
              <br />
              Analysis Platform
            </h1>
            <p className="text-xl text-muted-foreground max-w-3xl mx-auto leading-relaxed">
              Revolutionize your legal document review with AI-powered arbitration clause detection, 
              real-time analysis, and comprehensive risk assessment tools.
            </p>
          </div>
          
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Button size="lg" className="min-w-[200px]">
              <Upload className="w-5 h-5 mr-2" />
              Analyze Document
            </Button>
            <Button variant="outline" size="lg" className="min-w-[200px]">
              <BarChart3 className="w-5 h-5 mr-2" />
              View Dashboard
            </Button>
          </div>
        </section>

        {/* Main Application Tabs */}
        <section className="mb-16">
          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full grid-cols-4 mb-8">
              <TabsTrigger value="upload" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
                <Upload className="w-4 h-4 mr-2" />
                <span className="hidden sm:inline">Upload</span>
              </TabsTrigger>
              <TabsTrigger value="analysis" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
                <Brain className="w-4 h-4 mr-2" />
                <span className="hidden sm:inline">Analysis</span>
              </TabsTrigger>
              <TabsTrigger value="results" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
                <BarChart3 className="w-4 h-4 mr-2" />
                <span className="hidden sm:inline">Results</span>
              </TabsTrigger>
              <TabsTrigger value="dashboard" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
                <Users className="w-4 h-4 mr-2" />
                <span className="hidden sm:inline">Dashboard</span>
              </TabsTrigger>
            </TabsList>

            <TabsContent value="upload" className="mt-0">
              <Card className="professional-card">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <FileText className="w-5 h-5" />
                    Document Upload & Analysis
                  </CardTitle>
                  <p className="text-muted-foreground">
                    Upload your legal documents for AI-powered analysis and arbitration clause detection.
                  </p>
                </CardHeader>
                <CardContent>
                  <QuickHelp.DocumentUpload>
                    <div>
                      <DocumentUploadZone
                        onFileUpload={handleFileUpload}
                        acceptedTypes={['.pdf', '.doc', '.docx', '.txt']}
                        maxFileSize={10 * 1024 * 1024}
                        maxFiles={5}
                      />
                    </div>
                  </QuickHelp.DocumentUpload>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="analysis" className="mt-0">
              <AnalysisProgressIndicator
                isActive={isAnalyzing}
                documentName={uploadedFiles[0]?.name || "Legal Document"}
                onComplete={handleAnalysisComplete}
              />
            </TabsContent>

            <TabsContent value="results" className="mt-0">
              {analysisResults ? (
                <QuickHelp.AnalysisResults>
                  <div>
                    <AnalysisResultsVisualization results={analysisResults} />
                  </div>
                </QuickHelp.AnalysisResults>
              ) : (
                <Card className="professional-card">
                  <CardContent className="text-center py-16">
                    <div className="space-y-4">
                      <div className="p-4 bg-muted/50 rounded-full w-16 h-16 mx-auto flex items-center justify-center">
                        <BarChart3 className="w-8 h-8 text-muted-foreground" />
                      </div>
                      <div>
                        <h3 className="text-lg font-medium mb-2">No Analysis Results</h3>
                        <p className="text-muted-foreground">
                          Upload and analyze a document to view detailed results here.
                        </p>
                      </div>
                      <Button onClick={() => setActiveTab('upload')}>
                        <Upload className="w-4 h-4 mr-2" />
                        Upload Document
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              )}
            </TabsContent>

            <TabsContent value="dashboard" className="mt-0">
              <ProfessionalDashboard />
            </TabsContent>
          </Tabs>
        </section>

        {/* Features Section */}
        <section id="features" className="section-spacing">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Powerful Features</h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Everything you need for comprehensive legal document analysis and risk assessment.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((feature, index) => (
              <Card key={index} className="interactive-card text-center">
                <CardContent className="p-6">
                  <div className={`mb-4 ${feature.color}`}>
                    {feature.icon}
                  </div>
                  <h3 className="text-lg font-semibold mb-2">{feature.title}</h3>
                  <p className="text-sm text-muted-foreground">{feature.description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        {/* Testimonials Section */}
        <section id="testimonials" className="section-spacing">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Trusted by Legal Professionals</h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              See what legal experts are saying about our AI-powered platform.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {testimonials.map((testimonial, index) => (
              <Card key={index} className="professional-card">
                <CardContent className="p-6">
                  <div className="flex items-center mb-4">
                    {[...Array(testimonial.rating)].map((_, i) => (
                      <Star key={i} className="w-4 h-4 text-yellow-500 fill-current" />
                    ))}
                  </div>
                  <p className="text-muted-foreground mb-4 italic">
                    "{testimonial.content}"
                  </p>
                  <div className="border-t pt-4">
                    <p className="font-semibold">{testimonial.name}</p>
                    <p className="text-sm text-muted-foreground">{testimonial.role}</p>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        {/* CTA Section */}
        <section className="text-center section-spacing">
          <Card className="professional-card max-w-4xl mx-auto">
            <CardContent className="p-12">
              <div className="space-y-6">
                <h2 className="text-3xl font-bold gradient-text">
                  Ready to Transform Your Legal Workflow?
                </h2>
                <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
                  Join thousands of legal professionals using AI to streamline document analysis 
                  and improve decision-making.
                </p>
                <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                  <Button size="lg" className="min-w-[200px]">
                    Start Free Trial
                    <ArrowRight className="w-4 h-4 ml-2" />
                  </Button>
                  <Button variant="outline" size="lg" className="min-w-[200px]">
                    Schedule Demo
                  </Button>
                </div>
                <div className="flex items-center justify-center gap-6 text-sm text-muted-foreground pt-4">
                  <div className="flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-success" />
                    No credit card required
                  </div>
                  <div className="flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-success" />
                    14-day free trial
                  </div>
                  <div className="flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-success" />
                    Enterprise ready
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-border mt-16">
        <div className="container-responsive py-12">
          <div className="text-center">
            <div className="flex items-center justify-center space-x-3 mb-4">
              <div className="p-2 rounded-lg bg-gradient-gold">
                <Scale className="w-6 h-6 text-background" />
              </div>
              <span className="text-xl font-bold gradient-text">LegalAI</span>
            </div>
            <p className="text-muted-foreground max-w-md mx-auto">
              Empowering legal professionals with advanced AI technology for smarter document analysis and risk assessment.
            </p>
            <div className="mt-6 text-sm text-muted-foreground">
              © 2024 LegalAI. Professional legal document analysis platform.
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default EnhancedIndex;