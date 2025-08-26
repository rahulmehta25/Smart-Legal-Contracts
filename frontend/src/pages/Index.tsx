import { useState } from 'react';
import { ParticleBackground } from '@/components/ParticleBackground';
import { TypewriterText } from '@/components/TypewriterText';
import { GlassCard } from '@/components/GlassCard';
import { NeuralNetwork } from '@/components/NeuralNetwork';
import { MagneticButton } from '@/components/MagneticButton';
import { 
  Brain, 
  FileText, 
  Zap, 
  Shield, 
  Upload, 
  BarChart3,
  CheckCircle,
  ArrowRight,
  Scale,
  Gavel
} from 'lucide-react';

const Index = () => {
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  const typewriterTexts = [
    "AI-Powered Arbitration Detection",
    "Revolutionary Legal Analytics", 
    "Intelligent Contract Analysis",
    "Advanced RAG Technology"
  ];

  const features = [
    {
      icon: Brain,
      title: "Neural RAG System",
      description: "Advanced neural networks analyze contract clauses with unprecedented accuracy using retrieval-augmented generation.",
      color: "neural"
    },
    {
      icon: Zap,
      title: "Real-time Analysis",
      description: "Instant arbitration clause detection with live updates via WebSocket connections for immediate insights.",
      color: "legal"
    },
    {
      icon: Shield,
      title: "Enterprise Security", 
      description: "Bank-grade encryption and JWT authentication ensure your sensitive legal documents remain protected.",
      color: "legal"
    },
    {
      icon: BarChart3,
      title: "Advanced Analytics",
      description: "Comprehensive dashboard with interactive charts and detailed reporting for legal professionals.",
      color: "neural"
    }
  ];

  const stats = [
    { value: "99.7%", label: "Detection Accuracy" },
    { value: "< 2s", label: "Analysis Speed" },
    { value: "500K+", label: "Documents Processed" },
    { value: "24/7", label: "System Uptime" }
  ];

  const handleFileUpload = async () => {
    setIsAnalyzing(true);
    // Simulate upload progress
    for (let i = 0; i <= 100; i += 10) {
      setUploadProgress(i);
      await new Promise(resolve => setTimeout(resolve, 200));
    }
    setIsAnalyzing(false);
    setUploadProgress(0);
  };

  return (
    <div className="min-h-screen bg-background relative overflow-hidden">
      <ParticleBackground />
      
      {/* Navigation */}
      <nav className="relative z-50 p-6">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-lg bg-gradient-gold">
              <Scale className="w-6 h-6 text-background" />
            </div>
            <span className="text-xl font-bold gradient-text">LegalAI</span>
          </div>
          
          <div className="hidden md:flex items-center space-x-8">
            <a href="#features" className="text-foreground/80 hover:text-primary transition-colors">Features</a>
            <a href="#analytics" className="text-foreground/80 hover:text-primary transition-colors">Analytics</a>
            <a href="#api" className="text-foreground/80 hover:text-primary transition-colors">API</a>
            <MagneticButton variant="hero" size="sm">
              Get Started
            </MagneticButton>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative z-10 py-20 px-6">
        <div className="max-w-7xl mx-auto text-center">
          <div className="mb-8">
            <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight">
              <TypewriterText 
                texts={typewriterTexts}
                speed={80}
                deleteSpeed={40}
                pauseTime={3000}
                className="gradient-text"
              />
            </h1>
            <p className="text-xl md:text-2xl text-foreground/80 max-w-3xl mx-auto mb-8">
              Revolutionize legal document analysis with our cutting-edge AI system that detects arbitration clauses with unmatched precision and speed.
            </p>
          </div>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-6 mb-16">
            <MagneticButton variant="hero" size="lg">
              <Upload className="w-5 h-5 mr-2" />
              Upload Document
            </MagneticButton>
            <MagneticButton variant="legal" size="lg">
              <Gavel className="w-5 h-5 mr-2" />
              Live Demo
            </MagneticButton>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8 mb-20">
            {stats.map((stat, index) => (
              <GlassCard key={index} className="text-center" hover3d>
                <div className="text-3xl md:text-4xl font-bold gradient-text mb-2">
                  {stat.value}
                </div>
                <div className="text-foreground/60">{stat.label}</div>
              </GlassCard>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="relative z-10 py-20 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16 px-4">
            <h2 className="text-4xl md:text-5xl font-bold gradient-text mb-6 break-words">
              Advanced AI Features
            </h2>
            <p className="text-xl text-foreground/80 max-w-3xl mx-auto">
              Advanced AI technology meets sophisticated legal expertise in our comprehensive arbitration detection platform.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <GlassCard key={index} className="text-center group h-full" hover3d magnetic>
                <div className="mb-6">
                  <div className={`w-16 h-16 mx-auto rounded-2xl bg-gradient-${feature.color} p-4 shadow-glow-primary flex items-center justify-center`}>
                    <feature.icon className="w-8 h-8 text-background" />
                  </div>
                </div>
                <h3 className="text-xl font-bold mb-4 gradient-text">{feature.title}</h3>
                <p className="text-foreground/70 leading-relaxed">{feature.description}</p>
              </GlassCard>
            ))}
          </div>
        </div>
      </section>

      {/* Neural Network Visualization */}
      <section className="relative z-10 py-20 px-6">
        <div className="max-w-7xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div>
              <h2 className="text-4xl font-bold gradient-text mb-6">
                Neural Architecture
              </h2>
              <p className="text-xl text-foreground/80 mb-8">
                Our sophisticated neural network processes legal documents through multiple layers of analysis, ensuring comprehensive arbitration clause detection.
              </p>
              
              <div className="space-y-4">
                <div className="flex items-center space-x-3">
                  <CheckCircle className="w-5 h-5 text-analysis-green" />
                  <span>Multi-layer transformer architecture</span>
                </div>
                <div className="flex items-center space-x-3">
                  <CheckCircle className="w-5 h-5 text-analysis-green" />
                  <span>Retrieval-augmented generation (RAG)</span>
                </div>
                <div className="flex items-center space-x-3">
                  <CheckCircle className="w-5 h-5 text-analysis-green" />
                  <span>Real-time inference optimization</span>
                </div>
              </div>

              <div className="mt-8">
                <MagneticButton variant="neural" size="lg">
                  Explore Architecture
                  <ArrowRight className="w-5 h-5 ml-2" />
                </MagneticButton>
              </div>
            </div>

            <GlassCard className="h-80" hover3d>
              <NeuralNetwork />
            </GlassCard>
          </div>
        </div>
      </section>

      {/* Upload Demo Section */}
      <section className="relative z-10 py-20 px-6">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-4xl font-bold gradient-text mb-6">
              Try It Now
            </h2>
            <p className="text-xl text-foreground/80">
              Upload a legal document and watch our AI analyze it in real-time.
            </p>
          </div>

          <GlassCard className="text-center" hover3d>
            <div className="border-2 border-dashed border-primary/30 rounded-2xl p-12 transition-all duration-300 hover:border-primary/60">
              {!isAnalyzing ? (
                <>
                  <FileText className="w-16 h-16 mx-auto mb-6 text-primary" />
                  <h3 className="text-2xl font-bold mb-4">Drop your document here</h3>
                  <p className="text-foreground/70 mb-6">
                    Supports PDF, DOC, DOCX files up to 10MB
                  </p>
                  <MagneticButton variant="hero" onClick={handleFileUpload}>
                    <Upload className="w-5 h-5 mr-2" />
                    Choose File
                  </MagneticButton>
                </>
              ) : (
                <div className="space-y-6">
                  <Brain className="w-16 h-16 mx-auto text-neural-purple animate-pulse" />
                  <h3 className="text-2xl font-bold">Analyzing Document...</h3>
                  <div className="w-full bg-muted rounded-full h-3">
                    <div 
                      className="bg-gradient-neural h-3 rounded-full transition-all duration-300"
                      style={{ width: `${uploadProgress}%` }}
                    />
                  </div>
                  <p className="text-foreground/70">
                    AI is processing your document for arbitration clauses...
                  </p>
                </div>
              )}
            </div>
          </GlassCard>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 py-12 px-6 border-t border-border/20">
        <div className="max-w-7xl mx-auto text-center">
          <div className="flex items-center justify-center space-x-3 mb-6">
            <div className="p-2 rounded-lg bg-gradient-gold">
              <Scale className="w-6 h-6 text-background" />
            </div>
            <span className="text-xl font-bold gradient-text">LegalAI</span>
          </div>
          <p className="text-foreground/60">
            © 2024 LegalAI. Revolutionizing arbitration clause detection with advanced AI technology.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
