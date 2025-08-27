/**
 * Demo Landing Page - Arbitration Clause Detector
 * Interactive showcase of all platform capabilities
 */

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  FileText, 
  Zap, 
  Shield, 
  Search, 
  Globe, 
  Mic, 
  Users, 
  BarChart3,
  ArrowRight,
  Play,
  CheckCircle,
  Star,
  Download,
  Eye,
  MessageSquare,
  Layers,
  Bot
} from 'lucide-react';
import Link from 'next/link';
import { useRouter } from 'next/router';

// Demo statistics (would come from API in real implementation)
const demoStats = {
  documentsAnalyzed: 10847,
  clausesDetected: 34521,
  accuracy: 94.8,
  languagesSupported: 6,
  activeUsers: 1247,
  contractsGenerated: 856
};

const features = [
  {
    id: 'ai-analysis',
    icon: Bot,
    title: 'AI-Powered Analysis',
    description: 'Advanced NLP models detect arbitration clauses with 94.8% accuracy',
    color: 'blue',
    demo: '/demo/ai-analysis'
  },
  {
    id: 'multilingual',
    icon: Globe,
    title: 'Multi-language Support',
    description: 'Analyze documents in English, Spanish, French, German, Chinese, Japanese',
    color: 'green',
    demo: '/demo/multilingual'
  },
  {
    id: 'voice-interface',
    icon: Mic,
    title: 'Voice Interface',
    description: 'Accessibility-first design with voice commands and screen reader support',
    color: 'purple',
    demo: '/demo/voice'
  },
  {
    id: 'collaboration',
    icon: Users,
    title: 'Real-time Collaboration',
    description: 'Teams can review and annotate documents together in real-time',
    color: 'orange',
    demo: '/demo/collaboration'
  },
  {
    id: 'contract-builder',
    icon: Layers,
    title: 'Contract Builder',
    description: 'Visual drag-and-drop interface for creating contracts with arbitration clauses',
    color: 'indigo',
    demo: '/demo/contract-builder'
  },
  {
    id: 'analytics',
    icon: BarChart3,
    title: 'Analytics Dashboard',
    description: 'Comprehensive insights and reporting on document analysis trends',
    color: 'pink',
    demo: '/demo/analytics'
  }
];

const demoScenarios = [
  {
    id: 'basic-analysis',
    title: 'Basic Document Analysis',
    description: 'Upload a Terms of Service document and see AI detection in action',
    duration: '2 min',
    difficulty: 'Beginner',
    url: '/demo/scenarios/basic'
  },
  {
    id: 'complex-contract',
    title: 'Complex Contract Review',
    description: 'Analyze enterprise agreements with multiple arbitration provisions',
    duration: '5 min',
    difficulty: 'Intermediate',
    url: '/demo/scenarios/complex'
  },
  {
    id: 'multilingual-analysis',
    title: 'Multi-language Processing',
    description: 'Test cross-language clause detection and translation',
    duration: '3 min',
    difficulty: 'Intermediate',
    url: '/demo/scenarios/multilingual'
  },
  {
    id: 'collaboration-workflow',
    title: 'Team Collaboration',
    description: 'Experience real-time document review and approval workflows',
    duration: '7 min',
    difficulty: 'Advanced',
    url: '/demo/scenarios/collaboration'
  }
];

export default function DemoLandingPage() {
  const router = useRouter();
  const [activeDemo, setActiveDemo] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [animatedStats, setAnimatedStats] = useState({
    documentsAnalyzed: 0,
    clausesDetected: 0,
    accuracy: 0,
    languagesSupported: 0,
    activeUsers: 0,
    contractsGenerated: 0
  });

  // Animate statistics on page load
  useEffect(() => {
    const duration = 2000; // 2 seconds
    const steps = 60; // 60 FPS
    const stepDuration = duration / steps;

    let step = 0;
    const timer = setInterval(() => {
      step++;
      const progress = step / steps;
      const easeOutQuart = 1 - Math.pow(1 - progress, 4);

      setAnimatedStats({
        documentsAnalyzed: Math.round(demoStats.documentsAnalyzed * easeOutQuart),
        clausesDetected: Math.round(demoStats.clausesDetected * easeOutQuart),
        accuracy: Number((demoStats.accuracy * easeOutQuart).toFixed(1)),
        languagesSupported: Math.round(demoStats.languagesSupported * easeOutQuart),
        activeUsers: Math.round(demoStats.activeUsers * easeOutQuart),
        contractsGenerated: Math.round(demoStats.contractsGenerated * easeOutQuart)
      });

      if (step >= steps) {
        clearInterval(timer);
      }
    }, stepDuration);

    return () => clearInterval(timer);
  }, []);

  const startDemo = async (demoUrl: string) => {
    setIsLoading(true);
    setActiveDemo(demoUrl);
    
    // Simulate demo preparation
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    router.push(demoUrl);
  };

  const quickStart = () => {
    router.push('/demo/quick-start');
  };

  return (
    <div id="demo-landing-page" className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-50">
      {/* Hero Section */}
      <section id="hero-section" className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-600/10 to-indigo-600/10"></div>
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-16 pb-20">
          <div className="text-center">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              className="mb-8"
            >
              <div id="hero-badge" className="inline-flex items-center px-4 py-2 bg-blue-100 text-blue-800 rounded-full text-sm font-medium mb-6">
                <Zap className="h-4 w-4 mr-2" />
                Interactive Demo Environment
              </div>
              <h1 id="hero-title" className="text-4xl sm:text-5xl lg:text-6xl font-bold text-gray-900 mb-6">
                Arbitration Clause Detector
                <span className="block text-blue-600">Interactive Demo</span>
              </h1>
              <p id="hero-description" className="text-xl text-gray-600 max-w-3xl mx-auto leading-relaxed">
                Experience the power of AI-driven legal document analysis. Upload documents, detect arbitration clauses, 
                and explore advanced features in this comprehensive demo environment.
              </p>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.2 }}
              className="flex flex-col sm:flex-row gap-4 justify-center mb-12"
            >
              <button
                id="quick-start-button"
                onClick={quickStart}
                className="inline-flex items-center px-8 py-4 bg-blue-600 text-white rounded-lg font-semibold text-lg hover:bg-blue-700 transform hover:scale-105 transition-all duration-200 shadow-lg"
              >
                <Play className="h-5 w-5 mr-2" />
                Start Interactive Demo
              </button>
              <button
                id="guided-tour-button"
                onClick={() => router.push('/demo/guided-tour')}
                className="inline-flex items-center px-8 py-4 bg-white text-blue-600 border-2 border-blue-600 rounded-lg font-semibold text-lg hover:bg-blue-50 transform hover:scale-105 transition-all duration-200"
              >
                <Eye className="h-5 w-5 mr-2" />
                Guided Tour
              </button>
            </motion.div>

            {/* Live Statistics */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.4 }}
              id="demo-stats"
              className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4 max-w-5xl mx-auto"
            >
              <div id="stat-documents" className="bg-white/80 backdrop-blur-sm rounded-lg p-4 border border-white/20">
                <div className="text-2xl font-bold text-blue-600">{animatedStats.documentsAnalyzed.toLocaleString()}</div>
                <div className="text-sm text-gray-600">Documents Analyzed</div>
              </div>
              <div id="stat-clauses" className="bg-white/80 backdrop-blur-sm rounded-lg p-4 border border-white/20">
                <div className="text-2xl font-bold text-green-600">{animatedStats.clausesDetected.toLocaleString()}</div>
                <div className="text-sm text-gray-600">Clauses Detected</div>
              </div>
              <div id="stat-accuracy" className="bg-white/80 backdrop-blur-sm rounded-lg p-4 border border-white/20">
                <div className="text-2xl font-bold text-purple-600">{animatedStats.accuracy}%</div>
                <div className="text-sm text-gray-600">Accuracy Rate</div>
              </div>
              <div id="stat-languages" className="bg-white/80 backdrop-blur-sm rounded-lg p-4 border border-white/20">
                <div className="text-2xl font-bold text-orange-600">{animatedStats.languagesSupported}</div>
                <div className="text-sm text-gray-600">Languages</div>
              </div>
              <div id="stat-users" className="bg-white/80 backdrop-blur-sm rounded-lg p-4 border border-white/20">
                <div className="text-2xl font-bold text-indigo-600">{animatedStats.activeUsers.toLocaleString()}</div>
                <div className="text-sm text-gray-600">Active Users</div>
              </div>
              <div id="stat-contracts" className="bg-white/80 backdrop-blur-sm rounded-lg p-4 border border-white/20">
                <div className="text-2xl font-bold text-pink-600">{animatedStats.contractsGenerated.toLocaleString()}</div>
                <div className="text-sm text-gray-600">Contracts Generated</div>
              </div>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Features Showcase */}
      <section id="features-section" className="py-20 bg-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 id="features-title" className="text-3xl font-bold text-gray-900 mb-4">
              Explore Platform Features
            </h2>
            <p id="features-description" className="text-lg text-gray-600 max-w-3xl mx-auto">
              Interactive demonstrations of our core capabilities. Click on any feature to see it in action.
            </p>
          </div>

          <div id="features-grid" className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <motion.div
                  key={feature.id}
                  initial={{ opacity: 0, y: 30 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: index * 0.1 }}
                  id={`feature-${feature.id}`}
                  className="group cursor-pointer"
                  onClick={() => startDemo(feature.demo)}
                >
                  <div className="bg-white rounded-xl shadow-lg border border-gray-100 p-6 h-full hover:shadow-xl hover:scale-105 transition-all duration-300">
                    <div className={`inline-flex items-center justify-center w-12 h-12 bg-${feature.color}-100 text-${feature.color}-600 rounded-lg mb-4 group-hover:scale-110 transition-transform duration-200`}>
                      <Icon className="h-6 w-6" />
                    </div>
                    <h3 className="text-xl font-semibold text-gray-900 mb-2">{feature.title}</h3>
                    <p className="text-gray-600 mb-4">{feature.description}</p>
                    <div className="flex items-center text-blue-600 font-medium group-hover:translate-x-2 transition-transform duration-200">
                      Try Demo
                      <ArrowRight className="h-4 w-4 ml-1" />
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Demo Scenarios */}
      <section id="scenarios-section" className="py-20 bg-gray-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 id="scenarios-title" className="text-3xl font-bold text-gray-900 mb-4">
              Demo Scenarios
            </h2>
            <p id="scenarios-description" className="text-lg text-gray-600 max-w-3xl mx-auto">
              Complete workflows demonstrating real-world usage of the Arbitration Clause Detector.
            </p>
          </div>

          <div id="scenarios-grid" className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {demoScenarios.map((scenario, index) => (
              <motion.div
                key={scenario.id}
                initial={{ opacity: 0, x: index % 2 === 0 ? -30 : 30 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                id={`scenario-${scenario.id}`}
                className="bg-white rounded-xl shadow-lg border border-gray-100 p-6 hover:shadow-xl transition-shadow duration-300"
              >
                <div className="flex items-start justify-between mb-4">
                  <h3 className="text-xl font-semibold text-gray-900">{scenario.title}</h3>
                  <div className="flex items-center space-x-2">
                    <span className="text-sm text-gray-500">{scenario.duration}</span>
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      scenario.difficulty === 'Beginner' ? 'bg-green-100 text-green-600' :
                      scenario.difficulty === 'Intermediate' ? 'bg-yellow-100 text-yellow-600' :
                      'bg-red-100 text-red-600'
                    }`}>
                      {scenario.difficulty}
                    </span>
                  </div>
                </div>
                <p className="text-gray-600 mb-6">{scenario.description}</p>
                <button
                  onClick={() => startDemo(scenario.url)}
                  className="w-full flex items-center justify-center px-4 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors duration-200"
                >
                  <Play className="h-4 w-4 mr-2" />
                  Start Scenario
                </button>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Quick Actions */}
      <section id="quick-actions-section" className="py-20 bg-blue-600 text-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 id="quick-actions-title" className="text-3xl font-bold mb-4">
              Ready to Get Started?
            </h2>
            <p id="quick-actions-description" className="text-xl text-blue-100 max-w-3xl mx-auto">
              Choose your preferred way to explore the Arbitration Clause Detector platform.
            </p>
          </div>

          <div id="quick-actions-grid" className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div id="action-upload" className="text-center">
              <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 hover:bg-white/20 transition-colors duration-200">
                <FileText className="h-12 w-12 mx-auto mb-4 text-blue-200" />
                <h3 className="text-xl font-semibold mb-2">Upload & Analyze</h3>
                <p className="text-blue-100 mb-4">Start with your own document or use our samples</p>
                <button
                  onClick={() => router.push('/demo/upload')}
                  className="bg-white text-blue-600 px-6 py-2 rounded-lg font-medium hover:bg-blue-50 transition-colors duration-200"
                >
                  Upload Document
                </button>
              </div>
            </div>

            <div id="action-api" className="text-center">
              <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 hover:bg-white/20 transition-colors duration-200">
                <Search className="h-12 w-12 mx-auto mb-4 text-blue-200" />
                <h3 className="text-xl font-semibold mb-2">API Playground</h3>
                <p className="text-blue-100 mb-4">Test our APIs with interactive examples</p>
                <button
                  onClick={() => router.push('/demo/api')}
                  className="bg-white text-blue-600 px-6 py-2 rounded-lg font-medium hover:bg-blue-50 transition-colors duration-200"
                >
                  Try API
                </button>
              </div>
            </div>

            <div id="action-download" className="text-center">
              <div className="bg-white/10 backdrop-blur-sm rounded-lg p-6 hover:bg-white/20 transition-colors duration-200">
                <Download className="h-12 w-12 mx-auto mb-4 text-blue-200" />
                <h3 className="text-xl font-semibold mb-2">Download SDK</h3>
                <p className="text-blue-100 mb-4">Integrate with your applications</p>
                <button
                  onClick={() => router.push('/demo/sdk')}
                  className="bg-white text-blue-600 px-6 py-2 rounded-lg font-medium hover:bg-blue-50 transition-colors duration-200"
                >
                  Get SDK
                </button>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Demo Loading Overlay */}
      <AnimatePresence>
        {isLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            id="demo-loading-overlay"
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center"
          >
            <div className="bg-white rounded-lg p-8 max-w-md mx-4 text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">Preparing Demo</h3>
              <p className="text-gray-600">Setting up {activeDemo} environment...</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}