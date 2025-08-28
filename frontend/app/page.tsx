'use client';

import { useState } from 'react';
import { cn } from '@/lib/utils';
import HealthCheck from '@/components/HealthCheck';
import WebSocketTest from '@/components/WebSocketTest';
import ApiOverview from '@/components/ApiOverview';
import TextAnalyzer from '@/components/TextAnalyzer';
import { 
  Activity,
  Wifi,
  Server,
  FileText,
  Upload,
  Code,
  Database,
  Zap
} from 'lucide-react';

export default function HomePage() {
  const [activeTab, setActiveTab] = useState<'overview' | 'health' | 'websocket' | 'analyzer'>('overview');

  const tabs = [
    {
      id: 'overview' as const,
      name: 'API Overview',
      icon: Server,
      description: 'Available endpoints and features',
    },
    {
      id: 'health' as const,
      name: 'Health Check',
      icon: Activity,
      description: 'Backend connectivity status',
    },
    {
      id: 'websocket' as const,
      name: 'WebSocket Test',
      icon: Wifi,
      description: 'Real-time communication test',
    },
    {
      id: 'analyzer' as const,
      name: 'Text Analyzer',
      icon: FileText,
      description: 'AI arbitration clause detection',
    },
  ];

  const features = [
    {
      name: 'FastAPI Backend',
      description: 'High-performance Python API with automatic documentation',
      icon: Code,
      color: 'text-green-600',
    },
    {
      name: 'WebSocket Support',
      description: 'Real-time bidirectional communication',
      icon: Zap,
      color: 'text-blue-600',
    },
    {
      name: 'Document Analysis',
      description: 'AI-powered arbitration clause detection',
      icon: FileText,
      color: 'text-purple-600',
    },
    {
      name: 'File Upload',
      description: 'Multi-format document processing support',
      icon: Upload,
      color: 'text-orange-600',
    },
    {
      name: 'Vector Storage',
      description: 'Semantic search and similarity matching',
      icon: Database,
      color: 'text-indigo-600',
    },
  ];

  return (
    <div id="homepage" className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      {/* Hero Section */}
      <div id="hero-section" className="text-center mb-12">
        <div id="hero-content" className="max-w-3xl mx-auto">
          <h2 id="hero-title" className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
            Frontend-Backend Integration Demo
          </h2>
          <p id="hero-description" className="text-xl text-gray-600 dark:text-gray-400 mb-8">
            A modern React frontend built with Next.js 14, demonstrating seamless integration 
            with our FastAPI backend for arbitration clause detection and document analysis.
          </p>
          <div id="hero-stats" className="flex justify-center space-x-8 text-sm text-gray-600 dark:text-gray-400">
            <div id="hero-stat-nextjs" className="text-center">
              <div className="font-semibold text-gray-900 dark:text-white">Next.js 14</div>
              <div>App Router</div>
            </div>
            <div id="hero-stat-react" className="text-center">
              <div className="font-semibold text-gray-900 dark:text-white">React 18</div>
              <div>TypeScript</div>
            </div>
            <div id="hero-stat-tailwind" className="text-center">
              <div className="font-semibold text-gray-900 dark:text-white">Tailwind CSS</div>
              <div>Responsive</div>
            </div>
            <div id="hero-stat-api" className="text-center">
              <div className="font-semibold text-gray-900 dark:text-white">FastAPI</div>
              <div>Python Backend</div>
            </div>
          </div>
        </div>
      </div>

      {/* Features Grid */}
      <div id="features-section" className="mb-12">
        <h3 id="features-title" className="text-2xl font-bold text-gray-900 dark:text-white text-center mb-8">
          Integration Features
        </h3>
        <div id="features-grid" className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <div
                key={feature.name}
                id={`feature-${index}`}
                className="glass-card p-6 hover:shadow-lg transition-shadow"
              >
                <div id={`feature-${index}-icon`} className="flex items-center mb-4">
                  <Icon className={cn('h-8 w-8 mr-3', feature.color)} />
                  <h4 className="text-lg font-semibold text-gray-900 dark:text-white">
                    {feature.name}
                  </h4>
                </div>
                <p id={`feature-${index}-description`} className="text-gray-600 dark:text-gray-400">
                  {feature.description}
                </p>
              </div>
            );
          })}
        </div>
      </div>

      {/* Integration Testing Section */}
      <div id="testing-section" className="mb-12">
        <div id="testing-header" className="text-center mb-8">
          <h3 id="testing-title" className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
            Integration Testing Dashboard
          </h3>
          <p id="testing-description" className="text-gray-600 dark:text-gray-400">
            Test the connection and functionality of our FastAPI backend integration
          </p>
        </div>

        {/* Tab Navigation */}
        <div id="tab-navigation" className="flex flex-wrap justify-center mb-8 bg-white/50 dark:bg-slate-800/50 rounded-lg p-1">
          {tabs.map((tab) => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                id={`tab-${tab.id}`}
                onClick={() => setActiveTab(tab.id)}
                className={cn(
                  'flex items-center space-x-2 px-4 py-3 rounded-md transition-colors text-sm font-medium',
                  activeTab === tab.id
                    ? 'bg-white dark:bg-slate-700 text-blue-600 dark:text-blue-400 shadow-sm'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white hover:bg-white/50 dark:hover:bg-slate-700/50'
                )}
              >
                <Icon className="h-4 w-4" />
                <span className="hidden sm:inline">{tab.name}</span>
                <span className="sm:hidden">{tab.name.split(' ')[0]}</span>
              </button>
            );
          })}
        </div>

        {/* Tab Content */}
        <div id="tab-content" className="grid gap-8">
          {activeTab === 'overview' && (
            <div id="overview-tab" className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <ApiOverview id="main-api-overview" />
              <HealthCheck id="overview-health-check" />
            </div>
          )}

          {activeTab === 'health' && (
            <div id="health-tab" className="max-w-2xl mx-auto">
              <HealthCheck id="detailed-health-check" autoRefresh={true} refreshInterval={10000} />
            </div>
          )}

          {activeTab === 'websocket' && (
            <div id="websocket-tab" className="max-w-4xl mx-auto">
              <WebSocketTest id="main-websocket-test" />
            </div>
          )}

          {activeTab === 'analyzer' && (
            <div id="analyzer-tab" className="max-w-4xl mx-auto">
              <TextAnalyzer id="main-text-analyzer" />
            </div>
          )}
        </div>
      </div>

      {/* Technical Details */}
      <div id="technical-section" className="glass-card p-8">
        <h3 id="technical-title" className="text-2xl font-bold text-gray-900 dark:text-white mb-6 text-center">
          Technical Implementation
        </h3>
        <div id="technical-grid" className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 text-sm">
          <div id="frontend-tech" className="space-y-2">
            <h4 className="font-semibold text-gray-900 dark:text-white">Frontend Stack</h4>
            <ul className="space-y-1 text-gray-600 dark:text-gray-400">
              <li>• Next.js 14 with App Router</li>
              <li>• React 18 with TypeScript</li>
              <li>• Tailwind CSS for styling</li>
              <li>• Custom hooks for API integration</li>
              <li>• Responsive design patterns</li>
            </ul>
          </div>
          <div id="backend-tech" className="space-y-2">
            <h4 className="font-semibold text-gray-900 dark:text-white">Backend Integration</h4>
            <ul className="space-y-1 text-gray-600 dark:text-gray-400">
              <li>• FastAPI with automatic docs</li>
              <li>• WebSocket real-time communication</li>
              <li>• CORS configuration</li>
              <li>• RESTful API endpoints</li>
              <li>• Error handling & validation</li>
            </ul>
          </div>
          <div id="deployment-tech" className="space-y-2">
            <h4 className="font-semibold text-gray-900 dark:text-white">Deployment Ready</h4>
            <ul className="space-y-1 text-gray-600 dark:text-gray-400">
              <li>• Vercel optimized configuration</li>
              <li>• Environment variable support</li>
              <li>• Production build settings</li>
              <li>• SEO and accessibility ready</li>
              <li>• Performance optimized</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}