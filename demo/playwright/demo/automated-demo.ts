import { chromium, Browser, Page } from 'playwright';
import { uploadDocument } from '../utils/upload';
import { takeScreenshot, generateMarketingScreenshot } from '../utils/screenshot';
import { smartWait, waitForAnalysisComplete } from '../utils/wait';
import { enableDemoMode } from '../utils/login';
import { SAMPLE_DOCUMENTS, DEMO_SCENARIOS } from '../utils/test-data';

interface DemoStep {
  name: string;
  action: () => Promise<void>;
  screenshot?: boolean;
  duration?: number;
  narration?: string;
}

class AutomatedDemo {
  private browser: Browser | null = null;
  private page: Page | null = null;
  private isRecording = false;
  private currentStep = 0;

  async initialize(options: { headless?: boolean; recordVideo?: boolean } = {}) {
    console.log('🚀 Initializing automated demo...');
    
    this.browser = await chromium.launch({
      headless: options.headless ?? false,
      slowMo: 1000, // Slow down for demo effect
      args: ['--start-maximized']
    });

    const context = await this.browser.newContext({
      viewport: { width: 1920, height: 1080 },
      recordVideo: options.recordVideo ? {
        dir: 'demo-videos/',
        size: { width: 1920, height: 1080 }
      } : undefined
    });

    this.page = await context.newPage();
    this.isRecording = options.recordVideo ?? false;
    
    // Enable demo mode
    await enableDemoMode(this.page);
    
    console.log('✅ Demo environment ready');
  }

  async runQuickDemo(): Promise<void> {
    if (!this.page) throw new Error('Demo not initialized');
    
    console.log('🎬 Starting Quick Demo (2 minutes)...');
    
    const steps: DemoStep[] = [
      {
        name: 'Homepage Introduction',
        action: async () => {
          await this.page!.goto('/');
          await smartWait(this.page!);
          await this.highlightElement('[data-testid="hero-section"]');
        },
        screenshot: true,
        duration: 10000,
        narration: 'Welcome to our AI-powered arbitration detection system'
      },
      
      {
        name: 'Navigate to Upload',
        action: async () => {
          await this.page!.click('[data-testid="upload-document-button"]');
          await smartWait(this.page!);
        },
        screenshot: true,
        duration: 5000,
        narration: 'Let\'s upload a document to analyze'
      },
      
      {
        name: 'Upload Document',
        action: async () => {
          await uploadDocument(this.page!, SAMPLE_DOCUMENTS.arbitration_clause.filename, {
            waitForAnalysis: false
          });
          await this.highlightElement('[data-testid="selected-file"]');
        },
        screenshot: true,
        duration: 8000,
        narration: 'We\'ve selected a terms of service document with arbitration clauses'
      },
      
      {
        name: 'Start Analysis',
        action: async () => {
          await this.page!.click('[data-testid="upload-button"]');
          await smartWait(this.page!);
        },
        screenshot: true,
        duration: 5000,
        narration: 'Now we start the AI analysis process'
      },
      
      {
        name: 'Analysis Progress',
        action: async () => {
          await this.page!.waitForSelector('[data-testid="analysis-progress"]');
          await this.highlightElement('[data-testid="analysis-progress"]');
        },
        screenshot: true,
        duration: 10000,
        narration: 'Our AI is analyzing the document for arbitration clauses'
      },
      
      {
        name: 'Show Results',
        action: async () => {
          await waitForAnalysisComplete(this.page!);
          await this.highlightElement('[data-testid="analysis-result"]');
          await this.highlightElement('[data-testid="confidence-score"]');
        },
        screenshot: true,
        duration: 15000,
        narration: 'Analysis complete! We found arbitration clauses with high confidence'
      },
      
      {
        name: 'Highlight Key Findings',
        action: async () => {
          await this.highlightElement('[data-testid="highlighted-clause"]');
          await this.page!.waitForTimeout(3000);
          await this.scrollToElement('[data-testid="detailed-findings"]');
        },
        screenshot: true,
        duration: 12000,
        narration: 'The system highlights the specific arbitration clauses found'
      },
      
      {
        name: 'Demonstrate Dashboard',
        action: async () => {
          await this.page!.goto('/dashboard');
          await smartWait(this.page!);
          await this.highlightElement('[data-testid="recent-analyses"]');
        },
        screenshot: true,
        duration: 10000,
        narration: 'All analyses are saved to your dashboard for future reference'
      }
    ];

    await this.executeSteps(steps);
    console.log('✅ Quick demo completed');
  }

  async runFullDemo(): Promise<void> {
    if (!this.page) throw new Error('Demo not initialized');
    
    console.log('🎬 Starting Full Demo (10 minutes)...');
    
    const steps: DemoStep[] = [
      {
        name: 'Homepage Overview',
        action: async () => {
          await this.page!.goto('/');
          await smartWait(this.page!);
          await this.tourHomepage();
        },
        screenshot: true,
        duration: 20000,
        narration: 'Complete overview of features and capabilities'
      },
      
      {
        name: 'Upload Multiple Documents',
        action: async () => {
          await this.page!.goto('/upload');
          await this.demoMultipleUploads();
        },
        screenshot: true,
        duration: 45000,
        narration: 'Demonstrating batch document processing'
      },
      
      {
        name: 'Compare Different Document Types',
        action: async () => {
          await this.demoDocumentComparison();
        },
        screenshot: true,
        duration: 60000,
        narration: 'Comparing documents with and without arbitration clauses'
      },
      
      {
        name: 'Show Analytics Dashboard',
        action: async () => {
          await this.page!.goto('/dashboard');
          await this.demoAnalyticsDashboard();
        },
        screenshot: true,
        duration: 40000,
        narration: 'Comprehensive analytics and reporting features'
      },
      
      {
        name: 'Export Functionality',
        action: async () => {
          await this.demoExportFeatures();
        },
        screenshot: true,
        duration: 25000,
        narration: 'Export results in multiple formats'
      },
      
      {
        name: 'Mobile Responsive Demo',
        action: async () => {
          await this.demoMobileExperience();
        },
        screenshot: true,
        duration: 30000,
        narration: 'Fully responsive design for all devices'
      },
      
      {
        name: 'API Integration Preview',
        action: async () => {
          await this.demoAPIIntegration();
        },
        screenshot: true,
        duration: 20000,
        narration: 'Enterprise API integration capabilities'
      }
    ];

    await this.executeSteps(steps);
    console.log('✅ Full demo completed');
  }

  async runTechnicalDemo(): Promise<void> {
    if (!this.page) throw new Error('Demo not initialized');
    
    console.log('🎬 Starting Technical Demo (15 minutes)...');
    
    const steps: DemoStep[] = [
      {
        name: 'Architecture Overview',
        action: async () => {
          await this.page!.goto('/technical');
          await this.demoArchitecture();
        },
        screenshot: true,
        duration: 60000,
        narration: 'Deep dive into system architecture and AI models'
      },
      
      {
        name: 'API Examples',
        action: async () => {
          await this.demoAPIExamples();
        },
        screenshot: true,
        duration: 90000,
        narration: 'Live API demonstrations and integration examples'
      },
      
      {
        name: 'Performance Metrics',
        action: async () => {
          await this.demoPerformanceMetrics();
        },
        screenshot: true,
        duration: 60000,
        narration: 'Performance benchmarks and scalability metrics'
      },
      
      {
        name: 'Security Features',
        action: async () => {
          await this.demoSecurityFeatures();
        },
        screenshot: true,
        duration: 45000,
        narration: 'Enterprise-grade security and compliance features'
      },
      
      {
        name: 'Customization Options',
        action: async () => {
          await this.demoCustomizationOptions();
        },
        screenshot: true,
        duration: 75000,
        narration: 'Customization and white-label options'
      },
      
      {
        name: 'Advanced Analytics',
        action: async () => {
          await this.demoAdvancedAnalytics();
        },
        screenshot: true,
        duration: 60000,
        narration: 'Advanced analytics and reporting capabilities'
      }
    ];

    await this.executeSteps(steps);
    console.log('✅ Technical demo completed');
  }

  private async executeSteps(steps: DemoStep[]): Promise<void> {
    for (const [index, step] of steps.entries()) {
      this.currentStep = index + 1;
      
      console.log(`📍 Step ${this.currentStep}/${steps.length}: ${step.name}`);
      
      if (step.narration) {
        await this.showNarration(step.narration);
      }
      
      try {
        await step.action();
        
        if (step.screenshot) {
          await takeScreenshot(this.page!, `demo-step-${this.currentStep}-${step.name.toLowerCase().replace(/\s+/g, '-')}`);
        }
        
        if (step.duration) {
          await this.page!.waitForTimeout(step.duration);
        }
        
      } catch (error) {
        console.error(`❌ Error in step "${step.name}":`, error);
        // Continue with next step
      }
    }
  }

  private async tourHomepage(): Promise<void> {
    // Scroll through and highlight key sections
    const sections = [
      '[data-testid="hero-section"]',
      '[data-testid="features-section"]',
      '[data-testid="stats-section"]',
      '[data-testid="testimonials-section"]',
      '[data-testid="cta-section"]'
    ];
    
    for (const section of sections) {
      await this.scrollToElement(section);
      await this.highlightElement(section);
      await this.page!.waitForTimeout(3000);
    }
  }

  private async demoMultipleUploads(): Promise<void> {
    const documents = [
      SAMPLE_DOCUMENTS.arbitration_clause,
      SAMPLE_DOCUMENTS.no_arbitration,
      SAMPLE_DOCUMENTS.complex_arbitration
    ];
    
    for (const doc of documents) {
      await uploadDocument(this.page!, doc.filename, {
        waitForAnalysis: true,
        expectedResult: doc.expectedResult as any
      });
      
      await this.highlightElement('[data-testid="analysis-result"]');
      await this.page!.waitForTimeout(5000);
      
      // Reset for next upload
      await this.page!.reload();
      await smartWait(this.page!);
    }
  }

  private async demoDocumentComparison(): Promise<void> {
    // Demonstrate side-by-side comparison
    await this.page!.goto('/compare');
    await smartWait(this.page!);
    
    // Upload two different documents for comparison
    await this.highlightElement('[data-testid="comparison-interface"]');
    await this.page!.waitForTimeout(5000);
  }

  private async demoAnalyticsDashboard(): Promise<void> {
    const dashboardElements = [
      '[data-testid="stats-overview"]',
      '[data-testid="analytics-charts"]',
      '[data-testid="recent-analyses"]',
      '[data-testid="document-library"]'
    ];
    
    for (const element of dashboardElements) {
      await this.scrollToElement(element);
      await this.highlightElement(element);
      await this.page!.waitForTimeout(4000);
    }
  }

  private async demoExportFeatures(): Promise<void> {
    const exportButton = this.page!.locator('[data-testid="export-results"]');
    if (await exportButton.isVisible()) {
      await exportButton.click();
      await smartWait(this.page!);
      
      await this.highlightElement('[data-testid="export-options"]');
      await this.page!.waitForTimeout(3000);
    }
  }

  private async demoMobileExperience(): Promise<void> {
    // Switch to mobile viewport
    await this.page!.setViewportSize({ width: 375, height: 667 });
    await this.page!.reload();
    await smartWait(this.page!);
    
    await takeScreenshot(this.page!, 'mobile-experience', { deviceType: 'mobile' });
    
    // Switch back to desktop
    await this.page!.setViewportSize({ width: 1920, height: 1080 });
  }

  private async demoAPIIntegration(): Promise<void> {
    // Show API documentation or examples
    await this.page!.goto('/api-docs');
    await smartWait(this.page!);
    
    await this.highlightElement('[data-testid="api-examples"]');
    await this.page!.waitForTimeout(5000);
  }

  private async demoArchitecture(): Promise<void> {
    // Technical architecture demonstration
    await this.highlightElement('[data-testid="architecture-diagram"]');
    await this.page!.waitForTimeout(10000);
  }

  private async demoAPIExamples(): Promise<void> {
    // Live API demonstrations
    await this.highlightElement('[data-testid="api-playground"]');
    await this.page!.waitForTimeout(15000);
  }

  private async demoPerformanceMetrics(): Promise<void> {
    // Performance benchmarks
    await this.highlightElement('[data-testid="performance-charts"]');
    await this.page!.waitForTimeout(10000);
  }

  private async demoSecurityFeatures(): Promise<void> {
    // Security and compliance features
    await this.highlightElement('[data-testid="security-features"]');
    await this.page!.waitForTimeout(8000);
  }

  private async demoCustomizationOptions(): Promise<void> {
    // Customization and white-label options
    await this.highlightElement('[data-testid="customization-options"]');
    await this.page!.waitForTimeout(12000);
  }

  private async demoAdvancedAnalytics(): Promise<void> {
    // Advanced analytics and reporting
    await this.highlightElement('[data-testid="advanced-analytics"]');
    await this.page!.waitForTimeout(10000);
  }

  private async highlightElement(selector: string): Promise<void> {
    try {
      await this.page!.evaluate((sel) => {
        const element = document.querySelector(sel);
        if (element) {
          element.style.outline = '3px solid #007acc';
          element.style.outlineOffset = '2px';
          element.style.transition = 'all 0.3s ease';
          element.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
      }, selector);
      
      await this.page!.waitForTimeout(2000);
      
      // Remove highlight
      await this.page!.evaluate((sel) => {
        const element = document.querySelector(sel);
        if (element) {
          element.style.outline = '';
          element.style.outlineOffset = '';
        }
      }, selector);
    } catch (error) {
      console.warn(`Could not highlight element: ${selector}`);
    }
  }

  private async scrollToElement(selector: string): Promise<void> {
    try {
      await this.page!.locator(selector).scrollIntoViewIfNeeded();
      await this.page!.waitForTimeout(1000);
    } catch (error) {
      console.warn(`Could not scroll to element: ${selector}`);
    }
  }

  private async showNarration(text: string): Promise<void> {
    // Display narration overlay
    await this.page!.evaluate((narration) => {
      const overlay = document.createElement('div');
      overlay.style.cssText = `
        position: fixed;
        top: 20px;
        left: 50%;
        transform: translateX(-50%);
        background: rgba(0, 0, 0, 0.8);
        color: white;
        padding: 15px 25px;
        border-radius: 8px;
        font-size: 18px;
        font-weight: 500;
        z-index: 10000;
        max-width: 600px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
      `;
      overlay.textContent = narration;
      overlay.id = 'demo-narration';
      
      // Remove existing narration
      const existing = document.getElementById('demo-narration');
      if (existing) existing.remove();
      
      document.body.appendChild(overlay);
      
      // Auto-remove after delay
      setTimeout(() => overlay.remove(), 4000);
    }, text);
    
    await this.page!.waitForTimeout(1000);
  }

  async cleanup(): Promise<void> {
    if (this.page && this.isRecording) {
      console.log('💾 Saving demo video...');
      await this.page.close();
    }
    
    if (this.browser) {
      await this.browser.close();
    }
    
    console.log('🧹 Demo cleanup completed');
  }
}

// Export for use in other scripts
export { AutomatedDemo };

// CLI usage
if (require.main === module) {
  const demoType = process.argv[2] || 'quick';
  const recordVideo = process.argv.includes('--record');
  const headless = process.argv.includes('--headless');
  
  const demo = new AutomatedDemo();
  
  (async () => {
    try {
      await demo.initialize({ recordVideo, headless });
      
      switch (demoType) {
        case 'quick':
          await demo.runQuickDemo();
          break;
        case 'full':
          await demo.runFullDemo();
          break;
        case 'technical':
          await demo.runTechnicalDemo();
          break;
        default:
          console.error('Unknown demo type. Use: quick, full, or technical');
          process.exit(1);
      }
      
    } catch (error) {
      console.error('❌ Demo failed:', error);
    } finally {
      await demo.cleanup();
    }
  })();
}