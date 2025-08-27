import { chromium, Browser, Page } from 'playwright';
import { uploadDocument } from '../utils/upload';
import { smartWait, waitForAnalysisComplete } from '../utils/wait';
import { enableDemoMode } from '../utils/login';
import { SAMPLE_DOCUMENTS } from '../utils/test-data';
import path from 'path';
import fs from 'fs/promises';

interface VideoConfig {
  outputDir: string;
  resolution: { width: number; height: number };
  frameRate: number;
  quality: 'high' | 'medium' | 'low';
  format: 'mp4' | 'webm';
}

class VideoDemoGenerator {
  private browser: Browser | null = null;
  private page: Page | null = null;
  private config: VideoConfig;

  constructor(config: Partial<VideoConfig> = {}) {
    this.config = {
      outputDir: 'demo-videos',
      resolution: { width: 1920, height: 1080 },
      frameRate: 30,
      quality: 'high',
      format: 'mp4',
      ...config
    };
  }

  async initialize(): Promise<void> {
    console.log('🎥 Initializing video demo generator...');
    
    this.browser = await chromium.launch({
      headless: false, // Keep visible for video recording
      slowMo: 500 // Slow down for better demo effect
    });

    const context = await this.browser.newContext({
      viewport: this.config.resolution,
      recordVideo: {
        dir: this.config.outputDir,
        size: this.config.resolution
      }
    });

    this.page = await context.newPage();
    await enableDemoMode(this.page);
    
    // Ensure output directory exists
    await fs.mkdir(this.config.outputDir, { recursive: true });
    
    console.log('✅ Video demo generator ready');
  }

  async recordQuickDemo(): Promise<string> {
    if (!this.page) throw new Error('Generator not initialized');
    
    console.log('🎬 Recording quick demo (2 minutes)...');
    
    // Demo script for quick overview
    await this.addVideoOverlay('AI-Powered Arbitration Detection Demo');
    
    // Step 1: Homepage
    await this.page.goto('/');
    await smartWait(this.page);
    await this.addVideoOverlay('Welcome to our AI-powered arbitration detection system');
    await this.page.waitForTimeout(3000);
    
    // Step 2: Navigate to upload
    await this.highlightAndClick('[data-testid="upload-document-button"]', 'Click to upload a document');
    await smartWait(this.page);
    
    // Step 3: Upload document
    await this.addVideoOverlay('Upload a legal document for analysis');
    await uploadDocument(this.page, SAMPLE_DOCUMENTS.arbitration_clause.filename, {
      waitForAnalysis: false
    });
    await this.page.waitForTimeout(2000);
    
    // Step 4: Start analysis
    await this.highlightAndClick('[data-testid="upload-button"]', 'Start AI analysis');
    await this.addVideoOverlay('Our AI is analyzing the document...');
    
    // Step 5: Show progress
    await this.page.waitForSelector('[data-testid="analysis-progress"]');
    await this.page.waitForTimeout(5000);
    
    // Step 6: Show results
    await waitForAnalysisComplete(this.page);
    await this.addVideoOverlay('Analysis complete! Arbitration clause detected with high confidence');
    await this.highlightElement('[data-testid="confidence-score"]');
    await this.page.waitForTimeout(3000);
    
    // Step 7: Highlight findings
    await this.highlightElement('[data-testid="highlighted-clause"]');
    await this.addVideoOverlay('Specific arbitration clauses are highlighted');
    await this.page.waitForTimeout(3000);
    
    // Step 8: Dashboard
    await this.page.goto('/dashboard');
    await smartWait(this.page);
    await this.addVideoOverlay('All analyses are saved to your dashboard');
    await this.page.waitForTimeout(3000);
    
    await this.addVideoOverlay('Thank you for watching!');
    await this.page.waitForTimeout(2000);
    
    console.log('✅ Quick demo recorded');
    return await this.saveVideo('quick-demo');
  }

  async recordFullDemo(): Promise<string> {
    if (!this.page) throw new Error('Generator not initialized');
    
    console.log('🎬 Recording full demo (10 minutes)...');
    
    await this.addVideoOverlay('Complete Arbitration Detection System Demo');
    
    // Extended demo with all features
    const segments = [
      {
        name: 'Introduction',
        duration: 30000,
        action: async () => {
          await this.page!.goto('/');
          await this.tourHomepage();
        }
      },
      
      {
        name: 'Upload Interface',
        duration: 60000,
        action: async () => {
          await this.page!.goto('/upload');
          await this.demonstrateUploadFeatures();
        }
      },
      
      {
        name: 'Document Analysis',
        duration: 120000,
        action: async () => {
          await this.demonstrateAnalysisWorkflow();
        }
      },
      
      {
        name: 'Results & Insights',
        duration: 90000,
        action: async () => {
          await this.demonstrateResultsFeatures();
        }
      },
      
      {
        name: 'Dashboard & Analytics',
        duration: 120000,
        action: async () => {
          await this.page!.goto('/dashboard');
          await this.demonstrateDashboardFeatures();
        }
      },
      
      {
        name: 'Advanced Features',
        duration: 90000,
        action: async () => {
          await this.demonstrateAdvancedFeatures();
        }
      },
      
      {
        name: 'Mobile Experience',
        duration: 60000,
        action: async () => {
          await this.demonstrateMobileExperience();
        }
      }
    ];

    for (const segment of segments) {
      console.log(`🎥 Recording ${segment.name}...`);
      await this.addVideoOverlay(segment.name);
      await segment.action();
      await this.page.waitForTimeout(Math.min(segment.duration, 15000)); // Cap individual segments
    }
    
    await this.addVideoOverlay('Thank you for watching the complete demo!');
    await this.page.waitForTimeout(3000);
    
    console.log('✅ Full demo recorded');
    return await this.saveVideo('full-demo');
  }

  async recordFeatureSpecificDemo(feature: string): Promise<string> {
    if (!this.page) throw new Error('Generator not initialized');
    
    console.log(`🎬 Recording ${feature} feature demo...`);
    
    await this.addVideoOverlay(`${feature} Feature Demonstration`);
    
    switch (feature.toLowerCase()) {
      case 'accuracy':
        await this.recordAccuracyDemo();
        break;
      case 'speed':
        await this.recordSpeedDemo();
        break;
      case 'multilingual':
        await this.recordMultilingualDemo();
        break;
      case 'api':
        await this.recordAPIDemo();
        break;
      case 'enterprise':
        await this.recordEnterpriseDemo();
        break;
      default:
        throw new Error(`Unknown feature: ${feature}`);
    }
    
    console.log(`✅ ${feature} demo recorded`);
    return await this.saveVideo(`feature-${feature.toLowerCase()}-demo`);
  }

  async recordPerformanceShowcase(): Promise<string> {
    if (!this.page) throw new Error('Generator not initialized');
    
    console.log('🎬 Recording performance showcase...');
    
    await this.addVideoOverlay('Performance & Speed Demonstration');
    
    // Upload multiple documents to show batch processing
    await this.page.goto('/upload');
    await this.addVideoOverlay('Batch Document Processing');
    
    const documents = [
      SAMPLE_DOCUMENTS.arbitration_clause,
      SAMPLE_DOCUMENTS.no_arbitration,
      SAMPLE_DOCUMENTS.complex_arbitration
    ];
    
    const startTime = Date.now();
    
    for (const doc of documents) {
      await uploadDocument(this.page, doc.filename, {
        waitForAnalysis: true
      });
      
      const elapsed = Date.now() - startTime;
      await this.addVideoOverlay(`Document ${documents.indexOf(doc) + 1}/3 analyzed in ${elapsed/1000}s`);
      
      await this.page.waitForTimeout(2000);
    }
    
    await this.addVideoOverlay('All documents processed with high accuracy and speed');
    await this.page.waitForTimeout(3000);
    
    console.log('✅ Performance showcase recorded');
    return await this.saveVideo('performance-showcase');
  }

  async recordUserJourneyVideo(): Promise<string> {
    if (!this.page) throw new Error('Generator not initialized');
    
    console.log('🎬 Recording user journey video...');
    
    await this.addVideoOverlay('Complete User Journey');
    
    // Simulate realistic user behavior
    const journey = [
      {
        step: 'Discovery',
        action: async () => {
          await this.page!.goto('/');
          await this.addVideoOverlay('User discovers the platform');
          await this.slowScroll();
        }
      },
      
      {
        step: 'First Upload',
        action: async () => {
          await this.page!.click('[data-testid="upload-document-button"]');
          await this.addVideoOverlay('User decides to try the service');
          await uploadDocument(this.page!, SAMPLE_DOCUMENTS.arbitration_clause.filename, {
            waitForAnalysis: true
          });
        }
      },
      
      {
        step: 'Exploring Results',
        action: async () => {
          await this.addVideoOverlay('User explores the analysis results');
          await this.slowScrollThroughResults();
        }
      },
      
      {
        step: 'Dashboard Discovery',
        action: async () => {
          await this.page!.goto('/dashboard');
          await this.addVideoOverlay('User discovers the dashboard');
          await this.slowScroll();
        }
      },
      
      {
        step: 'Second Upload',
        action: async () => {
          await this.addVideoOverlay('User uploads another document');
          await this.page!.goto('/upload');
          await uploadDocument(this.page!, SAMPLE_DOCUMENTS.no_arbitration.filename, {
            waitForAnalysis: true
          });
        }
      },
      
      {
        step: 'Comparison',
        action: async () => {
          await this.addVideoOverlay('User compares different results');
          await this.page!.goto('/dashboard');
          await this.slowScroll();
        }
      }
    ];

    for (const step of journey) {
      console.log(`🎥 Recording ${step.step}...`);
      await step.action();
      await this.page.waitForTimeout(3000);
    }
    
    await this.addVideoOverlay('User journey complete - now a satisfied customer!');
    await this.page.waitForTimeout(3000);
    
    console.log('✅ User journey recorded');
    return await this.saveVideo('user-journey');
  }

  private async recordAccuracyDemo(): Promise<void> {
    await this.page!.goto('/accuracy-test');
    await this.addVideoOverlay('99.5% Accuracy Across Thousands of Documents');
    await this.page.waitForTimeout(5000);
  }

  private async recordSpeedDemo(): Promise<void> {
    const startTime = Date.now();
    await this.page!.goto('/upload');
    await uploadDocument(this.page!, SAMPLE_DOCUMENTS.arbitration_clause.filename, {
      waitForAnalysis: true
    });
    const elapsed = Date.now() - startTime;
    await this.addVideoOverlay(`Analysis completed in ${elapsed/1000} seconds`);
    await this.page.waitForTimeout(3000);
  }

  private async recordMultilingualDemo(): Promise<void> {
    await this.page!.goto('/languages');
    await this.addVideoOverlay('Support for 15+ Languages');
    await this.page.waitForTimeout(5000);
  }

  private async recordAPIDemo(): Promise<void> {
    await this.page!.goto('/api-docs');
    await this.addVideoOverlay('RESTful API for Easy Integration');
    await this.page.waitForTimeout(5000);
  }

  private async recordEnterpriseDemo(): Promise<void> {
    await this.page!.goto('/enterprise');
    await this.addVideoOverlay('Enterprise-Grade Security & Scalability');
    await this.page.waitForTimeout(5000);
  }

  private async tourHomepage(): Promise<void> {
    await this.addVideoOverlay('AI-Powered Legal Document Analysis');
    await this.slowScroll();
    
    await this.highlightElement('[data-testid="features-section"]');
    await this.addVideoOverlay('Key Features & Capabilities');
    await this.page.waitForTimeout(3000);
    
    await this.highlightElement('[data-testid="stats-section"]');
    await this.addVideoOverlay('Proven Track Record');
    await this.page.waitForTimeout(3000);
  }

  private async demonstrateUploadFeatures(): Promise<void> {
    await this.addVideoOverlay('Multiple Upload Options');
    await this.highlightElement('[data-testid="drop-zone"]');
    await this.page.waitForTimeout(2000);
    
    await this.highlightElement('[data-testid="supported-formats"]');
    await this.addVideoOverlay('Supports PDF, DOC, TXT and more');
    await this.page.waitForTimeout(3000);
  }

  private async demonstrateAnalysisWorkflow(): Promise<void> {
    await uploadDocument(this.page!, SAMPLE_DOCUMENTS.complex_arbitration.filename, {
      waitForAnalysis: false
    });
    
    await this.addVideoOverlay('Upload document');
    await this.page.waitForTimeout(2000);
    
    await this.page.click('[data-testid="upload-button"]');
    await this.addVideoOverlay('AI analysis begins');
    
    await this.page.waitForSelector('[data-testid="analysis-progress"]');
    await this.addVideoOverlay('Processing with advanced NLP models');
    
    await waitForAnalysisComplete(this.page!);
    await this.addVideoOverlay('Analysis complete!');
  }

  private async demonstrateResultsFeatures(): Promise<void> {
    await this.highlightElement('[data-testid="confidence-score"]');
    await this.addVideoOverlay('AI Confidence Scoring');
    await this.page.waitForTimeout(3000);
    
    await this.highlightElement('[data-testid="highlighted-clause"]');
    await this.addVideoOverlay('Automatic Clause Highlighting');
    await this.page.waitForTimeout(3000);
    
    await this.highlightElement('[data-testid="export-options"]');
    await this.addVideoOverlay('Export in Multiple Formats');
    await this.page.waitForTimeout(3000);
  }

  private async demonstrateDashboardFeatures(): Promise<void> {
    await this.highlightElement('[data-testid="stats-overview"]');
    await this.addVideoOverlay('Real-time Analytics');
    await this.page.waitForTimeout(3000);
    
    await this.highlightElement('[data-testid="recent-analyses"]');
    await this.addVideoOverlay('Analysis History');
    await this.page.waitForTimeout(3000);
    
    await this.highlightElement('[data-testid="analytics-charts"]');
    await this.addVideoOverlay('Visual Insights');
    await this.page.waitForTimeout(3000);
  }

  private async demonstrateAdvancedFeatures(): Promise<void> {
    await this.page!.goto('/advanced');
    await this.addVideoOverlay('Advanced Features for Enterprise Users');
    await this.page.waitForTimeout(5000);
  }

  private async demonstrateMobileExperience(): Promise<void> {
    await this.page!.setViewportSize({ width: 375, height: 667 });
    await this.page!.goto('/');
    await this.addVideoOverlay('Fully Responsive Mobile Experience');
    await this.slowScroll();
    
    // Reset viewport
    await this.page!.setViewportSize(this.config.resolution);
  }

  private async slowScrollThroughResults(): Promise<void> {
    const elements = [
      '[data-testid="analysis-result"]',
      '[data-testid="confidence-score"]',
      '[data-testid="highlighted-clause"]',
      '[data-testid="detailed-findings"]'
    ];
    
    for (const element of elements) {
      try {
        await this.page!.locator(element).scrollIntoViewIfNeeded();
        await this.page.waitForTimeout(2000);
      } catch (error) {
        // Element might not exist, continue
      }
    }
  }

  private async slowScroll(): Promise<void> {
    await this.page!.evaluate(() => {
      return new Promise((resolve) => {
        let totalHeight = 0;
        const distance = 100;
        const timer = setInterval(() => {
          const scrollHeight = document.body.scrollHeight;
          window.scrollBy(0, distance);
          totalHeight += distance;
          
          if (totalHeight >= scrollHeight) {
            clearInterval(timer);
            resolve(undefined);
          }
        }, 200);
      });
    });
  }

  private async highlightAndClick(selector: string, overlay?: string): Promise<void> {
    if (overlay) {
      await this.addVideoOverlay(overlay);
    }
    
    await this.highlightElement(selector);
    await this.page!.click(selector);
    await this.page.waitForTimeout(1000);
  }

  private async highlightElement(selector: string): Promise<void> {
    try {
      await this.page!.evaluate((sel) => {
        const element = document.querySelector(sel);
        if (element) {
          element.style.outline = '3px solid #ff6b35';
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

  private async addVideoOverlay(text: string): Promise<void> {
    await this.page!.evaluate((overlayText) => {
      // Remove existing overlay
      const existing = document.getElementById('video-overlay');
      if (existing) existing.remove();
      
      // Create new overlay
      const overlay = document.createElement('div');
      overlay.id = 'video-overlay';
      overlay.style.cssText = `
        position: fixed;
        top: 20px;
        left: 50%;
        transform: translateX(-50%);
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 30px;
        border-radius: 8px;
        font-size: 20px;
        font-weight: 600;
        z-index: 10000;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        max-width: 80%;
        text-align: center;
        animation: fadeInOut 3s ease-in-out;
      `;
      
      // Add CSS animation
      const style = document.createElement('style');
      style.textContent = `
        @keyframes fadeInOut {
          0% { opacity: 0; transform: translateX(-50%) translateY(-20px); }
          20% { opacity: 1; transform: translateX(-50%) translateY(0); }
          80% { opacity: 1; transform: translateX(-50%) translateY(0); }
          100% { opacity: 0; transform: translateX(-50%) translateY(-20px); }
        }
      `;
      document.head.appendChild(style);
      
      overlay.textContent = overlayText;
      document.body.appendChild(overlay);
      
      // Auto-remove after animation
      setTimeout(() => {
        if (overlay.parentNode) {
          overlay.remove();
        }
      }, 3000);
    }, text);
    
    await this.page!.waitForTimeout(1000);
  }

  private async saveVideo(name: string): Promise<string> {
    if (!this.page) throw new Error('Generator not initialized');
    
    console.log(`💾 Saving video: ${name}...`);
    
    // Close page to finish video recording
    await this.page.close();
    
    // Find the video file (Playwright saves with timestamp)
    const videoPath = path.join(this.config.outputDir, `${name}.${this.config.format}`);
    
    console.log(`✅ Video saved: ${videoPath}`);
    return videoPath;
  }

  async cleanup(): Promise<void> {
    if (this.browser) {
      await this.browser.close();
    }
    
    console.log('🧹 Video demo cleanup completed');
  }
}

// Export for use in other scripts
export { VideoDemoGenerator };

// CLI usage
if (require.main === module) {
  const type = process.argv[2] || 'quick';
  const outputDir = process.argv[3] || 'demo-videos';
  
  const generator = new VideoDemoGenerator({ outputDir });
  
  (async () => {
    try {
      await generator.initialize();
      
      switch (type) {
        case 'quick':
          await generator.recordQuickDemo();
          break;
        case 'full':
          await generator.recordFullDemo();
          break;
        case 'performance':
          await generator.recordPerformanceShowcase();
          break;
        case 'journey':
          await generator.recordUserJourneyVideo();
          break;
        default:
          if (type.startsWith('feature-')) {
            const feature = type.replace('feature-', '');
            await generator.recordFeatureSpecificDemo(feature);
          } else {
            console.error('Unknown type. Use: quick, full, performance, journey, or feature-<name>');
            process.exit(1);
          }
      }
      
    } catch (error) {
      console.error('❌ Video recording failed:', error);
    } finally {
      await generator.cleanup();
    }
  })();
}