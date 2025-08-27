import { chromium, Browser, Page } from 'playwright';
import { uploadDocument } from '../utils/upload';
import { 
  takeScreenshot, 
  generateMarketingScreenshot, 
  createScreenshotGallery,
  takeResponsiveScreenshots 
} from '../utils/screenshot';
import { smartWait } from '../utils/wait';
import { enableDemoMode } from '../utils/login';
import { SAMPLE_DOCUMENTS } from '../utils/test-data';
import path from 'path';
import fs from 'fs/promises';

interface ScreenshotConfig {
  outputDir: string;
  themes: ('light' | 'dark')[];
  devices: ('desktop' | 'tablet' | 'mobile')[];
  quality: 'high' | 'medium' | 'low';
}

class ScreenshotGenerator {
  private browser: Browser | null = null;
  private page: Page | null = null;
  private config: ScreenshotConfig;

  constructor(config: Partial<ScreenshotConfig> = {}) {
    this.config = {
      outputDir: 'marketing-screenshots',
      themes: ['light', 'dark'],
      devices: ['desktop', 'tablet', 'mobile'],
      quality: 'high',
      ...config
    };
  }

  async initialize(): Promise<void> {
    console.log('📸 Initializing screenshot generator...');
    
    this.browser = await chromium.launch({
      headless: true
    });

    const context = await this.browser.newContext({
      viewport: { width: 1920, height: 1080 }
    });

    this.page = await context.newPage();
    await enableDemoMode(this.page);
    
    // Ensure output directory exists
    await fs.mkdir(this.config.outputDir, { recursive: true });
    
    console.log('✅ Screenshot generator ready');
  }

  async generateMarketingScreenshots(): Promise<void> {
    if (!this.page) throw new Error('Generator not initialized');
    
    console.log('🎨 Generating marketing screenshots...');
    
    const scenarios = [
      {
        name: 'hero-showcase',
        description: 'Homepage hero section with key value propositions',
        setup: async () => {
          await this.page!.goto('/');
          await smartWait(this.page!);
        },
        annotations: [
          {
            selector: '[data-testid="hero-title"]',
            text: 'AI-Powered Arbitration Detection',
            position: 'bottom' as const
          },
          {
            selector: '[data-testid="upload-document-button"]',
            text: 'Upload & Analyze in Seconds',
            position: 'right' as const
          },
          {
            selector: '[data-testid="accuracy-badge"]',
            text: '99.5% Accuracy Rate',
            position: 'left' as const
          }
        ]
      },
      
      {
        name: 'upload-interface',
        description: 'Clean and intuitive upload interface',
        setup: async () => {
          await this.page!.goto('/upload');
          await smartWait(this.page!);
        },
        annotations: [
          {
            selector: '[data-testid="drop-zone"]',
            text: 'Drag & Drop Support',
            position: 'top' as const
          },
          {
            selector: '[data-testid="supported-formats"]',
            text: 'Multiple File Formats',
            position: 'right' as const
          }
        ]
      },
      
      {
        name: 'analysis-results',
        description: 'Comprehensive analysis results with confidence scoring',
        setup: async () => {
          await this.page!.goto('/upload');
          await uploadDocument(this.page!, SAMPLE_DOCUMENTS.arbitration_clause.filename, {
            waitForAnalysis: true
          });
        },
        annotations: [
          {
            selector: '[data-testid="confidence-score"]',
            text: 'AI Confidence Scoring',
            position: 'right' as const
          },
          {
            selector: '[data-testid="highlighted-clause"]',
            text: 'Automatic Clause Detection',
            position: 'left' as const
          },
          {
            selector: '[data-testid="analysis-explanation"]',
            text: 'Clear Explanations',
            position: 'bottom' as const
          }
        ]
      },
      
      {
        name: 'dashboard-analytics',
        description: 'Powerful analytics and reporting dashboard',
        setup: async () => {
          await this.page!.goto('/dashboard');
          await smartWait(this.page!);
        },
        annotations: [
          {
            selector: '[data-testid="stats-overview"]',
            text: 'Real-time Analytics',
            position: 'top' as const
          },
          {
            selector: '[data-testid="analytics-charts"]',
            text: 'Visual Insights',
            position: 'right' as const
          },
          {
            selector: '[data-testid="recent-analyses"]',
            text: 'Complete History',
            position: 'left' as const
          }
        ]
      },
      
      {
        name: 'mobile-experience',
        description: 'Fully responsive mobile experience',
        setup: async () => {
          await this.page!.setViewportSize({ width: 375, height: 667 });
          await this.page!.goto('/');
          await smartWait(this.page!);
        },
        annotations: [
          {
            selector: '[data-testid="mobile-menu-button"]',
            text: 'Mobile-Optimized Interface',
            position: 'bottom' as const
          }
        ]
      }
    ];

    for (const theme of this.config.themes) {
      console.log(`📸 Generating ${theme} theme screenshots...`);
      
      await createScreenshotGallery(this.page, scenarios, {
        mode: theme,
        fullPage: true,
        animations: 'disabled'
      });
    }
    
    console.log('✅ Marketing screenshots generated');
  }

  async generateFeatureShowcase(): Promise<void> {
    if (!this.page) throw new Error('Generator not initialized');
    
    console.log('🌟 Generating feature showcase screenshots...');
    
    const features = [
      {
        name: 'accuracy-demonstration',
        title: 'Industry-Leading Accuracy',
        setup: async () => {
          await this.page!.goto('/accuracy-test');
          await smartWait(this.page!);
        }
      },
      
      {
        name: 'speed-benchmark',
        title: 'Lightning-Fast Analysis',
        setup: async () => {
          await this.page!.goto('/performance');
          await smartWait(this.page!);
        }
      },
      
      {
        name: 'multi-language-support',
        title: 'Global Language Support',
        setup: async () => {
          await this.page!.goto('/languages');
          await smartWait(this.page!);
        }
      },
      
      {
        name: 'enterprise-integration',
        title: 'Enterprise API Integration',
        setup: async () => {
          await this.page!.goto('/enterprise');
          await smartWait(this.page!);
        }
      },
      
      {
        name: 'compliance-features',
        title: 'Compliance & Security',
        setup: async () => {
          await this.page!.goto('/compliance');
          await smartWait(this.page!);
        }
      }
    ];

    for (const feature of features) {
      console.log(`📸 Capturing ${feature.name}...`);
      
      try {
        await feature.setup();
        
        for (const theme of this.config.themes) {
          await takeScreenshot(this.page, `feature-${feature.name}`, {
            mode: theme,
            fullPage: true
          });
        }
      } catch (error) {
        console.warn(`⚠️ Could not capture ${feature.name}:`, error);
      }
    }
    
    console.log('✅ Feature showcase screenshots generated');
  }

  async generateComparisonScreenshots(): Promise<void> {
    if (!this.page) throw new Error('Generator not initialized');
    
    console.log('⚖️ Generating before/after comparison screenshots...');
    
    const comparisons = [
      {
        name: 'document-analysis-workflow',
        before: async () => {
          await this.page!.goto('/upload');
          await smartWait(this.page!);
        },
        after: async () => {
          await uploadDocument(this.page!, SAMPLE_DOCUMENTS.arbitration_clause.filename, {
            waitForAnalysis: true
          });
        }
      },
      
      {
        name: 'empty-vs-populated-dashboard',
        before: async () => {
          await this.page!.evaluate(() => {
            localStorage.clear();
            sessionStorage.clear();
          });
          await this.page!.goto('/dashboard');
          await smartWait(this.page!);
        },
        after: async () => {
          // Simulate populated dashboard
          await this.page!.goto('/dashboard');
          await smartWait(this.page!);
        }
      }
    ];

    for (const comparison of comparisons) {
      console.log(`📸 Capturing ${comparison.name} comparison...`);
      
      try {
        // Before screenshot
        await comparison.before();
        await takeScreenshot(this.page, `${comparison.name}-before`);
        
        // After screenshot
        await comparison.after();
        await takeScreenshot(this.page, `${comparison.name}-after`);
        
      } catch (error) {
        console.warn(`⚠️ Could not capture ${comparison.name}:`, error);
      }
    }
    
    console.log('✅ Comparison screenshots generated');
  }

  async generateResponsiveShowcase(): Promise<void> {
    if (!this.page) throw new Error('Generator not initialized');
    
    console.log('📱 Generating responsive design showcase...');
    
    const pages = [
      { name: 'homepage', url: '/' },
      { name: 'upload', url: '/upload' },
      { name: 'dashboard', url: '/dashboard' },
      { name: 'results', url: '/results' }
    ];

    for (const page of pages) {
      console.log(`📸 Capturing responsive ${page.name}...`);
      
      try {
        await this.page!.goto(page.url);
        await smartWait(this.page!);
        
        await takeResponsiveScreenshots(this.page, `responsive-${page.name}`, {
          fullPage: true
        });
      } catch (error) {
        console.warn(`⚠️ Could not capture responsive ${page.name}:`, error);
      }
    }
    
    console.log('✅ Responsive showcase screenshots generated');
  }

  async generateUserJourneyScreenshots(): Promise<void> {
    if (!this.page) throw new Error('Generator not initialized');
    
    console.log('🚶 Generating user journey screenshots...');
    
    const journeySteps = [
      {
        name: 'step-1-landing',
        title: 'User lands on homepage',
        action: async () => {
          await this.page!.goto('/');
          await smartWait(this.page!);
        }
      },
      
      {
        name: 'step-2-upload',
        title: 'User navigates to upload',
        action: async () => {
          await this.page!.click('[data-testid="upload-document-button"]');
          await smartWait(this.page!);
        }
      },
      
      {
        name: 'step-3-select-file',
        title: 'User selects document',
        action: async () => {
          await uploadDocument(this.page!, SAMPLE_DOCUMENTS.arbitration_clause.filename, {
            waitForAnalysis: false
          });
        }
      },
      
      {
        name: 'step-4-analysis',
        title: 'Analysis in progress',
        action: async () => {
          await this.page!.click('[data-testid="upload-button"]');
          await this.page!.waitForSelector('[data-testid="analysis-progress"]');
        }
      },
      
      {
        name: 'step-5-results',
        title: 'Results displayed',
        action: async () => {
          await this.page!.waitForSelector('[data-testid="analysis-complete"]');
          await smartWait(this.page!);
        }
      },
      
      {
        name: 'step-6-dashboard',
        title: 'User views dashboard',
        action: async () => {
          await this.page!.goto('/dashboard');
          await smartWait(this.page!);
        }
      }
    ];

    for (const [index, step] of journeySteps.entries()) {
      console.log(`📸 Capturing journey step ${index + 1}: ${step.title}...`);
      
      try {
        await step.action();
        
        await generateMarketingScreenshot(this.page, step.name, [
          {
            selector: 'body',
            text: `${index + 1}. ${step.title}`,
            position: 'top'
          }
        ]);
        
      } catch (error) {
        console.warn(`⚠️ Could not capture step ${step.name}:`, error);
      }
    }
    
    console.log('✅ User journey screenshots generated');
  }

  async generateSocialMediaAssets(): Promise<void> {
    if (!this.page) throw new Error('Generator not initialized');
    
    console.log('📱 Generating social media assets...');
    
    const socialFormats = [
      { name: 'twitter-card', width: 1200, height: 675 },
      { name: 'linkedin-post', width: 1200, height: 627 },
      { name: 'facebook-cover', width: 1200, height: 630 },
      { name: 'instagram-story', width: 1080, height: 1920 },
      { name: 'youtube-thumbnail', width: 1280, height: 720 }
    ];

    for (const format of socialFormats) {
      console.log(`📸 Creating ${format.name} asset...`);
      
      try {
        await this.page!.setViewportSize({ 
          width: format.width, 
          height: format.height 
        });
        
        await this.page!.goto('/');
        await smartWait(this.page!);
        
        await generateMarketingScreenshot(this.page, `social-${format.name}`, [
          {
            selector: '[data-testid="hero-title"]',
            text: 'AI-Powered Legal Document Analysis',
            position: 'bottom'
          }
        ], {
          fullPage: false,
          clip: { x: 0, y: 0, width: format.width, height: format.height }
        });
        
      } catch (error) {
        console.warn(`⚠️ Could not create ${format.name}:`, error);
      }
    }
    
    // Reset viewport
    await this.page!.setViewportSize({ width: 1920, height: 1080 });
    
    console.log('✅ Social media assets generated');
  }

  async generateDocumentationScreenshots(): Promise<void> {
    if (!this.page) throw new Error('Generator not initialized');
    
    console.log('📚 Generating documentation screenshots...');
    
    const docPages = [
      { name: 'api-docs', url: '/api-docs', title: 'API Documentation' },
      { name: 'getting-started', url: '/docs/getting-started', title: 'Getting Started Guide' },
      { name: 'integration-guide', url: '/docs/integration', title: 'Integration Guide' },
      { name: 'tutorials', url: '/docs/tutorials', title: 'Video Tutorials' },
      { name: 'faq', url: '/docs/faq', title: 'Frequently Asked Questions' }
    ];

    for (const docPage of docPages) {
      console.log(`📸 Capturing ${docPage.name}...`);
      
      try {
        await this.page!.goto(docPage.url);
        await smartWait(this.page!);
        
        await takeScreenshot(this.page, `docs-${docPage.name}`, {
          fullPage: true
        });
        
      } catch (error) {
        console.warn(`⚠️ Could not capture ${docPage.name}:`, error);
      }
    }
    
    console.log('✅ Documentation screenshots generated');
  }

  async generateAll(): Promise<void> {
    console.log('🎬 Generating complete screenshot suite...');
    
    await this.generateMarketingScreenshots();
    await this.generateFeatureShowcase();
    await this.generateComparisonScreenshots();
    await this.generateResponsiveShowcase();
    await this.generateUserJourneyScreenshots();
    await this.generateSocialMediaAssets();
    await this.generateDocumentationScreenshots();
    
    console.log('✅ Complete screenshot suite generated');
  }

  async cleanup(): Promise<void> {
    if (this.browser) {
      await this.browser.close();
    }
    
    console.log('🧹 Screenshot generator cleanup completed');
  }
}

// Export for use in other scripts
export { ScreenshotGenerator };

// CLI usage
if (require.main === module) {
  const type = process.argv[2] || 'all';
  const outputDir = process.argv[3] || 'marketing-screenshots';
  
  const generator = new ScreenshotGenerator({ outputDir });
  
  (async () => {
    try {
      await generator.initialize();
      
      switch (type) {
        case 'marketing':
          await generator.generateMarketingScreenshots();
          break;
        case 'features':
          await generator.generateFeatureShowcase();
          break;
        case 'responsive':
          await generator.generateResponsiveShowcase();
          break;
        case 'journey':
          await generator.generateUserJourneyScreenshots();
          break;
        case 'social':
          await generator.generateSocialMediaAssets();
          break;
        case 'docs':
          await generator.generateDocumentationScreenshots();
          break;
        case 'all':
          await generator.generateAll();
          break;
        default:
          console.error('Unknown type. Use: marketing, features, responsive, journey, social, docs, or all');
          process.exit(1);
      }
      
    } catch (error) {
      console.error('❌ Screenshot generation failed:', error);
    } finally {
      await generator.cleanup();
    }
  })();
}