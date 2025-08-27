import { chromium, Browser, Page } from 'playwright';
import { uploadDocument } from '../utils/upload';
import { takeScreenshot } from '../utils/screenshot';
import { smartWait, waitForAnalysisComplete } from '../utils/wait';
import { enableDemoMode } from '../utils/login';
import { SAMPLE_DOCUMENTS } from '../utils/test-data';

interface InteractionScenario {
  name: string;
  description: string;
  steps: InteractionStep[];
}

interface InteractionStep {
  action: string;
  selector?: string;
  text?: string;
  wait?: number;
  screenshot?: boolean;
  highlight?: boolean;
}

class InteractionDemoGenerator {
  private browser: Browser | null = null;
  private page: Page | null = null;

  async initialize(): Promise<void> {
    console.log('🎮 Initializing interaction demo generator...');
    
    this.browser = await chromium.launch({
      headless: false,
      slowMo: 1000,
      args: ['--start-maximized']
    });

    const context = await this.browser.newContext({
      viewport: { width: 1920, height: 1080 }
    });

    this.page = await context.newPage();
    await enableDemoMode(this.page);
    
    console.log('✅ Interaction demo generator ready');
  }

  async runFullInteractionDemo(): Promise<void> {
    if (!this.page) throw new Error('Generator not initialized');
    
    console.log('🎬 Running full interaction demonstration...');
    
    const scenarios: InteractionScenario[] = [
      {
        name: 'User Onboarding Flow',
        description: 'Complete new user experience from landing to first analysis',
        steps: [
          { action: 'navigate', selector: '/', wait: 2000, screenshot: true },
          { action: 'scroll', selector: '[data-testid="features-section"]', wait: 1000 },
          { action: 'click', selector: '[data-testid="start-demo-button"]', wait: 2000 },
          { action: 'navigate', selector: '/upload', wait: 2000, screenshot: true },
          { action: 'upload', text: SAMPLE_DOCUMENTS.arbitration_clause.filename, wait: 3000 },
          { action: 'click', selector: '[data-testid="upload-button"]', wait: 2000 },
          { action: 'wait-analysis', wait: 5000, screenshot: true },
          { action: 'scroll', selector: '[data-testid="analysis-result"]', wait: 2000 },
          { action: 'click', selector: '[data-testid="view-dashboard"]', wait: 2000 },
          { action: 'screenshot', wait: 1000 }
        ]
      },
      
      {
        name: 'Power User Workflow',
        description: 'Advanced user performing batch analysis and comparison',
        steps: [
          { action: 'navigate', selector: '/dashboard', wait: 2000, screenshot: true },
          { action: 'click', selector: '[data-testid="batch-upload"]', wait: 1000 },
          { action: 'upload-multiple', text: 'multiple-docs', wait: 5000 },
          { action: 'wait-batch-analysis', wait: 10000, screenshot: true },
          { action: 'click', selector: '[data-testid="compare-results"]', wait: 2000 },
          { action: 'screenshot', wait: 1000 },
          { action: 'click', selector: '[data-testid="export-report"]', wait: 3000 },
          { action: 'screenshot', wait: 1000 }
        ]
      },
      
      {
        name: 'Mobile User Experience',
        description: 'Complete mobile workflow demonstration',
        steps: [
          { action: 'set-mobile-viewport', wait: 1000 },
          { action: 'navigate', selector: '/', wait: 2000, screenshot: true },
          { action: 'click', selector: '[data-testid="mobile-menu-button"]', wait: 1000 },
          { action: 'screenshot', wait: 1000 },
          { action: 'click', selector: '[data-testid="mobile-upload-link"]', wait: 2000 },
          { action: 'upload', text: SAMPLE_DOCUMENTS.no_arbitration.filename, wait: 3000 },
          { action: 'wait-analysis', wait: 5000, screenshot: true },
          { action: 'swipe-results', wait: 2000 },
          { action: 'screenshot', wait: 1000 }
        ]
      },
      
      {
        name: 'Error Handling Demo',
        description: 'Demonstration of error states and recovery',
        steps: [
          { action: 'navigate', selector: '/upload', wait: 2000 },
          { action: 'simulate-network-error', wait: 1000 },
          { action: 'upload', text: SAMPLE_DOCUMENTS.arbitration_clause.filename, wait: 2000 },
          { action: 'click', selector: '[data-testid="upload-button"]', wait: 2000 },
          { action: 'screenshot', wait: 2000 }, // Error state
          { action: 'restore-network', wait: 1000 },
          { action: 'click', selector: '[data-testid="retry-button"]', wait: 2000 },
          { action: 'wait-analysis', wait: 5000, screenshot: true }
        ]
      },
      
      {
        name: 'Advanced Features Tour',
        description: 'Showcase of advanced features and integrations',
        steps: [
          { action: 'navigate', selector: '/advanced', wait: 2000, screenshot: true },
          { action: 'click', selector: '[data-testid="api-demo"]', wait: 2000 },
          { action: 'type', selector: '[data-testid="api-endpoint"]', text: '/api/analyze', wait: 1000 },
          { action: 'click', selector: '[data-testid="send-request"]', wait: 3000 },
          { action: 'screenshot', wait: 1000 },
          { action: 'click', selector: '[data-testid="webhook-setup"]', wait: 2000 },
          { action: 'screenshot', wait: 1000 },
          { action: 'click', selector: '[data-testid="custom-models"]', wait: 2000 },
          { action: 'screenshot', wait: 1000 }
        ]
      }
    ];

    for (const scenario of scenarios) {
      console.log(`🎮 Running ${scenario.name}...`);
      await this.runScenario(scenario);
      await this.page.waitForTimeout(2000); // Pause between scenarios
    }
    
    console.log('✅ Full interaction demo completed');
  }

  async runAccessibilityDemo(): Promise<void> {
    if (!this.page) throw new Error('Generator not initialized');
    
    console.log('♿ Running accessibility demonstration...');
    
    const accessibilityScenario: InteractionScenario = {
      name: 'Accessibility Features Demo',
      description: 'Demonstration of keyboard navigation and screen reader support',
      steps: [
        { action: 'navigate', selector: '/', wait: 2000, screenshot: true },
        { action: 'keyboard-navigation', wait: 5000 },
        { action: 'screenshot', wait: 1000 },
        { action: 'high-contrast-mode', wait: 2000 },
        { action: 'screenshot', wait: 1000 },
        { action: 'zoom-test', wait: 3000 },
        { action: 'screenshot', wait: 1000 },
        { action: 'screen-reader-simulation', wait: 5000 },
        { action: 'screenshot', wait: 1000 }
      ]
    };

    await this.runScenario(accessibilityScenario);
    console.log('✅ Accessibility demo completed');
  }

  async runPerformanceDemo(): Promise<void> {
    if (!this.page) throw new Error('Generator not initialized');
    
    console.log('⚡ Running performance demonstration...');
    
    // Monitor performance metrics
    await this.page.goto('/');
    
    const performanceMetrics = await this.page.evaluate(() => {
      return new Promise((resolve) => {
        new PerformanceObserver((list) => {
          const entries = list.getEntries();
          const metrics = {
            fcp: 0,
            lcp: 0,
            cls: 0,
            fid: 0
          };
          
          entries.forEach((entry) => {
            if (entry.entryType === 'paint' && entry.name === 'first-contentful-paint') {
              metrics.fcp = entry.startTime;
            }
            if (entry.entryType === 'largest-contentful-paint') {
              metrics.lcp = entry.startTime;
            }
            if (entry.entryType === 'layout-shift' && !(entry as any).hadRecentInput) {
              metrics.cls += (entry as any).value;
            }
          });
          
          setTimeout(() => resolve(metrics), 3000);
        }).observe({ entryTypes: ['paint', 'largest-contentful-paint', 'layout-shift'] });
      });
    });
    
    console.log('📊 Performance metrics:', performanceMetrics);
    
    // Demonstrate fast upload and analysis
    const startTime = Date.now();
    await this.page.goto('/upload');
    await uploadDocument(this.page, SAMPLE_DOCUMENTS.arbitration_clause.filename, {
      waitForAnalysis: true
    });
    const totalTime = Date.now() - startTime;
    
    await this.addOverlay(`Analysis completed in ${totalTime/1000} seconds`);
    await takeScreenshot(this.page, 'performance-demo-result');
    
    console.log(`✅ Performance demo completed - Total time: ${totalTime}ms`);
  }

  async runMultilingualDemo(): Promise<void> {
    if (!this.page) throw new Error('Generator not initialized');
    
    console.log('🌍 Running multilingual demonstration...');
    
    const languages = [
      { code: 'en', name: 'English', doc: SAMPLE_DOCUMENTS.arbitration_clause.filename },
      { code: 'es', name: 'Spanish', doc: 'software_license_es.txt' },
      { code: 'fr', name: 'French', doc: 'software_license_fr.txt' },
      { code: 'de', name: 'German', doc: 'software_license_de.txt' }
    ];
    
    for (const lang of languages) {
      console.log(`🌐 Testing ${lang.name} language support...`);
      
      await this.page.goto('/upload');
      await this.addOverlay(`Testing ${lang.name} Document Analysis`);
      
      try {
        await uploadDocument(this.page, lang.doc, {
          waitForAnalysis: true
        });
        
        await takeScreenshot(this.page, `multilingual-${lang.code}-result`);
      } catch (error) {
        console.warn(`⚠️ Could not test ${lang.name}:`, error);
      }
      
      await this.page.waitForTimeout(2000);
    }
    
    console.log('✅ Multilingual demo completed');
  }

  private async runScenario(scenario: InteractionScenario): Promise<void> {
    console.log(`📝 Running scenario: ${scenario.description}`);
    
    for (const [index, step] of scenario.steps.entries()) {
      console.log(`  ${index + 1}. ${step.action}${step.selector ? ` (${step.selector})` : ''}`);
      
      try {
        await this.executeStep(step);
        
        if (step.wait) {
          await this.page!.waitForTimeout(step.wait);
        }
        
        if (step.screenshot) {
          await takeScreenshot(this.page!, `${scenario.name.toLowerCase().replace(/\s+/g, '-')}-step-${index + 1}`);
        }
        
      } catch (error) {
        console.error(`❌ Error in step ${index + 1}:`, error);
        // Continue with next step
      }
    }
  }

  private async executeStep(step: InteractionStep): Promise<void> {
    if (!this.page) return;
    
    switch (step.action) {
      case 'navigate':
        await this.page.goto(step.selector || '/');
        await smartWait(this.page);
        break;
        
      case 'click':
        if (step.selector) {
          if (step.highlight) {
            await this.highlightElement(step.selector);
          }
          await this.page.click(step.selector);
        }
        break;
        
      case 'type':
        if (step.selector && step.text) {
          await this.page.fill(step.selector, step.text);
        }
        break;
        
      case 'scroll':
        if (step.selector) {
          await this.page.locator(step.selector).scrollIntoViewIfNeeded();
        }
        break;
        
      case 'upload':
        if (step.text) {
          await uploadDocument(this.page, step.text, { waitForAnalysis: false });
        }
        break;
        
      case 'upload-multiple':
        await this.uploadMultipleDocuments();
        break;
        
      case 'wait-analysis':
        await waitForAnalysisComplete(this.page);
        break;
        
      case 'wait-batch-analysis':
        await this.waitForBatchAnalysis();
        break;
        
      case 'set-mobile-viewport':
        await this.page.setViewportSize({ width: 375, height: 667 });
        break;
        
      case 'swipe-results':
        await this.simulateSwipeGesture();
        break;
        
      case 'simulate-network-error':
        await this.page.route('**/api/**', route => route.abort());
        break;
        
      case 'restore-network':
        await this.page.unroute('**/api/**');
        break;
        
      case 'keyboard-navigation':
        await this.demonstrateKeyboardNavigation();
        break;
        
      case 'high-contrast-mode':
        await this.enableHighContrastMode();
        break;
        
      case 'zoom-test':
        await this.testZoomLevels();
        break;
        
      case 'screen-reader-simulation':
        await this.simulateScreenReaderNavigation();
        break;
        
      case 'screenshot':
        await takeScreenshot(this.page, `interaction-step-${Date.now()}`);
        break;
        
      default:
        console.warn(`Unknown action: ${step.action}`);
    }
  }

  private async uploadMultipleDocuments(): Promise<void> {
    const documents = [
      SAMPLE_DOCUMENTS.arbitration_clause.filename,
      SAMPLE_DOCUMENTS.no_arbitration.filename,
      SAMPLE_DOCUMENTS.complex_arbitration.filename
    ];
    
    for (const doc of documents) {
      await uploadDocument(this.page!, doc, { waitForAnalysis: false });
      await this.page!.waitForTimeout(1000);
    }
  }

  private async waitForBatchAnalysis(): Promise<void> {
    // Wait for multiple analysis processes to complete
    await this.page!.waitForSelector('[data-testid="batch-analysis-complete"]', { timeout: 30000 });
  }

  private async simulateSwipeGesture(): Promise<void> {
    // Simulate mobile swipe gesture
    const element = this.page!.locator('[data-testid="analysis-result"]');
    const box = await element.boundingBox();
    
    if (box) {
      await this.page!.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
      await this.page!.mouse.down();
      await this.page!.mouse.move(box.x + box.width / 2 - 100, box.y + box.height / 2);
      await this.page!.mouse.up();
    }
  }

  private async demonstrateKeyboardNavigation(): Promise<void> {
    // Demonstrate tab navigation through interactive elements
    const interactiveElements = [
      '[data-testid="main-navigation"] a',
      '[data-testid="upload-document-button"]',
      '[data-testid="start-demo-button"]'
    ];
    
    for (const selector of interactiveElements) {
      try {
        await this.page!.focus(selector);
        await this.page!.waitForTimeout(1000);
        await this.highlightElement(selector);
        await this.page!.waitForTimeout(500);
      } catch (error) {
        // Element might not exist, continue
      }
    }
  }

  private async enableHighContrastMode(): Promise<void> {
    await this.page!.addStyleTag({
      content: `
        * {
          filter: contrast(150%) !important;
        }
        body {
          background: black !important;
          color: yellow !important;
        }
        a, button {
          color: cyan !important;
          border: 2px solid white !important;
        }
      `
    });
  }

  private async testZoomLevels(): Promise<void> {
    const zoomLevels = [1.0, 1.5, 2.0];
    
    for (const zoom of zoomLevels) {
      await this.page!.evaluate((level) => {
        document.body.style.zoom = level.toString();
      }, zoom);
      
      await this.addOverlay(`Zoom level: ${zoom * 100}%`);
      await this.page!.waitForTimeout(1500);
    }
    
    // Reset zoom
    await this.page!.evaluate(() => {
      document.body.style.zoom = '1';
    });
  }

  private async simulateScreenReaderNavigation(): Promise<void> {
    // Highlight elements that would be read by screen reader
    const landmarks = [
      '[role="main"]',
      '[role="navigation"]',
      'h1, h2, h3',
      '[data-testid*="heading"]'
    ];
    
    for (const landmark of landmarks) {
      try {
        const elements = await this.page!.locator(landmark).all();
        for (const element of elements) {
          await element.scrollIntoViewIfNeeded();
          await this.highlightElement(landmark);
          await this.page!.waitForTimeout(1000);
        }
      } catch (error) {
        // Element might not exist, continue
      }
    }
  }

  private async highlightElement(selector: string): Promise<void> {
    try {
      await this.page!.evaluate((sel) => {
        const element = document.querySelector(sel);
        if (element) {
          element.style.outline = '3px solid #ff6b35';
          element.style.outlineOffset = '2px';
          element.style.transition = 'all 0.3s ease';
        }
      }, selector);
      
      await this.page!.waitForTimeout(1000);
      
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

  private async addOverlay(text: string): Promise<void> {
    await this.page!.evaluate((overlayText) => {
      // Remove existing overlay
      const existing = document.getElementById('interaction-overlay');
      if (existing) existing.remove();
      
      // Create new overlay
      const overlay = document.createElement('div');
      overlay.id = 'interaction-overlay';
      overlay.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: rgba(0, 0, 0, 0.8);
        color: white;
        padding: 15px 20px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: 500;
        z-index: 10000;
        max-width: 300px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
      `;
      overlay.textContent = overlayText;
      document.body.appendChild(overlay);
      
      // Auto-remove after delay
      setTimeout(() => overlay.remove(), 3000);
    }, text);
    
    await this.page!.waitForTimeout(500);
  }

  async cleanup(): Promise<void> {
    if (this.browser) {
      await this.browser.close();
    }
    
    console.log('🧹 Interaction demo cleanup completed');
  }
}

// Export for use in other scripts
export { InteractionDemoGenerator };

// CLI usage
if (require.main === module) {
  const type = process.argv[2] || 'full';
  
  const demo = new InteractionDemoGenerator();
  
  (async () => {
    try {
      await demo.initialize();
      
      switch (type) {
        case 'full':
          await demo.runFullInteractionDemo();
          break;
        case 'accessibility':
          await demo.runAccessibilityDemo();
          break;
        case 'performance':
          await demo.runPerformanceDemo();
          break;
        case 'multilingual':
          await demo.runMultilingualDemo();
          break;
        default:
          console.error('Unknown type. Use: full, accessibility, performance, or multilingual');
          process.exit(1);
      }
      
    } catch (error) {
      console.error('❌ Interaction demo failed:', error);
    } finally {
      await demo.cleanup();
    }
  })();
}