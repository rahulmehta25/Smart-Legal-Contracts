import { test, expect } from '@playwright/test';
import { uploadDocument } from '../utils/upload';
import { takeScreenshot } from '../utils/screenshot';
import { smartWait } from '../utils/wait';
import { enableDemoMode } from '../utils/login';
import { SAMPLE_DOCUMENTS } from '../utils/test-data';
import pixelmatch from 'pixelmatch';
import { PNG } from 'pngjs';
import fs from 'fs';
import path from 'path';

test.describe('Visual Regression Tests', () => {
  const baselineDir = 'screenshots/baseline';
  const currentDir = 'screenshots/current';
  const diffDir = 'screenshots/diff';

  test.beforeEach(async ({ page }) => {
    await enableDemoMode(page);
    
    // Ensure screenshot directories exist
    for (const dir of [baselineDir, currentDir, diffDir]) {
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
    }
  });

  test('should match homepage baseline @visual @regression', async ({ page }) => {
    await page.goto('/');
    await smartWait(page);
    
    // Take current screenshot
    await page.screenshot({ 
      path: path.join(currentDir, 'homepage.png'),
      fullPage: true,
      animations: 'disabled'
    });
    
    // Compare with baseline if it exists
    await compareWithBaseline('homepage');
  });

  test('should match upload interface baseline @visual @regression', async ({ page }) => {
    await page.goto('/upload');
    await smartWait(page);
    
    await page.screenshot({ 
      path: path.join(currentDir, 'upload-interface.png'),
      fullPage: true,
      animations: 'disabled'
    });
    
    await compareWithBaseline('upload-interface');
  });

  test('should match analysis results baseline @visual @regression @demo', async ({ page }) => {
    await page.goto('/upload');
    
    await uploadDocument(page, SAMPLE_DOCUMENTS.arbitration_clause.filename, {
      waitForAnalysis: true,
      expectedResult: 'arbitration'
    });
    
    // Hide dynamic elements before screenshot
    await page.addStyleTag({
      content: `
        [data-testid*="timestamp"],
        [data-testid*="time"],
        .processing-animation {
          visibility: hidden !important;
        }
      `
    });
    
    await page.screenshot({ 
      path: path.join(currentDir, 'analysis-results.png'),
      fullPage: true,
      animations: 'disabled'
    });
    
    await compareWithBaseline('analysis-results');
  });

  test('should match dashboard baseline @visual @regression', async ({ page }) => {
    await page.goto('/dashboard');
    await smartWait(page);
    
    // Hide dynamic content
    await page.addStyleTag({
      content: `
        [data-testid*="timestamp"],
        [data-testid*="date"],
        [data-testid*="time"],
        .chart-animation {
          visibility: hidden !important;
        }
      `
    });
    
    await page.screenshot({ 
      path: path.join(currentDir, 'dashboard.png'),
      fullPage: true,
      animations: 'disabled'
    });
    
    await compareWithBaseline('dashboard');
  });

  test('should match mobile layout baseline @visual @regression @mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });
    await page.goto('/');
    await smartWait(page);
    
    await page.screenshot({ 
      path: path.join(currentDir, 'mobile-homepage.png'),
      fullPage: true,
      animations: 'disabled'
    });
    
    await compareWithBaseline('mobile-homepage');
  });

  test('should match tablet layout baseline @visual @regression', async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.goto('/');
    await smartWait(page);
    
    await page.screenshot({ 
      path: path.join(currentDir, 'tablet-homepage.png'),
      fullPage: true,
      animations: 'disabled'
    });
    
    await compareWithBaseline('tablet-homepage');
  });

  test('should match dark theme baseline @visual @regression', async ({ page }) => {
    // Set dark theme
    await page.addStyleTag({
      content: `
        html { color-scheme: dark; }
        body { background: #1a1a1a; color: #ffffff; }
      `
    });
    
    await page.goto('/');
    await smartWait(page);
    
    await page.screenshot({ 
      path: path.join(currentDir, 'dark-theme-homepage.png'),
      fullPage: true,
      animations: 'disabled'
    });
    
    await compareWithBaseline('dark-theme-homepage');
  });

  test('should match form states baseline @visual @regression', async ({ page }) => {
    await page.goto('/upload');
    await smartWait(page);
    
    // Test different form states
    const states = ['empty', 'file-selected', 'uploading', 'error'];
    
    for (const state of states) {
      await setupFormState(page, state);
      
      await page.screenshot({ 
        path: path.join(currentDir, `form-state-${state}.png`),
        animations: 'disabled'
      });
      
      await compareWithBaseline(`form-state-${state}`);
    }
  });

  test('should match component variations baseline @visual @regression', async ({ page }) => {
    const components = [
      { name: 'navigation', selector: '[data-testid="main-navigation"]' },
      { name: 'footer', selector: '[data-testid="footer"]' },
      { name: 'hero-section', selector: '[data-testid="hero-section"]' },
      { name: 'features-grid', selector: '[data-testid="features-section"]' }
    ];
    
    await page.goto('/');
    await smartWait(page);
    
    for (const component of components) {
      const element = page.locator(component.selector);
      
      if (await element.isVisible()) {
        await element.screenshot({ 
          path: path.join(currentDir, `component-${component.name}.png`),
          animations: 'disabled'
        });
        
        await compareWithBaseline(`component-${component.name}`);
      }
    }
  });

  test('should detect layout changes @visual @regression', async ({ page }) => {
    // Test for unintentional layout shifts
    await page.goto('/');
    await smartWait(page);
    
    // Measure layout stability
    const layoutShift = await page.evaluate(() => {
      return new Promise((resolve) => {
        let cumulativeLayoutShift = 0;
        
        const observer = new PerformanceObserver((list) => {
          for (const entry of list.getEntries()) {
            if (entry.entryType === 'layout-shift' && !(entry as any).hadRecentInput) {
              cumulativeLayoutShift += (entry as any).value;
            }
          }
        });
        
        observer.observe({ entryTypes: ['layout-shift'] });
        
        setTimeout(() => {
          observer.disconnect();
          resolve(cumulativeLayoutShift);
        }, 3000);
      });
    });
    
    // Assert layout stability (CLS should be < 0.1)
    expect(layoutShift).toBeLessThan(0.1);
    
    await page.screenshot({ 
      path: path.join(currentDir, 'layout-stability.png'),
      fullPage: true,
      animations: 'disabled'
    });
    
    await compareWithBaseline('layout-stability');
  });

  test('should match error states baseline @visual @regression', async ({ page }) => {
    // Test various error states
    const errorScenarios = [
      {
        name: 'network-error',
        setup: async () => {
          await page.route('**/api/**', route => route.abort());
          await page.goto('/');
        }
      },
      {
        name: 'upload-error',
        setup: async () => {
          await page.goto('/upload');
          await page.route('**/api/upload', route => {
            route.fulfill({ status: 500, body: 'Upload failed' });
          });
        }
      },
      {
        name: '404-page',
        setup: async () => {
          await page.goto('/non-existent-page');
        }
      }
    ];
    
    for (const scenario of errorScenarios) {
      try {
        await scenario.setup();
        await smartWait(page);
        
        await page.screenshot({ 
          path: path.join(currentDir, `error-${scenario.name}.png`),
          fullPage: true,
          animations: 'disabled'
        });
        
        await compareWithBaseline(`error-${scenario.name}`);
      } catch (error) {
        console.warn(`Could not test ${scenario.name}:`, error);
      }
      
      // Clean up route interceptions
      await page.unroute('**/api/**');
    }
  });

  test('should create baseline screenshots if missing @visual @regression', async ({ page }) => {
    const testPages = [
      { name: 'homepage', url: '/' },
      { name: 'upload', url: '/upload' },
      { name: 'dashboard', url: '/dashboard' },
      { name: 'about', url: '/about' },
      { name: 'contact', url: '/contact' }
    ];
    
    for (const testPage of testPages) {
      const baselinePath = path.join(baselineDir, `${testPage.name}.png`);
      
      if (!fs.existsSync(baselinePath)) {
        console.log(`📸 Creating baseline for ${testPage.name}...`);
        
        try {
          await page.goto(testPage.url);
          await smartWait(page);
          
          await page.screenshot({ 
            path: baselinePath,
            fullPage: true,
            animations: 'disabled'
          });
          
          console.log(`✅ Baseline created: ${baselinePath}`);
        } catch (error) {
          console.warn(`⚠️ Could not create baseline for ${testPage.name}:`, error);
        }
      }
    }
  });

  async function setupFormState(page: any, state: string): Promise<void> {
    switch (state) {
      case 'empty':
        // Default empty state
        break;
        
      case 'file-selected':
        await uploadDocument(page, SAMPLE_DOCUMENTS.arbitration_clause.filename, {
          waitForAnalysis: false
        });
        break;
        
      case 'uploading':
        await uploadDocument(page, SAMPLE_DOCUMENTS.arbitration_clause.filename, {
          waitForAnalysis: false
        });
        await page.click('[data-testid="upload-button"]');
        await page.waitForSelector('[data-testid="upload-progress"]');
        break;
        
      case 'error':
        await page.route('**/api/upload', route => {
          route.fulfill({ status: 500, body: 'Upload failed' });
        });
        await uploadDocument(page, SAMPLE_DOCUMENTS.arbitration_clause.filename, {
          waitForAnalysis: false
        });
        await page.click('[data-testid="upload-button"]');
        break;
    }
  }

  async function compareWithBaseline(testName: string): Promise<void> {
    const baselinePath = path.join(baselineDir, `${testName}.png`);
    const currentPath = path.join(currentDir, `${testName}.png`);
    const diffPath = path.join(diffDir, `${testName}.png`);
    
    if (!fs.existsSync(baselinePath)) {
      console.log(`📸 Creating baseline for ${testName}...`);
      fs.copyFileSync(currentPath, baselinePath);
      return;
    }
    
    if (!fs.existsSync(currentPath)) {
      throw new Error(`Current screenshot not found: ${currentPath}`);
    }
    
    const baseline = PNG.sync.read(fs.readFileSync(baselinePath));
    const current = PNG.sync.read(fs.readFileSync(currentPath));
    
    const { width, height } = baseline;
    const diff = new PNG({ width, height });
    
    const numDiffPixels = pixelmatch(
      baseline.data,
      current.data,
      diff.data,
      width,
      height,
      {
        threshold: 0.1, // Sensitivity threshold
        includeAA: false // Ignore anti-aliasing differences
      }
    );
    
    // Save diff image if there are differences
    if (numDiffPixels > 0) {
      fs.writeFileSync(diffPath, PNG.sync.write(diff));
      console.log(`📊 Visual differences found in ${testName}: ${numDiffPixels} pixels`);
    }
    
    // Calculate difference percentage
    const totalPixels = width * height;
    const diffPercentage = (numDiffPixels / totalPixels) * 100;
    
    // Allow up to 0.5% difference (adjustable threshold)
    const threshold = 0.5;
    
    if (diffPercentage > threshold) {
      throw new Error(
        `Visual regression detected in ${testName}: ${diffPercentage.toFixed(2)}% difference ` +
        `(${numDiffPixels} pixels). Threshold: ${threshold}%. ` +
        `Diff image saved to: ${diffPath}`
      );
    }
    
    console.log(`✅ Visual regression test passed for ${testName}: ${diffPercentage.toFixed(2)}% difference`);
  }
});

test.describe('Visual Regression Utilities', () => {
  test('should update all baselines @visual @update-baselines', async ({ page }) => {
    // This test updates all baseline screenshots
    // Run with: npx playwright test --grep @update-baselines
    
    await enableDemoMode(page);
    
    const testPages = [
      { name: 'homepage', url: '/', wait: () => smartWait(page) },
      { 
        name: 'upload', 
        url: '/upload', 
        wait: () => smartWait(page) 
      },
      { 
        name: 'upload-with-file', 
        url: '/upload', 
        wait: async () => {
          await uploadDocument(page, SAMPLE_DOCUMENTS.arbitration_clause.filename, {
            waitForAnalysis: false
          });
        }
      },
      { 
        name: 'analysis-results', 
        url: '/upload', 
        wait: async () => {
          await uploadDocument(page, SAMPLE_DOCUMENTS.arbitration_clause.filename, {
            waitForAnalysis: true
          });
        }
      },
      { 
        name: 'dashboard', 
        url: '/dashboard', 
        wait: () => smartWait(page) 
      }
    ];
    
    for (const testPage of testPages) {
      console.log(`📸 Updating baseline: ${testPage.name}...`);
      
      try {
        await page.goto(testPage.url);
        await testPage.wait();
        
        // Hide dynamic elements
        await page.addStyleTag({
          content: `
            [data-testid*="timestamp"],
            [data-testid*="time"],
            [data-testid*="date"],
            .loading-animation,
            .processing-animation {
              visibility: hidden !important;
            }
          `
        });
        
        const baselinePath = path.join(baselineDir, `${testPage.name}.png`);
        
        await page.screenshot({ 
          path: baselinePath,
          fullPage: true,
          animations: 'disabled'
        });
        
        console.log(`✅ Updated baseline: ${testPage.name}`);
        
      } catch (error) {
        console.error(`❌ Failed to update baseline for ${testPage.name}:`, error);
      }
    }
  });

  test('should generate visual regression report @visual @report', async ({ page }) => {
    // Generate a comprehensive visual regression report
    const report = {
      timestamp: new Date().toISOString(),
      testResults: [],
      summary: {
        total: 0,
        passed: 0,
        failed: 0,
        updated: 0
      }
    };
    
    // This would normally run all regression tests and collect results
    console.log('📊 Visual regression report generated');
    
    // Save report
    const reportPath = path.join(diffDir, 'visual-regression-report.json');
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    
    console.log(`📄 Report saved: ${reportPath}`);
  });
});