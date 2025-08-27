# Playwright Visual Testing & Demo Automation Suite

A comprehensive Playwright-based testing and demonstration framework for the AI-powered arbitration detection system.

## 🎯 Overview

This suite provides:
- **Visual regression testing** with pixel-perfect screenshot comparison
- **Automated demo generation** with video recording capabilities
- **Responsive design testing** across multiple devices and viewports
- **Accessibility testing** with keyboard navigation and screen reader simulation
- **Performance monitoring** with Core Web Vitals measurement
- **Interactive demonstrations** for sales and marketing purposes

## 📁 Directory Structure

```
demo/playwright/
├── tests/                          # Test specifications
│   ├── homepage.spec.ts            # Homepage visual tests
│   ├── upload-flow.spec.ts         # Document upload workflow tests
│   ├── analysis-results.spec.ts    # Analysis results display tests
│   ├── dashboard.spec.ts           # Dashboard interface tests
│   ├── responsive.spec.ts          # Responsive design tests
│   └── visual-regression.spec.ts   # Visual regression testing
├── demo/                           # Demo automation scripts
│   ├── automated-demo.ts           # Main demo automation class
│   ├── screenshot-generator.ts     # Marketing screenshot generator
│   ├── video-demo.ts              # Video demo recorder
│   ├── interaction-demo.ts        # Interactive feature demos
│   ├── quick-demo.js              # 2-minute quick demo
│   ├── full-demo.js               # 10-minute comprehensive demo
│   ├── performance-demo.js        # Performance showcase
│   └── accessibility-demo.js      # Accessibility features demo
├── utils/                          # Utility functions
│   ├── upload.ts                  # Document upload helpers
│   ├── login.ts                   # Authentication utilities
│   ├── wait.ts                    # Smart waiting strategies
│   ├── screenshot.ts              # Screenshot utilities
│   └── test-data.ts               # Test data and fixtures
├── screenshots/                    # Generated screenshots
│   ├── baseline/                  # Baseline images for regression
│   ├── current/                   # Current test screenshots
│   └── diff/                      # Visual difference images
├── demo-videos/                    # Generated demo videos
├── marketing-screenshots/          # Marketing assets
├── reports/                        # Test reports and artifacts
├── playwright.config.ts           # Playwright configuration
├── global-setup.ts               # Global test setup
├── global-teardown.ts             # Global test cleanup
└── package.json                   # Dependencies and scripts
```

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
npm install

# Install Playwright browsers
npm run install-browsers
```

### Running Tests

```bash
# Run all visual tests
npm test

# Run tests with UI mode
npm run test:ui

# Run specific test suites
npm run test:visual          # Visual regression tests
npm run test:mobile          # Mobile responsive tests
npm run test:accessibility   # Accessibility tests
```

### Running Demos

```bash
# Quick 2-minute demo
npm run demo:quick

# Full 10-minute comprehensive demo
npm run demo:full

# Performance showcase
npm run performance:test

# Accessibility features demo
npm run audit:accessibility

# Generate marketing screenshots
npm run demo:screenshots

# Record video demos
npm run demo:video

# Interactive feature demonstrations
npm run demo:interactive
```

## 🎬 Demo Scripts

### Quick Demo (2 minutes)
Perfect for initial product introductions:
```bash
node demo/quick-demo.js [--record]
```

### Full Demo (10 minutes)
Comprehensive feature walkthrough:
```bash
node demo/full-demo.js [--record]
```

### Feature-Specific Demos
```bash
# Performance showcase
node demo/performance-demo.js

# Accessibility features
node demo/accessibility-demo.js

# Multilingual support
node demo/interaction-demo.js multilingual
```

## 📸 Screenshot Generation

### Marketing Assets
```bash
# Generate all marketing screenshots
npm run demo:screenshots marketing

# Generate specific types
npm run demo:screenshots features
npm run demo:screenshots responsive
npm run demo:screenshots social
```

### Visual Regression Baselines
```bash
# Update baseline screenshots
npx playwright test --grep @update-baselines

# Generate visual regression report
npx playwright test --grep @report
```

## 🎥 Video Recording

### Automated Video Demos
```bash
# Record quick demo video
node demo/video-demo.js quick

# Record full feature walkthrough
node demo/video-demo.js full

# Record performance showcase
node demo/video-demo.js performance

# Record user journey
node demo/video-demo.js journey
```

## 📱 Responsive Testing

Tests across multiple viewports:
- **Mobile**: 375×667 (iPhone), 393×851 (Android)
- **Tablet**: 768×1024 (iPad Portrait), 1024×768 (iPad Landscape)
- **Desktop**: 1280×720, 1920×1080, 2560×1440

```bash
# Run responsive tests
npm run test:mobile

# Generate responsive screenshots
npm run demo:screenshots responsive
```

## ♿ Accessibility Testing

Comprehensive accessibility validation:
- Keyboard navigation
- Screen reader compatibility
- Color contrast compliance
- Focus management
- ARIA attributes
- Touch target sizing

```bash
# Run accessibility tests
npm run test:accessibility

# Generate accessibility report
npm run audit:accessibility
```

## 🔍 Visual Regression Testing

Pixel-perfect visual comparisons:
- Baseline screenshot management
- Automated diff generation
- Configurable similarity thresholds
- Cross-browser compatibility

```bash
# Run visual regression tests
npm run test:regression

# Update baselines (use with caution)
npm run update-screenshots
```

## ⚡ Performance Testing

Core Web Vitals monitoring:
- First Contentful Paint (FCP)
- Largest Contentful Paint (LCP)
- Cumulative Layout Shift (CLS)
- Time to Interactive (TTI)
- Document analysis speed

```bash
# Run performance tests
npm run performance:test

# Generate performance report
npm run performance:report
```

## 🎯 Test Categories

### Visual Tests (`@visual`)
Screenshot-based visual validation

### Demo Tests (`@demo`)
Interactive demonstrations with real data

### Regression Tests (`@regression`)
Baseline comparison testing

### Mobile Tests (`@mobile`)
Mobile-specific functionality and layout

### Accessibility Tests (`@accessibility`)
WCAG compliance validation

## 📊 Reports and Artifacts

### HTML Reports
```bash
# Generate comprehensive HTML report
npm run generate-report

# View latest report
npx playwright show-report
```

### Test Artifacts
- Screenshots (baseline, current, diff)
- Video recordings
- Performance metrics
- Accessibility audit results
- Coverage reports

## 🛠️ Configuration

### Playwright Config
Key configuration options in `playwright.config.ts`:
- Multiple browser support (Chrome, Firefox, Safari)
- Device emulation
- Video recording settings
- Screenshot comparison thresholds
- Parallel execution settings

### Environment Variables
```bash
BASE_URL=http://localhost:3000    # Application base URL
CI=true                          # Enable CI mode
HEADLESS=true                    # Run in headless mode
```

## 🎨 Marketing Assets

### Social Media Assets
Generated in various formats:
- Twitter cards (1200×675)
- LinkedIn posts (1200×627)
- Facebook covers (1200×630)
- Instagram stories (1080×1920)
- YouTube thumbnails (1280×720)

### Screenshot Types
- Hero sections with annotations
- Feature showcases
- Workflow demonstrations
- Before/after comparisons
- Mobile responsive views
- Dashboard analytics

## 🔄 Continuous Integration

### GitHub Actions Integration
```yaml
# Example CI configuration
- name: Run Playwright Tests
  run: |
    npm ci
    npm run install-browsers
    npm run test:ci
    
- name: Upload Test Results
  uses: actions/upload-artifact@v3
  with:
    name: playwright-report
    path: reports/
```

### Visual Regression in CI
- Automatic baseline updates on main branch
- Pull request visual comparisons
- Failure notifications with diff images

## 🚨 Troubleshooting

### Common Issues

**Tests failing due to timing**
```bash
# Increase timeout in playwright.config.ts
use: {
  actionTimeout: 30000,
  navigationTimeout: 60000
}
```

**Screenshots don't match baseline**
```bash
# Update baselines if changes are intentional
npm run update-screenshots
```

**Browser launch failures**
```bash
# Reinstall browsers
npx playwright install --force
```

### Debug Mode
```bash
# Run tests in debug mode
npm run test:debug

# Run with headed browser
npm run test:headed
```

## 📋 Best Practices

### Writing Visual Tests
1. **Use stable selectors**: Prefer `data-testid` attributes
2. **Hide dynamic content**: Timestamps, animations, loading states
3. **Wait for stability**: Use smart waiting strategies
4. **Mask sensitive data**: Hide personal or variable information
5. **Test critical paths**: Focus on user-facing features

### Demo Creation
1. **Slow down actions**: Use `slowMo` for better visibility
2. **Add narration**: Include explanatory overlays
3. **Highlight interactions**: Draw attention to key elements
4. **Handle errors gracefully**: Include error state demonstrations
5. **Keep it focused**: Stick to core value propositions

### Screenshot Management
1. **Organize by feature**: Group related screenshots
2. **Use descriptive names**: Include viewport and theme info
3. **Version control baselines**: Track changes in git
4. **Regular cleanup**: Remove outdated screenshots
5. **Optimize file sizes**: Balance quality and storage

## 🤝 Contributing

### Adding New Tests
1. Create test file in appropriate category
2. Use existing utilities and patterns
3. Add proper test tags (`@visual`, `@demo`, etc.)
4. Document test purpose and scope
5. Update this README if needed

### Adding Demo Scenarios
1. Extend existing demo classes
2. Follow naming conventions
3. Include error handling
4. Add progress logging
5. Test across different viewports

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review Playwright documentation
3. Check existing GitHub issues
4. Create new issue with reproduction steps

## 📄 License

This testing suite is part of the arbitration detection system project.