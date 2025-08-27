import { Page } from '@playwright/test';
import { createTestUser, enableDemoMode } from './login';
import { uploadTestDocument } from './upload';

/**
 * Sample document content for testing
 */
export const SAMPLE_DOCUMENTS = {
  arbitration_clause: {
    filename: 'sample_tos_with_arbitration.txt',
    content: `
TERMS OF SERVICE

1. ACCEPTANCE OF TERMS
By accessing and using this service, you accept and agree to be bound by the terms and provision of this agreement.

2. DISPUTE RESOLUTION
Any dispute, controversy or claim arising out of or relating to this contract, or the breach, termination or validity thereof, shall be settled by binding arbitration. The arbitration shall be conducted in accordance with the rules of the American Arbitration Association. The arbitration shall take place in New York, NY, and the arbitrator's decision shall be final and binding.

You hereby waive your right to participate in a class action lawsuit or class-wide arbitration against the Company.

3. GOVERNING LAW
This agreement shall be governed by the laws of the State of New York.
    `,
    expectedResult: 'arbitration'
  },
  
  no_arbitration: {
    filename: 'sample_contract_without_arbitration.txt',
    content: `
SERVICE AGREEMENT

1. SCOPE OF SERVICES
The Provider agrees to provide the services described in this agreement.

2. PAYMENT TERMS
Payment is due within 30 days of invoice date.

3. DISPUTE RESOLUTION
Any disputes arising under this agreement shall be resolved through good faith negotiation between the parties. If negotiation fails, disputes may be resolved through mediation or in the courts of competent jurisdiction.

4. TERMINATION
Either party may terminate this agreement with 30 days written notice.
    `,
    expectedResult: 'no-arbitration'
  },
  
  complex_arbitration: {
    filename: 'complex_arbitration_example.txt',
    content: `
COMPREHENSIVE TERMS OF SERVICE

SECTION 15: DISPUTE RESOLUTION AND ARBITRATION

15.1 Informal Resolution
Before filing a claim against Company, you agree to try to resolve the dispute informally by contacting us at legal@company.com. We'll try to resolve the dispute informally by contacting you via email.

15.2 Binding Arbitration
If we can't resolve the dispute informally, you and Company agree that any dispute, claim or controversy arising out of or relating to this Agreement or the breach, termination, enforcement, interpretation or validity thereof or the use of the Services (collectively, "Disputes") will be resolved solely by binding, individual arbitration and not in a class, representative or consolidated action or proceeding.

15.3 Arbitration Rules
The arbitration will be administered by JAMS pursuant to its Streamlined Arbitration Rules and Procedures. The arbitrator, and not any federal, state or local court or agency, shall have exclusive authority to resolve all disputes arising out of or relating to the interpretation, applicability, enforceability or formation of this Agreement, including, but not limited to any claim that all or any part of this Agreement is void or voidable.

15.4 Exception for Small Claims
Either you or Company may assert claims, if they qualify, in small claims court in San Francisco (CA) or any United States county where you live or work.

15.5 Waiver of Class Actions
YOU AND COMPANY AGREE THAT EACH MAY BRING CLAIMS AGAINST THE OTHER ONLY IN YOUR OR ITS INDIVIDUAL CAPACITY, AND NOT AS A PLAINTIFF OR CLASS MEMBER IN ANY PURPORTED CLASS OR REPRESENTATIVE PROCEEDING.
    `,
    expectedResult: 'arbitration'
  },
  
  uncertain_case: {
    filename: 'uncertain_arbitration_case.txt',
    content: `
TERMS AND CONDITIONS

DISPUTE RESOLUTION
For disputes under $1,000, either party may choose to resolve the matter through binding arbitration or small claims court. For disputes over $1,000, the parties agree to first attempt mediation through a neutral third party. If mediation is unsuccessful, the matter may proceed to arbitration at the option of either party, or may be resolved through the courts.

Certain types of disputes, including but not limited to intellectual property claims and injunctive relief, are excluded from the arbitration requirement and may be brought in court.

The arbitration procedures, if chosen, shall follow the rules of the National Arbitration Forum, unless both parties agree to alternative procedures.
    `,
    expectedResult: 'uncertain'
  }
};

/**
 * Create test documents for upload testing
 */
export async function createSampleDocuments() {
  const fs = require('fs').promises;
  const path = require('path');
  
  const docsDir = path.join(__dirname, '../../sample-data/documents');
  
  // Ensure directory exists
  try {
    await fs.mkdir(docsDir, { recursive: true });
  } catch (error) {
    // Directory might already exist
  }
  
  // Create each sample document
  for (const [key, doc] of Object.entries(SAMPLE_DOCUMENTS)) {
    const filePath = path.join(docsDir, doc.filename);
    await fs.writeFile(filePath, doc.content.trim());
    console.log(`📄 Created sample document: ${doc.filename}`);
  }
}

/**
 * Demo flow scenarios for testing
 */
export const DEMO_SCENARIOS = {
  quickDemo: {
    name: 'Quick Demo (2 minutes)',
    steps: [
      'Navigate to homepage',
      'Upload document with arbitration clause',
      'Show analysis results',
      'Highlight key findings',
      'Display confidence score'
    ],
    duration: 120000 // 2 minutes
  },
  
  fullDemo: {
    name: 'Full Demo (10 minutes)',
    steps: [
      'Homepage overview',
      'Upload multiple documents',
      'Compare different document types',
      'Show detailed analysis',
      'Demonstrate confidence scoring',
      'Show dashboard analytics',
      'Export results',
      'Mobile responsive view'
    ],
    duration: 600000 // 10 minutes
  },
  
  technicalDemo: {
    name: 'Technical Demo (15 minutes)',
    steps: [
      'API integration examples',
      'Bulk document processing',
      'Custom confidence thresholds',
      'Multi-language support',
      'Advanced analytics',
      'Reporting features',
      'Compliance checking'
    ],
    duration: 900000 // 15 minutes
  }
};

/**
 * Test user profiles for different scenarios
 */
export const TEST_PROFILES = {
  legalProfessional: {
    name: 'Legal Professional',
    interests: ['accuracy', 'detailed-analysis', 'compliance', 'reporting'],
    documents: ['complex_arbitration', 'uncertain_case'],
    features: ['confidence-scoring', 'clause-highlighting', 'export-pdf']
  },
  
  businessUser: {
    name: 'Business User', 
    interests: ['speed', 'simplicity', 'cost-savings'],
    documents: ['arbitration_clause', 'no_arbitration'],
    features: ['quick-analysis', 'dashboard', 'bulk-processing']
  },
  
  developer: {
    name: 'Developer',
    interests: ['api', 'integration', 'automation', 'scalability'],
    documents: ['arbitration_clause', 'complex_arbitration'],
    features: ['api-access', 'webhooks', 'batch-processing']
  }
};

/**
 * Setup test environment with sample data
 */
export async function setupTestEnvironment(page: Page) {
  console.log('🔧 Setting up test environment...');
  
  // Create sample documents
  await createSampleDocuments();
  
  // Enable demo mode for easier testing
  await enableDemoMode(page);
  
  // Navigate to home page
  await page.goto('/');
  
  console.log('✅ Test environment ready');
}

/**
 * Generate realistic demo data
 */
export async function generateDemoData(page: Page) {
  console.log('📊 Generating demo data...');
  
  // Upload sample documents for analysis history
  for (const doc of Object.values(SAMPLE_DOCUMENTS)) {
    try {
      await uploadTestDocument(page, doc.filename);
      await page.waitForTimeout(1000); // Prevent rate limiting
    } catch (error) {
      console.warn(`Failed to upload ${doc.filename}:`, error);
    }
  }
  
  console.log('✅ Demo data generated');
}

/**
 * Performance test data
 */
export const PERFORMANCE_TEST_DATA = {
  smallDocument: {
    size: '1KB',
    content: SAMPLE_DOCUMENTS.arbitration_clause.content,
    expectedTime: 2000 // 2 seconds
  },
  
  mediumDocument: {
    size: '50KB',
    content: SAMPLE_DOCUMENTS.complex_arbitration.content.repeat(50),
    expectedTime: 5000 // 5 seconds
  },
  
  largeDocument: {
    size: '500KB', 
    content: SAMPLE_DOCUMENTS.complex_arbitration.content.repeat(500),
    expectedTime: 15000 // 15 seconds
  }
};

/**
 * Accessibility test scenarios
 */
export const ACCESSIBILITY_SCENARIOS = {
  keyboardNavigation: {
    name: 'Keyboard Navigation',
    steps: [
      'Tab through all interactive elements',
      'Use Enter/Space to activate buttons',
      'Navigate forms with Tab/Shift+Tab',
      'Use arrow keys in custom components'
    ]
  },
  
  screenReader: {
    name: 'Screen Reader Support',
    checks: [
      'All images have alt text',
      'Form labels are properly associated',
      'ARIA attributes are correct',
      'Reading order is logical'
    ]
  },
  
  colorContrast: {
    name: 'Color Contrast',
    requirements: [
      'Text contrast ratio ≥ 4.5:1',
      'Large text contrast ratio ≥ 3:1',
      'Interactive elements are distinguishable',
      'Error states are clearly marked'
    ]
  }
};

/**
 * Mobile test scenarios
 */
export const MOBILE_SCENARIOS = {
  touchInteraction: {
    name: 'Touch Interaction',
    tests: [
      'Tap targets are at least 44px',
      'Swipe gestures work correctly',
      'Pinch to zoom functions',
      'Touch feedback is provided'
    ]
  },
  
  responsiveLayout: {
    name: 'Responsive Layout',
    breakpoints: [
      { name: 'mobile', width: 375 },
      { name: 'tablet', width: 768 },
      { name: 'desktop', width: 1024 }
    ]
  }
};

/**
 * Reset test environment
 */
export async function resetTestEnvironment(page: Page) {
  console.log('🔄 Resetting test environment...');
  
  // Clear local storage
  await page.evaluate(() => {
    localStorage.clear();
    sessionStorage.clear();
  });
  
  // Clear cookies
  await page.context().clearCookies();
  
  // Navigate to home
  await page.goto('/');
  
  console.log('✅ Test environment reset');
}