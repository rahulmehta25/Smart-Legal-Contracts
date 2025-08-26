/**
 * Test data factories and fixtures for comprehensive testing
 */

export interface TestUser {
  username: string;
  email: string;
  password: string;
  firstName?: string;
  lastName?: string;
}

export interface TestDocument {
  filename: string;
  content: string;
  type: 'terms_of_service' | 'privacy_policy' | 'contract' | 'agreement';
  hasArbitration: boolean;
  expectedAnalysis?: {
    confidence: number;
    arbitrationClauses: string[];
  };
}

export interface APIResponse {
  status: number;
  data?: any;
  error?: string;
}

export class TestData {
  /**
   * Generate test users
   */
  static users = {
    validUser: (): TestUser => ({
      username: `testuser_${Date.now()}`,
      email: `test_${Date.now()}@example.com`,
      password: 'SecurePassword123!',
      firstName: 'Test',
      lastName: 'User'
    }),

    invalidUser: (): TestUser => ({
      username: '', // Invalid empty username
      email: 'invalid-email', // Invalid email format
      password: '123', // Too short password
    }),

    adminUser: (): TestUser => ({
      username: 'admin',
      email: 'admin@arbitration-detector.com',
      password: 'AdminPassword123!',
    })
  };

  /**
   * Test documents with different arbitration scenarios
   */
  static documents = {
    withArbitration: (): TestDocument => ({
      filename: 'tos_with_arbitration.txt',
      content: `
Terms of Service

1. Acceptance of Terms
By using our service, you agree to these terms.

2. Dispute Resolution
Any disputes arising from this agreement shall be resolved through binding arbitration 
in accordance with the American Arbitration Association rules. You waive your right 
to a jury trial and agree that all disputes will be settled by a single arbitrator.

3. Limitation of Liability
Our liability is limited as described herein.
      `,
      type: 'terms_of_service',
      hasArbitration: true,
      expectedAnalysis: {
        confidence: 0.95,
        arbitrationClauses: [
          'binding arbitration in accordance with the American Arbitration Association',
          'waive your right to a jury trial'
        ]
      }
    }),

    withoutArbitration: (): TestDocument => ({
      filename: 'tos_without_arbitration.txt',
      content: `
Terms of Service

1. Acceptance of Terms
By using our service, you agree to these terms.

2. Dispute Resolution
Any disputes will be resolved in the courts of [Jurisdiction].

3. Limitation of Liability
Our liability is limited as described herein.
      `,
      type: 'terms_of_service',
      hasArbitration: false,
      expectedAnalysis: {
        confidence: 0.05,
        arbitrationClauses: []
      }
    }),

    hiddenArbitration: (): TestDocument => ({
      filename: 'complex_arbitration.txt',
      content: `
Software License Agreement

IMPORTANT - READ CAREFULLY: This Software License Agreement ("Agreement") is a legal 
agreement between you and Company Inc. By installing, copying, or using the Software, 
you agree to be bound by the terms of this Agreement.

SECTION 15: DISPUTE RESOLUTION
15.1 Informal Resolution: Prior to initiating any formal proceedings, the parties 
     agree to attempt to resolve disputes through good faith negotiations.

15.2 Binding Individual Arbitration: IF THE INFORMAL RESOLUTION PROCESS DOES NOT 
     RESOLVE THE DISPUTE WITHIN SIXTY (60) DAYS, ANY REMAINING DISPUTE SHALL BE 
     RESOLVED THROUGH FINAL AND BINDING ARBITRATION, RATHER THAN IN COURT, except 
     as otherwise provided herein. The arbitration shall be administered by JAMS 
     pursuant to its Comprehensive Arbitration Rules and Procedures.

15.3 Class Action Waiver: THE PARTIES AGREE THAT ANY ARBITRATION SHALL BE LIMITED 
     TO THE DISPUTE BETWEEN THE PARTIES INDIVIDUALLY. TO THE FULL EXTENT PERMITTED 
     BY LAW, (A) NO ARBITRATION SHALL BE JOINED WITH ANY OTHER PROCEEDING; (B) THERE 
     IS NO RIGHT OR AUTHORITY FOR ANY DISPUTE TO BE ARBITRATED ON A CLASS-ACTION BASIS 
     OR TO UTILIZE CLASS ACTION PROCEDURES; AND (C) THERE IS NO RIGHT OR AUTHORITY FOR 
     ANY DISPUTE TO BE BROUGHT IN A PURPORTED REPRESENTATIVE CAPACITY ON BEHALF OF THE 
     GENERAL PUBLIC OR ANY OTHER PERSONS.
      `,
      type: 'agreement',
      hasArbitration: true,
      expectedAnalysis: {
        confidence: 0.98,
        arbitrationClauses: [
          'FINAL AND BINDING ARBITRATION',
          'JAMS pursuant to its Comprehensive Arbitration Rules',
          'Class Action Waiver'
        ]
      }
    }),

    largeDocument: (): TestDocument => {
      const sections = Array.from({ length: 50 }, (_, i) => `
Section ${i + 1}: Lorem ipsum dolor sit amet, consectetur adipiscing elit. 
Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris.
      `);
      
      // Insert arbitration clause in the middle
      sections[25] = `
Section 26: ARBITRATION AGREEMENT
All disputes, claims, or controversies arising from or relating to this Agreement 
shall be resolved by binding arbitration administered by the American Arbitration 
Association under its Commercial Arbitration Rules.
      `;

      return {
        filename: 'large_document.txt',
        content: sections.join('\n'),
        type: 'contract',
        hasArbitration: true,
        expectedAnalysis: {
          confidence: 0.92,
          arbitrationClauses: ['binding arbitration administered by the American Arbitration Association']
        }
      };
    }
  };

  /**
   * WebSocket test messages
   */
  static webSocketMessages = {
    connectionTest: () => ({
      type: 'connection_test',
      timestamp: new Date().toISOString(),
      data: 'Testing connection'
    }),

    analysisRequest: (documentId: string) => ({
      type: 'analysis_request',
      documentId,
      timestamp: new Date().toISOString(),
      options: {
        realTimeUpdates: true,
        includeConfidenceScores: true
      }
    }),

    userPresence: (userId: string, status: 'online' | 'offline') => ({
      type: 'user_presence',
      userId,
      status,
      timestamp: new Date().toISOString()
    }),

    collaborationCursor: (x: number, y: number, userId: string) => ({
      type: 'cursor_position',
      position: { x, y },
      userId,
      timestamp: new Date().toISOString()
    })
  };

  /**
   * API test data
   */
  static api = {
    healthResponse: () => ({
      status: 'healthy',
      service: 'Arbitration RAG API',
      version: '1.0.0'
    }),

    analysisResponse: (hasArbitration = true) => ({
      id: `analysis_${Date.now()}`,
      result: {
        hasArbitrationClause: hasArbitration,
        confidence: hasArbitration ? 0.95 : 0.05,
        clauses: hasArbitration ? ['binding arbitration', 'class action waiver'] : [],
        metadata: {
          processingTime: Math.random() * 1000 + 500,
          modelVersion: '1.0.0'
        }
      },
      timestamp: new Date().toISOString()
    }),

    errorResponse: (status = 500, message = 'Internal Server Error') => ({
      detail: message,
      status_code: status,
      timestamp: new Date().toISOString()
    })
  };

  /**
   * Form validation test cases
   */
  static validation = {
    email: {
      valid: ['test@example.com', 'user+tag@domain.co.uk', 'valid.email@test-domain.org'],
      invalid: ['invalid-email', '@domain.com', 'user@', 'user space@domain.com']
    },
    
    password: {
      valid: ['SecurePassword123!', 'MyP@ssw0rd', 'C0mpl3xP@ss!'],
      invalid: ['123', 'password', 'PASSWORD123', 'Pass123'] // Missing special chars, too short, etc.
    },

    username: {
      valid: ['testuser', 'user123', 'valid_user', 'User-Name'],
      invalid: ['', 'us', 'user@invalid', 'spaces in name', '123456789012345678901'] // Too short/long, invalid chars
    }
  };

  /**
   * Performance test scenarios
   */
  static performance = {
    largeFileUpload: () => ({
      size: 10 * 1024 * 1024, // 10MB
      type: 'application/pdf',
      expectedProcessingTime: 30000 // 30 seconds max
    }),

    concurrentUsers: (count = 10) => 
      Array.from({ length: count }, (_, i) => ({
        id: `user_${i}`,
        actions: ['login', 'upload_document', 'analyze', 'logout']
      })),

    apiLoadTest: () => ({
      requestsPerSecond: 100,
      duration: 60000, // 1 minute
      endpoints: ['/health', '/api/v1', '/api/v1/analysis/quick-analyze']
    })
  };

  /**
   * Accessibility test scenarios
   */
  static accessibility = {
    keyboardNavigation: [
      { key: 'Tab', expectedFocus: 'next focusable element' },
      { key: 'Enter', expectedAction: 'activate element' },
      { key: 'Space', expectedAction: 'activate button/checkbox' },
      { key: 'Escape', expectedAction: 'close modal/menu' }
    ],

    screenReaderLabels: [
      { selector: 'input[type="email"]', expectedLabel: /email/i },
      { selector: 'input[type="password"]', expectedLabel: /password/i },
      { selector: 'button[type="submit"]', expectedLabel: /submit|send|login/i }
    ],

    colorContrast: {
      minimumRatio: 4.5, // WCAG AA standard
      elementsToCheck: ['button', 'a', '.text-primary', '.text-secondary']
    }
  };

  /**
   * Error scenarios for testing
   */
  static errorScenarios = {
    networkFailure: {
      status: 0,
      message: 'Network Error'
    },

    serverError: {
      status: 500,
      message: 'Internal Server Error'
    },

    authenticationError: {
      status: 401,
      message: 'Unauthorized'
    },

    validationError: {
      status: 400,
      message: 'Bad Request - Invalid data'
    },

    rateLimitError: {
      status: 429,
      message: 'Too Many Requests'
    }
  };
}

export default TestData;