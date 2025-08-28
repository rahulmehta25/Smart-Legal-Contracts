#!/usr/bin/env node

/**
 * Health Check Utility for Vercel Deployment
 * 
 * This script verifies that the deployed application is working correctly
 * by testing various endpoints and configurations.
 */

const https = require('https');
const http = require('http');
const { URL } = require('url');

// Configuration
const config = {
  frontend_url: process.env.FRONTEND_URL || 'https://your-app.vercel.app',
  backend_url: process.env.BACKEND_URL || 'https://your-backend-api.com',
  timeout: 10000, // 10 seconds
  retries: 3,
};

// Colors for console output
const colors = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
};

// Utility functions
const log = {
  info: (msg) => console.log(`${colors.blue}[INFO]${colors.reset} ${msg}`),
  success: (msg) => console.log(`${colors.green}[SUCCESS]${colors.reset} ${msg}`),
  warning: (msg) => console.log(`${colors.yellow}[WARNING]${colors.reset} ${msg}`),
  error: (msg) => console.log(`${colors.red}[ERROR]${colors.reset} ${msg}`),
  section: (msg) => console.log(`\n${colors.magenta}=== ${msg} ===${colors.reset}`),
};

// HTTP request helper
function makeRequest(url, options = {}) {
  return new Promise((resolve, reject) => {
    const urlObj = new URL(url);
    const requestLib = urlObj.protocol === 'https:' ? https : http;
    
    const reqOptions = {
      hostname: urlObj.hostname,
      port: urlObj.port,
      path: urlObj.pathname + urlObj.search,
      method: options.method || 'GET',
      timeout: config.timeout,
      headers: {
        'User-Agent': 'Health-Check/1.0.0',
        'Accept': 'application/json, text/html, */*',
        ...options.headers,
      },
    };

    const req = requestLib.request(reqOptions, (res) => {
      let data = '';
      res.on('data', (chunk) => {
        data += chunk;
      });
      
      res.on('end', () => {
        resolve({
          status: res.statusCode,
          headers: res.headers,
          data: data,
          url: url,
        });
      });
    });

    req.on('timeout', () => {
      req.destroy();
      reject(new Error(`Request timeout for ${url}`));
    });

    req.on('error', (err) => {
      reject(err);
    });

    if (options.body) {
      req.write(options.body);
    }
    
    req.end();
  });
}

// Test functions
async function testFrontendHealth() {
  log.section('Frontend Health Check');
  
  const tests = [
    {
      name: 'Frontend Root',
      url: config.frontend_url,
      expectedStatus: 200,
    },
    {
      name: 'Frontend Health via Proxy',
      url: `${config.frontend_url}/api/health`,
      expectedStatus: 200,
    },
    {
      name: 'API v1 Proxy',
      url: `${config.frontend_url}/api/v1`,
      expectedStatus: 200,
    },
  ];

  const results = [];
  
  for (const test of tests) {
    try {
      log.info(`Testing ${test.name}...`);
      const response = await makeRequest(test.url);
      
      if (response.status === test.expectedStatus) {
        log.success(`${test.name}: ${response.status}`);
        results.push({ ...test, status: 'PASS', response });
      } else {
        log.warning(`${test.name}: Expected ${test.expectedStatus}, got ${response.status}`);
        results.push({ ...test, status: 'WARN', response });
      }
    } catch (error) {
      log.error(`${test.name}: ${error.message}`);
      results.push({ ...test, status: 'FAIL', error: error.message });
    }
  }
  
  return results;
}

async function testBackendHealth() {
  log.section('Backend Health Check');
  
  const tests = [
    {
      name: 'Backend Root',
      url: config.backend_url,
      expectedStatus: 200,
    },
    {
      name: 'Backend Health',
      url: `${config.backend_url}/health`,
      expectedStatus: 200,
    },
    {
      name: 'Backend Detailed Health',
      url: `${config.backend_url}/health/detailed`,
      expectedStatus: 200,
    },
    {
      name: 'API v1 Overview',
      url: `${config.backend_url}/api/v1`,
      expectedStatus: 200,
    },
    {
      name: 'WebSocket Stats',
      url: `${config.backend_url}/api/websocket/stats`,
      expectedStatus: 200,
    },
  ];

  const results = [];
  
  for (const test of tests) {
    try {
      log.info(`Testing ${test.name}...`);
      const response = await makeRequest(test.url);
      
      if (response.status === test.expectedStatus) {
        log.success(`${test.name}: ${response.status}`);
        
        // Try to parse JSON response for additional validation
        if (response.headers['content-type']?.includes('application/json')) {
          try {
            const jsonData = JSON.parse(response.data);
            log.info(`  Response: ${JSON.stringify(jsonData, null, 2).substring(0, 200)}...`);
          } catch (e) {
            log.warning(`  Could not parse JSON response`);
          }
        }
        
        results.push({ ...test, status: 'PASS', response });
      } else {
        log.warning(`${test.name}: Expected ${test.expectedStatus}, got ${response.status}`);
        results.push({ ...test, status: 'WARN', response });
      }
    } catch (error) {
      log.error(`${test.name}: ${error.message}`);
      results.push({ ...test, status: 'FAIL', error: error.message });
    }
  }
  
  return results;
}

async function testCorsConfiguration() {
  log.section('CORS Configuration Check');
  
  try {
    log.info('Testing CORS headers...');
    const response = await makeRequest(`${config.backend_url}/health`, {
      headers: {
        'Origin': config.frontend_url,
        'Access-Control-Request-Method': 'GET',
        'Access-Control-Request-Headers': 'Content-Type, Authorization',
      },
    });
    
    const corsHeaders = {
      'access-control-allow-origin': response.headers['access-control-allow-origin'],
      'access-control-allow-methods': response.headers['access-control-allow-methods'],
      'access-control-allow-headers': response.headers['access-control-allow-headers'],
    };
    
    log.success('CORS headers found:');
    Object.entries(corsHeaders).forEach(([key, value]) => {
      if (value) {
        log.info(`  ${key}: ${value}`);
      } else {
        log.warning(`  ${key}: Not set`);
      }
    });
    
    return { status: 'PASS', headers: corsHeaders };
  } catch (error) {
    log.error(`CORS test failed: ${error.message}`);
    return { status: 'FAIL', error: error.message };
  }
}

async function testEnvironmentConfiguration() {
  log.section('Environment Configuration Check');
  
  const envChecks = [
    {
      name: 'Frontend URL',
      value: config.frontend_url,
      valid: config.frontend_url && config.frontend_url.startsWith('http'),
    },
    {
      name: 'Backend URL',
      value: config.backend_url,
      valid: config.backend_url && config.backend_url.startsWith('http'),
    },
  ];
  
  const results = [];
  
  envChecks.forEach((check) => {
    if (check.valid) {
      log.success(`${check.name}: ${check.value}`);
      results.push({ ...check, status: 'PASS' });
    } else {
      log.error(`${check.name}: Invalid or missing - ${check.value}`);
      results.push({ ...check, status: 'FAIL' });
    }
  });
  
  return results;
}

async function testApiIntegration() {
  log.section('API Integration Test');
  
  try {
    log.info('Testing document upload endpoint...');
    
    // Test with a simple text upload
    const testData = JSON.stringify({
      text: 'This is a test document for arbitration clause detection.',
    });
    
    const response = await makeRequest(`${config.frontend_url}/api/v1/analysis/text`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(testData),
      },
      body: testData,
    });
    
    if (response.status >= 200 && response.status < 300) {
      log.success(`API integration test: ${response.status}`);
      
      try {
        const jsonResponse = JSON.parse(response.data);
        log.info(`  Analysis result: ${jsonResponse.has_arbitration ? 'Found' : 'Not found'} arbitration clauses`);
      } catch (e) {
        log.warning('  Could not parse analysis response');
      }
      
      return { status: 'PASS', response };
    } else {
      log.warning(`API integration test: Unexpected status ${response.status}`);
      return { status: 'WARN', response };
    }
  } catch (error) {
    log.error(`API integration test failed: ${error.message}`);
    return { status: 'FAIL', error: error.message };
  }
}

// Main health check function
async function runHealthCheck() {
  console.log(`${colors.magenta}
╔══════════════════════════════════════════════════════════╗
║                    Health Check Report                   ║
║              Arbitration Detector Frontend               ║
╚══════════════════════════════════════════════════════════╝${colors.reset}`);
  
  log.info(`Frontend URL: ${config.frontend_url}`);
  log.info(`Backend URL: ${config.backend_url}`);
  log.info(`Timeout: ${config.timeout}ms`);
  
  const results = {
    timestamp: new Date().toISOString(),
    frontend_url: config.frontend_url,
    backend_url: config.backend_url,
    tests: {},
  };
  
  try {
    // Run all tests
    results.tests.environment = await testEnvironmentConfiguration();
    results.tests.frontend = await testFrontendHealth();
    results.tests.backend = await testBackendHealth();
    results.tests.cors = await testCorsConfiguration();
    results.tests.integration = await testApiIntegration();
    
    // Summary
    log.section('Health Check Summary');
    
    let totalTests = 0;
    let passedTests = 0;
    let failedTests = 0;
    let warningTests = 0;
    
    Object.entries(results.tests).forEach(([category, categoryResults]) => {
      if (Array.isArray(categoryResults)) {
        categoryResults.forEach((result) => {
          totalTests++;
          if (result.status === 'PASS') passedTests++;
          else if (result.status === 'FAIL') failedTests++;
          else if (result.status === 'WARN') warningTests++;
        });
      } else {
        totalTests++;
        if (categoryResults.status === 'PASS') passedTests++;
        else if (categoryResults.status === 'FAIL') failedTests++;
        else if (categoryResults.status === 'WARN') warningTests++;
      }
    });
    
    log.info(`Total tests: ${totalTests}`);
    log.success(`Passed: ${passedTests}`);
    log.warning(`Warnings: ${warningTests}`);
    log.error(`Failed: ${failedTests}`);
    
    const healthStatus = failedTests === 0 ? 'HEALTHY' : 'UNHEALTHY';
    const statusColor = healthStatus === 'HEALTHY' ? colors.green : colors.red;
    
    console.log(`\n${statusColor}Overall Status: ${healthStatus}${colors.reset}\n`);
    
    results.summary = {
      status: healthStatus,
      total: totalTests,
      passed: passedTests,
      warnings: warningTests,
      failed: failedTests,
    };
    
    // Save results if requested
    if (process.env.SAVE_RESULTS) {
      const fs = require('fs');
      const filename = `health-check-${Date.now()}.json`;
      fs.writeFileSync(filename, JSON.stringify(results, null, 2));
      log.info(`Results saved to: ${filename}`);
    }
    
    // Exit with appropriate code
    process.exit(failedTests > 0 ? 1 : 0);
    
  } catch (error) {
    log.error(`Health check failed: ${error.message}`);
    process.exit(1);
  }
}

// Handle command line arguments
function parseArgs() {
  const args = process.argv.slice(2);
  
  args.forEach((arg) => {
    if (arg.startsWith('--frontend-url=')) {
      config.frontend_url = arg.split('=')[1];
    } else if (arg.startsWith('--backend-url=')) {
      config.backend_url = arg.split('=')[1];
    } else if (arg.startsWith('--timeout=')) {
      config.timeout = parseInt(arg.split('=')[1]);
    } else if (arg === '--help') {
      console.log(`
Health Check Utility for Vercel Deployment

Usage: node health-check.js [options]

Options:
  --frontend-url=URL  Frontend URL to test (default: ${config.frontend_url})
  --backend-url=URL   Backend URL to test (default: ${config.backend_url})
  --timeout=MS        Request timeout in milliseconds (default: ${config.timeout})
  --help              Show this help message

Environment Variables:
  FRONTEND_URL        Frontend URL
  BACKEND_URL         Backend URL
  SAVE_RESULTS        Save results to JSON file
      `);
      process.exit(0);
    }
  });
}

// Main execution
if (require.main === module) {
  parseArgs();
  runHealthCheck().catch((error) => {
    log.error(`Unhandled error: ${error.message}`);
    process.exit(1);
  });
}

module.exports = {
  runHealthCheck,
  testFrontendHealth,
  testBackendHealth,
  testCorsConfiguration,
  testEnvironmentConfiguration,
  testApiIntegration,
};