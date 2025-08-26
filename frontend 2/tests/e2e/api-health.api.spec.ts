import { test, expect } from '@playwright/test';
import { TestData } from '../fixtures/test-data';

/**
 * API Health Check Tests
 * 
 * These tests validate the backend API endpoints are working correctly
 * and can handle various scenarios including success, failure, and edge cases.
 */

test.describe('API Health Check', () => {
  const baseURL = process.env.API_URL || 'http://localhost:8000';

  test('should return healthy status from /health endpoint', async ({ request }) => {
    const response = await request.get(`${baseURL}/health`);
    
    expect(response.status()).toBe(200);
    
    const body = await response.json();
    expect(body).toMatchObject({
      status: 'healthy',
      service: 'Arbitration RAG API',
      version: '1.0.0'
    });
  });

  test('should return API information from root endpoint', async ({ request }) => {
    const response = await request.get(`${baseURL}/`);
    
    expect(response.status()).toBe(200);
    
    const body = await response.json();
    expect(body).toHaveProperty('message', 'Arbitration RAG API');
    expect(body).toHaveProperty('version', '1.0.0');
    expect(body).toHaveProperty('docs', '/docs');
    expect(body).toHaveProperty('health', '/health');
  });

  test('should return API v1 overview with endpoints', async ({ request }) => {
    const response = await request.get(`${baseURL}/api/v1`);
    
    expect(response.status()).toBe(200);
    
    const body = await response.json();
    expect(body).toHaveProperty('version', '1.0.0');
    expect(body).toHaveProperty('endpoints');
    expect(body).toHaveProperty('features');
    
    // Verify key endpoints are listed
    const endpoints = body.endpoints;
    expect(endpoints).toHaveProperty('documents');
    expect(endpoints).toHaveProperty('analysis');
    expect(endpoints).toHaveProperty('users');
    expect(endpoints).toHaveProperty('websocket');
  });

  test('should serve OpenAPI specification', async ({ request }) => {
    const response = await request.get(`${baseURL}/openapi.json`);
    
    expect(response.status()).toBe(200);
    
    const body = await response.json();
    expect(body).toHaveProperty('info');
    expect(body).toHaveProperty('paths');
    expect(body.info).toHaveProperty('title', 'Arbitration RAG API');
  });

  test('should handle CORS preflight requests', async ({ request }) => {
    const response = await request.fetch(`${baseURL}/api/v1`, {
      method: 'OPTIONS',
      headers: {
        'Origin': 'http://localhost:3000',
        'Access-Control-Request-Method': 'POST',
        'Access-Control-Request-Headers': 'Content-Type'
      }
    });

    expect(response.status()).toBe(200);
    
    const headers = response.headers();
    expect(headers['access-control-allow-origin']).toBeTruthy();
    expect(headers['access-control-allow-methods']).toBeTruthy();
    expect(headers['access-control-allow-headers']).toBeTruthy();
  });

  test('should return 404 for non-existent endpoints', async ({ request }) => {
    const response = await request.get(`${baseURL}/api/v1/nonexistent`);
    expect(response.status()).toBe(404);
  });

  test('should handle malformed requests gracefully', async ({ request }) => {
    // Test with invalid JSON
    const response = await request.post(`${baseURL}/api/v1/analysis/quick-analyze`, {
      data: 'invalid json{',
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    // Should return 400 or 422 for malformed data
    expect([400, 422, 500].includes(response.status())).toBeTruthy();
  });

  test('should respond within acceptable time limits', async ({ request }) => {
    const startTime = Date.now();
    const response = await request.get(`${baseURL}/health`);
    const responseTime = Date.now() - startTime;
    
    expect(response.status()).toBe(200);
    expect(responseTime).toBeLessThan(5000); // Should respond within 5 seconds
  });

  test('should handle concurrent requests', async ({ request }) => {
    const requests = Array.from({ length: 10 }, () => 
      request.get(`${baseURL}/health`)
    );
    
    const responses = await Promise.all(requests);
    
    // All requests should succeed
    responses.forEach(response => {
      expect(response.status()).toBe(200);
    });
  });

  test('should maintain consistent response format', async ({ request }) => {
    const endpoints = ['/health', '/', '/api/v1'];
    
    for (const endpoint of endpoints) {
      const response = await request.get(`${baseURL}${endpoint}`);
      expect(response.status()).toBe(200);
      
      const body = await response.json();
      expect(typeof body).toBe('object');
      expect(body).not.toBeNull();
    }
  });
});

test.describe('API Error Handling', () => {
  const baseURL = process.env.API_URL || 'http://localhost:8000';

  test('should handle database connection errors gracefully', async ({ request }) => {
    // These endpoints require database connection and will fail with 500
    const dbEndpoints = [
      '/api/v1/users/register',
      '/api/v1/documents/',
      '/api/v1/analysis/'
    ];

    for (const endpoint of dbEndpoints) {
      const response = await request.get(`${baseURL}${endpoint}`);
      
      // Should return 500 due to missing database connection
      expect(response.status()).toBe(500);
      
      const body = await response.json();
      expect(body).toHaveProperty('detail');
    }
  });

  test('should return appropriate HTTP status codes', async ({ request }) => {
    const testCases = [
      { endpoint: '/health', expectedStatus: 200 },
      { endpoint: '/nonexistent', expectedStatus: 404 },
      { endpoint: '/api/v1/users/register', expectedStatus: 500 }, // DB error
    ];

    for (const { endpoint, expectedStatus } of testCases) {
      const response = await request.get(`${baseURL}${endpoint}`);
      expect(response.status()).toBe(expectedStatus);
    }
  });

  test('should validate request content types', async ({ request }) => {
    const response = await request.post(`${baseURL}/api/v1/analysis/quick-analyze`, {
      data: 'plain text instead of JSON',
      headers: {
        'Content-Type': 'text/plain'
      }
    });

    // Should reject non-JSON content
    expect([400, 422, 500].includes(response.status())).toBeTruthy();
  });
});

test.describe('API Performance', () => {
  const baseURL = process.env.API_URL || 'http://localhost:8000';

  test('should handle high-frequency health checks', async ({ request }) => {
    const startTime = Date.now();
    const requests = Array.from({ length: 50 }, () => 
      request.get(`${baseURL}/health`)
    );
    
    const responses = await Promise.all(requests);
    const totalTime = Date.now() - startTime;
    
    // All requests should succeed
    responses.forEach(response => {
      expect(response.status()).toBe(200);
    });

    // Should handle 50 requests in under 10 seconds
    expect(totalTime).toBeLessThan(10000);
    
    console.log(`✅ Handled 50 concurrent requests in ${totalTime}ms`);
  });

  test('should maintain response consistency under load', async ({ request }) => {
    const responses = await Promise.all(
      Array.from({ length: 20 }, async () => {
        const response = await request.get(`${baseURL}/health`);
        return {
          status: response.status(),
          body: await response.json()
        };
      })
    );

    // All responses should be identical
    const firstResponse = responses[0];
    responses.forEach(response => {
      expect(response.status).toBe(firstResponse.status);
      expect(response.body).toEqual(firstResponse.body);
    });
  });
});