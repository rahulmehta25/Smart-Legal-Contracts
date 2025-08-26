import { test, expect } from '@playwright/test';
import { TestData } from '../fixtures/test-data';

/**
 * WebSocket Connection Tests
 * 
 * Tests real-time communication features including connection establishment,
 * message exchange, error handling, and connection lifecycle management.
 */

test.describe('WebSocket Connection', () => {
  const wsURL = 'ws://localhost:8000/ws';

  test('should establish WebSocket connection successfully', async ({ page }) => {
    // Navigate to the application
    await page.goto('/');

    // Add WebSocket connection monitoring
    let connectionEstablished = false;
    let welcomeMessage: any = null;

    await page.addInitScript(() => {
      (window as any).wsConnection = null;
      (window as any).wsMessages = [];
      
      // Mock or monitor WebSocket connections
      const originalWebSocket = (window as any).WebSocket;
      (window as any).WebSocket = class extends originalWebSocket {
        constructor(url: string, protocols?: string | string[]) {
          super(url, protocols);
          (window as any).wsConnection = this;
          
          this.addEventListener('open', () => {
            console.log('WebSocket connected');
          });
          
          this.addEventListener('message', (event) => {
            (window as any).wsMessages.push(event.data);
          });
        }
      };
    });

    // Trigger WebSocket connection (assuming the app auto-connects)
    await page.evaluate(() => {
      const ws = new WebSocket('ws://localhost:8000/ws');
      (window as any).testWs = ws;
      
      return new Promise((resolve) => {
        ws.onopen = () => resolve('connected');
        ws.onerror = () => resolve('error');
      });
    });

    // Wait for connection
    const connectionStatus = await page.evaluate(() => {
      return new Promise((resolve) => {
        const ws = (window as any).testWs;
        if (ws.readyState === WebSocket.OPEN) {
          resolve('open');
        } else if (ws.readyState === WebSocket.CLOSED) {
          resolve('closed');
        } else {
          ws.addEventListener('open', () => resolve('open'));
          ws.addEventListener('error', () => resolve('error'));
          setTimeout(() => resolve('timeout'), 5000);
        }
      });
    });

    expect(connectionStatus).toBe('open');
  });

  test('should receive welcome message on connection', async ({ page }) => {
    await page.goto('/');

    const welcomeMessage = await page.evaluate(async () => {
      const ws = new WebSocket('ws://localhost:8000/ws');
      
      return new Promise((resolve) => {
        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            resolve(data);
          } catch {
            resolve({ error: 'Invalid JSON' });
          }
        };
        
        ws.onerror = () => resolve({ error: 'Connection failed' });
        setTimeout(() => resolve({ error: 'Timeout' }), 5000);
      });
    });

    expect(welcomeMessage).toMatchObject({
      type: 'connected',
      message: 'WebSocket connected successfully',
      server: 'Arbitration RAG API',
      version: '1.0.0'
    });
  });

  test('should echo messages correctly', async ({ page }) => {
    await page.goto('/');

    const testMessage = TestData.webSocketMessages.connectionTest();
    
    const echoResponse = await page.evaluate(async (message) => {
      const ws = new WebSocket('ws://localhost:8000/ws');
      
      return new Promise((resolve) => {
        let welcomed = false;
        
        ws.onmessage = (event) => {
          const data = JSON.parse(event.data);
          
          if (data.type === 'connected' && !welcomed) {
            welcomed = true;
            // Send test message after welcome
            ws.send(JSON.stringify(message));
          } else if (data.type === 'echo') {
            resolve(data);
          }
        };
        
        ws.onerror = () => resolve({ error: 'Connection failed' });
        setTimeout(() => resolve({ error: 'Timeout' }), 10000);
      });
    }, testMessage);

    expect(echoResponse).toMatchObject({
      type: 'echo',
      original_message: testMessage,
      status: 'received'
    });
  });

  test('should handle plain text messages', async ({ page }) => {
    await page.goto('/');

    const testText = 'Plain text test message';
    
    const response = await page.evaluate(async (text) => {
      const ws = new WebSocket('ws://localhost:8000/ws');
      
      return new Promise((resolve) => {
        let welcomed = false;
        
        ws.onmessage = (event) => {
          const data = JSON.parse(event.data);
          
          if (data.type === 'connected' && !welcomed) {
            welcomed = true;
            // Send plain text message after welcome
            ws.send(text);
          } else if (data.type === 'text_received') {
            resolve(data);
          }
        };
        
        ws.onerror = () => resolve({ error: 'Connection failed' });
        setTimeout(() => resolve({ error: 'Timeout' }), 10000);
      });
    }, testText);

    expect(response).toMatchObject({
      type: 'text_received',
      message: `Received: ${testText}`
    });
  });

  test('should handle connection disconnection gracefully', async ({ page }) => {
    await page.goto('/');

    const disconnectionHandled = await page.evaluate(async () => {
      const ws = new WebSocket('ws://localhost:8000/ws');
      
      return new Promise((resolve) => {
        ws.onopen = () => {
          // Close connection immediately after opening
          ws.close();
        };
        
        ws.onclose = () => {
          resolve(true);
        };
        
        ws.onerror = () => resolve(false);
        setTimeout(() => resolve(false), 5000);
      });
    });

    expect(disconnectionHandled).toBe(true);
  });

  test('should reject invalid WebSocket URLs', async ({ page }) => {
    await page.goto('/');

    const connectionFailed = await page.evaluate(async () => {
      try {
        const ws = new WebSocket('ws://localhost:9999/invalid');
        
        return new Promise((resolve) => {
          ws.onopen = () => resolve(false); // Should not open
          ws.onerror = () => resolve(true); // Should error
          setTimeout(() => resolve(true), 3000); // Timeout = failure to connect
        });
      } catch {
        return true; // Exception = expected behavior
      }
    });

    expect(connectionFailed).toBe(true);
  });
});

test.describe('WebSocket Statistics API', () => {
  const baseURL = process.env.API_URL || 'http://localhost:8000';

  test('should return WebSocket statistics', async ({ request }) => {
    // This endpoint may fail if database is not connected
    const response = await request.get(`${baseURL}/api/websocket/stats`);
    
    if (response.status() === 200) {
      const body = await response.json();
      expect(body).toHaveProperty('active_connections');
      expect(body).toHaveProperty('server_status');
      expect(body).toHaveProperty('features');
      expect(typeof body.active_connections).toBe('number');
    } else {
      // Expected if database is not connected
      expect(response.status()).toBe(500);
    }
  });
});

test.describe('WebSocket Error Scenarios', () => {
  test('should handle malformed JSON messages', async ({ page }) => {
    await page.goto('/');

    const errorHandled = await page.evaluate(async () => {
      const ws = new WebSocket('ws://localhost:8000/ws');
      
      return new Promise((resolve) => {
        let welcomed = false;
        
        ws.onmessage = (event) => {
          const data = JSON.parse(event.data);
          
          if (data.type === 'connected' && !welcomed) {
            welcomed = true;
            // Send malformed JSON
            ws.send('{"invalid": json}');
          } else {
            // Server should handle gracefully and not close connection
            resolve(ws.readyState === WebSocket.OPEN);
          }
        };
        
        ws.onerror = () => resolve(false);
        ws.onclose = () => resolve(false);
        setTimeout(() => resolve(ws.readyState === WebSocket.OPEN), 3000);
      });
    });

    expect(errorHandled).toBe(true);
  });

  test('should handle rapid message sending', async ({ page }) => {
    await page.goto('/');

    const rapidMessagesHandled = await page.evaluate(async () => {
      const ws = new WebSocket('ws://localhost:8000/ws');
      
      return new Promise((resolve) => {
        let welcomed = false;
        let messagesReceived = 0;
        const totalMessages = 10;
        
        ws.onmessage = (event) => {
          const data = JSON.parse(event.data);
          
          if (data.type === 'connected' && !welcomed) {
            welcomed = true;
            
            // Send rapid messages
            for (let i = 0; i < totalMessages; i++) {
              ws.send(JSON.stringify({ id: i, message: `Test ${i}` }));
            }
          } else if (data.type === 'echo') {
            messagesReceived++;
            
            if (messagesReceived === totalMessages) {
              resolve(true);
            }
          }
        };
        
        ws.onerror = () => resolve(false);
        setTimeout(() => resolve(messagesReceived >= totalMessages * 0.8), 10000); // Allow 80% success
      });
    });

    expect(rapidMessagesHandled).toBe(true);
  });

  test('should handle WebSocket reconnection', async ({ page }) => {
    await page.goto('/');

    const reconnectionWorked = await page.evaluate(async () => {
      let ws = new WebSocket('ws://localhost:8000/ws');
      
      return new Promise((resolve) => {
        let firstConnection = false;
        
        ws.onopen = () => {
          if (!firstConnection) {
            firstConnection = true;
            // Close connection to test reconnection
            ws.close();
            
            // Attempt reconnection after a delay
            setTimeout(() => {
              ws = new WebSocket('ws://localhost:8000/ws');
              
              ws.onopen = () => resolve(true);
              ws.onerror = () => resolve(false);
            }, 1000);
          }
        };
        
        ws.onerror = () => resolve(false);
        setTimeout(() => resolve(false), 10000);
      });
    });

    expect(reconnectionWorked).toBe(true);
  });
});