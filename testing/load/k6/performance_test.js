/**
 * K6 Performance Testing Script
 * Enterprise-scale load testing with 10,000+ virtual users
 */

import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Rate, Trend, Counter, Gauge } from 'k6/metrics';
import { SharedArray } from 'k6/data';
import { randomItem, randomIntBetween } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';
import ws from 'k6/ws';
import { htmlReport } from 'https://raw.githubusercontent.com/benc-uk/k6-reporter/main/dist/bundle.js';

// Custom metrics
const errorRate = new Rate('errors');
const apiLatency = new Trend('api_latency');
const dbQueryTime = new Trend('db_query_time');
const cacheHitRate = new Rate('cache_hits');
const throughput = new Counter('throughput_requests');
const concurrentUsers = new Gauge('concurrent_users');

// Test data
const testUsers = new SharedArray('users', function () {
    return Array.from({ length: 10000 }, (_, i) => ({
        id: i + 1,
        username: `user${i + 1}`,
        password: 'password123',
        email: `user${i + 1}@example.com`
    }));
});

// Configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const WS_URL = __ENV.WS_URL || 'ws://localhost:8000';

/**
 * Test Scenarios Configuration
 */
export const options = {
    scenarios: {
        // Scenario 1: Gradual ramp-up to 10,000 users
        gradual_load: {
            executor: 'ramping-vus',
            startVUs: 0,
            stages: [
                { duration: '2m', target: 1000 },   // Ramp up to 1000 users
                { duration: '3m', target: 3000 },   // Ramp up to 3000 users
                { duration: '5m', target: 5000 },   // Ramp up to 5000 users
                { duration: '5m', target: 10000 },  // Ramp up to 10000 users
                { duration: '10m', target: 10000 }, // Stay at 10000 users
                { duration: '5m', target: 0 },      // Ramp down
            ],
            gracefulRampDown: '30s',
            exec: 'mainScenario'
        },
        
        // Scenario 2: Spike test
        spike_test: {
            executor: 'ramping-vus',
            startVUs: 100,
            stages: [
                { duration: '30s', target: 100 },   // Baseline
                { duration: '10s', target: 5000 },  // Spike to 5000
                { duration: '2m', target: 5000 },   // Hold spike
                { duration: '30s', target: 100 },   // Back to baseline
            ],
            gracefulRampDown: '10s',
            exec: 'spikeScenario',
            startTime: '35m'  // Start after gradual load test
        },
        
        // Scenario 3: Stress test
        stress_test: {
            executor: 'ramping-arrival-rate',
            startRate: 100,
            timeUnit: '1s',
            preAllocatedVUs: 2000,
            maxVUs: 15000,
            stages: [
                { duration: '5m', target: 500 },   // 500 RPS
                { duration: '5m', target: 1000 },  // 1000 RPS
                { duration: '5m', target: 2000 },  // 2000 RPS
                { duration: '5m', target: 3000 },  // 3000 RPS
            ],
            exec: 'stressScenario',
            startTime: '45m'
        },
        
        // Scenario 4: Constant load (soak test)
        soak_test: {
            executor: 'constant-vus',
            vus: 5000,
            duration: '30m',
            exec: 'soakScenario',
            startTime: '70m'
        }
    },
    
    thresholds: {
        // Performance thresholds (targets)
        http_req_duration: [
            'p(50)<100',   // 50% of requests under 100ms
            'p(95)<500',   // 95% of requests under 500ms
            'p(99)<2000',  // 99% of requests under 2s
        ],
        http_req_failed: ['rate<0.01'],        // Error rate < 1%
        errors: ['rate<0.05'],                 // Custom error rate < 5%
        api_latency: ['p(95)<300'],           // API latency P95 < 300ms
        cache_hits: ['rate>0.8'],              // Cache hit rate > 80%
        throughput_requests: ['count>100000'], // Total throughput > 100k requests
    },
    
    // Tags for organizing metrics
    tags: {
        test_type: 'performance',
        environment: __ENV.ENVIRONMENT || 'staging'
    }
};

/**
 * Setup function - runs once before tests
 */
export function setup() {
    console.log('Setting up test environment...');
    
    // Health check
    const healthCheck = http.get(`${BASE_URL}/health`);
    check(healthCheck, {
        'System is healthy': (r) => r.status === 200
    });
    
    // Warm up cache
    console.log('Warming up cache...');
    for (let i = 0; i < 10; i++) {
        http.get(`${BASE_URL}/api/warm-cache/${i}`);
    }
    
    return {
        startTime: new Date().toISOString(),
        testData: generateTestData()
    };
}

/**
 * Main test scenario
 */
export function mainScenario(data) {
    const user = randomItem(testUsers);
    const token = authenticateUser(user);
    
    if (!token) {
        errorRate.add(1);
        return;
    }
    
    // Update concurrent users metric
    concurrentUsers.add(__VU);
    
    // User journey
    group('User Journey', () => {
        group('Dashboard', () => {
            loadDashboard(token);
        });
        
        group('Search', () => {
            performSearch(token);
        });
        
        group('CRUD Operations', () => {
            crudOperations(token);
        });
        
        group('Analytics', () => {
            viewAnalytics(token);
        });
        
        group('File Operations', () => {
            if (Math.random() > 0.7) {
                fileOperations(token);
            }
        });
    });
    
    sleep(randomIntBetween(1, 3));
}

/**
 * Spike test scenario
 */
export function spikeScenario(data) {
    const startTime = Date.now();
    
    // Rapid fire requests
    for (let i = 0; i < 10; i++) {
        const response = http.get(`${BASE_URL}/api/spike-test`, {
            tags: { scenario: 'spike' }
        });
        
        check(response, {
            'Spike response OK': (r) => r.status === 200
        });
    }
    
    apiLatency.add(Date.now() - startTime);
}

/**
 * Stress test scenario
 */
export function stressScenario(data) {
    // Heavy operations
    const payload = {
        data: Array(1000).fill(0).map(() => ({
            id: randomIntBetween(1, 100000),
            value: Math.random() * 1000,
            timestamp: Date.now()
        }))
    };
    
    const response = http.post(`${BASE_URL}/api/batch-process`, 
        JSON.stringify(payload), {
        headers: { 'Content-Type': 'application/json' },
        tags: { scenario: 'stress' }
    });
    
    check(response, {
        'Batch process successful': (r) => r.status === 200
    });
    
    throughput.add(1);
}

/**
 * Soak test scenario
 */
export function soakScenario(data) {
    // Sustained load with regular operations
    const operations = [
        () => http.get(`${BASE_URL}/api/status`),
        () => http.get(`${BASE_URL}/api/metrics`),
        () => http.post(`${BASE_URL}/api/log`, JSON.stringify({ 
            message: 'Soak test', 
            timestamp: Date.now() 
        }))
    ];
    
    const operation = randomItem(operations);
    const response = operation();
    
    check(response, {
        'Soak operation successful': (r) => r.status < 400
    });
    
    sleep(randomIntBetween(2, 5));
}

/**
 * Helper Functions
 */

function authenticateUser(user) {
    const response = http.post(`${BASE_URL}/api/auth/login`,
        JSON.stringify({
            username: user.username,
            password: user.password
        }), {
        headers: { 'Content-Type': 'application/json' },
        tags: { operation: 'login' }
    });
    
    const success = check(response, {
        'Login successful': (r) => r.status === 200,
        'Token received': (r) => r.json('token') !== undefined
    });
    
    if (!success) {
        errorRate.add(1);
        return null;
    }
    
    return response.json('token');
}

function loadDashboard(token) {
    const startTime = Date.now();
    
    const response = http.get(`${BASE_URL}/api/dashboard`, {
        headers: { 'Authorization': `Bearer ${token}` },
        tags: { operation: 'dashboard' }
    });
    
    const latency = Date.now() - startTime;
    apiLatency.add(latency);
    
    check(response, {
        'Dashboard loaded': (r) => r.status === 200,
        'Dashboard fast': (r) => latency < 500
    });
    
    // Check if response was cached
    if (response.headers['X-Cache-Hit'] === 'true') {
        cacheHitRate.add(1);
    } else {
        cacheHitRate.add(0);
    }
}

function performSearch(token) {
    const searchTerms = ['product', 'user', 'order', 'analytics', 'report'];
    const searchTerm = randomItem(searchTerms);
    
    const response = http.get(
        `${BASE_URL}/api/search?q=${searchTerm}&limit=50`, {
        headers: { 'Authorization': `Bearer ${token}` },
        tags: { operation: 'search' }
    });
    
    check(response, {
        'Search successful': (r) => r.status === 200,
        'Search results returned': (r) => r.json('results') !== undefined
    });
}

function crudOperations(token) {
    // Create
    const createResponse = http.post(
        `${BASE_URL}/api/entities`,
        JSON.stringify({
            name: `Entity_${randomIntBetween(1, 10000)}`,
            type: randomItem(['A', 'B', 'C']),
            value: Math.random() * 10000
        }), {
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
        },
        tags: { operation: 'create' }
    });
    
    if (createResponse.status === 201) {
        const entityId = createResponse.json('id');
        
        // Read
        const readResponse = http.get(
            `${BASE_URL}/api/entities/${entityId}`, {
            headers: { 'Authorization': `Bearer ${token}` },
            tags: { operation: 'read' }
        });
        
        check(readResponse, {
            'Entity retrieved': (r) => r.status === 200
        });
        
        // Update
        const updateResponse = http.put(
            `${BASE_URL}/api/entities/${entityId}`,
            JSON.stringify({ value: Math.random() * 5000 }), {
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${token}`
            },
            tags: { operation: 'update' }
        });
        
        check(updateResponse, {
            'Entity updated': (r) => r.status === 200
        });
        
        // Delete
        const deleteResponse = http.del(
            `${BASE_URL}/api/entities/${entityId}`, {
            headers: { 'Authorization': `Bearer ${token}` },
            tags: { operation: 'delete' }
        });
        
        check(deleteResponse, {
            'Entity deleted': (r) => r.status === 204
        });
    }
}

function viewAnalytics(token) {
    const startTime = Date.now();
    
    const response = http.post(
        `${BASE_URL}/api/analytics/aggregate`,
        JSON.stringify({
            start_date: '2024-01-01',
            end_date: '2024-12-31',
            group_by: randomItem(['day', 'week', 'month']),
            metrics: ['revenue', 'users', 'transactions']
        }), {
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
        },
        tags: { operation: 'analytics' }
    });
    
    const queryTime = Date.now() - startTime;
    dbQueryTime.add(queryTime);
    
    check(response, {
        'Analytics loaded': (r) => r.status === 200,
        'Query performant': () => queryTime < 1000
    });
}

function fileOperations(token) {
    // Upload file
    const file = open('./test-file.bin', 'b');
    const uploadResponse = http.post(
        `${BASE_URL}/api/files/upload`, {
        file: http.file(file, 'test-file.bin', 'application/octet-stream')
    }, {
        headers: { 'Authorization': `Bearer ${token}` },
        tags: { operation: 'file_upload' }
    });
    
    if (uploadResponse.status === 200) {
        const fileId = uploadResponse.json('file_id');
        
        // Download file
        const downloadResponse = http.get(
            `${BASE_URL}/api/files/${fileId}/download`, {
            headers: { 'Authorization': `Bearer ${token}` },
            tags: { operation: 'file_download' }
        });
        
        check(downloadResponse, {
            'File downloaded': (r) => r.status === 200
        });
    }
}

/**
 * WebSocket test function
 */
export function websocketTest() {
    const url = `${WS_URL}/ws`;
    const params = { tags: { scenario: 'websocket' } };
    
    const response = ws.connect(url, params, function (socket) {
        socket.on('open', () => {
            console.log('WebSocket connected');
            socket.send(JSON.stringify({ type: 'subscribe', channel: 'updates' }));
        });
        
        socket.on('message', (data) => {
            const message = JSON.parse(data);
            check(message, {
                'Valid message received': (m) => m.type !== undefined
            });
        });
        
        socket.on('error', (e) => {
            console.error('WebSocket error:', e);
            errorRate.add(1);
        });
        
        // Keep connection open for 30 seconds
        socket.setTimeout(() => {
            socket.close();
        }, 30000);
    });
    
    check(response, {
        'WebSocket connection successful': (r) => r.status === 101
    });
}

/**
 * Generate test data
 */
function generateTestData() {
    return {
        products: Array.from({ length: 1000 }, (_, i) => ({
            id: i + 1,
            name: `Product ${i + 1}`,
            price: Math.random() * 1000,
            category: randomItem(['Electronics', 'Clothing', 'Food', 'Books'])
        })),
        orders: Array.from({ length: 5000 }, (_, i) => ({
            id: i + 1,
            userId: randomIntBetween(1, 10000),
            total: Math.random() * 5000,
            status: randomItem(['pending', 'processing', 'completed', 'cancelled'])
        }))
    };
}

/**
 * Teardown function - runs once after tests
 */
export function teardown(data) {
    console.log('Test completed at:', new Date().toISOString());
    console.log('Test started at:', data.startTime);
    
    // Generate HTML report
    // This would be done externally in practice
}

/**
 * Handle summary generation
 */
export function handleSummary(data) {
    return {
        'summary.html': htmlReport(data),
        'summary.json': JSON.stringify(data, null, 2),
        stdout: textSummary(data, { indent: ' ', enableColors: true })
    };
}

function textSummary(data, options) {
    // Simple text summary
    const summary = [];
    summary.push('=== Performance Test Summary ===\n');
    summary.push(`Total Requests: ${data.metrics.http_reqs.values.count}\n`);
    summary.push(`Failed Requests: ${data.metrics.http_req_failed.values.rate * 100}%\n`);
    summary.push(`Avg Response Time: ${data.metrics.http_req_duration.values.avg}ms\n`);
    summary.push(`P95 Response Time: ${data.metrics.http_req_duration.values['p(95)']}ms\n`);
    summary.push(`P99 Response Time: ${data.metrics.http_req_duration.values['p(99)']}ms\n`);
    
    return summary.join('');
}