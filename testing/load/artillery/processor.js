/**
 * Artillery Processor Script
 * Custom logic for load testing scenarios
 */

const crypto = require('crypto');
const fs = require('fs');
const path = require('path');

// Store for session data
const sessions = new Map();

// Metrics tracking
const customMetrics = {
    cacheHits: 0,
    cacheMisses: 0,
    slowRequests: 0,
    errorsByType: {}
};

/**
 * Before request hook - modify requests before sending
 */
function beforeRequest(requestParams, context, ee, next) {
    // Add correlation ID to all requests
    requestParams.headers = requestParams.headers || {};
    requestParams.headers['X-Correlation-ID'] = generateCorrelationId();
    
    // Add timestamp
    requestParams.headers['X-Request-Time'] = Date.now().toString();
    
    // Add user session if exists
    if (context.vars.userId) {
        requestParams.headers['X-User-ID'] = context.vars.userId;
    }
    
    return next();
}

/**
 * After response hook - process responses
 */
function afterResponse(requestParams, response, context, ee, next) {
    // Track response time
    const responseTime = response.timings.phases.firstByte;
    
    // Check for slow requests
    if (responseTime > 2000) {
        customMetrics.slowRequests++;
        console.log(`Slow request detected: ${requestParams.url} took ${responseTime}ms`);
    }
    
    // Track cache hits/misses
    if (response.headers['x-cache-hit'] === 'true') {
        customMetrics.cacheHits++;
    } else {
        customMetrics.cacheMisses++;
    }
    
    // Track errors by type
    if (response.statusCode >= 400) {
        const errorType = `${response.statusCode}`;
        customMetrics.errorsByType[errorType] = (customMetrics.errorsByType[errorType] || 0) + 1;
    }
    
    // Store session data
    if (response.body && response.body.sessionId) {
        sessions.set(context.vars.userId, response.body.sessionId);
    }
    
    return next();
}

/**
 * Set authentication header
 */
function setAuthHeader(context, events, done) {
    if (context.vars.authToken) {
        context.vars.headers = context.vars.headers || {};
        context.vars.headers['Authorization'] = `Bearer ${context.vars.authToken}`;
    }
    return done();
}

/**
 * Generate correlation ID
 */
function generateCorrelationId() {
    return crypto.randomBytes(16).toString('hex');
}

/**
 * Generate random user data
 */
function generateUserData(context, events, done) {
    context.vars.userData = {
        firstName: randomFirstName(),
        lastName: randomLastName(),
        email: `user${Math.random().toString(36).substring(7)}@example.com`,
        age: Math.floor(Math.random() * 50) + 18,
        country: randomCountry()
    };
    return done();
}

/**
 * Simulate complex data processing
 */
function processComplexData(context, events, done) {
    // Simulate data processing delay
    const processingTime = Math.random() * 100;
    
    setTimeout(() => {
        context.vars.processedData = {
            timestamp: Date.now(),
            processingTime: processingTime,
            result: Math.random() * 1000
        };
        done();
    }, processingTime);
}

/**
 * Validate response data
 */
function validateResponse(context, events, done) {
    const response = context.vars.response;
    
    if (!response) {
        return done(new Error('No response received'));
    }
    
    // Custom validation logic
    if (response.statusCode === 200) {
        try {
            const body = JSON.parse(response.body);
            
            // Validate response structure
            if (!body.data || !body.timestamp) {
                return done(new Error('Invalid response structure'));
            }
            
            // Store validated data
            context.vars.validatedData = body.data;
        } catch (error) {
            return done(new Error('Failed to parse response'));
        }
    }
    
    return done();
}

/**
 * Generate file for upload
 */
function generateFile(context, events, done) {
    const fileSize = context.vars.fileSize || 1024 * 1024; // Default 1MB
    const buffer = crypto.randomBytes(fileSize);
    
    const fileName = `test_${Date.now()}.bin`;
    const filePath = path.join('/tmp', fileName);
    
    fs.writeFile(filePath, buffer, (err) => {
        if (err) {
            return done(err);
        }
        
        context.vars.filePath = filePath;
        context.vars.fileName = fileName;
        done();
    });
}

/**
 * Clean up uploaded file
 */
function cleanupFile(context, events, done) {
    if (context.vars.filePath) {
        fs.unlink(context.vars.filePath, (err) => {
            if (err) {
                console.error('Failed to delete file:', err);
            }
            done();
        });
    } else {
        done();
    }
}

/**
 * Calculate statistics
 */
function calculateStats(context, events, done) {
    const cacheHitRate = customMetrics.cacheHits / 
        (customMetrics.cacheHits + customMetrics.cacheMisses);
    
    context.vars.stats = {
        cacheHitRate: (cacheHitRate * 100).toFixed(2) + '%',
        slowRequests: customMetrics.slowRequests,
        errorCount: Object.values(customMetrics.errorsByType).reduce((a, b) => a + b, 0)
    };
    
    console.log('Current stats:', context.vars.stats);
    
    return done();
}

/**
 * Simulate database query
 */
function simulateDbQuery(context, events, done) {
    // Simulate variable query time
    const queryTime = Math.random() * 500 + 50; // 50-550ms
    
    setTimeout(() => {
        context.vars.dbResult = {
            rows: Math.floor(Math.random() * 1000),
            queryTime: queryTime,
            cached: Math.random() > 0.5
        };
        done();
    }, queryTime);
}

/**
 * WebSocket message handler
 */
function handleWebSocketMessage(context, events, done) {
    // Process WebSocket message
    if (context.vars.wsMessage) {
        const message = JSON.parse(context.vars.wsMessage);
        
        switch (message.type) {
            case 'notification':
                context.vars.notifications = context.vars.notifications || [];
                context.vars.notifications.push(message);
                break;
            
            case 'update':
                context.vars.lastUpdate = message.data;
                break;
            
            case 'error':
                console.error('WebSocket error:', message.error);
                break;
        }
    }
    
    return done();
}

/**
 * Rate limiter simulation
 */
function checkRateLimit(context, events, done) {
    const userId = context.vars.userId;
    const now = Date.now();
    
    // Simple rate limiting logic
    if (!context.vars.requestTimes) {
        context.vars.requestTimes = [];
    }
    
    // Keep only requests from last minute
    context.vars.requestTimes = context.vars.requestTimes.filter(
        time => now - time < 60000
    );
    
    // Check if rate limit exceeded (100 requests per minute)
    if (context.vars.requestTimes.length >= 100) {
        return done(new Error('Rate limit exceeded'));
    }
    
    context.vars.requestTimes.push(now);
    return done();
}

/**
 * Helper functions
 */
function randomFirstName() {
    const names = ['John', 'Jane', 'Bob', 'Alice', 'Charlie', 'Diana'];
    return names[Math.floor(Math.random() * names.length)];
}

function randomLastName() {
    const names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia'];
    return names[Math.floor(Math.random() * names.length)];
}

function randomCountry() {
    const countries = ['USA', 'UK', 'Canada', 'Australia', 'Germany', 'France'];
    return countries[Math.floor(Math.random() * countries.length)];
}

/**
 * Export functions for Artillery
 */
module.exports = {
    beforeRequest,
    afterResponse,
    setAuthHeader,
    generateUserData,
    processComplexData,
    validateResponse,
    generateFile,
    cleanupFile,
    calculateStats,
    simulateDbQuery,
    handleWebSocketMessage,
    checkRateLimit
};