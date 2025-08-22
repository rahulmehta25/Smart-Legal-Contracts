// Edge Router - Main request routing and orchestration
export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    const startTime = Date.now();
    
    try {
      // Add request ID for tracing
      const requestId = crypto.randomUUID();
      request.headers.set('X-Request-ID', requestId);
      
      // Get edge location
      const colo = request.cf?.colo || 'UNKNOWN';
      const country = request.cf?.country || 'XX';
      const region = request.cf?.region || 'Unknown';
      
      // Rate limiting check
      const rateLimitResponse = await checkRateLimit(request, env);
      if (rateLimitResponse) return rateLimitResponse;
      
      // DDoS protection
      const ddosCheck = await performDDoSCheck(request, env);
      if (ddosCheck.blocked) {
        return new Response('Access Denied', { status: 403 });
      }
      
      // Authentication for protected routes
      if (isProtectedRoute(url.pathname)) {
        const authResponse = await env.AUTH.fetch(request);
        if (authResponse.status !== 200) {
          return authResponse;
        }
        const user = await authResponse.json();
        request.headers.set('X-User-ID', user.id);
      }
      
      // Route based on path pattern
      let response;
      
      if (url.pathname.startsWith('/api/')) {
        response = await handleAPIRequest(request, env, ctx);
      } else if (url.pathname.startsWith('/ml/')) {
        response = await env.ML_INFERENCE.fetch(request);
      } else if (url.pathname.startsWith('/static/')) {
        response = await handleStaticContent(request, env, ctx);
      } else if (url.pathname.startsWith('/websocket')) {
        response = await handleWebSocket(request, env);
      } else {
        response = await handleDynamicContent(request, env, ctx);
      }
      
      // Add performance headers
      const duration = Date.now() - startTime;
      response.headers.set('X-Edge-Location', colo);
      response.headers.set('X-Edge-Region', `${country}-${region}`);
      response.headers.set('X-Response-Time', `${duration}ms`);
      response.headers.set('X-Request-ID', requestId);
      
      // Log metrics
      ctx.waitUntil(logMetrics(request, response, duration, env));
      
      return response;
      
    } catch (error) {
      console.error('Edge router error:', error);
      return new Response('Internal Server Error', { 
        status: 500,
        headers: {
          'X-Edge-Location': request.cf?.colo || 'UNKNOWN',
          'X-Error-ID': crypto.randomUUID()
        }
      });
    }
  }
};

async function checkRateLimit(request, env) {
  const ip = request.headers.get('CF-Connecting-IP');
  const key = `rate:${ip}`;
  
  const count = await env.RATE_LIMITS.get(key);
  const limit = 100; // requests per minute
  
  if (count && parseInt(count) > limit) {
    return new Response('Rate limit exceeded', { 
      status: 429,
      headers: {
        'Retry-After': '60',
        'X-RateLimit-Limit': limit.toString(),
        'X-RateLimit-Remaining': '0'
      }
    });
  }
  
  // Increment counter
  const newCount = count ? parseInt(count) + 1 : 1;
  await env.RATE_LIMITS.put(key, newCount.toString(), { 
    expirationTtl: 60 
  });
  
  return null;
}

async function performDDoSCheck(request, env) {
  const ip = request.headers.get('CF-Connecting-IP');
  const userAgent = request.headers.get('User-Agent');
  const asn = request.cf?.asn;
  
  // Check IP reputation
  const reputation = await env.EDGE_CONFIG.get(`reputation:${ip}`);
  if (reputation === 'blocked') {
    return { blocked: true, reason: 'IP reputation' };
  }
  
  // Check for suspicious patterns
  const suspiciousPatterns = [
    /bot|crawler|spider/i,
    /curl|wget|python/i,
    /scanner|nikto|sqlmap/i
  ];
  
  if (userAgent && suspiciousPatterns.some(pattern => pattern.test(userAgent))) {
    // Additional verification for legitimate bots
    const allowedBots = ['Googlebot', 'bingbot', 'Slackbot'];
    if (!allowedBots.some(bot => userAgent.includes(bot))) {
      return { blocked: true, reason: 'Suspicious user agent' };
    }
  }
  
  // Check request patterns
  const requestPattern = await analyzeRequestPattern(request, env);
  if (requestPattern.suspicious) {
    return { blocked: true, reason: 'Suspicious request pattern' };
  }
  
  return { blocked: false };
}

function isProtectedRoute(pathname) {
  const protectedPaths = [
    '/api/admin',
    '/api/user',
    '/dashboard',
    '/settings'
  ];
  
  return protectedPaths.some(path => pathname.startsWith(path));
}

async function handleAPIRequest(request, env, ctx) {
  const url = new URL(request.url);
  const cacheKey = `api:${url.pathname}:${url.search}`;
  
  // Check cache for GET requests
  if (request.method === 'GET') {
    const cached = await env.CACHE.get(cacheKey);
    if (cached) {
      const response = new Response(cached, {
        headers: {
          'Content-Type': 'application/json',
          'X-Cache': 'HIT'
        }
      });
      return response;
    }
  }
  
  // Forward to origin
  const originUrl = `${env.ORIGIN_URL}${url.pathname}${url.search}`;
  const originRequest = new Request(originUrl, request);
  
  const response = await fetch(originRequest);
  
  // Cache successful GET responses
  if (request.method === 'GET' && response.status === 200) {
    const responseText = await response.text();
    ctx.waitUntil(
      env.CACHE.put(cacheKey, responseText, { 
        expirationTtl: 300 // 5 minutes
      })
    );
    
    return new Response(responseText, {
      status: response.status,
      headers: {
        ...response.headers,
        'X-Cache': 'MISS'
      }
    });
  }
  
  return response;
}

async function handleStaticContent(request, env, ctx) {
  const url = new URL(request.url);
  const cacheKey = `static:${url.pathname}`;
  
  // Check edge cache
  const cached = await env.CACHE.get(cacheKey);
  if (cached) {
    return new Response(cached, {
      headers: {
        'Content-Type': getContentType(url.pathname),
        'Cache-Control': 'public, max-age=31536000',
        'X-Cache': 'HIT'
      }
    });
  }
  
  // Fetch from origin or CDN
  const cdnUrl = `${env.CDN_URL}${url.pathname}`;
  const response = await fetch(cdnUrl);
  
  if (response.status === 200) {
    const content = await response.arrayBuffer();
    
    // Apply transformations if needed
    let transformedContent = content;
    if (shouldTransform(url.pathname)) {
      transformedContent = await applyTransformations(content, request, env);
    }
    
    // Cache at edge
    ctx.waitUntil(
      env.CACHE.put(cacheKey, transformedContent, {
        expirationTtl: 86400 // 24 hours
      })
    );
    
    return new Response(transformedContent, {
      headers: {
        'Content-Type': getContentType(url.pathname),
        'Cache-Control': 'public, max-age=31536000',
        'X-Cache': 'MISS'
      }
    });
  }
  
  return response;
}

async function handleDynamicContent(request, env, ctx) {
  const url = new URL(request.url);
  
  // A/B testing logic
  const experiment = await getActiveExperiment(url.pathname, env);
  if (experiment) {
    const variant = selectVariant(request, experiment);
    request.headers.set('X-Experiment-ID', experiment.id);
    request.headers.set('X-Variant', variant);
    
    // Log impression
    ctx.waitUntil(logExperimentImpression(experiment.id, variant, env));
  }
  
  // Geo-targeting
  const country = request.cf?.country || 'US';
  const geoConfig = await env.EDGE_CONFIG.get(`geo:${country}`);
  if (geoConfig) {
    const config = JSON.parse(geoConfig);
    request.headers.set('X-Geo-Config', JSON.stringify(config));
  }
  
  // Forward to appropriate origin based on routing rules
  const routingRules = await getRoutingRules(env);
  const destination = selectDestination(request, routingRules);
  
  const originRequest = new Request(destination, request);
  const response = await fetch(originRequest);
  
  // Apply edge-side includes (ESI) if needed
  if (response.headers.get('Surrogate-Control')?.includes('ESI/1.0')) {
    const body = await response.text();
    const processedBody = await processESI(body, env);
    return new Response(processedBody, response);
  }
  
  return response;
}

async function handleWebSocket(request, env) {
  const upgradeHeader = request.headers.get('Upgrade');
  if (upgradeHeader !== 'websocket') {
    return new Response('Expected websocket', { status: 400 });
  }
  
  // Select WebSocket origin based on load
  const wsOrigin = await selectWebSocketOrigin(env);
  
  // Create WebSocket pair
  const webSocketPair = new WebSocketPair();
  const [client, server] = Object.values(webSocketPair);
  
  // Connect to origin WebSocket
  const originWS = new WebSocket(wsOrigin);
  
  // Relay messages
  server.accept();
  server.addEventListener('message', event => {
    originWS.send(event.data);
  });
  
  originWS.addEventListener('message', event => {
    server.send(event.data);
  });
  
  // Handle connection close
  server.addEventListener('close', () => originWS.close());
  originWS.addEventListener('close', () => server.close());
  
  return new Response(null, {
    status: 101,
    webSocket: client
  });
}

async function analyzeRequestPattern(request, env) {
  const ip = request.headers.get('CF-Connecting-IP');
  const url = new URL(request.url);
  
  // Check for SQL injection patterns
  const sqlPatterns = [
    /(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|CREATE|ALTER)\b)/gi,
    /(\'|\"|;|--|\*|\/\*|\*\/)/g
  ];
  
  const params = url.searchParams.toString();
  if (sqlPatterns.some(pattern => pattern.test(params))) {
    return { suspicious: true, type: 'sql_injection' };
  }
  
  // Check for XSS patterns
  const xssPatterns = [
    /<script[^>]*>.*?<\/script>/gi,
    /javascript:/gi,
    /on\w+\s*=/gi
  ];
  
  if (xssPatterns.some(pattern => pattern.test(params))) {
    return { suspicious: true, type: 'xss' };
  }
  
  // Check request frequency
  const frequencyKey = `freq:${ip}:${Date.now() / 1000 | 0}`;
  const frequency = await env.RATE_LIMITS.get(frequencyKey);
  
  if (frequency && parseInt(frequency) > 10) {
    return { suspicious: true, type: 'high_frequency' };
  }
  
  return { suspicious: false };
}

function getContentType(pathname) {
  const ext = pathname.split('.').pop().toLowerCase();
  const types = {
    'html': 'text/html',
    'css': 'text/css',
    'js': 'application/javascript',
    'json': 'application/json',
    'png': 'image/png',
    'jpg': 'image/jpeg',
    'jpeg': 'image/jpeg',
    'gif': 'image/gif',
    'svg': 'image/svg+xml',
    'webp': 'image/webp',
    'woff': 'font/woff',
    'woff2': 'font/woff2',
    'ttf': 'font/ttf',
    'eot': 'application/vnd.ms-fontobject'
  };
  
  return types[ext] || 'application/octet-stream';
}

function shouldTransform(pathname) {
  return pathname.endsWith('.html') || 
         pathname.endsWith('.css') || 
         pathname.endsWith('.js');
}

async function applyTransformations(content, request, env) {
  // Get transformation rules
  const rules = await env.EDGE_CONFIG.get('transform:rules');
  if (!rules) return content;
  
  const transformRules = JSON.parse(rules);
  let transformed = new TextDecoder().decode(content);
  
  // Apply each transformation
  for (const rule of transformRules) {
    if (rule.type === 'replace') {
      transformed = transformed.replace(
        new RegExp(rule.pattern, 'g'),
        rule.replacement
      );
    } else if (rule.type === 'inject') {
      if (rule.position === 'head') {
        transformed = transformed.replace('</head>', `${rule.content}</head>`);
      } else if (rule.position === 'body') {
        transformed = transformed.replace('</body>', `${rule.content}</body>`);
      }
    }
  }
  
  return new TextEncoder().encode(transformed);
}

async function getActiveExperiment(pathname, env) {
  const experiments = await env.EXPERIMENTS.get('active');
  if (!experiments) return null;
  
  const activeExperiments = JSON.parse(experiments);
  
  // Find matching experiment for path
  return activeExperiments.find(exp => {
    const pattern = new RegExp(exp.pathPattern);
    return pattern.test(pathname) && exp.status === 'active';
  });
}

function selectVariant(request, experiment) {
  // Use consistent hashing based on user ID or session
  const userId = request.headers.get('X-User-ID') || 
                 request.headers.get('CF-Connecting-IP');
  
  const hash = hashCode(userId + experiment.id);
  const bucket = Math.abs(hash) % 100;
  
  let accumulated = 0;
  for (const variant of experiment.variants) {
    accumulated += variant.percentage;
    if (bucket < accumulated) {
      return variant.name;
    }
  }
  
  return 'control';
}

function hashCode(str) {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return hash;
}

async function logExperimentImpression(experimentId, variant, env) {
  const timestamp = Date.now();
  const key = `exp:${experimentId}:${variant}:${timestamp}`;
  
  await env.EXPERIMENTS.put(key, '1', {
    expirationTtl: 86400 // 24 hours
  });
}

async function getRoutingRules(env) {
  const rules = await env.EDGE_CONFIG.get('routing:rules');
  return rules ? JSON.parse(rules) : [];
}

function selectDestination(request, rules) {
  const url = new URL(request.url);
  const country = request.cf?.country || 'US';
  
  for (const rule of rules) {
    if (rule.type === 'geo' && rule.countries.includes(country)) {
      return rule.destination;
    }
    
    if (rule.type === 'path' && new RegExp(rule.pattern).test(url.pathname)) {
      return rule.destination;
    }
    
    if (rule.type === 'header' && request.headers.get(rule.header) === rule.value) {
      return rule.destination;
    }
  }
  
  // Default destination
  return process.env.ORIGIN_URL || 'https://origin.example.com';
}

async function selectWebSocketOrigin(env) {
  const origins = await env.EDGE_CONFIG.get('ws:origins');
  if (!origins) return 'wss://ws.example.com';
  
  const originList = JSON.parse(origins);
  
  // Simple round-robin selection
  const index = Math.floor(Math.random() * originList.length);
  return originList[index];
}

async function processESI(html, env) {
  const esiPattern = /<esi:include\s+src="([^"]+)"\s*\/>/g;
  let processed = html;
  
  const includes = [...html.matchAll(esiPattern)];
  
  for (const match of includes) {
    const [tag, src] = match;
    
    try {
      const response = await fetch(src);
      const content = await response.text();
      processed = processed.replace(tag, content);
    } catch (error) {
      console.error(`ESI include failed for ${src}:`, error);
      processed = processed.replace(tag, '<!-- ESI include failed -->');
    }
  }
  
  return processed;
}

async function logMetrics(request, response, duration, env) {
  const metrics = {
    timestamp: Date.now(),
    method: request.method,
    path: new URL(request.url).pathname,
    status: response.status,
    duration: duration,
    colo: request.cf?.colo || 'UNKNOWN',
    country: request.cf?.country || 'XX',
    cacheStatus: response.headers.get('X-Cache') || 'NONE'
  };
  
  // Send to analytics engine
  if (env.ANALYTICS) {
    await env.ANALYTICS.writeDataPoint(metrics);
  }
  
  // Store in KV for aggregation
  const metricsKey = `metrics:${Date.now()}:${crypto.randomUUID()}`;
  await env.EDGE_CONFIG.put(metricsKey, JSON.stringify(metrics), {
    expirationTtl: 3600 // 1 hour
  });
}