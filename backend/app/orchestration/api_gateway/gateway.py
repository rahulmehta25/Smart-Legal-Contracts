"""
API Gateway Implementation

Central API gateway that handles all incoming requests and routes them 
to appropriate microservices with authentication, rate limiting, and monitoring.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import time
import uuid
from urllib.parse import urlparse, parse_qs
import aiohttp
import jwt

logger = logging.getLogger(__name__)


class RouteMethod(Enum):
    GET = "GET"
    POST = "POST" 
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"
    OPTIONS = "OPTIONS"
    HEAD = "HEAD"


class LoadBalanceStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"
    IP_HASH = "ip_hash"


@dataclass
class ServiceEndpoint:
    """Represents a backend service endpoint"""
    host: str
    port: int
    protocol: str = "http"
    health_check_path: str = "/health"
    weight: float = 1.0
    max_connections: int = 100
    current_connections: int = 0
    response_time: float = 0.0
    success_rate: float = 1.0
    last_health_check: Optional[datetime] = None
    is_healthy: bool = True
    
    @property
    def url(self) -> str:
        return f"{self.protocol}://{self.host}:{self.port}"
    
    @property
    def load_score(self) -> float:
        """Calculate load score for load balancing decisions"""
        connection_ratio = self.current_connections / self.max_connections
        return (connection_ratio * 0.5) + (self.response_time / 1000 * 0.3) + ((1 - self.success_rate) * 0.2)


@dataclass  
class RouteRule:
    """Defines routing rules for requests"""
    path: str
    methods: List[RouteMethod]
    service_name: str
    target_path: Optional[str] = None
    authentication_required: bool = True
    rate_limit: Optional[Dict[str, Any]] = None
    timeout: float = 30.0
    retry_attempts: int = 3
    circuit_breaker: bool = True
    load_balance_strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN
    middleware: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def matches(self, path: str, method: str) -> bool:
        """Check if request matches this route rule"""
        # Simple path matching - could be enhanced with regex/wildcards
        path_match = path.startswith(self.path) or path == self.path
        method_match = RouteMethod(method.upper()) in self.methods
        return path_match and method_match


@dataclass
class RequestContext:
    """Context for processing a request through the gateway"""
    request_id: str
    method: str
    path: str
    headers: Dict[str, str]
    query_params: Dict[str, List[str]]
    body: Optional[bytes] = None
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    client_ip: str = ""
    user_agent: str = ""
    start_time: float = field(default_factory=time.time)
    route_rule: Optional[RouteRule] = None
    target_endpoint: Optional[ServiceEndpoint] = None
    
    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time


class APIGateway:
    """
    Central API Gateway for routing requests to microservices
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.routes: List[RouteRule] = []
        self.services: Dict[str, List[ServiceEndpoint]] = {}
        self.middleware_stack: List[Callable] = []
        
        # Load balancing state
        self._round_robin_counters: Dict[str, int] = {}
        
        # Circuit breaker state
        self._circuit_breaker_state: Dict[str, Dict[str, Any]] = {}
        
        # Rate limiting state  
        self._rate_limit_buckets: Dict[str, Dict[str, Any]] = {}
        
        # Monitoring
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "active_connections": 0
        }
        
        self._running = False
        self._tasks: Set[asyncio.Task] = set()
        
    async def start(self):
        """Start the API gateway"""
        if self._running:
            return
            
        self._running = True
        logger.info("Starting API Gateway")
        
        # Start background tasks
        health_check_task = asyncio.create_task(self._health_check_loop())
        stats_task = asyncio.create_task(self._stats_collection_loop())
        
        self._tasks.update([health_check_task, stats_task])
        
        # Register default routes for new features
        await self._register_default_routes()
        
        logger.info("API Gateway started successfully")
    
    async def stop(self):
        """Stop the API gateway"""
        if not self._running:
            return
            
        self._running = False
        logger.info("Stopping API Gateway")
        
        # Cancel all background tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._tasks.clear()
        logger.info("API Gateway stopped")
    
    def add_route(self, route: RouteRule):
        """Add a new route rule"""
        self.routes.append(route)
        logger.info(f"Added route: {route.path} -> {route.service_name}")
    
    def add_service(self, service_name: str, endpoints: List[ServiceEndpoint]):
        """Add service endpoints"""
        self.services[service_name] = endpoints
        logger.info(f"Added service {service_name} with {len(endpoints)} endpoints")
    
    def add_middleware(self, middleware: Callable):
        """Add middleware to the processing stack"""
        self.middleware_stack.append(middleware)
        logger.info(f"Added middleware: {middleware.__name__}")
    
    async def handle_request(
        self,
        method: str,
        path: str, 
        headers: Dict[str, str],
        query_params: Dict[str, List[str]] = None,
        body: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """Handle incoming request through the gateway"""
        request_id = str(uuid.uuid4())
        
        # Create request context
        context = RequestContext(
            request_id=request_id,
            method=method,
            path=path,
            headers=headers,
            query_params=query_params or {},
            body=body,
            client_ip=headers.get('x-forwarded-for', headers.get('remote-addr', '')),
            user_agent=headers.get('user-agent', '')
        )
        
        self.stats["total_requests"] += 1
        self.stats["active_connections"] += 1
        
        try:
            # Find matching route
            route_rule = self._find_matching_route(path, method)
            if not route_rule:
                return self._create_error_response(404, "Route not found", request_id)
            
            context.route_rule = route_rule
            
            # Apply middleware stack
            for middleware in self.middleware_stack:
                try:
                    result = await middleware(context)
                    if result is not None:  # Middleware can short-circuit
                        return result
                except Exception as e:
                    logger.error(f"Middleware {middleware.__name__} failed: {e}")
                    return self._create_error_response(500, "Middleware error", request_id)
            
            # Check authentication if required
            if route_rule.authentication_required:
                auth_result = await self._authenticate_request(context)
                if auth_result is not None:
                    return auth_result
            
            # Check rate limits
            rate_limit_result = await self._check_rate_limit(context)
            if rate_limit_result is not None:
                return rate_limit_result
            
            # Select target endpoint
            endpoint = await self._select_endpoint(route_rule.service_name, route_rule.load_balance_strategy)
            if not endpoint:
                return self._create_error_response(503, "Service unavailable", request_id)
            
            context.target_endpoint = endpoint
            
            # Check circuit breaker
            if route_rule.circuit_breaker:
                circuit_state = self._get_circuit_breaker_state(route_rule.service_name)
                if circuit_state.get('state') == 'open':
                    return self._create_error_response(503, "Circuit breaker open", request_id)
            
            # Forward request to service
            response = await self._forward_request(context)
            
            # Update circuit breaker on success
            if route_rule.circuit_breaker:
                self._update_circuit_breaker(route_rule.service_name, success=True)
            
            # Update statistics
            self.stats["successful_requests"] += 1
            self._update_response_time(context.elapsed_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Request {request_id} failed: {e}")
            
            # Update circuit breaker on failure
            if context.route_rule and context.route_rule.circuit_breaker:
                self._update_circuit_breaker(context.route_rule.service_name, success=False)
            
            self.stats["failed_requests"] += 1
            return self._create_error_response(500, "Internal server error", request_id)
        
        finally:
            self.stats["active_connections"] -= 1
    
    def _find_matching_route(self, path: str, method: str) -> Optional[RouteRule]:
        """Find the first matching route rule"""
        for route in self.routes:
            if route.matches(path, method):
                return route
        return None
    
    async def _authenticate_request(self, context: RequestContext) -> Optional[Dict[str, Any]]:
        """Authenticate request using JWT token"""
        auth_header = context.headers.get('authorization', '')
        
        if not auth_header.startswith('Bearer '):
            return self._create_error_response(401, "Missing or invalid authorization header", context.request_id)
        
        token = auth_header[7:]  # Remove 'Bearer ' prefix
        
        try:
            # Decode JWT token (in production, use proper secret key)
            payload = jwt.decode(token, options={"verify_signature": False})
            context.user_id = payload.get('user_id')
            context.tenant_id = payload.get('tenant_id')
            
            return None  # Authentication successful
            
        except jwt.InvalidTokenError as e:
            return self._create_error_response(401, f"Invalid token: {e}", context.request_id)
    
    async def _check_rate_limit(self, context: RequestContext) -> Optional[Dict[str, Any]]:
        """Check rate limiting for the request"""
        if not context.route_rule.rate_limit:
            return None
        
        # Use user_id or client_ip as rate limit key
        rate_key = context.user_id or context.client_ip
        if not rate_key:
            return None
        
        bucket_key = f"{context.route_rule.service_name}:{rate_key}"
        current_time = time.time()
        
        # Get or create rate limit bucket
        if bucket_key not in self._rate_limit_buckets:
            self._rate_limit_buckets[bucket_key] = {
                'tokens': context.route_rule.rate_limit.get('tokens', 100),
                'last_refill': current_time,
                'max_tokens': context.route_rule.rate_limit.get('tokens', 100),
                'refill_rate': context.route_rule.rate_limit.get('refill_rate', 1.0)  # tokens per second
            }
        
        bucket = self._rate_limit_buckets[bucket_key]
        
        # Refill tokens based on elapsed time
        time_elapsed = current_time - bucket['last_refill']
        tokens_to_add = int(time_elapsed * bucket['refill_rate'])
        bucket['tokens'] = min(bucket['max_tokens'], bucket['tokens'] + tokens_to_add)
        bucket['last_refill'] = current_time
        
        # Check if request can be processed
        if bucket['tokens'] >= 1:
            bucket['tokens'] -= 1
            return None  # Request allowed
        else:
            return self._create_error_response(429, "Rate limit exceeded", context.request_id)
    
    async def _select_endpoint(self, service_name: str, strategy: LoadBalanceStrategy) -> Optional[ServiceEndpoint]:
        """Select endpoint based on load balancing strategy"""
        endpoints = self.services.get(service_name, [])
        healthy_endpoints = [ep for ep in endpoints if ep.is_healthy]
        
        if not healthy_endpoints:
            return None
        
        if strategy == LoadBalanceStrategy.ROUND_ROBIN:
            return self._round_robin_select(service_name, healthy_endpoints)
        elif strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
            return min(healthy_endpoints, key=lambda x: x.current_connections)
        elif strategy == LoadBalanceStrategy.WEIGHTED:
            return self._weighted_select(healthy_endpoints)
        else:
            return healthy_endpoints[0]
    
    def _round_robin_select(self, service_name: str, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Round-robin endpoint selection"""
        if service_name not in self._round_robin_counters:
            self._round_robin_counters[service_name] = 0
        
        index = self._round_robin_counters[service_name] % len(endpoints)
        self._round_robin_counters[service_name] += 1
        
        return endpoints[index]
    
    def _weighted_select(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Weighted endpoint selection based on load scores"""
        # Invert load scores so lower load = higher probability
        weights = [1.0 / (ep.load_score + 0.1) for ep in endpoints]
        total_weight = sum(weights)
        
        import random
        r = random.uniform(0, total_weight)
        cumulative = 0
        
        for i, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return endpoints[i]
        
        return endpoints[-1]
    
    async def _forward_request(self, context: RequestContext) -> Dict[str, Any]:
        """Forward request to target service endpoint"""
        endpoint = context.target_endpoint
        route_rule = context.route_rule
        
        # Build target URL
        target_path = route_rule.target_path or context.path
        target_url = f"{endpoint.url}{target_path}"
        
        # Add query parameters
        if context.query_params:
            query_string = "&".join([
                f"{key}={value}"
                for key, values in context.query_params.items()
                for value in values
            ])
            target_url += f"?{query_string}"
        
        # Prepare headers
        headers = dict(context.headers)
        headers['x-request-id'] = context.request_id
        headers['x-forwarded-for'] = context.client_ip
        
        # Track connection
        endpoint.current_connections += 1
        
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                
                async with session.request(
                    method=context.method,
                    url=target_url,
                    headers=headers,
                    data=context.body,
                    timeout=aiohttp.ClientTimeout(total=route_rule.timeout)
                ) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    # Update endpoint metrics
                    endpoint.response_time = (endpoint.response_time * 0.9) + (response_time * 0.1)
                    
                    response_data = await response.read()
                    
                    return {
                        "status_code": response.status,
                        "headers": dict(response.headers),
                        "body": response_data,
                        "request_id": context.request_id,
                        "processing_time": context.elapsed_time,
                        "service_response_time": response_time
                    }
        
        except asyncio.TimeoutError:
            raise Exception("Request timeout")
        except Exception as e:
            raise Exception(f"Failed to forward request: {e}")
        
        finally:
            endpoint.current_connections -= 1
    
    def _get_circuit_breaker_state(self, service_name: str) -> Dict[str, Any]:
        """Get circuit breaker state for service"""
        if service_name not in self._circuit_breaker_state:
            self._circuit_breaker_state[service_name] = {
                'state': 'closed',  # closed, open, half-open
                'failure_count': 0,
                'last_failure_time': None,
                'success_count': 0
            }
        
        return self._circuit_breaker_state[service_name]
    
    def _update_circuit_breaker(self, service_name: str, success: bool):
        """Update circuit breaker state"""
        state = self._get_circuit_breaker_state(service_name)
        
        if success:
            state['success_count'] += 1
            state['failure_count'] = 0
            
            if state['state'] == 'half-open' and state['success_count'] >= 3:
                state['state'] = 'closed'
                state['success_count'] = 0
        else:
            state['failure_count'] += 1
            state['last_failure_time'] = time.time()
            
            if state['failure_count'] >= 5:  # Threshold
                state['state'] = 'open'
        
        # Check if circuit should move from open to half-open
        if (state['state'] == 'open' and 
            state['last_failure_time'] and
            time.time() - state['last_failure_time'] > 60):  # 1 minute timeout
            state['state'] = 'half-open'
            state['success_count'] = 0
    
    def _create_error_response(self, status_code: int, message: str, request_id: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "status_code": status_code,
            "headers": {"content-type": "application/json"},
            "body": json.dumps({
                "error": message,
                "request_id": request_id,
                "timestamp": datetime.now().isoformat()
            }).encode(),
            "request_id": request_id
        }
    
    def _update_response_time(self, response_time: float):
        """Update average response time statistic"""
        current_avg = self.stats["average_response_time"]
        # Exponential moving average
        self.stats["average_response_time"] = (current_avg * 0.9) + (response_time * 0.1)
    
    async def _health_check_loop(self):
        """Background task for health checking service endpoints"""
        while self._running:
            try:
                health_check_tasks = []
                
                for service_name, endpoints in self.services.items():
                    for endpoint in endpoints:
                        task = asyncio.create_task(self._check_endpoint_health(endpoint))
                        health_check_tasks.append(task)
                
                if health_check_tasks:
                    await asyncio.gather(*health_check_tasks, return_exceptions=True)
                
                await asyncio.sleep(30)  # Health check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(10)
    
    async def _check_endpoint_health(self, endpoint: ServiceEndpoint):
        """Check health of individual endpoint"""
        try:
            health_url = f"{endpoint.url}{endpoint.health_check_path}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(health_url, timeout=5) as response:
                    endpoint.is_healthy = response.status == 200
                    endpoint.last_health_check = datetime.now()
                    
                    if endpoint.is_healthy:
                        # Update success rate
                        endpoint.success_rate = min(1.0, endpoint.success_rate + 0.1)
                    else:
                        endpoint.success_rate = max(0.0, endpoint.success_rate - 0.1)
        
        except Exception:
            endpoint.is_healthy = False
            endpoint.last_health_check = datetime.now()
            endpoint.success_rate = max(0.0, endpoint.success_rate - 0.2)
    
    async def _stats_collection_loop(self):
        """Background task for collecting and logging statistics"""
        while self._running:
            try:
                stats_summary = {
                    "timestamp": datetime.now().isoformat(),
                    "gateway_stats": self.stats.copy(),
                    "service_health": {}
                }
                
                # Collect service health stats
                for service_name, endpoints in self.services.items():
                    healthy_count = sum(1 for ep in endpoints if ep.is_healthy)
                    total_count = len(endpoints)
                    avg_response_time = sum(ep.response_time for ep in endpoints) / total_count if endpoints else 0
                    
                    stats_summary["service_health"][service_name] = {
                        "healthy_endpoints": healthy_count,
                        "total_endpoints": total_count,
                        "health_percentage": (healthy_count / total_count * 100) if total_count > 0 else 0,
                        "average_response_time": avg_response_time
                    }
                
                logger.info(f"Gateway stats: {json.dumps(stats_summary, indent=2)}")
                
                await asyncio.sleep(300)  # Log stats every 5 minutes
                
            except Exception as e:
                logger.error(f"Stats collection error: {e}")
                await asyncio.sleep(60)
    
    async def _register_default_routes(self):
        """Register default routes for new features"""
        # Voice interface routes
        self.add_route(RouteRule(
            path="/api/v1/voice",
            methods=[RouteMethod.POST, RouteMethod.GET],
            service_name="voice-interface",
            rate_limit={"tokens": 10, "refill_rate": 0.1},  # 10 requests, refill 0.1/sec
            timeout=45.0  # Voice processing can take longer
        ))
        
        # Document comparison routes
        self.add_route(RouteRule(
            path="/api/v1/documents/compare",
            methods=[RouteMethod.POST],
            service_name="document-comparison",
            rate_limit={"tokens": 5, "refill_rate": 0.05},  # More intensive operation
            timeout=60.0
        ))
        
        # White-label routes
        self.add_route(RouteRule(
            path="/api/v1/tenants",
            methods=[RouteMethod.GET, RouteMethod.POST, RouteMethod.PUT, RouteMethod.DELETE],
            service_name="whitelabel-service",
            load_balance_strategy=LoadBalanceStrategy.WEIGHTED
        ))
        
        # Compliance automation routes
        self.add_route(RouteRule(
            path="/api/v1/compliance",
            methods=[RouteMethod.POST, RouteMethod.GET],
            service_name="compliance-automation",
            timeout=30.0
        ))
        
        # Visualization routes
        self.add_route(RouteRule(
            path="/api/v1/visualizations",
            methods=[RouteMethod.POST, RouteMethod.GET],
            service_name="visualization-engine",
            rate_limit={"tokens": 20, "refill_rate": 0.2},
            timeout=30.0
        ))
        
        logger.info("Registered default routes for new features")
    
    def get_gateway_status(self) -> Dict[str, Any]:
        """Get comprehensive gateway status"""
        return {
            "gateway_stats": self.stats,
            "registered_routes": len(self.routes),
            "registered_services": len(self.services),
            "total_endpoints": sum(len(endpoints) for endpoints in self.services.values()),
            "healthy_endpoints": sum(
                len([ep for ep in endpoints if ep.is_healthy]) 
                for endpoints in self.services.values()
            ),
            "circuit_breaker_states": {
                service: state["state"] 
                for service, state in self._circuit_breaker_state.items()
            },
            "middleware_count": len(self.middleware_stack)
        }