"""
Rate Limiting Middleware

Advanced rate limiting with:
- Multiple algorithms (Token Bucket, Sliding Window)
- User-based and IP-based limits
- Dynamic rate limiting based on user tier
- Distributed rate limiting with Redis
- Rate limiting bypass for privileged users
"""

import time
import math
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from starlette.status import HTTP_429_TOO_MANY_REQUESTS

import redis
from ..core.config import get_settings
from ..auth.permissions import Role


class RateLimitAlgorithm(Enum):
    """Rate limiting algorithms"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimit:
    """Rate limit configuration"""
    max_requests: int
    window_seconds: int
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.SLIDING_WINDOW
    burst_limit: Optional[int] = None  # For token bucket
    leak_rate: Optional[float] = None  # For leaky bucket


@dataclass
class RateLimitResult:
    """Rate limit check result"""
    allowed: bool
    requests_remaining: int
    reset_time: int
    retry_after: Optional[int] = None
    current_requests: int = 0


class RateLimitingMiddleware(BaseHTTPMiddleware):
    """
    Advanced rate limiting middleware with multiple algorithms and strategies
    """
    
    def __init__(self, app, redis_client: Optional[redis.Redis] = None):
        super().__init__(app)
        self.settings = get_settings()
        self.redis_client = redis_client or redis.Redis(
            host=self.settings.REDIS_HOST,
            port=self.settings.REDIS_PORT,
            password=self.settings.REDIS_PASSWORD,
            decode_responses=True
        )
        
        # Default rate limits by user role/tier
        self.role_limits = {
            Role.SUPER_ADMIN: RateLimit(10000, 3600),  # 10k/hour
            Role.ADMIN: RateLimit(5000, 3600),         # 5k/hour
            Role.MANAGER: RateLimit(2000, 3600),       # 2k/hour
            Role.ANALYST: RateLimit(1000, 3600),       # 1k/hour
            Role.CLIENT_ADMIN: RateLimit(1500, 3600),  # 1.5k/hour
            Role.CLIENT_USER: RateLimit(500, 3600),    # 500/hour
            Role.API_USER: RateLimit(10000, 3600),     # 10k/hour for API users
            Role.DEMO_USER: RateLimit(100, 3600),      # 100/hour for demo
            Role.GUEST: RateLimit(50, 3600)            # 50/hour for guests
        }
        
        # IP-based rate limits (fallback)
        self.default_ip_limit = RateLimit(1000, 3600)  # 1k/hour per IP
        
        # Endpoint-specific limits
        self.endpoint_limits = {
            "/auth/login": RateLimit(10, 900),         # 10/15min
            "/auth/register": RateLimit(5, 3600),      # 5/hour
            "/auth/forgot-password": RateLimit(5, 3600), # 5/hour
            "/auth/reset-password": RateLimit(5, 3600),  # 5/hour
            "/api/documents/upload": RateLimit(100, 3600), # 100/hour
            "/api/analysis/bulk": RateLimit(10, 3600)    # 10/hour for bulk operations
        }
        
        # Routes exempt from rate limiting
        self.exempt_routes = {
            "/health", "/metrics", "/docs", "/redoc", "/openapi.json"
        }
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP with proxy support"""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "127.0.0.1"
    
    def _get_rate_limit_key(self, request: Request, user_data: Optional[Dict[str, Any]] = None) -> str:
        """Generate rate limit key based on user or IP"""
        if user_data and user_data.get("user_id"):
            return f"rate_limit:user:{user_data['user_id']}"
        else:
            client_ip = self._get_client_ip(request)
            return f"rate_limit:ip:{client_ip}"
    
    def _get_endpoint_key(self, request: Request, user_data: Optional[Dict[str, Any]] = None) -> str:
        """Generate endpoint-specific rate limit key"""
        base_key = self._get_rate_limit_key(request, user_data)
        path = request.url.path
        return f"{base_key}:endpoint:{path}"
    
    def _get_rate_limit_config(self, request: Request, user_data: Optional[Dict[str, Any]] = None) -> RateLimit:
        """Get rate limit configuration for the request"""
        path = request.url.path
        
        # Check endpoint-specific limits first
        if path in self.endpoint_limits:
            return self.endpoint_limits[path]
        
        # Check user role-based limits
        if user_data and user_data.get("roles"):
            user_roles = user_data["roles"]
            # Use the highest privilege role's limit
            for role in [Role.SUPER_ADMIN, Role.ADMIN, Role.MANAGER, Role.ANALYST, 
                        Role.CLIENT_ADMIN, Role.CLIENT_USER, Role.API_USER]:
                if role.value in user_roles:
                    return self.role_limits[role]
        
        # Default IP-based limit
        return self.default_ip_limit
    
    def _sliding_window_check(self, key: str, limit: RateLimit) -> RateLimitResult:
        """Sliding window rate limit algorithm"""
        now = time.time()
        window_start = now - limit.window_seconds
        
        try:
            pipe = self.redis_client.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current requests
            pipe.zcard(key)
            
            # Add current request with timestamp
            request_id = f"{now}:{hashlib.md5(str(now).encode()).hexdigest()[:8]}"
            pipe.zadd(key, {request_id: now})
            
            # Set expiry
            pipe.expire(key, limit.window_seconds + 10)
            
            results = pipe.execute()
            current_requests = results[1] + 1  # +1 for current request
            
            # Check limit
            allowed = current_requests <= limit.max_requests
            requests_remaining = max(0, limit.max_requests - current_requests)
            reset_time = int(now + limit.window_seconds)
            
            if not allowed:
                # Remove the request we just added
                self.redis_client.zrem(key, request_id)
                retry_after = int(limit.window_seconds)
            else:
                retry_after = None
            
            return RateLimitResult(
                allowed=allowed,
                requests_remaining=requests_remaining,
                reset_time=reset_time,
                retry_after=retry_after,
                current_requests=current_requests
            )
        
        except redis.RedisError:
            # If Redis is unavailable, allow request (fail open)
            return RateLimitResult(
                allowed=True,
                requests_remaining=limit.max_requests,
                reset_time=int(now + limit.window_seconds)
            )
    
    def _fixed_window_check(self, key: str, limit: RateLimit) -> RateLimitResult:
        """Fixed window rate limit algorithm"""
        now = time.time()
        window_key = f"{key}:{int(now // limit.window_seconds)}"
        
        try:
            pipe = self.redis_client.pipeline()
            
            # Get current count
            pipe.get(window_key)
            
            # Increment
            pipe.incr(window_key)
            
            # Set expiry on first increment
            pipe.expire(window_key, limit.window_seconds)
            
            results = pipe.execute()
            current_requests = int(results[0] or 0) + 1
            
            # Check limit
            allowed = current_requests <= limit.max_requests
            requests_remaining = max(0, limit.max_requests - current_requests)
            
            # Calculate reset time (end of current window)
            window_number = int(now // limit.window_seconds)
            reset_time = int((window_number + 1) * limit.window_seconds)
            
            if not allowed:
                # Decrement since we're not allowing this request
                self.redis_client.decr(window_key)
                retry_after = reset_time - int(now)
            else:
                retry_after = None
            
            return RateLimitResult(
                allowed=allowed,
                requests_remaining=requests_remaining,
                reset_time=reset_time,
                retry_after=retry_after,
                current_requests=current_requests
            )
        
        except redis.RedisError:
            # If Redis is unavailable, allow request (fail open)
            return RateLimitResult(
                allowed=True,
                requests_remaining=limit.max_requests,
                reset_time=int(now + limit.window_seconds)
            )
    
    def _token_bucket_check(self, key: str, limit: RateLimit) -> RateLimitResult:
        """Token bucket rate limit algorithm"""
        now = time.time()
        bucket_key = f"{key}:bucket"
        
        # Token bucket parameters
        capacity = limit.burst_limit or limit.max_requests
        refill_rate = limit.max_requests / limit.window_seconds  # tokens per second
        
        try:
            # Get bucket state
            bucket_data = self.redis_client.hmget(bucket_key, ["tokens", "last_refill"])
            current_tokens = float(bucket_data[0] or capacity)
            last_refill = float(bucket_data[1] or now)
            
            # Calculate tokens to add based on time elapsed
            time_elapsed = now - last_refill
            tokens_to_add = time_elapsed * refill_rate
            current_tokens = min(capacity, current_tokens + tokens_to_add)
            
            # Check if we can consume a token
            if current_tokens >= 1:
                # Consume token
                current_tokens -= 1
                allowed = True
                retry_after = None
            else:
                allowed = False
                # Calculate when next token will be available
                retry_after = int(math.ceil((1 - current_tokens) / refill_rate))
            
            # Update bucket state
            self.redis_client.hmset(bucket_key, {
                "tokens": current_tokens,
                "last_refill": now
            })
            self.redis_client.expire(bucket_key, limit.window_seconds * 2)
            
            return RateLimitResult(
                allowed=allowed,
                requests_remaining=int(current_tokens),
                reset_time=int(now + limit.window_seconds),
                retry_after=retry_after,
                current_requests=capacity - int(current_tokens)
            )
        
        except redis.RedisError:
            # If Redis is unavailable, allow request (fail open)
            return RateLimitResult(
                allowed=True,
                requests_remaining=capacity,
                reset_time=int(now + limit.window_seconds)
            )
    
    def _check_rate_limit(self, key: str, limit: RateLimit) -> RateLimitResult:
        """Check rate limit using specified algorithm"""
        if limit.algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
            return self._sliding_window_check(key, limit)
        elif limit.algorithm == RateLimitAlgorithm.FIXED_WINDOW:
            return self._fixed_window_check(key, limit)
        elif limit.algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
            return self._token_bucket_check(key, limit)
        else:
            # Default to sliding window
            return self._sliding_window_check(key, limit)
    
    def _is_exempt_route(self, path: str) -> bool:
        """Check if route is exempt from rate limiting"""
        if path in self.exempt_routes:
            return True
        
        # Check prefixes
        exempt_prefixes = ["/health", "/metrics", "/static"]
        for prefix in exempt_prefixes:
            if path.startswith(prefix):
                return True
        
        return False
    
    def _has_rate_limit_bypass(self, user_data: Optional[Dict[str, Any]]) -> bool:
        """Check if user has rate limit bypass permission"""
        if not user_data:
            return False
        
        permissions = user_data.get("permissions", [])
        return "api:rate_limit_override" in permissions
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Main middleware dispatch method"""
        
        path = request.url.path
        
        # Skip exempt routes
        if self._is_exempt_route(path):
            return await call_next(request)
        
        # Get user data if available
        user_data = getattr(request.state, "current_user", None)
        
        # Check for rate limit bypass
        if self._has_rate_limit_bypass(user_data):
            return await call_next(request)
        
        # Get rate limit configuration
        rate_limit = self._get_rate_limit_config(request, user_data)
        
        # Generate rate limit key
        global_key = self._get_rate_limit_key(request, user_data)
        
        # Check global rate limit
        global_result = self._check_rate_limit(global_key, rate_limit)
        
        if not global_result.allowed:
            return self._create_rate_limit_response(global_result, "global")
        
        # Check endpoint-specific rate limit if exists
        if path in self.endpoint_limits:
            endpoint_key = self._get_endpoint_key(request, user_data)
            endpoint_limit = self.endpoint_limits[path]
            endpoint_result = self._check_rate_limit(endpoint_key, endpoint_limit)
            
            if not endpoint_result.allowed:
                return self._create_rate_limit_response(endpoint_result, "endpoint")
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(rate_limit.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(global_result.requests_remaining)
        response.headers["X-RateLimit-Reset"] = str(global_result.reset_time)
        response.headers["X-RateLimit-Window"] = str(rate_limit.window_seconds)
        
        return response
    
    def _create_rate_limit_response(self, result: RateLimitResult, limit_type: str) -> JSONResponse:
        """Create rate limit exceeded response"""
        
        headers = {
            "X-RateLimit-Limit": str(result.current_requests + result.requests_remaining),
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(result.reset_time),
            "Retry-After": str(result.retry_after) if result.retry_after else str(60)
        }
        
        content = {
            "detail": f"Rate limit exceeded for {limit_type}",
            "error_code": "RATE_LIMIT_EXCEEDED",
            "limit_type": limit_type,
            "retry_after": result.retry_after,
            "reset_time": result.reset_time
        }
        
        return JSONResponse(
            status_code=HTTP_429_TOO_MANY_REQUESTS,
            content=content,
            headers=headers
        )


class AdaptiveRateLimitingMiddleware(RateLimitingMiddleware):
    """
    Adaptive rate limiting that adjusts limits based on system load and user behavior
    """
    
    def __init__(self, app, redis_client: Optional[redis.Redis] = None):
        super().__init__(app, redis_client)
        self.base_limits = self.role_limits.copy()
        self.adaptation_factor = 1.0
        self.last_adaptation_check = time.time()
        self.adaptation_interval = 300  # 5 minutes
    
    def _get_system_load(self) -> float:
        """Get current system load (placeholder - implement based on your metrics)"""
        try:
            # Example: Get load from Redis metrics or system monitoring
            load_key = "system:load:current"
            load = self.redis_client.get(load_key)
            return float(load or 0.5)  # Default to medium load
        except Exception:
            return 0.5  # Default load
    
    def _adapt_rate_limits(self):
        """Adapt rate limits based on system conditions"""
        now = time.time()
        
        if now - self.last_adaptation_check < self.adaptation_interval:
            return
        
        system_load = self._get_system_load()
        
        # Adjust limits based on load
        if system_load > 0.8:  # High load
            self.adaptation_factor = 0.5  # Reduce limits by 50%
        elif system_load > 0.6:  # Medium-high load
            self.adaptation_factor = 0.75  # Reduce limits by 25%
        elif system_load < 0.3:  # Low load
            self.adaptation_factor = 1.5  # Increase limits by 50%
        else:  # Normal load
            self.adaptation_factor = 1.0
        
        # Apply adaptation factor to role limits
        for role, base_limit in self.base_limits.items():
            adapted_max = int(base_limit.max_requests * self.adaptation_factor)
            self.role_limits[role] = RateLimit(
                adapted_max,
                base_limit.window_seconds,
                base_limit.algorithm,
                base_limit.burst_limit,
                base_limit.leak_rate
            )
        
        self.last_adaptation_check = now
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Dispatch with adaptive rate limiting"""
        
        # Adapt rate limits based on current conditions
        self._adapt_rate_limits()
        
        # Use parent dispatch method
        return await super().dispatch(request, call_next)