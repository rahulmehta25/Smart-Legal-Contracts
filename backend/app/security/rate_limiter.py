"""
Advanced Rate Limiting and DDoS Protection System
Implements multi-layer rate limiting with adaptive throttling
OWASP Rate Limiting Cheat Sheet compliant
"""

import time
import hashlib
import json
import math
from typing import Optional, Dict, Any, List, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import threading
import ipaddress
import asyncio
from functools import wraps

import redis
from fastapi import HTTPException, status, Request, Response
from fastapi.responses import JSONResponse
import numpy as np


class RateLimitType(str, Enum):
    """Types of rate limiting strategies"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"
    ADAPTIVE = "adaptive"
    DISTRIBUTED = "distributed"


class RateLimitScope(str, Enum):
    """Scopes for rate limiting"""
    GLOBAL = "global"           # System-wide
    IP = "ip"                   # Per IP address
    USER = "user"               # Per authenticated user
    API_KEY = "api_key"         # Per API key
    ENDPOINT = "endpoint"       # Per endpoint
    COMBINED = "combined"       # Multiple factors


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    requests: int              # Number of requests allowed
    window: int                # Time window in seconds
    burst: Optional[int] = None  # Burst allowance
    strategy: RateLimitType = RateLimitType.SLIDING_WINDOW
    scope: RateLimitScope = RateLimitScope.IP
    penalty_duration: int = 300  # Penalty duration in seconds when limit exceeded
    adaptive: bool = False      # Enable adaptive throttling


@dataclass
class RateLimitStatus:
    """Current rate limit status"""
    allowed: bool
    limit: int
    remaining: int
    reset_at: datetime
    retry_after: Optional[int] = None
    identifier: str = None
    metadata: Dict[str, Any] = None


class RateLimiter:
    """
    Advanced rate limiting system with:
    - Multiple rate limiting algorithms
    - IP-based and user-based limits
    - Adaptive throttling based on system load
    - DDoS protection mechanisms
    - Distributed rate limiting support
    """
    
    def __init__(self,
                 redis_client: Optional[redis.Redis] = None,
                 default_config: Optional[RateLimitConfig] = None):
        
        # Redis for distributed rate limiting
        self.redis_client = redis_client or redis.Redis(
            host='localhost',
            port=6379,
            decode_responses=True
        )
        
        # Default configuration
        self.default_config = default_config or RateLimitConfig(
            requests=100,
            window=60,
            burst=10,
            strategy=RateLimitType.SLIDING_WINDOW
        )
        
        # Endpoint-specific configurations
        self.endpoint_configs = {}
        
        # In-memory cache for performance
        self.cache = {}
        self.cache_lock = threading.Lock()
        
        # Metrics for adaptive throttling
        self.metrics = {
            "total_requests": 0,
            "blocked_requests": 0,
            "response_times": deque(maxlen=1000),
            "cpu_usage": deque(maxlen=100),
            "memory_usage": deque(maxlen=100)
        }
        
        # IP reputation tracking
        self.ip_reputation = defaultdict(lambda: {"score": 100, "violations": 0})
        
        # DDoS protection
        self.ddos_protection = DDoSProtection()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitor_system)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
    
    # ========== Core Rate Limiting ==========
    
    def check_rate_limit(self,
                        identifier: str,
                        config: Optional[RateLimitConfig] = None,
                        endpoint: Optional[str] = None) -> RateLimitStatus:
        """
        Check if request is within rate limits
        """
        
        config = config or self.endpoint_configs.get(endpoint, self.default_config)
        
        # Check DDoS protection first
        if self.ddos_protection.is_under_attack():
            config = self._apply_ddos_config(config)
        
        # Apply adaptive throttling if enabled
        if config.adaptive:
            config = self._apply_adaptive_config(config)
        
        # Check rate limit based on strategy
        if config.strategy == RateLimitType.FIXED_WINDOW:
            return self._fixed_window_limit(identifier, config)
        elif config.strategy == RateLimitType.SLIDING_WINDOW:
            return self._sliding_window_limit(identifier, config)
        elif config.strategy == RateLimitType.TOKEN_BUCKET:
            return self._token_bucket_limit(identifier, config)
        elif config.strategy == RateLimitType.LEAKY_BUCKET:
            return self._leaky_bucket_limit(identifier, config)
        elif config.strategy == RateLimitType.ADAPTIVE:
            return self._adaptive_limit(identifier, config)
        else:
            return self._sliding_window_limit(identifier, config)
    
    # ========== Rate Limiting Algorithms ==========
    
    def _fixed_window_limit(self, identifier: str, config: RateLimitConfig) -> RateLimitStatus:
        """
        Fixed window rate limiting
        Simple but can have thundering herd problem
        """
        
        key = f"rate_limit:fixed:{identifier}"
        window_start = int(time.time() / config.window) * config.window
        window_key = f"{key}:{window_start}"
        
        # Increment counter
        count = self.redis_client.incr(window_key)
        
        # Set expiry on first request
        if count == 1:
            self.redis_client.expire(window_key, config.window)
        
        # Check limit
        allowed = count <= config.requests
        remaining = max(0, config.requests - count)
        reset_at = datetime.fromtimestamp(window_start + config.window)
        
        if not allowed:
            retry_after = int(reset_at.timestamp() - time.time())
            self._record_violation(identifier)
        else:
            retry_after = None
        
        return RateLimitStatus(
            allowed=allowed,
            limit=config.requests,
            remaining=remaining,
            reset_at=reset_at,
            retry_after=retry_after,
            identifier=identifier
        )
    
    def _sliding_window_limit(self, identifier: str, config: RateLimitConfig) -> RateLimitStatus:
        """
        Sliding window rate limiting using sorted sets
        More accurate than fixed window
        """
        
        key = f"rate_limit:sliding:{identifier}"
        now = time.time()
        window_start = now - config.window
        
        # Remove old entries
        self.redis_client.zremrangebyscore(key, 0, window_start)
        
        # Count requests in window
        count = self.redis_client.zcard(key)
        
        # Check limit
        allowed = count < config.requests
        
        if allowed:
            # Add current request
            self.redis_client.zadd(key, {str(now): now})
            self.redis_client.expire(key, config.window)
            remaining = config.requests - count - 1
            retry_after = None
        else:
            remaining = 0
            # Calculate retry after based on oldest request
            oldest = self.redis_client.zrange(key, 0, 0, withscores=True)
            if oldest:
                retry_after = int(oldest[0][1] + config.window - now)
            else:
                retry_after = config.window
            self._record_violation(identifier)
        
        reset_at = datetime.fromtimestamp(now + config.window)
        
        return RateLimitStatus(
            allowed=allowed,
            limit=config.requests,
            remaining=remaining,
            reset_at=reset_at,
            retry_after=retry_after,
            identifier=identifier
        )
    
    def _token_bucket_limit(self, identifier: str, config: RateLimitConfig) -> RateLimitStatus:
        """
        Token bucket algorithm
        Allows burst traffic while maintaining average rate
        """
        
        key = f"rate_limit:token:{identifier}"
        now = time.time()
        
        # Get bucket state
        bucket_data = self.redis_client.hgetall(key)
        
        if bucket_data:
            tokens = float(bucket_data.get('tokens', config.requests))
            last_refill = float(bucket_data.get('last_refill', now))
        else:
            tokens = config.requests
            last_refill = now
        
        # Calculate tokens to add
        time_passed = now - last_refill
        refill_rate = config.requests / config.window
        tokens_to_add = time_passed * refill_rate
        
        # Update tokens (with burst limit)
        max_tokens = config.burst or config.requests
        tokens = min(tokens + tokens_to_add, max_tokens)
        
        # Check if request allowed
        allowed = tokens >= 1
        
        if allowed:
            tokens -= 1
            remaining = int(tokens)
            retry_after = None
        else:
            remaining = 0
            retry_after = int((1 - tokens) / refill_rate)
            self._record_violation(identifier)
        
        # Update bucket state
        self.redis_client.hset(key, mapping={
            'tokens': tokens,
            'last_refill': now
        })
        self.redis_client.expire(key, config.window * 2)
        
        reset_at = datetime.fromtimestamp(now + config.window)
        
        return RateLimitStatus(
            allowed=allowed,
            limit=config.requests,
            remaining=remaining,
            reset_at=reset_at,
            retry_after=retry_after,
            identifier=identifier
        )
    
    def _leaky_bucket_limit(self, identifier: str, config: RateLimitConfig) -> RateLimitStatus:
        """
        Leaky bucket algorithm
        Smooths out traffic by processing at constant rate
        """
        
        key = f"rate_limit:leaky:{identifier}"
        now = time.time()
        
        # Get bucket state
        bucket_data = self.redis_client.hgetall(key)
        
        if bucket_data:
            volume = float(bucket_data.get('volume', 0))
            last_leak = float(bucket_data.get('last_leak', now))
        else:
            volume = 0
            last_leak = now
        
        # Calculate leak
        leak_rate = config.requests / config.window
        time_passed = now - last_leak
        leaked = time_passed * leak_rate
        
        # Update volume
        volume = max(0, volume - leaked)
        
        # Check if request fits
        bucket_capacity = config.burst or config.requests
        allowed = volume < bucket_capacity
        
        if allowed:
            volume += 1
            remaining = int(bucket_capacity - volume)
            retry_after = None
        else:
            remaining = 0
            retry_after = int((volume - bucket_capacity + 1) / leak_rate)
            self._record_violation(identifier)
        
        # Update bucket state
        self.redis_client.hset(key, mapping={
            'volume': volume,
            'last_leak': now
        })
        self.redis_client.expire(key, config.window * 2)
        
        reset_at = datetime.fromtimestamp(now + (volume / leak_rate))
        
        return RateLimitStatus(
            allowed=allowed,
            limit=config.requests,
            remaining=remaining,
            reset_at=reset_at,
            retry_after=retry_after,
            identifier=identifier
        )
    
    def _adaptive_limit(self, identifier: str, config: RateLimitConfig) -> RateLimitStatus:
        """
        Adaptive rate limiting based on system load
        """
        
        # Get current system load
        load_factor = self._calculate_load_factor()
        
        # Adjust limits based on load
        adjusted_config = RateLimitConfig(
            requests=int(config.requests * (1 - load_factor * 0.5)),
            window=config.window,
            burst=config.burst,
            strategy=RateLimitType.SLIDING_WINDOW,
            scope=config.scope
        )
        
        # Use sliding window with adjusted limits
        status = self._sliding_window_limit(identifier, adjusted_config)
        
        # Add load information to metadata
        status.metadata = {
            "load_factor": load_factor,
            "original_limit": config.requests,
            "adjusted_limit": adjusted_config.requests
        }
        
        return status
    
    # ========== IP Reputation Management ==========
    
    def _record_violation(self, identifier: str):
        """Record rate limit violation"""
        
        self.metrics["blocked_requests"] += 1
        
        # Update IP reputation if identifier is IP
        if self._is_ip(identifier):
            self.ip_reputation[identifier]["violations"] += 1
            self.ip_reputation[identifier]["score"] = max(
                0,
                self.ip_reputation[identifier]["score"] - 10
            )
            
            # Block if reputation too low
            if self.ip_reputation[identifier]["score"] < 20:
                self._block_ip(identifier)
    
    def _is_ip(self, identifier: str) -> bool:
        """Check if identifier is an IP address"""
        try:
            ipaddress.ip_address(identifier)
            return True
        except:
            return False
    
    def _block_ip(self, ip: str, duration: int = 3600):
        """Block IP address"""
        
        key = f"blocked_ip:{ip}"
        self.redis_client.setex(key, duration, "1")
    
    def is_ip_blocked(self, ip: str) -> bool:
        """Check if IP is blocked"""
        
        return self.redis_client.exists(f"blocked_ip:{ip}") > 0
    
    def get_ip_reputation(self, ip: str) -> Dict[str, Any]:
        """Get IP reputation score"""
        
        return dict(self.ip_reputation[ip])
    
    # ========== Adaptive Throttling ==========
    
    def _calculate_load_factor(self) -> float:
        """Calculate system load factor for adaptive throttling"""
        
        # Calculate based on response times
        if self.metrics["response_times"]:
            avg_response_time = np.mean(self.metrics["response_times"])
            response_factor = min(1.0, avg_response_time / 1000)  # 1 second threshold
        else:
            response_factor = 0
        
        # Calculate based on blocked requests ratio
        if self.metrics["total_requests"] > 0:
            block_ratio = self.metrics["blocked_requests"] / self.metrics["total_requests"]
        else:
            block_ratio = 0
        
        # Calculate based on CPU/memory (if available)
        if self.metrics["cpu_usage"]:
            cpu_factor = np.mean(self.metrics["cpu_usage"]) / 100
        else:
            cpu_factor = 0
        
        # Weighted average
        load_factor = (response_factor * 0.4 + block_ratio * 0.3 + cpu_factor * 0.3)
        
        return min(1.0, load_factor)
    
    def _apply_adaptive_config(self, config: RateLimitConfig) -> RateLimitConfig:
        """Apply adaptive adjustments to configuration"""
        
        load_factor = self._calculate_load_factor()
        
        # Reduce limits under high load
        if load_factor > 0.7:
            adjustment = 1 - (load_factor - 0.7) * 2  # Up to 60% reduction
            config.requests = int(config.requests * adjustment)
        
        return config
    
    def _apply_ddos_config(self, config: RateLimitConfig) -> RateLimitConfig:
        """Apply DDoS protection configuration"""
        
        # Significantly reduce limits during attack
        config.requests = max(1, config.requests // 10)
        config.window = config.window * 2  # Increase window
        config.burst = 1  # No burst allowed
        
        return config
    
    # ========== System Monitoring ==========
    
    def _monitor_system(self):
        """Background thread for system monitoring"""
        
        while True:
            try:
                # Collect CPU usage (platform-specific)
                import psutil
                cpu_percent = psutil.cpu_percent(interval=1)
                self.metrics["cpu_usage"].append(cpu_percent)
                
                # Collect memory usage
                memory_percent = psutil.virtual_memory().percent
                self.metrics["memory_usage"].append(memory_percent)
                
                # Clean old data
                self._cleanup_old_data()
                
                time.sleep(10)
            except:
                time.sleep(10)
    
    def _cleanup_old_data(self):
        """Clean up old rate limiting data"""
        
        # Clean old IP reputation data
        cutoff = time.time() - 86400  # 24 hours
        for ip in list(self.ip_reputation.keys()):
            if self.ip_reputation[ip].get("last_seen", 0) < cutoff:
                del self.ip_reputation[ip]
    
    def record_request(self, response_time: float):
        """Record request metrics"""
        
        self.metrics["total_requests"] += 1
        self.metrics["response_times"].append(response_time)
    
    # ========== Configuration Management ==========
    
    def configure_endpoint(self, endpoint: str, config: RateLimitConfig):
        """Configure rate limiting for specific endpoint"""
        
        self.endpoint_configs[endpoint] = config
    
    def get_configuration(self, endpoint: Optional[str] = None) -> RateLimitConfig:
        """Get rate limit configuration"""
        
        if endpoint and endpoint in self.endpoint_configs:
            return self.endpoint_configs[endpoint]
        return self.default_config


class DDoSProtection:
    """
    DDoS protection mechanisms
    """
    
    def __init__(self):
        self.attack_detected = False
        self.attack_start = None
        self.request_counts = defaultdict(int)
        self.syn_flood_detector = SynFloodDetector()
        
    def is_under_attack(self) -> bool:
        """Check if system is under DDoS attack"""
        
        # Multiple detection methods
        if self._detect_request_flood():
            self.attack_detected = True
            self.attack_start = datetime.utcnow()
        elif self._detect_slowloris():
            self.attack_detected = True
            self.attack_start = datetime.utcnow()
        elif self.syn_flood_detector.detect():
            self.attack_detected = True
            self.attack_start = datetime.utcnow()
        
        # Auto-clear after 5 minutes
        if self.attack_detected and self.attack_start:
            if (datetime.utcnow() - self.attack_start).seconds > 300:
                self.attack_detected = False
                self.attack_start = None
        
        return self.attack_detected
    
    def _detect_request_flood(self) -> bool:
        """Detect request flood attack"""
        
        # Check request rate in last minute
        total_requests = sum(self.request_counts.values())
        return total_requests > 10000  # Threshold
    
    def _detect_slowloris(self) -> bool:
        """Detect slowloris attack"""
        
        # Would check for slow/incomplete requests
        # Implementation depends on web server
        return False
    
    def record_request(self, ip: str):
        """Record request for DDoS detection"""
        
        current_minute = int(time.time() / 60)
        self.request_counts[current_minute] += 1
        
        # Clean old data
        cutoff = current_minute - 5
        for minute in list(self.request_counts.keys()):
            if minute < cutoff:
                del self.request_counts[minute]


class SynFloodDetector:
    """Detect SYN flood attacks"""
    
    def __init__(self):
        self.syn_counts = defaultdict(int)
        self.threshold = 100
    
    def detect(self) -> bool:
        """Detect SYN flood"""
        
        # Would monitor TCP SYN packets
        # Requires system-level access
        return False


# ========== FastAPI Integration ==========

class RateLimitMiddleware:
    """FastAPI middleware for rate limiting"""
    
    def __init__(self, app, rate_limiter: RateLimiter):
        self.app = app
        self.rate_limiter = rate_limiter
    
    async def __call__(self, request: Request, call_next):
        # Get identifier (IP or user)
        identifier = self._get_identifier(request)
        
        # Check if IP is blocked
        if self.rate_limiter.is_ip_blocked(identifier):
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": "IP blocked due to violations"}
            )
        
        # Check rate limit
        endpoint = request.url.path
        status = self.rate_limiter.check_rate_limit(identifier, endpoint=endpoint)
        
        if not status.allowed:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded",
                    "retry_after": status.retry_after
                },
                headers={
                    "X-RateLimit-Limit": str(status.limit),
                    "X-RateLimit-Remaining": str(status.remaining),
                    "X-RateLimit-Reset": status.reset_at.isoformat(),
                    "Retry-After": str(status.retry_after)
                }
            )
        
        # Process request
        start_time = time.time()
        response = await call_next(request)
        response_time = (time.time() - start_time) * 1000
        
        # Record metrics
        self.rate_limiter.record_request(response_time)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(status.limit)
        response.headers["X-RateLimit-Remaining"] = str(status.remaining)
        response.headers["X-RateLimit-Reset"] = status.reset_at.isoformat()
        
        return response
    
    def _get_identifier(self, request: Request) -> str:
        """Get identifier for rate limiting"""
        
        # Check for authenticated user
        if hasattr(request.state, "user") and request.state.user:
            return f"user:{request.state.user.id}"
        
        # Check for API key
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"api_key:{api_key}"
        
        # Use IP address
        client_ip = request.client.host
        
        # Check for proxy headers
        if "X-Forwarded-For" in request.headers:
            client_ip = request.headers["X-Forwarded-For"].split(",")[0].strip()
        elif "X-Real-IP" in request.headers:
            client_ip = request.headers["X-Real-IP"]
        
        return client_ip


def rate_limit(
    requests: int = 100,
    window: int = 60,
    strategy: RateLimitType = RateLimitType.SLIDING_WINDOW
):
    """
    Decorator for endpoint-specific rate limiting
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            # Get rate limiter
            rate_limiter = request.app.state.rate_limiter
            
            # Create config
            config = RateLimitConfig(
                requests=requests,
                window=window,
                strategy=strategy
            )
            
            # Get identifier
            identifier = request.client.host
            
            # Check rate limit
            status = rate_limiter.check_rate_limit(identifier, config)
            
            if not status.allowed:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                    headers={"Retry-After": str(status.retry_after)}
                )
            
            return await func(request, *args, **kwargs)
        
        return wrapper
    return decorator