"""
Rate Limiting for GraphQL API
"""

import time
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum


class RateLimitError(Exception):
    """Exception raised when rate limit is exceeded"""
    
    def __init__(self, message: str, retry_after: int):
        super().__init__(message)
        self.retry_after = retry_after


class RateLimitStrategy(Enum):
    """Rate limiting strategies"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitRule:
    """Rate limit rule configuration"""
    name: str
    max_requests: int
    window_seconds: int
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    burst_limit: Optional[int] = None
    cost_per_request: int = 1


@dataclass
class RateLimitResult:
    """Result of rate limit check"""
    allowed: bool
    remaining: int
    reset_time: datetime
    retry_after: Optional[int] = None


class RateLimiter:
    """Multi-strategy rate limiter for GraphQL operations"""
    
    def __init__(self, storage_backend=None):
        self.storage = storage_backend or InMemoryStorage()
        self.rules = {}
        
        # Default rules for different operation types
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default rate limiting rules"""
        self.rules = {
            # Global API limits
            "global": RateLimitRule(
                name="global",
                max_requests=1000,
                window_seconds=3600,  # 1 hour
                strategy=RateLimitStrategy.SLIDING_WINDOW
            ),
            
            # Query operations
            "query": RateLimitRule(
                name="query",
                max_requests=100,
                window_seconds=60,  # 1 minute
                strategy=RateLimitStrategy.SLIDING_WINDOW
            ),
            
            # Mutation operations (more restrictive)
            "mutation": RateLimitRule(
                name="mutation",
                max_requests=20,
                window_seconds=60,
                strategy=RateLimitStrategy.SLIDING_WINDOW
            ),
            
            # Subscription operations
            "subscription": RateLimitRule(
                name="subscription",
                max_requests=10,
                window_seconds=60,
                strategy=RateLimitStrategy.SLIDING_WINDOW
            ),
            
            # High-cost operations
            "analysis": RateLimitRule(
                name="analysis",
                max_requests=10,
                window_seconds=300,  # 5 minutes
                strategy=RateLimitStrategy.TOKEN_BUCKET,
                cost_per_request=5
            ),
            
            "upload": RateLimitRule(
                name="upload",
                max_requests=5,
                window_seconds=60,
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                cost_per_request=3
            ),
            
            "search": RateLimitRule(
                name="search",
                max_requests=30,
                window_seconds=60,
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                cost_per_request=2
            ),
            
            # Authentication operations
            "auth": RateLimitRule(
                name="auth",
                max_requests=5,
                window_seconds=300,  # 5 minutes
                strategy=RateLimitStrategy.FIXED_WINDOW
            ),
            
            # Admin operations
            "admin": RateLimitRule(
                name="admin",
                max_requests=50,
                window_seconds=60,
                strategy=RateLimitStrategy.SLIDING_WINDOW
            )
        }
    
    def add_rule(self, rule: RateLimitRule):
        """Add a custom rate limiting rule"""
        self.rules[rule.name] = rule
    
    async def check_rate_limit(
        self,
        identifier: str,
        rule_name: str,
        cost: int = 1,
        context: Optional[Dict[str, Any]] = None
    ) -> RateLimitResult:
        """Check if request is within rate limits"""
        if rule_name not in self.rules:
            # No rule found, allow request
            return RateLimitResult(
                allowed=True,
                remaining=float('inf'),
                reset_time=datetime.utcnow() + timedelta(hours=1)
            )
        
        rule = self.rules[rule_name]
        key = f"rate_limit:{rule_name}:{identifier}"
        
        if rule.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return await self._check_sliding_window(key, rule, cost)
        elif rule.strategy == RateLimitStrategy.FIXED_WINDOW:
            return await self._check_fixed_window(key, rule, cost)
        elif rule.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return await self._check_token_bucket(key, rule, cost)
        elif rule.strategy == RateLimitStrategy.LEAKY_BUCKET:
            return await self._check_leaky_bucket(key, rule, cost)
        else:
            return RateLimitResult(
                allowed=True,
                remaining=rule.max_requests,
                reset_time=datetime.utcnow() + timedelta(seconds=rule.window_seconds)
            )
    
    async def _check_sliding_window(self, key: str, rule: RateLimitRule, cost: int) -> RateLimitResult:
        """Check sliding window rate limit"""
        now = time.time()
        window_start = now - rule.window_seconds
        
        # Get existing requests in window
        requests = await self.storage.get_requests_in_window(key, window_start, now)
        current_count = sum(req.get('cost', 1) for req in requests)
        
        # Check if adding this request would exceed limit
        if current_count + cost > rule.max_requests:
            # Calculate retry after
            oldest_request = min(requests, key=lambda r: r['timestamp'], default={'timestamp': now})
            retry_after = int(oldest_request['timestamp'] + rule.window_seconds - now)
            
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_time=datetime.fromtimestamp(oldest_request['timestamp'] + rule.window_seconds),
                retry_after=max(retry_after, 1)
            )
        
        # Record this request
        await self.storage.record_request(key, now, cost, rule.window_seconds)
        
        return RateLimitResult(
            allowed=True,
            remaining=rule.max_requests - (current_count + cost),
            reset_time=datetime.fromtimestamp(now + rule.window_seconds)
        )
    
    async def _check_fixed_window(self, key: str, rule: RateLimitRule, cost: int) -> RateLimitResult:
        """Check fixed window rate limit"""
        now = time.time()
        window_start = int(now // rule.window_seconds) * rule.window_seconds
        window_key = f"{key}:{window_start}"
        
        current_count = await self.storage.get_counter(window_key)
        
        if current_count + cost > rule.max_requests:
            reset_time = datetime.fromtimestamp(window_start + rule.window_seconds)
            retry_after = int(reset_time.timestamp() - now)
            
            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_time=reset_time,
                retry_after=max(retry_after, 1)
            )
        
        await self.storage.increment_counter(window_key, cost, rule.window_seconds)
        
        return RateLimitResult(
            allowed=True,
            remaining=rule.max_requests - (current_count + cost),
            reset_time=datetime.fromtimestamp(window_start + rule.window_seconds)
        )
    
    async def _check_token_bucket(self, key: str, rule: RateLimitRule, cost: int) -> RateLimitResult:
        """Check token bucket rate limit"""
        now = time.time()
        bucket_data = await self.storage.get_bucket_data(key)
        
        if not bucket_data:
            # Initialize bucket
            bucket_data = {
                'tokens': rule.max_requests,
                'last_refill': now
            }
        
        # Calculate tokens to add based on time elapsed
        time_elapsed = now - bucket_data['last_refill']
        tokens_to_add = int(time_elapsed * rule.max_requests / rule.window_seconds)
        
        # Update bucket
        bucket_data['tokens'] = min(rule.max_requests, bucket_data['tokens'] + tokens_to_add)
        bucket_data['last_refill'] = now
        
        if bucket_data['tokens'] < cost:
            # Not enough tokens
            retry_after = int((cost - bucket_data['tokens']) * rule.window_seconds / rule.max_requests)
            
            return RateLimitResult(
                allowed=False,
                remaining=bucket_data['tokens'],
                reset_time=datetime.fromtimestamp(now + retry_after),
                retry_after=max(retry_after, 1)
            )
        
        # Consume tokens
        bucket_data['tokens'] -= cost
        await self.storage.set_bucket_data(key, bucket_data, rule.window_seconds * 2)
        
        return RateLimitResult(
            allowed=True,
            remaining=bucket_data['tokens'],
            reset_time=datetime.fromtimestamp(now + rule.window_seconds)
        )
    
    async def _check_leaky_bucket(self, key: str, rule: RateLimitRule, cost: int) -> RateLimitResult:
        """Check leaky bucket rate limit"""
        now = time.time()
        bucket_data = await self.storage.get_bucket_data(key)
        
        if not bucket_data:
            bucket_data = {
                'level': 0,
                'last_leak': now
            }
        
        # Calculate leak based on time elapsed
        time_elapsed = now - bucket_data['last_leak']
        leak_amount = time_elapsed * rule.max_requests / rule.window_seconds
        
        # Update bucket level
        bucket_data['level'] = max(0, bucket_data['level'] - leak_amount)
        bucket_data['last_leak'] = now
        
        if bucket_data['level'] + cost > rule.max_requests:
            # Bucket would overflow
            retry_after = int((bucket_data['level'] + cost - rule.max_requests) * rule.window_seconds / rule.max_requests)
            
            return RateLimitResult(
                allowed=False,
                remaining=int(rule.max_requests - bucket_data['level']),
                reset_time=datetime.fromtimestamp(now + retry_after),
                retry_after=max(retry_after, 1)
            )
        
        # Add to bucket
        bucket_data['level'] += cost
        await self.storage.set_bucket_data(key, bucket_data, rule.window_seconds * 2)
        
        return RateLimitResult(
            allowed=True,
            remaining=int(rule.max_requests - bucket_data['level']),
            reset_time=datetime.fromtimestamp(now + rule.window_seconds)
        )
    
    def get_rate_limit_identifier(self, context: Dict[str, Any]) -> str:
        """Get rate limit identifier from context"""
        # Try to get user ID first
        user = context.get("user")
        if user:
            return f"user:{user.id}"
        
        # Fall back to IP address
        request = context.get("request")
        if request:
            client_ip = self._get_client_ip(request)
            return f"ip:{client_ip}"
        
        # Default identifier
        return "anonymous"
    
    def _get_client_ip(self, request) -> str:
        """Extract client IP from request"""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fall back to direct connection
        return getattr(request.client, "host", "unknown")
    
    def get_operation_type(self, context: Dict[str, Any]) -> str:
        """Determine operation type for rate limiting"""
        operation = context.get("operation")
        if not operation:
            return "query"
        
        operation_type = operation.operation.value if hasattr(operation, 'operation') else "query"
        
        # Map specific operations to custom rule names
        field_name = self._get_root_field_name(operation)
        
        if field_name:
            # Map specific fields to rule names
            field_to_rule = {
                "quickAnalysis": "analysis",
                "requestAnalysis": "analysis",
                "uploadDocument": "upload",
                "searchDocuments": "search",
                "searchClauses": "search",
                "loginUser": "auth",
                "registerUser": "auth",
                "systemStats": "admin",
                "clearCache": "admin",
                "refreshStats": "admin",
            }
            
            if field_name in field_to_rule:
                return field_to_rule[field_name]
        
        return operation_type
    
    def _get_root_field_name(self, operation) -> Optional[str]:
        """Get the root field name from operation"""
        if not hasattr(operation, 'selection_set') or not operation.selection_set:
            return None
        
        for selection in operation.selection_set.selections:
            if hasattr(selection, 'name'):
                return selection.name.value
        
        return None
    
    def calculate_request_cost(self, context: Dict[str, Any]) -> int:
        """Calculate cost of request based on complexity"""
        complexity_analysis = context.get("complexity_analysis", {})
        complexity = complexity_analysis.get("complexity", 1)
        
        # Map complexity to cost
        if complexity <= 10:
            return 1
        elif complexity <= 50:
            return 2
        elif complexity <= 100:
            return 3
        elif complexity <= 500:
            return 5
        else:
            return 10


# Storage backends for rate limiting data
class InMemoryStorage:
    """In-memory storage backend for rate limiting"""
    
    def __init__(self):
        self.requests = {}
        self.counters = {}
        self.buckets = {}
    
    async def get_requests_in_window(self, key: str, start: float, end: float) -> List[Dict[str, Any]]:
        """Get requests in time window"""
        if key not in self.requests:
            return []
        
        return [
            req for req in self.requests[key]
            if start <= req['timestamp'] <= end
        ]
    
    async def record_request(self, key: str, timestamp: float, cost: int, ttl: int):
        """Record a request"""
        if key not in self.requests:
            self.requests[key] = []
        
        self.requests[key].append({
            'timestamp': timestamp,
            'cost': cost
        })
        
        # Clean up old requests
        cutoff = timestamp - ttl
        self.requests[key] = [
            req for req in self.requests[key]
            if req['timestamp'] > cutoff
        ]
    
    async def get_counter(self, key: str) -> int:
        """Get counter value"""
        return self.counters.get(key, 0)
    
    async def increment_counter(self, key: str, amount: int, ttl: int):
        """Increment counter"""
        self.counters[key] = self.counters.get(key, 0) + amount
        
        # Note: TTL cleanup would need a separate mechanism in production
    
    async def get_bucket_data(self, key: str) -> Optional[Dict[str, Any]]:
        """Get bucket data"""
        return self.buckets.get(key)
    
    async def set_bucket_data(self, key: str, data: Dict[str, Any], ttl: int):
        """Set bucket data"""
        self.buckets[key] = data


class RedisStorage:
    """Redis storage backend for rate limiting"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def get_requests_in_window(self, key: str, start: float, end: float) -> List[Dict[str, Any]]:
        """Get requests in time window using Redis sorted sets"""
        results = await self.redis.zrangebyscore(
            key, start, end, withscores=True
        )
        
        requests = []
        for member, timestamp in results:
            try:
                import json
                request_data = json.loads(member)
                request_data['timestamp'] = timestamp
                requests.append(request_data)
            except json.JSONDecodeError:
                # Handle legacy format
                requests.append({
                    'timestamp': timestamp,
                    'cost': 1
                })
        
        return requests
    
    async def record_request(self, key: str, timestamp: float, cost: int, ttl: int):
        """Record request in Redis sorted set"""
        import json
        
        request_data = json.dumps({'cost': cost})
        
        # Add to sorted set
        await self.redis.zadd(key, {request_data: timestamp})
        
        # Set expiration
        await self.redis.expire(key, ttl)
        
        # Clean up old entries
        cutoff = timestamp - ttl
        await self.redis.zremrangebyscore(key, 0, cutoff)
    
    async def get_counter(self, key: str) -> int:
        """Get counter from Redis"""
        value = await self.redis.get(key)
        return int(value) if value else 0
    
    async def increment_counter(self, key: str, amount: int, ttl: int):
        """Increment counter in Redis"""
        pipe = self.redis.pipeline()
        pipe.incrby(key, amount)
        pipe.expire(key, ttl)
        await pipe.execute()
    
    async def get_bucket_data(self, key: str) -> Optional[Dict[str, Any]]:
        """Get bucket data from Redis hash"""
        data = await self.redis.hgetall(key)
        if not data:
            return None
        
        return {
            'tokens': float(data.get('tokens', 0)),
            'last_refill': float(data.get('last_refill', 0)),
            'level': float(data.get('level', 0)),
            'last_leak': float(data.get('last_leak', 0))
        }
    
    async def set_bucket_data(self, key: str, data: Dict[str, Any], ttl: int):
        """Set bucket data in Redis hash"""
        await self.redis.hset(key, mapping=data)
        await self.redis.expire(key, ttl)


# Rate limiting middleware
class RateLimitMiddleware:
    """Middleware for applying rate limiting to GraphQL operations"""
    
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
    
    async def process_request(self, info, **kwargs):
        """Process request and check rate limits"""
        context = info.context
        
        # Get rate limit identifier
        identifier = self.rate_limiter.get_rate_limit_identifier(context)
        
        # Get operation type
        operation_type = self.rate_limiter.get_operation_type(context)
        
        # Calculate request cost
        cost = self.rate_limiter.calculate_request_cost(context)
        
        # Check global rate limit
        global_result = await self.rate_limiter.check_rate_limit(
            identifier, "global", cost, context
        )
        
        if not global_result.allowed:
            raise RateLimitError(
                f"Global rate limit exceeded. Try again in {global_result.retry_after} seconds.",
                global_result.retry_after
            )
        
        # Check operation-specific rate limit
        operation_result = await self.rate_limiter.check_rate_limit(
            identifier, operation_type, cost, context
        )
        
        if not operation_result.allowed:
            raise RateLimitError(
                f"{operation_type.title()} rate limit exceeded. Try again in {operation_result.retry_after} seconds.",
                operation_result.retry_after
            )
        
        # Add rate limit info to context
        context["rate_limit"] = {
            "global": global_result,
            "operation": operation_result,
            "identifier": identifier,
            "cost": cost
        }


# Rate limit decorators
def rate_limit(rule_name: str, cost: int = 1):
    """Decorator for applying rate limits to resolvers"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract info from arguments
            info = None
            for arg in args:
                if hasattr(arg, 'context'):
                    info = arg
                    break
            
            if info:
                # Apply rate limiting
                rate_limiter = info.context.get('rate_limiter')
                if rate_limiter:
                    identifier = rate_limiter.get_rate_limit_identifier(info.context)
                    result = await rate_limiter.check_rate_limit(
                        identifier, rule_name, cost, info.context
                    )
                    
                    if not result.allowed:
                        raise RateLimitError(
                            f"Rate limit exceeded for {rule_name}. Try again in {result.retry_after} seconds.",
                            result.retry_after
                        )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


# Global rate limiter instance
default_rate_limiter = RateLimiter()