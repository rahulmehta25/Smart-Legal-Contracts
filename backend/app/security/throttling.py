"""
Advanced Request Throttling System
Implements intelligent throttling with burst handling and priority queues
OWASP API Security compliant
"""

import time
import asyncio
import heapq
from typing import Optional, Dict, Any, List, Callable, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import json
import hashlib

import redis
import numpy as np
from fastapi import HTTPException, status, Request


class ThrottleStrategy(str, Enum):
    """Throttling strategies"""
    AGGRESSIVE = "aggressive"      # Strict throttling
    MODERATE = "moderate"          # Balanced approach
    LENIENT = "lenient"           # Permissive throttling
    ADAPTIVE = "adaptive"          # Adjusts based on load
    PRIORITY = "priority"          # Priority-based queuing


class RequestPriority(str, Enum):
    """Request priority levels"""
    CRITICAL = "critical"          # System-critical requests
    HIGH = "high"                  # Premium users
    NORMAL = "normal"              # Regular users
    LOW = "low"                    # Free tier
    BACKGROUND = "background"      # Batch/async operations


@dataclass
class ThrottleConfig:
    """Throttle configuration"""
    max_concurrent: int = 100      # Max concurrent requests
    queue_size: int = 1000         # Max queue size
    timeout: int = 30              # Request timeout in seconds
    burst_size: int = 20          # Burst allowance
    burst_window: int = 10        # Burst window in seconds
    strategy: ThrottleStrategy = ThrottleStrategy.MODERATE
    enable_priority: bool = True  # Enable priority queuing
    adaptive_threshold: float = 0.8  # Load threshold for adaptive throttling


@dataclass
class ThrottleStatus:
    """Throttle status information"""
    allowed: bool
    queue_position: Optional[int] = None
    estimated_wait: Optional[float] = None
    current_load: float = 0.0
    active_requests: int = 0
    queued_requests: int = 0
    rejected_count: int = 0


@dataclass
class QueuedRequest:
    """Queued request information"""
    request_id: str
    priority: RequestPriority
    timestamp: float
    identifier: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    callback: Optional[Callable] = None
    
    def __lt__(self, other):
        """Priority comparison for heap queue"""
        priority_order = {
            RequestPriority.CRITICAL: 0,
            RequestPriority.HIGH: 1,
            RequestPriority.NORMAL: 2,
            RequestPriority.LOW: 3,
            RequestPriority.BACKGROUND: 4
        }
        
        if self.priority != other.priority:
            return priority_order[self.priority] < priority_order[other.priority]
        return self.timestamp < other.timestamp  # FIFO within same priority


class RequestThrottler:
    """
    Advanced request throttling system with:
    - Concurrent request limiting
    - Priority-based queuing
    - Burst handling
    - Adaptive throttling based on system load
    - Circuit breaker pattern
    - Request coalescing
    """
    
    def __init__(self,
                 redis_client: Optional[redis.Redis] = None,
                 config: Optional[ThrottleConfig] = None):
        
        self.redis_client = redis_client or redis.Redis(
            host='localhost',
            port=6379,
            decode_responses=True
        )
        
        self.config = config or ThrottleConfig()
        
        # Active request tracking
        self.active_requests = set()
        self.active_lock = threading.Lock()
        
        # Priority queue for pending requests
        self.request_queue = []
        self.queue_lock = threading.Lock()
        
        # Burst tracking
        self.burst_tracker = defaultdict(deque)
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreaker()
        
        # Request coalescing
        self.coalesce_cache = {}
        self.coalesce_lock = threading.Lock()
        
        # Metrics
        self.metrics = {
            "total_requests": 0,
            "throttled_requests": 0,
            "rejected_requests": 0,
            "completed_requests": 0,
            "average_wait_time": 0,
            "current_load": 0.0
        }
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Start metrics collection
        self.metrics_thread = threading.Thread(target=self._collect_metrics)
        self.metrics_thread.daemon = True
        self.metrics_thread.start()
    
    # ========== Main Throttling Logic ==========
    
    async def throttle_request(self,
                              request_id: str,
                              identifier: str,
                              priority: Optional[RequestPriority] = None,
                              metadata: Optional[Dict[str, Any]] = None) -> ThrottleStatus:
        """
        Throttle incoming request
        """
        
        self.metrics["total_requests"] += 1
        
        # Check circuit breaker
        if self.circuit_breaker.is_open():
            self.metrics["rejected_requests"] += 1
            return ThrottleStatus(
                allowed=False,
                current_load=1.0,
                rejected_count=self.metrics["rejected_requests"]
            )
        
        # Determine priority
        if priority is None:
            priority = self._determine_priority(identifier, metadata)
        
        # Check if can process immediately
        if self._can_process_now(priority):
            self._add_active_request(request_id)
            return ThrottleStatus(
                allowed=True,
                current_load=self._calculate_load(),
                active_requests=len(self.active_requests)
            )
        
        # Check burst allowance
        if self._check_burst_allowance(identifier):
            self._add_active_request(request_id)
            return ThrottleStatus(
                allowed=True,
                current_load=self._calculate_load(),
                active_requests=len(self.active_requests)
            )
        
        # Try to queue request
        queue_status = await self._queue_request(
            request_id, identifier, priority, metadata
        )
        
        if queue_status:
            self.metrics["throttled_requests"] += 1
            return ThrottleStatus(
                allowed=False,
                queue_position=queue_status["position"],
                estimated_wait=queue_status["estimated_wait"],
                current_load=self._calculate_load(),
                queued_requests=len(self.request_queue)
            )
        
        # Reject if queue is full
        self.metrics["rejected_requests"] += 1
        return ThrottleStatus(
            allowed=False,
            current_load=self._calculate_load(),
            rejected_count=self.metrics["rejected_requests"]
        )
    
    def complete_request(self, request_id: str):
        """Mark request as completed"""
        
        with self.active_lock:
            self.active_requests.discard(request_id)
        
        self.metrics["completed_requests"] += 1
        
        # Process next queued request
        self._process_next_request()
    
    # ========== Priority Management ==========
    
    def _determine_priority(self,
                          identifier: str,
                          metadata: Optional[Dict[str, Any]]) -> RequestPriority:
        """Determine request priority based on various factors"""
        
        # Check for explicit priority in metadata
        if metadata and "priority" in metadata:
            return RequestPriority(metadata["priority"])
        
        # Check user tier (would need user info)
        user_tier = self._get_user_tier(identifier)
        
        tier_priority = {
            "enterprise": RequestPriority.HIGH,
            "pro": RequestPriority.NORMAL,
            "basic": RequestPriority.NORMAL,
            "free": RequestPriority.LOW
        }
        
        return tier_priority.get(user_tier, RequestPriority.NORMAL)
    
    def _get_user_tier(self, identifier: str) -> str:
        """Get user tier from identifier"""
        
        # Look up user tier from cache or database
        tier_key = f"user_tier:{identifier}"
        tier = self.redis_client.get(tier_key)
        
        return tier or "free"
    
    # ========== Queue Management ==========
    
    async def _queue_request(self,
                            request_id: str,
                            identifier: str,
                            priority: RequestPriority,
                            metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Queue request for later processing"""
        
        with self.queue_lock:
            # Check queue size
            if len(self.request_queue) >= self.config.queue_size:
                # Try to remove lower priority requests
                if not self._make_room_in_queue(priority):
                    return None
            
            # Create queued request
            queued = QueuedRequest(
                request_id=request_id,
                priority=priority,
                timestamp=time.time(),
                identifier=identifier,
                metadata=metadata or {}
            )
            
            # Add to priority queue
            heapq.heappush(self.request_queue, queued)
            
            # Calculate position and wait time
            position = self._calculate_queue_position(request_id)
            estimated_wait = self._estimate_wait_time(position, priority)
            
            return {
                "position": position,
                "estimated_wait": estimated_wait
            }
    
    def _make_room_in_queue(self, priority: RequestPriority) -> bool:
        """Try to make room in queue by removing lower priority requests"""
        
        if not self.config.enable_priority:
            return False
        
        # Find and remove lowest priority request
        lowest_priority_requests = [
            r for r in self.request_queue 
            if r.priority == RequestPriority.BACKGROUND or r.priority == RequestPriority.LOW
        ]
        
        if lowest_priority_requests and priority in [RequestPriority.CRITICAL, RequestPriority.HIGH]:
            # Remove oldest low priority request
            oldest = min(lowest_priority_requests, key=lambda r: r.timestamp)
            self.request_queue.remove(oldest)
            heapq.heapify(self.request_queue)
            return True
        
        return False
    
    def _calculate_queue_position(self, request_id: str) -> int:
        """Calculate position in queue"""
        
        for i, req in enumerate(sorted(self.request_queue)):
            if req.request_id == request_id:
                return i + 1
        return len(self.request_queue)
    
    def _estimate_wait_time(self, position: int, priority: RequestPriority) -> float:
        """Estimate wait time based on queue position and processing rate"""
        
        # Calculate average processing time
        avg_processing_time = 1.0  # Would calculate from metrics
        
        # Adjust for priority
        priority_factor = {
            RequestPriority.CRITICAL: 0.5,
            RequestPriority.HIGH: 0.7,
            RequestPriority.NORMAL: 1.0,
            RequestPriority.LOW: 1.5,
            RequestPriority.BACKGROUND: 2.0
        }
        
        factor = priority_factor.get(priority, 1.0)
        
        return position * avg_processing_time * factor
    
    def _process_queue(self):
        """Background thread to process queued requests"""
        
        while True:
            try:
                # Check if can process more requests
                if len(self.active_requests) < self.config.max_concurrent:
                    self._process_next_request()
                
                time.sleep(0.1)  # Check every 100ms
            except Exception as e:
                print(f"Queue processing error: {e}")
                time.sleep(1)
    
    def _process_next_request(self):
        """Process next request from queue"""
        
        with self.queue_lock:
            if not self.request_queue:
                return
            
            # Get highest priority request
            next_request = heapq.heappop(self.request_queue)
            
            # Check if request is still valid (not timed out)
            if time.time() - next_request.timestamp > self.config.timeout:
                # Request timed out in queue
                return
            
            # Add to active requests
            self._add_active_request(next_request.request_id)
            
            # Execute callback if provided
            if next_request.callback:
                threading.Thread(
                    target=next_request.callback,
                    args=(next_request.request_id,)
                ).start()
    
    # ========== Burst Handling ==========
    
    def _check_burst_allowance(self, identifier: str) -> bool:
        """Check if request can use burst allowance"""
        
        now = time.time()
        window_start = now - self.config.burst_window
        
        # Clean old entries
        burst_history = self.burst_tracker[identifier]
        while burst_history and burst_history[0] < window_start:
            burst_history.popleft()
        
        # Check burst limit
        if len(burst_history) < self.config.burst_size:
            burst_history.append(now)
            return True
        
        return False
    
    # ========== Load Management ==========
    
    def _can_process_now(self, priority: RequestPriority) -> bool:
        """Check if request can be processed immediately"""
        
        with self.active_lock:
            current_count = len(self.active_requests)
        
        # Always allow critical requests
        if priority == RequestPriority.CRITICAL:
            return current_count < self.config.max_concurrent * 1.2  # 20% overhead for critical
        
        # Apply strategy
        if self.config.strategy == ThrottleStrategy.AGGRESSIVE:
            limit = self.config.max_concurrent * 0.8
        elif self.config.strategy == ThrottleStrategy.MODERATE:
            limit = self.config.max_concurrent
        elif self.config.strategy == ThrottleStrategy.LENIENT:
            limit = self.config.max_concurrent * 1.2
        elif self.config.strategy == ThrottleStrategy.ADAPTIVE:
            limit = self._calculate_adaptive_limit()
        else:
            limit = self.config.max_concurrent
        
        return current_count < limit
    
    def _calculate_adaptive_limit(self) -> int:
        """Calculate adaptive limit based on system load"""
        
        load = self._calculate_load()
        
        if load < 0.5:
            # Low load - allow more requests
            return int(self.config.max_concurrent * 1.2)
        elif load < 0.8:
            # Normal load
            return self.config.max_concurrent
        else:
            # High load - reduce limit
            return int(self.config.max_concurrent * 0.8)
    
    def _calculate_load(self) -> float:
        """Calculate current system load"""
        
        with self.active_lock:
            active_ratio = len(self.active_requests) / max(1, self.config.max_concurrent)
        
        with self.queue_lock:
            queue_ratio = len(self.request_queue) / max(1, self.config.queue_size)
        
        # Weighted average
        load = (active_ratio * 0.7) + (queue_ratio * 0.3)
        
        self.metrics["current_load"] = load
        
        return min(1.0, load)
    
    def _add_active_request(self, request_id: str):
        """Add request to active set"""
        
        with self.active_lock:
            self.active_requests.add(request_id)
    
    # ========== Request Coalescing ==========
    
    def coalesce_request(self,
                        cache_key: str,
                        request_func: Callable,
                        ttl: int = 60) -> Any:
        """
        Coalesce multiple identical requests into one
        Useful for expensive operations
        """
        
        with self.coalesce_lock:
            # Check if request is already in progress
            if cache_key in self.coalesce_cache:
                future = self.coalesce_cache[cache_key]["future"]
                return future  # Return same future to all callers
            
            # Create new future
            future = asyncio.Future()
            self.coalesce_cache[cache_key] = {
                "future": future,
                "timestamp": time.time(),
                "ttl": ttl
            }
        
        # Execute request
        try:
            result = request_func()
            future.set_result(result)
        except Exception as e:
            future.set_exception(e)
        finally:
            # Clean up after TTL
            threading.Timer(ttl, lambda: self._cleanup_coalesce(cache_key)).start()
        
        return future
    
    def _cleanup_coalesce(self, cache_key: str):
        """Clean up coalesced request cache"""
        
        with self.coalesce_lock:
            self.coalesce_cache.pop(cache_key, None)
    
    # ========== Metrics Collection ==========
    
    def _collect_metrics(self):
        """Background thread to collect metrics"""
        
        while True:
            try:
                # Calculate average wait time
                if self.metrics["throttled_requests"] > 0:
                    # Would calculate from actual wait times
                    pass
                
                # Export metrics to monitoring system
                self._export_metrics()
                
                time.sleep(60)  # Collect every minute
            except:
                time.sleep(60)
    
    def _export_metrics(self):
        """Export metrics to monitoring system"""
        
        # Would send to Prometheus, CloudWatch, etc.
        metrics_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": self.metrics,
            "active_requests": len(self.active_requests),
            "queued_requests": len(self.request_queue),
            "load": self._calculate_load()
        }
        
        # Store in Redis for dashboard
        self.redis_client.lpush(
            "throttle_metrics",
            json.dumps(metrics_data)
        )
        self.redis_client.ltrim("throttle_metrics", 0, 1000)  # Keep last 1000
    
    def get_status(self) -> Dict[str, Any]:
        """Get current throttler status"""
        
        return {
            "active_requests": len(self.active_requests),
            "queued_requests": len(self.request_queue),
            "current_load": self._calculate_load(),
            "metrics": self.metrics,
            "circuit_breaker": self.circuit_breaker.get_status(),
            "config": {
                "max_concurrent": self.config.max_concurrent,
                "queue_size": self.config.queue_size,
                "strategy": self.config.strategy
            }
        }


class CircuitBreaker:
    """
    Circuit breaker pattern implementation
    Prevents cascading failures
    """
    
    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 expected_exception: type = Exception):
        
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
        self.state_lock = threading.Lock()
    
    def is_open(self) -> bool:
        """Check if circuit breaker is open"""
        
        with self.state_lock:
            if self.state == "open":
                # Check if should transition to half-open
                if self.last_failure_time:
                    time_since_failure = time.time() - self.last_failure_time
                    if time_since_failure > self.recovery_timeout:
                        self.state = "half-open"
                        return False
                return True
            
            return False
    
    def record_success(self):
        """Record successful request"""
        
        with self.state_lock:
            if self.state == "half-open":
                # Transition back to closed
                self.state = "closed"
                self.failure_count = 0
    
    def record_failure(self):
        """Record failed request"""
        
        with self.state_lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            elif self.state == "half-open":
                # Failed during recovery, go back to open
                self.state = "open"
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        
        return {
            "state": self.state,
            "failure_count": self.failure_count,
            "last_failure": self.last_failure_time
        }
    
    def reset(self):
        """Reset circuit breaker"""
        
        with self.state_lock:
            self.state = "closed"
            self.failure_count = 0
            self.last_failure_time = None