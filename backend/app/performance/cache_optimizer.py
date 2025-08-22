"""
Cache Optimization System
Implements multi-level caching with Redis, CDN, and application-level caches
"""

import time
import hashlib
import pickle
import json
import redis
import functools
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import OrderedDict, defaultdict
from threading import Lock, Thread
import asyncio
import aioredis
from enum import Enum

class CacheLevel(Enum):
    """Cache levels in order of proximity"""
    L1_MEMORY = "memory"      # In-process memory cache
    L2_REDIS = "redis"         # Redis cache
    L3_CDN = "cdn"            # CDN cache
    L4_BROWSER = "browser"     # Browser cache

@dataclass
class CacheEntry:
    """Individual cache entry"""
    key: str
    value: Any
    size: int
    ttl: int
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    cache_level: CacheLevel = CacheLevel.L1_MEMORY

@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size: int = 0
    avg_hit_time: float = 0.0
    avg_miss_time: float = 0.0
    hit_rate: float = 0.0

class CacheOptimizer:
    """Multi-level cache optimization system"""
    
    def __init__(self,
                 memory_size_mb: int = 100,
                 redis_url: str = "redis://localhost:6379",
                 default_ttl: int = 3600):
        
        self.memory_size_mb = memory_size_mb
        self.memory_size_bytes = memory_size_mb * 1024 * 1024
        self.default_ttl = default_ttl
        
        # L1 Memory Cache
        self.memory_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.memory_size_used = 0
        self.memory_lock = Lock()
        
        # L2 Redis Cache
        self.redis_client = redis.Redis.from_url(redis_url, decode_responses=False)
        self.redis_pipeline = self.redis_client.pipeline()
        
        # Cache statistics
        self.stats: Dict[CacheLevel, CacheStats] = {
            level: CacheStats() for level in CacheLevel
        }
        
        # Cache strategies
        self.eviction_policy = "LRU"  # LRU, LFU, FIFO
        self.write_policy = "write-through"  # write-through, write-back
        
        # Optimization settings
        self.prefetch_enabled = True
        self.compression_enabled = True
        self.adaptive_ttl_enabled = True
        
        # Pattern tracking for optimization
        self.access_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.miss_patterns: Dict[str, int] = defaultdict(int)
        
        # Start background optimization
        self._start_optimization_worker()
    
    def cache(self, 
             ttl: Optional[int] = None,
             key_prefix: str = "",
             cache_levels: List[CacheLevel] = None,
             compress: bool = None) -> Callable:
        """Decorator for caching function results"""
        
        if cache_levels is None:
            cache_levels = [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS]
        
        if compress is None:
            compress = self.compression_enabled
        
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_key(func, args, kwargs, key_prefix)
                
                # Try to get from cache
                result = self._get_from_cache(cache_key, cache_levels)
                
                if result is not None:
                    return result
                
                # Cache miss - execute function
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Store in cache
                self._set_in_cache(
                    cache_key, 
                    result, 
                    ttl or self.default_ttl,
                    cache_levels,
                    compress
                )
                
                # Track patterns for optimization
                self._track_pattern(cache_key, execution_time)
                
                return result
            
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Async version
                cache_key = self._generate_key(func, args, kwargs, key_prefix)
                
                result = await self._async_get_from_cache(cache_key, cache_levels)
                
                if result is not None:
                    return result
                
                start_time = time.time()
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                await self._async_set_in_cache(
                    cache_key, 
                    result, 
                    ttl or self.default_ttl,
                    cache_levels,
                    compress
                )
                
                self._track_pattern(cache_key, execution_time)
                
                return result
            
            # Return appropriate wrapper
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return wrapper
        
        return decorator
    
    def _generate_key(self, func: Callable, args: tuple, kwargs: dict, prefix: str) -> str:
        """Generate unique cache key"""
        key_parts = [
            prefix,
            func.__module__,
            func.__name__,
            str(args),
            str(sorted(kwargs.items()))
        ]
        
        key_string = ":".join(filter(None, key_parts))
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_from_cache(self, key: str, cache_levels: List[CacheLevel]) -> Optional[Any]:
        """Get value from cache hierarchy"""
        start_time = time.time()
        
        # Try L1 Memory Cache
        if CacheLevel.L1_MEMORY in cache_levels:
            value = self._get_from_memory(key)
            if value is not None:
                self._record_hit(CacheLevel.L1_MEMORY, time.time() - start_time)
                return value
        
        # Try L2 Redis Cache
        if CacheLevel.L2_REDIS in cache_levels:
            value = self._get_from_redis(key)
            if value is not None:
                self._record_hit(CacheLevel.L2_REDIS, time.time() - start_time)
                
                # Promote to L1 if not present
                if CacheLevel.L1_MEMORY in cache_levels:
                    self._set_in_memory(key, value, self.default_ttl)
                
                return value
        
        # Cache miss
        self._record_miss(cache_levels[0] if cache_levels else CacheLevel.L1_MEMORY, 
                         time.time() - start_time)
        return None
    
    def _set_in_cache(self, 
                     key: str, 
                     value: Any, 
                     ttl: int,
                     cache_levels: List[CacheLevel],
                     compress: bool):
        """Set value in cache hierarchy"""
        
        # Serialize value
        serialized = self._serialize(value, compress)
        
        # Write-through: write to all cache levels
        if self.write_policy == "write-through":
            if CacheLevel.L1_MEMORY in cache_levels:
                self._set_in_memory(key, value, ttl)
            
            if CacheLevel.L2_REDIS in cache_levels:
                self._set_in_redis(key, serialized, ttl)
        
        # Write-back: write only to L1, background sync to L2
        elif self.write_policy == "write-back":
            if CacheLevel.L1_MEMORY in cache_levels:
                self._set_in_memory(key, value, ttl)
                # Queue for background write to Redis
                self._queue_write_back(key, serialized, ttl)
    
    def _get_from_memory(self, key: str) -> Optional[Any]:
        """Get from L1 memory cache"""
        with self.memory_lock:
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                
                # Check TTL
                if self._is_expired(entry):
                    del self.memory_cache[key]
                    self.memory_size_used -= entry.size
                    return None
                
                # Update LRU
                self.memory_cache.move_to_end(key)
                entry.last_accessed = datetime.now()
                entry.access_count += 1
                
                return entry.value
        
        return None
    
    def _set_in_memory(self, key: str, value: Any, ttl: int):
        """Set in L1 memory cache"""
        size = self._calculate_size(value)
        
        with self.memory_lock:
            # Evict if necessary
            while self.memory_size_used + size > self.memory_size_bytes:
                self._evict_from_memory()
            
            # Store entry
            entry = CacheEntry(
                key=key,
                value=value,
                size=size,
                ttl=ttl,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                cache_level=CacheLevel.L1_MEMORY
            )
            
            self.memory_cache[key] = entry
            self.memory_size_used += size
    
    def _get_from_redis(self, key: str) -> Optional[Any]:
        """Get from L2 Redis cache"""
        try:
            data = self.redis_client.get(key)
            if data:
                return self._deserialize(data)
        except Exception as e:
            print(f"Redis get error: {e}")
        
        return None
    
    def _set_in_redis(self, key: str, value: bytes, ttl: int):
        """Set in L2 Redis cache"""
        try:
            self.redis_client.setex(key, ttl, value)
        except Exception as e:
            print(f"Redis set error: {e}")
    
    async def _async_get_from_cache(self, key: str, cache_levels: List[CacheLevel]) -> Optional[Any]:
        """Async get from cache"""
        # Try memory first (sync is ok for memory)
        if CacheLevel.L1_MEMORY in cache_levels:
            value = self._get_from_memory(key)
            if value is not None:
                return value
        
        # Try Redis async
        if CacheLevel.L2_REDIS in cache_levels:
            try:
                async with aioredis.from_url(self.redis_url) as redis:
                    data = await redis.get(key)
                    if data:
                        return self._deserialize(data)
            except Exception as e:
                print(f"Async Redis error: {e}")
        
        return None
    
    async def _async_set_in_cache(self, 
                                  key: str, 
                                  value: Any, 
                                  ttl: int,
                                  cache_levels: List[CacheLevel],
                                  compress: bool):
        """Async set in cache"""
        serialized = self._serialize(value, compress)
        
        # Set in memory (sync is ok)
        if CacheLevel.L1_MEMORY in cache_levels:
            self._set_in_memory(key, value, ttl)
        
        # Set in Redis async
        if CacheLevel.L2_REDIS in cache_levels:
            try:
                async with aioredis.from_url(self.redis_url) as redis:
                    await redis.setex(key, ttl, serialized)
            except Exception as e:
                print(f"Async Redis set error: {e}")
    
    def _evict_from_memory(self):
        """Evict entry from memory cache based on policy"""
        if not self.memory_cache:
            return
        
        if self.eviction_policy == "LRU":
            # Remove least recently used
            key, entry = self.memory_cache.popitem(last=False)
        elif self.eviction_policy == "LFU":
            # Remove least frequently used
            key = min(self.memory_cache, key=lambda k: self.memory_cache[k].access_count)
            entry = self.memory_cache.pop(key)
        else:  # FIFO
            # Remove oldest
            key, entry = self.memory_cache.popitem(last=False)
        
        self.memory_size_used -= entry.size
        self.stats[CacheLevel.L1_MEMORY].evictions += 1
    
    def _serialize(self, value: Any, compress: bool) -> bytes:
        """Serialize and optionally compress value"""
        data = pickle.dumps(value)
        
        if compress:
            import zlib
            data = zlib.compress(data)
        
        return data
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize and decompress value"""
        try:
            # Try decompression first
            import zlib
            data = zlib.decompress(data)
        except:
            pass  # Not compressed
        
        return pickle.loads(data)
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value"""
        try:
            return len(pickle.dumps(value))
        except:
            return 1000  # Default size
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired"""
        age = (datetime.now() - entry.created_at).total_seconds()
        return age > entry.ttl
    
    def _record_hit(self, level: CacheLevel, response_time: float):
        """Record cache hit"""
        stats = self.stats[level]
        stats.hits += 1
        
        # Update average hit time
        total_time = stats.avg_hit_time * (stats.hits - 1) + response_time
        stats.avg_hit_time = total_time / stats.hits
        
        # Update hit rate
        total = stats.hits + stats.misses
        stats.hit_rate = stats.hits / total if total > 0 else 0
    
    def _record_miss(self, level: CacheLevel, response_time: float):
        """Record cache miss"""
        stats = self.stats[level]
        stats.misses += 1
        
        # Update average miss time
        total_time = stats.avg_miss_time * (stats.misses - 1) + response_time
        stats.avg_miss_time = total_time / stats.misses
        
        # Update hit rate
        total = stats.hits + stats.misses
        stats.hit_rate = stats.hits / total if total > 0 else 0
    
    def _track_pattern(self, key: str, execution_time: float):
        """Track access patterns for optimization"""
        self.access_patterns[key].append(datetime.now())
        
        # Keep only recent accesses
        cutoff = datetime.now() - timedelta(hours=24)
        self.access_patterns[key] = [
            dt for dt in self.access_patterns[key] 
            if dt > cutoff
        ]
        
        # Adaptive TTL based on access frequency
        if self.adaptive_ttl_enabled:
            self._adjust_ttl(key, execution_time)
    
    def _adjust_ttl(self, key: str, execution_time: float):
        """Adjust TTL based on access patterns"""
        accesses = self.access_patterns[key]
        
        if len(accesses) < 2:
            return
        
        # Calculate access frequency
        time_span = (accesses[-1] - accesses[0]).total_seconds()
        if time_span > 0:
            frequency = len(accesses) / time_span  # accesses per second
            
            # Adjust TTL based on frequency and computation cost
            if frequency > 0.1:  # High frequency
                new_ttl = min(7200, self.default_ttl * 2)  # Increase TTL
            elif frequency < 0.001:  # Low frequency
                new_ttl = max(300, self.default_ttl // 2)  # Decrease TTL
            else:
                new_ttl = self.default_ttl
            
            # Consider computation cost
            if execution_time > 1.0:  # Expensive computation
                new_ttl = int(new_ttl * 1.5)
            
            # Update entry TTL if in memory
            with self.memory_lock:
                if key in self.memory_cache:
                    self.memory_cache[key].ttl = new_ttl
    
    def _queue_write_back(self, key: str, value: bytes, ttl: int):
        """Queue write-back to Redis"""
        # In production, use a proper queue like Celery
        Thread(target=self._set_in_redis, args=(key, value, ttl), daemon=True).start()
    
    def _start_optimization_worker(self):
        """Start background optimization worker"""
        def worker():
            while True:
                time.sleep(60)  # Run every minute
                self._optimize_cache()
        
        Thread(target=worker, daemon=True).start()
    
    def _optimize_cache(self):
        """Perform cache optimization"""
        # Analyze patterns and optimize
        self._prefetch_predicted()
        self._rebalance_cache_levels()
        self._cleanup_expired()
    
    def _prefetch_predicted(self):
        """Prefetch predicted cache misses"""
        if not self.prefetch_enabled:
            return
        
        # Analyze miss patterns
        frequent_misses = sorted(
            self.miss_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Prefetch top misses
        for key, count in frequent_misses:
            if count > 5:  # Threshold for prefetching
                # In production, trigger computation and caching
                pass
    
    def _rebalance_cache_levels(self):
        """Rebalance data across cache levels"""
        # Move frequently accessed items to L1
        # Move less frequently accessed items to L2
        
        with self.memory_lock:
            # Find least used entries in L1
            if len(self.memory_cache) > 100:
                least_used = sorted(
                    self.memory_cache.items(),
                    key=lambda x: x[1].access_count
                )[:10]
                
                for key, entry in least_used:
                    if entry.access_count < 2:
                        # Move to Redis only
                        serialized = self._serialize(entry.value, self.compression_enabled)
                        self._set_in_redis(key, serialized, entry.ttl)
                        
                        # Remove from memory
                        del self.memory_cache[key]
                        self.memory_size_used -= entry.size
    
    def _cleanup_expired(self):
        """Clean up expired entries"""
        with self.memory_lock:
            expired_keys = [
                key for key, entry in self.memory_cache.items()
                if self._is_expired(entry)
            ]
            
            for key in expired_keys:
                entry = self.memory_cache[key]
                del self.memory_cache[key]
                self.memory_size_used -= entry.size
    
    def get_cache_report(self) -> Dict:
        """Generate cache performance report"""
        report = {
            'configuration': {
                'memory_size_mb': self.memory_size_mb,
                'eviction_policy': self.eviction_policy,
                'write_policy': self.write_policy,
                'compression_enabled': self.compression_enabled,
                'adaptive_ttl_enabled': self.adaptive_ttl_enabled
            },
            'memory_cache': {
                'entries': len(self.memory_cache),
                'size_used_mb': self.memory_size_used / (1024 * 1024),
                'size_limit_mb': self.memory_size_mb,
                'utilization': (self.memory_size_used / self.memory_size_bytes) * 100
            },
            'statistics': {}
        }
        
        # Add statistics for each cache level
        for level, stats in self.stats.items():
            report['statistics'][level.value] = {
                'hits': stats.hits,
                'misses': stats.misses,
                'hit_rate': f"{stats.hit_rate * 100:.2f}%",
                'evictions': stats.evictions,
                'avg_hit_time_ms': stats.avg_hit_time * 1000,
                'avg_miss_time_ms': stats.avg_miss_time * 1000
            }
        
        # Add optimization recommendations
        report['recommendations'] = self._generate_recommendations()
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate cache optimization recommendations"""
        recommendations = []
        
        # Check hit rate
        for level, stats in self.stats.items():
            if stats.hit_rate < 0.8 and stats.hits + stats.misses > 100:
                recommendations.append(
                    f"{level.value} cache hit rate is low ({stats.hit_rate * 100:.1f}%), "
                    f"consider increasing cache size or TTL"
                )
        
        # Check eviction rate
        l1_stats = self.stats[CacheLevel.L1_MEMORY]
        if l1_stats.evictions > l1_stats.hits * 0.1:
            recommendations.append(
                "High eviction rate in memory cache, consider increasing memory size"
            )
        
        # Check response times
        if l1_stats.avg_hit_time > 0.001:  # 1ms
            recommendations.append(
                "L1 cache response time is high, consider optimizing serialization"
            )
        
        return recommendations
    
    def warm_cache(self, data_loader: Callable, keys: List[str]):
        """Warm cache with preloaded data"""
        for key in keys:
            try:
                value = data_loader(key)
                self._set_in_cache(
                    key, 
                    value, 
                    self.default_ttl,
                    [CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS],
                    self.compression_enabled
                )
            except Exception as e:
                print(f"Failed to warm cache for key {key}: {e}")
    
    def invalidate(self, pattern: str = None):
        """Invalidate cache entries"""
        if pattern:
            # Invalidate by pattern
            with self.memory_lock:
                keys_to_remove = [
                    key for key in self.memory_cache 
                    if pattern in key
                ]
                
                for key in keys_to_remove:
                    entry = self.memory_cache[key]
                    del self.memory_cache[key]
                    self.memory_size_used -= entry.size
            
            # Also invalidate in Redis
            for key in self.redis_client.scan_iter(match=f"*{pattern}*"):
                self.redis_client.delete(key)
        else:
            # Clear all caches
            self.clear_all()
    
    def clear_all(self):
        """Clear all cache levels"""
        with self.memory_lock:
            self.memory_cache.clear()
            self.memory_size_used = 0
        
        try:
            self.redis_client.flushdb()
        except Exception as e:
            print(f"Failed to clear Redis: {e}")

# Example usage
if __name__ == "__main__":
    # Create cache optimizer
    cache = CacheOptimizer(memory_size_mb=100)
    
    # Example function to cache
    @cache.cache(ttl=300, cache_levels=[CacheLevel.L1_MEMORY, CacheLevel.L2_REDIS])
    def expensive_computation(x: int, y: int) -> int:
        """Simulate expensive computation"""
        time.sleep(0.1)  # Simulate work
        return x * y + sum(range(1000000))
    
    # Test caching
    import timeit
    
    # First call - cache miss
    t1 = timeit.timeit(lambda: expensive_computation(5, 10), number=1)
    print(f"First call (miss): {t1:.4f}s")
    
    # Second call - cache hit
    t2 = timeit.timeit(lambda: expensive_computation(5, 10), number=1)
    print(f"Second call (hit): {t2:.4f}s")
    
    # Generate report
    report = cache.get_cache_report()
    print(json.dumps(report, indent=2))