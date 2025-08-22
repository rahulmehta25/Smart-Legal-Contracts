"""
Enterprise Advanced Caching Module

Implements multi-tier caching architecture with intelligent cache warming,
invalidation strategies, and performance optimization.

Target: Support 10,000+ concurrent users with <50ms query response time
"""

import redis
import memcache
import hashlib
import json
import pickle
import asyncio
import time
import logging
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import threading
from contextlib import asynccontextmanager
import weakref
import sys

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache tier levels"""
    L1_APPLICATION = "l1_application"  # In-memory application cache
    L2_REDIS = "l2_redis"              # Redis distributed cache
    L3_CDN = "l3_cdn"                  # CDN edge cache
    

class CacheStrategy(Enum):
    """Cache invalidation strategies"""
    TTL = "ttl"                        # Time-to-live expiration
    LRU = "lru"                        # Least recently used
    LFU = "lfu"                        # Least frequently used
    WRITE_THROUGH = "write_through"    # Update cache on write
    WRITE_BEHIND = "write_behind"      # Async update cache after write
    REFRESH_AHEAD = "refresh_ahead"    # Refresh before expiration
    

class CachePattern(Enum):
    """Cache access patterns"""
    CACHE_ASIDE = "cache_aside"        # Check cache, load if miss
    READ_THROUGH = "read_through"      # Cache loads data automatically
    WRITE_THROUGH = "write_through"    # Cache updates data automatically
    WRITE_BEHIND = "write_behind"      # Cache updates data asynchronously
    

@dataclass
class CacheConfig:
    """Configuration for cache tiers"""
    level: CacheLevel
    strategy: CacheStrategy
    ttl: timedelta
    max_size: int
    compression: bool = False
    serialization: str = "json"  # json, pickle, msgpack
    key_prefix: str = ""
    host: Optional[str] = None
    port: Optional[int] = None
    password: Optional[str] = None
    

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    accessed_at: datetime
    access_count: int
    ttl: timedelta
    size_bytes: int
    compressed: bool = False
    

@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    errors: int = 0
    total_requests: int = 0
    avg_response_time: float = 0.0
    memory_usage: int = 0
    hit_ratio: float = 0.0
    

class L1ApplicationCache:
    """
    Level 1: High-performance in-memory application cache
    
    Features:
    - Thread-safe operations
    - Multiple eviction policies (LRU, LFU, TTL)
    - Compression support
    - Memory usage tracking
    """
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []  # For LRU
        self.access_counts: Dict[str, int] = defaultdict(int)  # For LFU
        self.lock = threading.RLock()
        self.stats = CacheStats()
        self.memory_usage = 0
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from L1 cache"""
        start_time = time.time()
        
        with self.lock:
            prefixed_key = f"{self.config.key_prefix}{key}"
            
            if prefixed_key in self.cache:
                entry = self.cache[prefixed_key]
                
                # Check TTL
                if self._is_expired(entry):
                    self._remove_entry(prefixed_key)
                    self.stats.misses += 1
                    return None
                
                # Update access metadata
                entry.accessed_at = datetime.now()
                entry.access_count += 1
                self.access_counts[prefixed_key] += 1
                
                # Update LRU order
                if prefixed_key in self.access_order:
                    self.access_order.remove(prefixed_key)
                self.access_order.append(prefixed_key)
                
                self.stats.hits += 1
                self.stats.total_requests += 1
                
                response_time = time.time() - start_time
                self._update_avg_response_time(response_time)
                
                return entry.value
            else:
                self.stats.misses += 1
                self.stats.total_requests += 1
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> bool:
        """Set value in L1 cache"""
        
        with self.lock:
            prefixed_key = f"{self.config.key_prefix}{key}"
            
            # Serialize and possibly compress the value
            serialized_value = self._serialize_value(value)
            size_bytes = sys.getsizeof(serialized_value)
            
            # Check if we need to evict entries
            if len(self.cache) >= self.config.max_size:
                self._evict_entries()
            
            # Create cache entry
            entry = CacheEntry(
                key=prefixed_key,
                value=serialized_value,
                created_at=datetime.now(),
                accessed_at=datetime.now(),
                access_count=1,
                ttl=ttl or self.config.ttl,
                size_bytes=size_bytes,
                compressed=self.config.compression
            )
            
            # Remove old entry if exists
            if prefixed_key in self.cache:
                self._remove_entry(prefixed_key)
            
            # Add new entry
            self.cache[prefixed_key] = entry
            self.access_order.append(prefixed_key)
            self.access_counts[prefixed_key] = 1
            self.memory_usage += size_bytes
            
            return True
    
    def delete(self, key: str) -> bool:
        """Delete value from L1 cache"""
        
        with self.lock:
            prefixed_key = f"{self.config.key_prefix}{key}"
            
            if prefixed_key in self.cache:
                self._remove_entry(prefixed_key)
                return True
            return False
    
    def clear(self):
        """Clear all cache entries"""
        
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.access_counts.clear()
            self.memory_usage = 0
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired"""
        return datetime.now() - entry.created_at > entry.ttl
    
    def _remove_entry(self, key: str):
        """Remove entry and update metadata"""
        if key in self.cache:
            entry = self.cache[key]
            self.memory_usage -= entry.size_bytes
            del self.cache[key]
            
        if key in self.access_order:
            self.access_order.remove(key)
            
        if key in self.access_counts:
            del self.access_counts[key]
    
    def _evict_entries(self):
        """Evict entries based on strategy"""
        
        if self.config.strategy == CacheStrategy.LRU:
            self._evict_lru()
        elif self.config.strategy == CacheStrategy.LFU:
            self._evict_lfu()
        elif self.config.strategy == CacheStrategy.TTL:
            self._evict_expired()
    
    def _evict_lru(self):
        """Evict least recently used entries"""
        while len(self.cache) >= self.config.max_size and self.access_order:
            oldest_key = self.access_order[0]
            self._remove_entry(oldest_key)
            self.stats.evictions += 1
    
    def _evict_lfu(self):
        """Evict least frequently used entries"""
        if self.access_counts:
            # Find key with minimum access count
            min_key = min(self.access_counts.keys(), key=lambda k: self.access_counts[k])
            self._remove_entry(min_key)
            self.stats.evictions += 1
    
    def _evict_expired(self):
        """Evict expired entries"""
        expired_keys = []
        for key, entry in self.cache.items():
            if self._is_expired(entry):
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_entry(key)
            self.stats.evictions += 1
    
    def _serialize_value(self, value: Any) -> Any:
        """Serialize value based on configuration"""
        if self.config.serialization == "json":
            try:
                return json.dumps(value)
            except:
                return pickle.dumps(value)
        elif self.config.serialization == "pickle":
            return pickle.dumps(value)
        else:
            return value
    
    def _deserialize_value(self, value: Any) -> Any:
        """Deserialize value based on configuration"""
        if isinstance(value, str) and self.config.serialization == "json":
            try:
                return json.loads(value)
            except:
                return pickle.loads(value)
        elif isinstance(value, bytes) and self.config.serialization == "pickle":
            return pickle.loads(value)
        else:
            return value
    
    def _update_avg_response_time(self, response_time: float):
        """Update average response time"""
        if self.stats.total_requests > 0:
            self.stats.avg_response_time = (
                (self.stats.avg_response_time * (self.stats.total_requests - 1) + response_time) /
                self.stats.total_requests
            )
        else:
            self.stats.avg_response_time = response_time
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        with self.lock:
            self.stats.memory_usage = self.memory_usage
            if self.stats.total_requests > 0:
                self.stats.hit_ratio = self.stats.hits / self.stats.total_requests
            return self.stats


class L2RedisCache:
    """
    Level 2: Redis distributed cache
    
    Features:
    - Distributed caching across multiple nodes
    - Pub/Sub for cache invalidation
    - Pipeline operations for performance
    - Cluster support
    """
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.stats = CacheStats()
        
        # Initialize Redis connection
        try:
            if config.host and config.port:
                self.redis_client = redis.Redis(
                    host=config.host,
                    port=config.port,
                    password=config.password,
                    decode_responses=False,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    health_check_interval=30
                )
            else:
                self.redis_client = redis.Redis(decode_responses=False)
                
            # Test connection
            self.redis_client.ping()
            logger.info("Redis cache connection established")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        if not self.redis_client:
            return None
            
        start_time = time.time()
        
        try:
            prefixed_key = f"{self.config.key_prefix}{key}"
            cached_data = self.redis_client.get(prefixed_key)
            
            if cached_data:
                self.stats.hits += 1
                self.stats.total_requests += 1
                
                # Deserialize data
                value = self._deserialize_value(cached_data)
                
                response_time = time.time() - start_time
                self._update_avg_response_time(response_time)
                
                return value
            else:
                self.stats.misses += 1
                self.stats.total_requests += 1
                return None
                
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self.stats.errors += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> bool:
        """Set value in Redis cache"""
        if not self.redis_client:
            return False
            
        try:
            prefixed_key = f"{self.config.key_prefix}{key}"
            serialized_value = self._serialize_value(value)
            
            ttl_seconds = int((ttl or self.config.ttl).total_seconds())
            
            result = self.redis_client.setex(
                prefixed_key,
                ttl_seconds,
                serialized_value
            )
            
            return bool(result)
            
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            self.stats.errors += 1
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from Redis cache"""
        if not self.redis_client:
            return False
            
        try:
            prefixed_key = f"{self.config.key_prefix}{key}"
            result = self.redis_client.delete(prefixed_key)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            self.stats.errors += 1
            return False
    
    async def mget(self, keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from Redis"""
        if not self.redis_client or not keys:
            return {}
            
        try:
            prefixed_keys = [f"{self.config.key_prefix}{key}" for key in keys]
            results = self.redis_client.mget(prefixed_keys)
            
            response = {}
            for i, result in enumerate(results):
                if result:
                    response[keys[i]] = self._deserialize_value(result)
                    self.stats.hits += 1
                else:
                    self.stats.misses += 1
                self.stats.total_requests += 1
            
            return response
            
        except Exception as e:
            logger.error(f"Redis mget error: {e}")
            self.stats.errors += 1
            return {}
    
    async def mset(self, data: Dict[str, Any], ttl: Optional[timedelta] = None) -> bool:
        """Set multiple values in Redis"""
        if not self.redis_client or not data:
            return False
            
        try:
            # Use pipeline for atomic operations
            pipe = self.redis_client.pipeline()
            
            ttl_seconds = int((ttl or self.config.ttl).total_seconds())
            
            for key, value in data.items():
                prefixed_key = f"{self.config.key_prefix}{key}"
                serialized_value = self._serialize_value(value)
                pipe.setex(prefixed_key, ttl_seconds, serialized_value)
            
            results = pipe.execute()
            return all(results)
            
        except Exception as e:
            logger.error(f"Redis mset error: {e}")
            self.stats.errors += 1
            return False
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate keys matching pattern"""
        if not self.redis_client:
            return 0
            
        try:
            pattern_key = f"{self.config.key_prefix}{pattern}"
            keys = self.redis_client.keys(pattern_key)
            
            if keys:
                return self.redis_client.delete(*keys)
            return 0
            
        except Exception as e:
            logger.error(f"Redis invalidate pattern error: {e}")
            self.stats.errors += 1
            return 0
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for Redis storage"""
        if self.config.serialization == "json":
            try:
                return json.dumps(value).encode('utf-8')
            except:
                return pickle.dumps(value)
        else:
            return pickle.dumps(value)
    
    def _deserialize_value(self, value: bytes) -> Any:
        """Deserialize value from Redis storage"""
        try:
            if self.config.serialization == "json":
                return json.loads(value.decode('utf-8'))
            else:
                return pickle.loads(value)
        except:
            return pickle.loads(value)
    
    def _update_avg_response_time(self, response_time: float):
        """Update average response time"""
        if self.stats.total_requests > 0:
            self.stats.avg_response_time = (
                (self.stats.avg_response_time * (self.stats.total_requests - 1) + response_time) /
                self.stats.total_requests
            )
        else:
            self.stats.avg_response_time = response_time
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        if self.stats.total_requests > 0:
            self.stats.hit_ratio = self.stats.hits / self.stats.total_requests
        return self.stats


class MultiTierCacheManager:
    """
    Enterprise multi-tier cache manager
    
    Features:
    - Automatic tier promotion/demotion
    - Cache warming strategies
    - Intelligent invalidation
    - Performance monitoring
    - Circuit breaker pattern
    """
    
    def __init__(self, l1_config: CacheConfig, l2_config: CacheConfig):
        self.l1_cache = L1ApplicationCache(l1_config)
        self.l2_cache = L2RedisCache(l2_config)
        self.cache_warming_enabled = True
        self.invalidation_patterns: Dict[str, List[str]] = defaultdict(list)
        self.performance_metrics: Dict[str, Dict] = defaultdict(dict)
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Circuit breaker for L2 cache
        self.l2_circuit_breaker = {
            "failures": 0,
            "failure_threshold": 5,
            "recovery_timeout": timedelta(minutes=1),
            "last_failure_time": None,
            "state": "closed"  # closed, open, half_open
        }
    
    async def get(self, key: str, loader: Optional[Callable] = None) -> Optional[Any]:
        """
        Get value from multi-tier cache with fallback loading
        
        Args:
            key: Cache key
            loader: Function to load data if not in cache
            
        Returns:
            Cached or loaded value
        """
        start_time = time.time()
        
        # Try L1 cache first
        value = self.l1_cache.get(key)
        if value is not None:
            logger.debug(f"Cache hit L1: {key}")
            return self.l1_cache._deserialize_value(value)
        
        # Try L2 cache if L1 miss
        if self._is_l2_available():
            value = await self.l2_cache.get(key)
            if value is not None:
                logger.debug(f"Cache hit L2: {key}")
                
                # Promote to L1 cache
                self.l1_cache.set(key, value)
                
                self._record_performance_metric(key, "l2_hit", time.time() - start_time)
                return value
        
        # Cache miss - use loader if provided
        if loader:
            try:
                value = await self._call_loader(loader, key)
                if value is not None:
                    # Store in both cache tiers
                    await self.set(key, value)
                    
                    self._record_performance_metric(key, "loader", time.time() - start_time)
                    return value
                    
            except Exception as e:
                logger.error(f"Loader failed for key {key}: {e}")
        
        self._record_performance_metric(key, "miss", time.time() - start_time)
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> bool:
        """Set value in all cache tiers"""
        
        success_l1 = self.l1_cache.set(key, value, ttl)
        success_l2 = False
        
        if self._is_l2_available():
            success_l2 = await self.l2_cache.set(key, value, ttl)
            
            if not success_l2:
                self._handle_l2_failure()
        
        return success_l1 or success_l2
    
    async def delete(self, key: str) -> bool:
        """Delete value from all cache tiers"""
        
        success_l1 = self.l1_cache.delete(key)
        success_l2 = False
        
        if self._is_l2_available():
            success_l2 = await self.l2_cache.delete(key)
        
        return success_l1 or success_l2
    
    async def mget(self, keys: List[str], loader: Optional[Callable] = None) -> Dict[str, Any]:
        """Get multiple values with intelligent cache tier optimization"""
        
        results = {}
        l1_misses = []
        
        # Check L1 cache for all keys
        for key in keys:
            value = self.l1_cache.get(key)
            if value is not None:
                results[key] = self.l1_cache._deserialize_value(value)
            else:
                l1_misses.append(key)
        
        # Check L2 cache for L1 misses
        if l1_misses and self._is_l2_available():
            l2_results = await self.l2_cache.mget(l1_misses)
            
            # Promote L2 hits to L1 and track remaining misses
            l2_misses = []
            for key in l1_misses:
                if key in l2_results:
                    value = l2_results[key]
                    results[key] = value
                    self.l1_cache.set(key, value)  # Promote to L1
                else:
                    l2_misses.append(key)
            
            l1_misses = l2_misses
        
        # Use loader for complete misses
        if l1_misses and loader:
            try:
                loaded_data = await self._call_loader_batch(loader, l1_misses)
                
                if loaded_data:
                    # Store loaded data in both cache tiers
                    await self.mset(loaded_data)
                    results.update(loaded_data)
                    
            except Exception as e:
                logger.error(f"Batch loader failed: {e}")
        
        return results
    
    async def mset(self, data: Dict[str, Any], ttl: Optional[timedelta] = None) -> bool:
        """Set multiple values in all cache tiers"""
        
        # Set in L1 cache
        l1_success = True
        for key, value in data.items():
            if not self.l1_cache.set(key, value, ttl):
                l1_success = False
        
        # Set in L2 cache
        l2_success = False
        if self._is_l2_available():
            l2_success = await self.l2_cache.mset(data, ttl)
            
            if not l2_success:
                self._handle_l2_failure()
        
        return l1_success or l2_success
    
    async def invalidate(self, key: str):
        """Invalidate key from all cache tiers"""
        await self.delete(key)
        
        # Trigger cascade invalidation if patterns are configured
        await self._cascade_invalidate(key)
    
    async def invalidate_pattern(self, pattern: str):
        """Invalidate keys matching pattern from all tiers"""
        
        # L1 cache pattern invalidation (simplified)
        keys_to_remove = []
        for cache_key in self.l1_cache.cache.keys():
            if pattern in cache_key:
                keys_to_remove.append(cache_key.replace(self.l1_cache.config.key_prefix, ""))
        
        for key in keys_to_remove:
            self.l1_cache.delete(key)
        
        # L2 cache pattern invalidation
        if self._is_l2_available():
            await self.l2_cache.invalidate_pattern(pattern)
    
    def add_invalidation_pattern(self, trigger_key: str, invalidate_patterns: List[str]):
        """Add cascade invalidation patterns"""
        self.invalidation_patterns[trigger_key].extend(invalidate_patterns)
    
    async def _cascade_invalidate(self, key: str):
        """Execute cascade invalidation based on patterns"""
        
        for trigger_pattern, invalidate_patterns in self.invalidation_patterns.items():
            if trigger_pattern in key:
                for pattern in invalidate_patterns:
                    await self.invalidate_pattern(pattern)
                    logger.debug(f"Cascade invalidated pattern: {pattern}")
    
    async def warm_cache(self, keys: List[str], loader: Callable, 
                        batch_size: int = 100, delay: float = 0.1):
        """
        Warm cache with specified keys
        
        Args:
            keys: Keys to warm
            loader: Function to load data
            batch_size: Number of keys to process in each batch
            delay: Delay between batches to avoid overwhelming the system
        """
        
        if not self.cache_warming_enabled:
            return
        
        logger.info(f"Starting cache warming for {len(keys)} keys")
        
        # Process keys in batches
        for i in range(0, len(keys), batch_size):
            batch_keys = keys[i:i + batch_size]
            
            try:
                # Check which keys are not in cache
                cache_results = await self.mget(batch_keys)
                missing_keys = [key for key in batch_keys if key not in cache_results]
                
                if missing_keys:
                    # Load missing data
                    loaded_data = await self._call_loader_batch(loader, missing_keys)
                    
                    if loaded_data:
                        await self.mset(loaded_data)
                        logger.debug(f"Warmed {len(loaded_data)} cache entries")
                
                # Delay between batches
                if delay > 0:
                    await asyncio.sleep(delay)
                    
            except Exception as e:
                logger.error(f"Cache warming batch failed: {e}")
        
        logger.info("Cache warming completed")
    
    def _is_l2_available(self) -> bool:
        """Check if L2 cache is available based on circuit breaker"""
        
        breaker = self.l2_circuit_breaker
        
        if breaker["state"] == "open":
            # Check if recovery timeout has passed
            if (breaker["last_failure_time"] and 
                datetime.now() - breaker["last_failure_time"] > breaker["recovery_timeout"]):
                breaker["state"] = "half_open"
                logger.info("L2 cache circuit breaker: half_open")
            else:
                return False
        
        return True
    
    def _handle_l2_failure(self):
        """Handle L2 cache failure for circuit breaker"""
        
        breaker = self.l2_circuit_breaker
        breaker["failures"] += 1
        breaker["last_failure_time"] = datetime.now()
        
        if breaker["failures"] >= breaker["failure_threshold"]:
            breaker["state"] = "open"
            logger.warning("L2 cache circuit breaker: OPEN")
    
    def _handle_l2_success(self):
        """Handle L2 cache success for circuit breaker"""
        
        breaker = self.l2_circuit_breaker
        
        if breaker["state"] == "half_open":
            breaker["state"] = "closed"
            breaker["failures"] = 0
            logger.info("L2 cache circuit breaker: CLOSED")
    
    async def _call_loader(self, loader: Callable, key: str) -> Any:
        """Call loader function with error handling"""
        
        if asyncio.iscoroutinefunction(loader):
            return await loader(key)
        else:
            # Run sync function in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.executor, loader, key)
    
    async def _call_loader_batch(self, loader: Callable, keys: List[str]) -> Dict[str, Any]:
        """Call batch loader function with error handling"""
        
        if asyncio.iscoroutinefunction(loader):
            return await loader(keys)
        else:
            # Run sync function in executor
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.executor, loader, keys)
    
    def _record_performance_metric(self, key: str, metric_type: str, response_time: float):
        """Record performance metrics for analysis"""
        
        if key not in self.performance_metrics:
            self.performance_metrics[key] = {
                "l1_hits": 0,
                "l2_hits": 0,
                "misses": 0,
                "total_requests": 0,
                "avg_response_time": 0.0
            }
        
        metrics = self.performance_metrics[key]
        metrics["total_requests"] += 1
        
        if metric_type == "l1_hit":
            metrics["l1_hits"] += 1
        elif metric_type == "l2_hit":
            metrics["l2_hits"] += 1
        elif metric_type == "miss":
            metrics["misses"] += 1
        
        # Update average response time
        metrics["avg_response_time"] = (
            (metrics["avg_response_time"] * (metrics["total_requests"] - 1) + response_time) /
            metrics["total_requests"]
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache performance statistics"""
        
        l1_stats = self.l1_cache.get_stats()
        l2_stats = self.l2_cache.get_stats()
        
        # Aggregate stats
        total_hits = l1_stats.hits + l2_stats.hits
        total_misses = l1_stats.misses + l2_stats.misses
        total_requests = total_hits + total_misses
        
        return {
            "l1_cache": {
                "hits": l1_stats.hits,
                "misses": l1_stats.misses,
                "hit_ratio": l1_stats.hit_ratio,
                "memory_usage": l1_stats.memory_usage,
                "avg_response_time": l1_stats.avg_response_time,
                "evictions": l1_stats.evictions
            },
            "l2_cache": {
                "hits": l2_stats.hits,
                "misses": l2_stats.misses,
                "hit_ratio": l2_stats.hit_ratio,
                "avg_response_time": l2_stats.avg_response_time,
                "errors": l2_stats.errors,
                "circuit_breaker_state": self.l2_circuit_breaker["state"]
            },
            "overall": {
                "total_hits": total_hits,
                "total_misses": total_misses,
                "total_requests": total_requests,
                "overall_hit_ratio": total_hits / total_requests if total_requests > 0 else 0,
                "cache_efficiency": self._calculate_cache_efficiency()
            }
        }
    
    def _calculate_cache_efficiency(self) -> float:
        """Calculate overall cache efficiency score"""
        
        l1_stats = self.l1_cache.get_stats()
        l2_stats = self.l2_cache.get_stats()
        
        # Weight L1 hits more heavily than L2 hits
        l1_weight = 1.0
        l2_weight = 0.7
        
        total_weighted_hits = (l1_stats.hits * l1_weight) + (l2_stats.hits * l2_weight)
        total_requests = l1_stats.total_requests + l2_stats.total_requests
        
        if total_requests > 0:
            return total_weighted_hits / total_requests
        return 0.0


# Cache configuration presets
CACHE_CONFIGS = {
    "l1_high_performance": CacheConfig(
        level=CacheLevel.L1_APPLICATION,
        strategy=CacheStrategy.LRU,
        ttl=timedelta(minutes=30),
        max_size=10000,
        compression=False,
        serialization="pickle",
        key_prefix="l1:"
    ),
    
    "l2_distributed": CacheConfig(
        level=CacheLevel.L2_REDIS,
        strategy=CacheStrategy.TTL,
        ttl=timedelta(hours=2),
        max_size=100000,
        compression=True,
        serialization="json",
        key_prefix="l2:",
        host="localhost",
        port=6379
    )
}


class CacheWarmer:
    """
    Intelligent cache warming service
    """
    
    def __init__(self, cache_manager: MultiTierCacheManager):
        self.cache_manager = cache_manager
        self.warming_strategies: List[Dict] = []
        
    def add_warming_strategy(self, name: str, loader: Callable, 
                           key_generator: Callable, interval: timedelta,
                           priority: int = 5):
        """Add a cache warming strategy"""
        
        strategy = {
            "name": name,
            "loader": loader,
            "key_generator": key_generator,
            "interval": interval,
            "priority": priority,
            "last_run": None
        }
        
        self.warming_strategies.append(strategy)
        logger.info(f"Added cache warming strategy: {name}")
    
    async def run_warming_cycle(self):
        """Run cache warming cycle for all strategies"""
        
        current_time = datetime.now()
        
        # Sort strategies by priority
        sorted_strategies = sorted(self.warming_strategies, key=lambda x: x["priority"], reverse=True)
        
        for strategy in sorted_strategies:
            last_run = strategy["last_run"]
            
            # Check if strategy should run
            if (not last_run or 
                current_time - last_run >= strategy["interval"]):
                
                try:
                    # Generate keys to warm
                    keys = await self._call_key_generator(strategy["key_generator"])
                    
                    if keys:
                        await self.cache_manager.warm_cache(
                            keys=keys,
                            loader=strategy["loader"],
                            batch_size=50,
                            delay=0.05
                        )
                    
                    strategy["last_run"] = current_time
                    logger.info(f"Completed warming strategy: {strategy['name']}")
                    
                except Exception as e:
                    logger.error(f"Cache warming strategy failed {strategy['name']}: {e}")
    
    async def _call_key_generator(self, key_generator: Callable) -> List[str]:
        """Call key generator function"""
        
        if asyncio.iscoroutinefunction(key_generator):
            return await key_generator()
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, key_generator)


async def setup_enterprise_caching(redis_host: str = "localhost", 
                                 redis_port: int = 6379,
                                 redis_password: str = None) -> MultiTierCacheManager:
    """
    Set up enterprise multi-tier caching
    
    Args:
        redis_host: Redis host
        redis_port: Redis port
        redis_password: Redis password
        
    Returns:
        MultiTierCacheManager: Configured cache manager
    """
    
    # Configure L1 cache
    l1_config = CACHE_CONFIGS["l1_high_performance"]
    
    # Configure L2 cache
    l2_config = CACHE_CONFIGS["l2_distributed"]
    l2_config.host = redis_host
    l2_config.port = redis_port
    l2_config.password = redis_password
    
    # Create cache manager
    cache_manager = MultiTierCacheManager(l1_config, l2_config)
    
    # Set up common invalidation patterns
    cache_manager.add_invalidation_pattern("documents", ["doc:*", "analysis:*"])
    cache_manager.add_invalidation_pattern("users", ["user:*", "auth:*"])
    cache_manager.add_invalidation_pattern("settings", ["config:*"])
    
    logger.info("Enterprise caching setup completed")
    return cache_manager


# Background task for cache maintenance
async def cache_maintenance_task(cache_manager: MultiTierCacheManager, 
                               cache_warmer: CacheWarmer):
    """
    Background task for cache maintenance and warming
    """
    
    while True:
        try:
            # Run cache warming cycle
            await cache_warmer.run_warming_cycle()
            
            # Clean up expired entries in L1 cache
            cache_manager.l1_cache._evict_expired()
            
            # Log performance statistics
            stats = cache_manager.get_performance_stats()
            logger.info(f"Cache stats - Overall hit ratio: {stats['overall']['overall_hit_ratio']:.2f}")
            
            # Sleep for 5 minutes before next maintenance cycle
            await asyncio.sleep(300)
            
        except Exception as e:
            logger.error(f"Cache maintenance failed: {e}")
            await asyncio.sleep(60)  # Wait 1 minute on error