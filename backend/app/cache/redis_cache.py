"""
Redis caching layer for expensive AI operations
"""
import os
import json
import hashlib
import pickle
from typing import Any, Optional, Callable, Union
from functools import wraps
from datetime import timedelta
import redis
from loguru import logger

from app.core.metrics import CACHE_HITS, CACHE_MISSES


class RedisCache:
    """Redis-based caching for AI operations"""
    
    # Default TTLs for different cache types (in seconds)
    TTL_EMBEDDINGS = 86400 * 7    # 7 days
    TTL_ANALYSIS = 86400 * 1      # 1 day
    TTL_SEARCH = 3600             # 1 hour
    TTL_DEFAULT = 3600            # 1 hour
    
    def __init__(self, redis_url: Optional[str] = None):
        """Initialize Redis connection"""
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self._client: Optional[redis.Redis] = None
        self._prefix = os.getenv('CACHE_PREFIX', 'slc')
    
    @property
    def client(self) -> redis.Redis:
        """Lazy initialization of Redis client"""
        if self._client is None:
            self._client = redis.from_url(
                self.redis_url,
                decode_responses=False,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
            )
        return self._client
    
    def _make_key(self, key_type: str, identifier: str) -> str:
        """Generate a cache key with prefix"""
        return f"{self._prefix}:{key_type}:{identifier}"
    
    def _hash_content(self, content: Union[str, bytes, dict]) -> str:
        """Generate a hash for content-based cache keys"""
        if isinstance(content, dict):
            content = json.dumps(content, sort_keys=True)
        if isinstance(content, str):
            content = content.encode('utf-8')
        return hashlib.sha256(content).hexdigest()[:16]
    
    def get(self, key_type: str, identifier: str) -> Optional[Any]:
        """
        Get a value from cache.
        
        Args:
            key_type: Type of cached data (embeddings, analysis, etc.)
            identifier: Unique identifier for the cached item
        
        Returns:
            Cached value or None if not found
        """
        try:
            key = self._make_key(key_type, identifier)
            data = self.client.get(key)
            
            if data is not None:
                CACHE_HITS.labels(cache_type=key_type).inc()
                return pickle.loads(data)
            
            CACHE_MISSES.labels(cache_type=key_type).inc()
            return None
            
        except redis.RedisError as e:
            logger.warning(f"Redis get error: {e}")
            CACHE_MISSES.labels(cache_type=key_type).inc()
            return None
        except pickle.PickleError as e:
            logger.warning(f"Cache deserialization error: {e}")
            return None
    
    def set(
        self,
        key_type: str,
        identifier: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set a value in cache.
        
        Args:
            key_type: Type of cached data
            identifier: Unique identifier
            value: Value to cache
            ttl: Time to live in seconds (default based on key_type)
        
        Returns:
            True if successful
        """
        try:
            key = self._make_key(key_type, identifier)
            data = pickle.dumps(value)
            
            if ttl is None:
                ttl = self._get_default_ttl(key_type)
            
            self.client.setex(key, ttl, data)
            return True
            
        except redis.RedisError as e:
            logger.warning(f"Redis set error: {e}")
            return False
        except pickle.PickleError as e:
            logger.warning(f"Cache serialization error: {e}")
            return False
    
    def _get_default_ttl(self, key_type: str) -> int:
        """Get default TTL for a cache type"""
        ttl_map = {
            'embeddings': self.TTL_EMBEDDINGS,
            'analysis': self.TTL_ANALYSIS,
            'search': self.TTL_SEARCH,
        }
        return ttl_map.get(key_type, self.TTL_DEFAULT)
    
    def delete(self, key_type: str, identifier: str) -> bool:
        """Delete a cache entry"""
        try:
            key = self._make_key(key_type, identifier)
            self.client.delete(key)
            return True
        except redis.RedisError as e:
            logger.warning(f"Redis delete error: {e}")
            return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching a pattern.
        
        Args:
            pattern: Pattern to match (e.g., "embeddings:doc_*")
        
        Returns:
            Number of keys deleted
        """
        try:
            full_pattern = f"{self._prefix}:{pattern}"
            keys = self.client.keys(full_pattern)
            if keys:
                return self.client.delete(*keys)
            return 0
        except redis.RedisError as e:
            logger.warning(f"Redis invalidate error: {e}")
            return 0
    
    def get_embeddings(self, document_id: str, chunk_hash: str) -> Optional[list]:
        """Get cached embeddings for a document chunk"""
        identifier = f"{document_id}:{chunk_hash}"
        return self.get('embeddings', identifier)
    
    def set_embeddings(
        self,
        document_id: str,
        chunk_hash: str,
        embeddings: list,
        ttl: Optional[int] = None
    ) -> bool:
        """Cache embeddings for a document chunk"""
        identifier = f"{document_id}:{chunk_hash}"
        return self.set('embeddings', identifier, embeddings, ttl or self.TTL_EMBEDDINGS)
    
    def get_analysis(self, document_id: str) -> Optional[dict]:
        """Get cached analysis results"""
        return self.get('analysis', document_id)
    
    def set_analysis(
        self,
        document_id: str,
        analysis: dict,
        ttl: Optional[int] = None
    ) -> bool:
        """Cache analysis results"""
        return self.set('analysis', document_id, analysis, ttl or self.TTL_ANALYSIS)
    
    def get_search_results(self, query_hash: str) -> Optional[list]:
        """Get cached search results"""
        return self.get('search', query_hash)
    
    def set_search_results(
        self,
        query_hash: str,
        results: list,
        ttl: Optional[int] = None
    ) -> bool:
        """Cache search results"""
        return self.set('search', query_hash, results, ttl or self.TTL_SEARCH)
    
    def cleanup_expired(self) -> dict:
        """
        Clean up expired entries (Redis handles this automatically,
        but this can be used to scan and report)
        """
        try:
            info = self.client.info('memory')
            return {
                'count': 0,  # Redis handles expiration
                'bytes': info.get('used_memory', 0),
                'evicted_keys': info.get('evicted_keys', 0),
            }
        except redis.RedisError as e:
            logger.warning(f"Redis info error: {e}")
            return {'count': 0, 'bytes': 0}
    
    def health_check(self) -> bool:
        """Check if Redis is healthy"""
        try:
            return self.client.ping()
        except redis.RedisError:
            return False


# Global cache instance
_cache: Optional[RedisCache] = None


def get_cache() -> RedisCache:
    """Get or create global cache instance"""
    global _cache
    if _cache is None:
        _cache = RedisCache()
    return _cache


def cache_result(
    key_type: str,
    ttl: Optional[int] = None,
    key_builder: Optional[Callable] = None
):
    """
    Decorator to cache function results.
    
    Args:
        key_type: Type of cache (embeddings, analysis, search)
        ttl: Time to live in seconds
        key_builder: Function to build cache key from args/kwargs
    
    Example:
        @cache_result('analysis', ttl=3600)
        async def analyze_document(doc_id: str, content: str):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache = get_cache()
            
            # Build cache key
            if key_builder:
                identifier = key_builder(*args, **kwargs)
            else:
                # Default: hash all arguments
                key_data = json.dumps({'args': args, 'kwargs': kwargs}, sort_keys=True, default=str)
                identifier = hashlib.sha256(key_data.encode()).hexdigest()[:16]
            
            # Try to get from cache
            cached = cache.get(key_type, identifier)
            if cached is not None:
                return cached
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            cache.set(key_type, identifier, result, ttl)
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            cache = get_cache()
            
            if key_builder:
                identifier = key_builder(*args, **kwargs)
            else:
                key_data = json.dumps({'args': args, 'kwargs': kwargs}, sort_keys=True, default=str)
                identifier = hashlib.sha256(key_data.encode()).hexdigest()[:16]
            
            cached = cache.get(key_type, identifier)
            if cached is not None:
                return cached
            
            result = func(*args, **kwargs)
            cache.set(key_type, identifier, result, ttl)
            
            return result
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


def invalidate_cache(key_type: str, identifier: str):
    """Invalidate a specific cache entry"""
    cache = get_cache()
    return cache.delete(key_type, identifier)
