"""
Redis Caching Layer for Arbitration Detection RAG System
High-performance caching for frequently accessed data and query results
"""

import json
import pickle
import hashlib
import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import redis
from redis.exceptions import ConnectionError, TimeoutError
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Redis cache configuration"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    socket_timeout: int = 30
    socket_connect_timeout: int = 30
    max_connections: int = 100
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    # Cache TTL settings (in seconds)
    default_ttl: int = 3600  # 1 hour
    document_ttl: int = 7200  # 2 hours
    chunk_ttl: int = 3600  # 1 hour
    detection_ttl: int = 1800  # 30 minutes
    search_results_ttl: int = 900  # 15 minutes
    analytics_ttl: int = 300  # 5 minutes


class CacheKey:
    """Cache key generator with consistent formatting"""
    
    @staticmethod
    def document(document_id: int) -> str:
        return f"doc:{document_id}"
    
    @staticmethod
    def document_chunks(document_id: int) -> str:
        return f"doc:{document_id}:chunks"
    
    @staticmethod
    def document_detections(document_id: int, detection_type: Optional[str] = None) -> str:
        if detection_type:
            return f"doc:{document_id}:detections:{detection_type}"
        return f"doc:{document_id}:detections"
    
    @staticmethod
    def chunk(chunk_id: int) -> str:
        return f"chunk:{chunk_id}"
    
    @staticmethod
    def chunk_embedding(chunk_id: int) -> str:
        return f"chunk:{chunk_id}:embedding"
    
    @staticmethod
    def detection(detection_id: int) -> str:
        return f"detection:{detection_id}"
    
    @staticmethod
    def pattern(pattern_id: int) -> str:
        return f"pattern:{pattern_id}"
    
    @staticmethod
    def search_results(query_hash: str) -> str:
        return f"search:{query_hash}"
    
    @staticmethod
    def document_stats(document_id: int) -> str:
        return f"stats:doc:{document_id}"
    
    @staticmethod
    def global_stats() -> str:
        return "stats:global"
    
    @staticmethod
    def detection_analytics(start_date: str, end_date: str) -> str:
        return f"analytics:detections:{start_date}:{end_date}"
    
    @staticmethod
    def active_patterns(pattern_type: Optional[str] = None, category: Optional[str] = None) -> str:
        key = "patterns:active"
        if pattern_type:
            key += f":{pattern_type}"
        if category:
            key += f":{category}"
        return key


class RedisCache:
    """Redis-based caching system with optimized serialization"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.client = self._create_client()
        self._health_check()
        
        logger.info("Redis cache initialized successfully")
    
    def _create_client(self) -> redis.Redis:
        """Create Redis client with connection pooling"""
        pool = redis.ConnectionPool(
            host=self.config.host,
            port=self.config.port,
            db=self.config.db,
            password=self.config.password,
            socket_timeout=self.config.socket_timeout,
            socket_connect_timeout=self.config.socket_connect_timeout,
            max_connections=self.config.max_connections,
            retry_on_timeout=self.config.retry_on_timeout,
            health_check_interval=self.config.health_check_interval
        )
        
        return redis.Redis(connection_pool=pool, decode_responses=False)
    
    def _health_check(self) -> bool:
        """Check Redis connection health"""
        try:
            self.client.ping()
            logger.info("Redis health check passed")
            return True
        except (ConnectionError, TimeoutError) as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    def _serialize_data(self, data: Any) -> bytes:
        """Serialize data for Redis storage with optimized format selection"""
        try:
            # Handle different data types optimally
            if isinstance(data, (dict, list)):
                # Use JSON for dict/list (human readable, good compression)
                return json.dumps(data, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
            elif isinstance(data, np.ndarray):
                # Use pickle for numpy arrays (preserves type and shape)
                return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            elif isinstance(data, (str, int, float, bool)):
                # Simple types as JSON
                return json.dumps(data).encode('utf-8')
            else:
                # Default to pickle for complex objects
                return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.error(f"Error serializing data: {e}")
            # Fallback to pickle
            return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _deserialize_data(self, data: bytes, data_type: Optional[str] = None) -> Any:
        """Deserialize data from Redis with type hints"""
        try:
            if data_type == "numpy":
                return pickle.loads(data)
            else:
                # Try JSON first (faster and more common)
                try:
                    return json.loads(data.decode('utf-8'))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # Fallback to pickle
                    return pickle.loads(data)
        except Exception as e:
            logger.error(f"Error deserializing data: {e}")
            return None
    
    @contextmanager
    def pipeline(self):
        """Context manager for Redis pipeline operations"""
        pipe = self.client.pipeline()
        try:
            yield pipe
            pipe.execute()
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            raise
    
    # Basic cache operations
    def get(self, key: str, data_type: Optional[str] = None) -> Optional[Any]:
        """Get value from cache"""
        try:
            data = self.client.get(key)
            if data is None:
                return None
            
            return self._deserialize_data(data, data_type)
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            return None
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None,
        data_type: Optional[str] = None
    ) -> bool:
        """Set value in cache with TTL"""
        try:
            serialized = self._serialize_data(value)
            ttl = ttl or self.config.default_ttl
            
            # Store data type hint for efficient deserialization
            if data_type:
                type_key = f"{key}:type"
                self.client.setex(type_key, ttl, data_type)
            
            return self.client.setex(key, ttl, serialized)
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            # Also delete type hint if exists
            type_key = f"{key}:type"
            with self.pipeline() as pipe:
                pipe.delete(key)
                pipe.delete(type_key)
            return True
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        try:
            return bool(self.client.exists(key))
        except Exception as e:
            logger.error(f"Error checking cache key {key}: {e}")
            return False
    
    def expire(self, key: str, ttl: int) -> bool:
        """Set expiration time for key"""
        try:
            return bool(self.client.expire(key, ttl))
        except Exception as e:
            logger.error(f"Error setting expiration for key {key}: {e}")
            return False
    
    def get_ttl(self, key: str) -> int:
        """Get TTL for key"""
        try:
            return self.client.ttl(key)
        except Exception as e:
            logger.error(f"Error getting TTL for key {key}: {e}")
            return -1
    
    # Hash operations for structured data
    def hget(self, name: str, key: str) -> Optional[Any]:
        """Get hash field value"""
        try:
            data = self.client.hget(name, key)
            if data is None:
                return None
            return self._deserialize_data(data)
        except Exception as e:
            logger.error(f"Error getting hash field {name}:{key}: {e}")
            return None
    
    def hset(self, name: str, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set hash field value"""
        try:
            serialized = self._serialize_data(value)
            result = self.client.hset(name, key, serialized)
            
            if ttl:
                self.client.expire(name, ttl)
            
            return bool(result)
        except Exception as e:
            logger.error(f"Error setting hash field {name}:{key}: {e}")
            return False
    
    def hgetall(self, name: str) -> Dict[str, Any]:
        """Get all hash fields"""
        try:
            data = self.client.hgetall(name)
            result = {}
            for key, value in data.items():
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                result[key_str] = self._deserialize_data(value)
            return result
        except Exception as e:
            logger.error(f"Error getting all hash fields for {name}: {e}")
            return {}
    
    def hdel(self, name: str, *keys: str) -> int:
        """Delete hash fields"""
        try:
            return self.client.hdel(name, *keys)
        except Exception as e:
            logger.error(f"Error deleting hash fields {name}:{keys}: {e}")
            return 0
    
    # Set operations for collections
    def sadd(self, name: str, *values: Any) -> int:
        """Add members to set"""
        try:
            serialized_values = [self._serialize_data(v) for v in values]
            return self.client.sadd(name, *serialized_values)
        except Exception as e:
            logger.error(f"Error adding to set {name}: {e}")
            return 0
    
    def smembers(self, name: str) -> set:
        """Get all set members"""
        try:
            members = self.client.smembers(name)
            return {self._deserialize_data(m) for m in members}
        except Exception as e:
            logger.error(f"Error getting set members {name}: {e}")
            return set()
    
    def srem(self, name: str, *values: Any) -> int:
        """Remove members from set"""
        try:
            serialized_values = [self._serialize_data(v) for v in values]
            return self.client.srem(name, *serialized_values)
        except Exception as e:
            logger.error(f"Error removing from set {name}: {e}")
            return 0
    
    # List operations for ordered data
    def lpush(self, name: str, *values: Any) -> int:
        """Push values to left of list"""
        try:
            serialized_values = [self._serialize_data(v) for v in values]
            return self.client.lpush(name, *serialized_values)
        except Exception as e:
            logger.error(f"Error left pushing to list {name}: {e}")
            return 0
    
    def rpush(self, name: str, *values: Any) -> int:
        """Push values to right of list"""
        try:
            serialized_values = [self._serialize_data(v) for v in values]
            return self.client.rpush(name, *serialized_values)
        except Exception as e:
            logger.error(f"Error right pushing to list {name}: {e}")
            return 0
    
    def lrange(self, name: str, start: int = 0, end: int = -1) -> List[Any]:
        """Get list range"""
        try:
            items = self.client.lrange(name, start, end)
            return [self._deserialize_data(item) for item in items]
        except Exception as e:
            logger.error(f"Error getting list range {name}: {e}")
            return []
    
    def ltrim(self, name: str, start: int, end: int) -> bool:
        """Trim list to specified range"""
        try:
            return bool(self.client.ltrim(name, start, end))
        except Exception as e:
            logger.error(f"Error trimming list {name}: {e}")
            return False
    
    # Cache invalidation patterns
    def invalidate_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern"""
        try:
            keys = self.client.keys(pattern)
            if keys:
                return self.client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Error invalidating pattern {pattern}: {e}")
            return 0
    
    def invalidate_document(self, document_id: int) -> int:
        """Invalidate all cache entries for a document"""
        pattern = f"*doc:{document_id}*"
        return self.invalidate_pattern(pattern)
    
    def invalidate_search_cache(self) -> int:
        """Invalidate all search result caches"""
        pattern = "search:*"
        return self.invalidate_pattern(pattern)
    
    def invalidate_stats_cache(self) -> int:
        """Invalidate all statistics caches"""
        pattern = "stats:*"
        return self.invalidate_pattern(pattern)
    
    # Cache statistics and monitoring
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache statistics and information"""
        try:
            info = self.client.info()
            return {
                'redis_version': info.get('redis_version'),
                'connected_clients': info.get('connected_clients'),
                'used_memory': info.get('used_memory'),
                'used_memory_human': info.get('used_memory_human'),
                'used_memory_peak': info.get('used_memory_peak'),
                'used_memory_peak_human': info.get('used_memory_peak_human'),
                'keyspace_hits': info.get('keyspace_hits'),
                'keyspace_misses': info.get('keyspace_misses'),
                'total_commands_processed': info.get('total_commands_processed'),
                'instantaneous_ops_per_sec': info.get('instantaneous_ops_per_sec')
            }
        except Exception as e:
            logger.error(f"Error getting cache info: {e}")
            return {}
    
    def get_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        try:
            info = self.client.info()
            hits = info.get('keyspace_hits', 0)
            misses = info.get('keyspace_misses', 0)
            total = hits + misses
            
            if total == 0:
                return 0.0
            
            return hits / total
        except Exception as e:
            logger.error(f"Error calculating hit rate: {e}")
            return 0.0
    
    def get_key_count_by_pattern(self, pattern: str) -> int:
        """Count keys matching pattern"""
        try:
            keys = self.client.keys(pattern)
            return len(keys)
        except Exception as e:
            logger.error(f"Error counting keys for pattern {pattern}: {e}")
            return 0
    
    # Bulk operations for better performance
    def mget(self, *keys: str) -> List[Optional[Any]]:
        """Get multiple keys"""
        try:
            values = self.client.mget(*keys)
            return [
                self._deserialize_data(value) if value is not None else None
                for value in values
            ]
        except Exception as e:
            logger.error(f"Error getting multiple keys: {e}")
            return [None] * len(keys)
    
    def mset(self, mapping: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple keys"""
        try:
            serialized_mapping = {
                key: self._serialize_data(value)
                for key, value in mapping.items()
            }
            
            with self.pipeline() as pipe:
                pipe.mset(serialized_mapping)
                if ttl:
                    for key in mapping.keys():
                        pipe.expire(key, ttl)
            
            return True
        except Exception as e:
            logger.error(f"Error setting multiple keys: {e}")
            return False
    
    # Maintenance operations
    def flush_db(self) -> bool:
        """Flush current database"""
        try:
            self.client.flushdb()
            logger.info("Cache database flushed")
            return True
        except Exception as e:
            logger.error(f"Error flushing database: {e}")
            return False
    
    def cleanup_expired_keys(self) -> int:
        """Manual cleanup of expired keys (Redis handles this automatically)"""
        # Redis automatically removes expired keys, but we can force scan for monitoring
        try:
            count = 0
            for key in self.client.scan_iter():
                ttl = self.client.ttl(key)
                if ttl == -2:  # Key expired and deleted
                    count += 1
            
            logger.info(f"Found {count} expired keys during scan")
            return count
        except Exception as e:
            logger.error(f"Error scanning for expired keys: {e}")
            return 0
    
    def close(self):
        """Close Redis connection"""
        try:
            self.client.close()
            logger.info("Redis cache connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")


class CacheDecorator:
    """Decorator for caching function results"""
    
    def __init__(self, cache: RedisCache, ttl: int = 3600, key_prefix: str = "func"):
        self.cache = cache
        self.ttl = ttl
        self.key_prefix = key_prefix
    
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            # Generate cache key from function name and arguments
            key_data = {
                'func': func.__name__,
                'args': args,
                'kwargs': kwargs
            }
            key_hash = hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
            cache_key = f"{self.key_prefix}:{func.__name__}:{key_hash}"
            
            # Try to get from cache
            result = self.cache.get(cache_key)
            if result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return result
            
            # Execute function and cache result
            logger.debug(f"Cache miss for {func.__name__}")
            result = func(*args, **kwargs)
            self.cache.set(cache_key, result, self.ttl)
            
            return result
        
        return wrapper


# Factory function
def create_cache(
    host: str = "localhost",
    port: int = 6379,
    db: int = 0,
    password: Optional[str] = None
) -> RedisCache:
    """Factory function to create Redis cache with default settings"""
    
    config = CacheConfig(
        host=host,
        port=port,
        db=db,
        password=password,
        max_connections=100,
        default_ttl=3600,
        document_ttl=7200,
        search_results_ttl=900
    )
    
    return RedisCache(config)


# Utility functions for common cache operations
def generate_query_hash(query_params: Dict[str, Any]) -> str:
    """Generate consistent hash for query parameters"""
    sorted_params = json.dumps(query_params, sort_keys=True)
    return hashlib.md5(sorted_params.encode()).hexdigest()


def create_cache_key(prefix: str, *args: Any) -> str:
    """Create cache key from prefix and arguments"""
    key_parts = [str(prefix)]
    for arg in args:
        if isinstance(arg, (dict, list)):
            key_parts.append(hashlib.md5(json.dumps(arg, sort_keys=True).encode()).hexdigest())
        else:
            key_parts.append(str(arg))
    
    return ":".join(key_parts)