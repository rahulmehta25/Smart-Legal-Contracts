"""
Database Module for Arbitration Detection RAG System
Provides optimized database access with connection pooling and caching
"""

import os
import logging
from typing import Optional, Dict, Any
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool, StaticPool

from .models import Base, DatabaseOptimizer
from .repository import ArbitrationRepository, DatabaseConfig
from .vector_store import VectorStoreManager, VectorStoreConfig, create_vector_store_config
from .cache import RedisCache, CacheConfig, create_cache

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Centralized database manager with optimized configuration"""
    
    def __init__(
        self,
        database_url: Optional[str] = None,
        vector_store_type: str = "chroma",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        enable_cache: bool = True,
        pool_size: int = 20,
        max_overflow: int = 30
    ):
        # Database configuration
        self.database_url = database_url or os.getenv(
            "DATABASE_URL", 
            "sqlite:///./arbitration_rag.db"
        )
        
        # Create optimized database configuration
        self.db_config = DatabaseConfig(
            database_url=self.database_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=30,
            pool_recycle=3600,
            echo=os.getenv("DB_ECHO", "false").lower() == "true",
            enable_wal_mode=True,
            cache_size=10000
        )
        
        # Vector store configuration
        self.vector_config = create_vector_store_config(
            store_type=vector_store_type,
            embedding_dimension=int(os.getenv("EMBEDDING_DIMENSION", "1536")),
            persist_directory=os.getenv("VECTOR_DB_PATH", "./vector_db")
        )
        
        # Cache configuration
        self.cache_config = None
        self.cache = None
        if enable_cache:
            try:
                self.cache_config = CacheConfig(
                    host=redis_host,
                    port=redis_port,
                    password=os.getenv("REDIS_PASSWORD"),
                    max_connections=100,
                    default_ttl=3600
                )
                self.cache = RedisCache(self.cache_config)
                logger.info("Redis cache enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize Redis cache: {e}")
                self.cache = None
        
        # Initialize repository
        self.repository = ArbitrationRepository(self.db_config, self.vector_config)
        
        logger.info("Database manager initialized successfully")
    
    def get_repository(self) -> ArbitrationRepository:
        """Get the main repository instance"""
        return self.repository
    
    def get_cache(self) -> Optional[RedisCache]:
        """Get the cache instance if available"""
        return self.cache
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        stats = {
            'database': {
                'url': self.database_url,
                'pool_size': self.db_config.pool_size,
                'max_overflow': self.db_config.max_overflow
            },
            'vector_store': self.repository.vector_store.get_stats(),
            'repository_stats': self.repository.get_document_statistics()
        }
        
        if self.cache:
            stats['cache'] = {
                'enabled': True,
                'info': self.cache.get_cache_info(),
                'hit_rate': self.cache.get_hit_rate()
            }
        else:
            stats['cache'] = {'enabled': False}
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components"""
        health = {
            'database': {'status': 'unknown'},
            'vector_store': {'status': 'unknown'},
            'cache': {'status': 'unknown'}
        }
        
        # Database health check
        try:
            with self.repository.get_session() as session:
                session.execute("SELECT 1")
            health['database'] = {'status': 'healthy'}
        except Exception as e:
            health['database'] = {'status': 'unhealthy', 'error': str(e)}
        
        # Vector store health check
        try:
            count = self.repository.vector_store.store.get_embedding_count()
            health['vector_store'] = {
                'status': 'healthy', 
                'embedding_count': count
            }
        except Exception as e:
            health['vector_store'] = {'status': 'unhealthy', 'error': str(e)}
        
        # Cache health check
        if self.cache:
            try:
                self.cache._health_check()
                health['cache'] = {'status': 'healthy'}
            except Exception as e:
                health['cache'] = {'status': 'unhealthy', 'error': str(e)}
        else:
            health['cache'] = {'status': 'disabled'}
        
        return health
    
    def optimize(self) -> bool:
        """Run optimization operations"""
        try:
            # Database optimization
            success = self.repository.optimize_database()
            
            # Cache cleanup
            if self.cache:
                self.cache.cleanup_expired_keys()
            
            logger.info("Database optimization completed")
            return success
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return False
    
    def close(self):
        """Clean up all resources"""
        try:
            self.repository.close()
            if self.cache:
                self.cache.close()
            logger.info("Database manager closed")
        except Exception as e:
            logger.error(f"Error closing database manager: {e}")


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def initialize_database(
    database_url: Optional[str] = None,
    vector_store_type: str = "chroma",
    redis_host: str = "localhost",
    redis_port: int = 6379,
    enable_cache: bool = True,
    pool_size: int = 20,
    max_overflow: int = 30
) -> DatabaseManager:
    """Initialize global database manager"""
    global _db_manager
    
    if _db_manager is not None:
        logger.warning("Database already initialized, returning existing instance")
        return _db_manager
    
    _db_manager = DatabaseManager(
        database_url=database_url,
        vector_store_type=vector_store_type,
        redis_host=redis_host,
        redis_port=redis_port,
        enable_cache=enable_cache,
        pool_size=pool_size,
        max_overflow=max_overflow
    )
    
    logger.info("Global database manager initialized")
    return _db_manager


def get_database() -> DatabaseManager:
    """Get global database manager instance"""
    global _db_manager
    
    if _db_manager is None:
        logger.info("Database not initialized, using default configuration")
        _db_manager = initialize_database()
    
    return _db_manager


def get_repository() -> ArbitrationRepository:
    """Get repository instance"""
    return get_database().get_repository()


def get_cache() -> Optional[RedisCache]:
    """Get cache instance"""
    return get_database().get_cache()


def close_database():
    """Close global database manager"""
    global _db_manager
    
    if _db_manager is not None:
        _db_manager.close()
        _db_manager = None
        logger.info("Global database manager closed")


# Context manager for database operations
class DatabaseSession:
    """Context manager for database sessions with automatic cleanup"""
    
    def __init__(self, repository: Optional[ArbitrationRepository] = None):
        self.repository = repository or get_repository()
        self.session = None
    
    def __enter__(self):
        self.session = self.repository.get_session()
        return self.session.__enter__()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            return self.session.__exit__(exc_type, exc_val, exc_tb)


# Performance monitoring decorator
def monitor_performance(operation_type: str):
    """Decorator to monitor database operation performance"""
    import time
    import functools
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            error_count = 0
            record_count = kwargs.get('record_count', 0)
            
            try:
                result = func(*args, **kwargs)
                
                # Try to determine record count from result
                if hasattr(result, '__len__'):
                    record_count = len(result)
                elif isinstance(result, bool):
                    record_count = 1 if result else 0
                
                return result
                
            except Exception as e:
                error_count = 1
                raise
                
            finally:
                duration_ms = int((time.time() - start_time) * 1000)
                
                # Log performance metric
                try:
                    repo = get_repository()
                    with repo.get_session() as session:
                        session.execute("""
                            INSERT INTO performance_metrics 
                            (operation_type, duration_ms, record_count, error_count)
                            VALUES (?, ?, ?, ?)
                        """, (operation_type, duration_ms, record_count, error_count))
                except:
                    # Don't fail the original operation if logging fails
                    pass
                
                if duration_ms > 1000:  # Log slow operations
                    logger.warning(f"Slow operation {operation_type}: {duration_ms}ms")
        
        return wrapper
    return decorator


# Configuration validation
def validate_database_config() -> Dict[str, Any]:
    """Validate database configuration and return recommendations"""
    recommendations = {
        'status': 'healthy',
        'warnings': [],
        'recommendations': []
    }
    
    # Check database URL
    database_url = os.getenv("DATABASE_URL", "sqlite:///./arbitration_rag.db")
    
    if database_url.startswith('sqlite'):
        recommendations['warnings'].append("Using SQLite - consider PostgreSQL for production")
        
        # Check if WAL mode is enabled
        if not os.path.exists(database_url.replace('sqlite:///', '') + '-wal'):
            recommendations['recommendations'].append("Enable WAL mode for better concurrency")
    
    # Check vector store configuration
    vector_store_type = os.getenv("VECTOR_STORE_TYPE", "chroma")
    if vector_store_type == "memory":
        recommendations['warnings'].append("Using memory vector store - data will be lost on restart")
    
    # Check Redis configuration
    redis_host = os.getenv("REDIS_HOST", "localhost")
    if redis_host == "localhost":
        recommendations['recommendations'].append("Configure Redis for production deployment")
    
    # Check pool size configuration
    pool_size = int(os.getenv("DB_POOL_SIZE", "20"))
    if pool_size < 10:
        recommendations['recommendations'].append("Consider increasing database pool size for better performance")
    
    return recommendations


# Export main components
__all__ = [
    'initialize_database',
    'get_database',
    'get_repository', 
    'get_cache',
    'close_database',
    'DatabaseManager',
    'DatabaseSession',
    'monitor_performance',
    'validate_database_config',
    'ArbitrationRepository',
    'RedisCache',
    'VectorStoreManager'
]