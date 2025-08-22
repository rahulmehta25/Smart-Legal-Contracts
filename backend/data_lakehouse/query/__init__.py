"""
Query Engines Integration Layer

Unified query layer supporting multiple engines:
- Presto/Trino for federated queries
- Spark SQL for large-scale processing
- Dremio for acceleration and caching
- Apache Drill for schema-free queries
- Query optimization and routing
- Result caching and materialized views
"""

from .query_manager import QueryEngineManager
from .presto_engine import PrestoEngine
from .spark_sql_engine import SparkSQLEngine
from .dremio_engine import DremioEngine
from .query_optimizer import QueryOptimizer
from .caching_layer import QueryCachingLayer

__all__ = [
    "QueryEngineManager",
    "PrestoEngine", 
    "SparkSQLEngine",
    "DremioEngine",
    "QueryOptimizer",
    "QueryCachingLayer"
]