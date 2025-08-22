"""
Enterprise Materialized Views Module

Implements pre-computed aggregations and materialized views for analytics performance.
Provides automatic refresh strategies and intelligent query rewriting.

Target: Support 10,000+ concurrent users with <50ms query response time
"""

from sqlalchemy import (
    create_engine, MetaData, text, Table, Column, Integer, String, DateTime, 
    Float, Boolean, inspect, func
)
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
import json
import time

logger = logging.getLogger(__name__)


class RefreshStrategy(Enum):
    """Materialized view refresh strategies"""
    IMMEDIATE = "immediate"  # Refresh immediately when base data changes
    SCHEDULED = "scheduled"  # Refresh on a schedule
    ON_DEMAND = "on_demand"  # Refresh only when requested
    INCREMENTAL = "incremental"  # Incremental refresh based on changes
    

class ViewType(Enum):
    """Types of materialized views"""
    AGGREGATION = "aggregation"  # SUM, COUNT, AVG aggregations
    ANALYTICS = "analytics"  # Complex analytics queries
    REPORTING = "reporting"  # Pre-built reports
    DENORMALIZED = "denormalized"  # Denormalized data for fast access
    

@dataclass
class MaterializedViewConfig:
    """Configuration for a materialized view"""
    name: str
    base_query: str
    view_type: ViewType
    refresh_strategy: RefreshStrategy
    refresh_interval: Optional[timedelta] = None  # For scheduled refresh
    dependencies: List[str] = field(default_factory=list)  # Base tables
    indexes: List[str] = field(default_factory=list)  # Indexes to create
    ttl: Optional[timedelta] = None  # Time to live for cached results
    partition_column: Optional[str] = None  # For partitioned views
    incremental_column: Optional[str] = None  # For incremental refresh
    metadata: Dict[str, Any] = field(default_factory=dict)
    

class MaterializedViewManager:
    """
    Advanced materialized view management for enterprise analytics
    
    Features:
    - Automatic view creation and management
    - Multiple refresh strategies (immediate, scheduled, incremental)
    - Query rewriting to use materialized views
    - Performance monitoring and optimization
    - Dependency tracking and cascade refresh
    """
    
    def __init__(self, engine, metadata: MetaData):
        self.engine = engine
        self.metadata = metadata
        self.views: Dict[str, MaterializedViewConfig] = {}
        self.refresh_schedule: Dict[str, datetime] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.query_rewrite_rules: List[Dict] = []
        self.performance_stats: Dict[str, Dict] = {}
        
    async def create_materialized_view(self, config: MaterializedViewConfig) -> bool:
        """
        Create a materialized view
        
        Args:
            config: View configuration
            
        Returns:
            bool: Success status
        """
        try:
            with self.engine.connect() as conn:
                # PostgreSQL materialized views
                if self.engine.dialect.name == 'postgresql':
                    return await self._create_postgres_materialized_view(conn, config)
                # MySQL doesn't have native materialized views, use tables
                elif self.engine.dialect.name == 'mysql':
                    return await self._create_mysql_materialized_table(conn, config)
                # SQLite uses regular tables
                else:
                    return await self._create_sqlite_materialized_table(conn, config)
                    
        except Exception as e:
            logger.error(f"Failed to create materialized view {config.name}: {e}")
            return False
    
    async def _create_postgres_materialized_view(self, conn, config: MaterializedViewConfig) -> bool:
        """Create PostgreSQL materialized view"""
        
        # Drop existing view if it exists
        drop_sql = f"DROP MATERIALIZED VIEW IF EXISTS {config.name} CASCADE"
        conn.execute(text(drop_sql))
        
        # Create materialized view
        create_sql = f"""
        CREATE MATERIALIZED VIEW {config.name} AS
        {config.base_query}
        """
        
        # Add WITH DATA clause for immediate population
        if config.refresh_strategy != RefreshStrategy.ON_DEMAND:
            create_sql += " WITH DATA"
        else:
            create_sql += " WITH NO DATA"
            
        conn.execute(text(create_sql))
        
        # Create indexes
        for index_def in config.indexes:
            try:
                conn.execute(text(f"CREATE INDEX CONCURRENTLY {index_def}"))
            except Exception as e:
                logger.warning(f"Index creation failed for {config.name}: {e}")
        
        # Store configuration
        self.views[config.name] = config
        
        # Set up refresh schedule
        if config.refresh_strategy == RefreshStrategy.SCHEDULED and config.refresh_interval:
            self.refresh_schedule[config.name] = datetime.now() + config.refresh_interval
        
        logger.info(f"Created PostgreSQL materialized view: {config.name}")
        return True
    
    async def _create_mysql_materialized_table(self, conn, config: MaterializedViewConfig) -> bool:
        """Create MySQL materialized table (MySQL doesn't have native materialized views)"""
        
        # Create table structure based on the query
        temp_table = f"{config.name}_temp"
        
        # Create temporary table to get structure
        create_temp_sql = f"""
        CREATE TEMPORARY TABLE {temp_table} AS
        {config.base_query}
        LIMIT 0
        """
        conn.execute(text(create_temp_sql))
        
        # Get column information
        columns_sql = f"""
        SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_DEFAULT
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = '{temp_table}'
        ORDER BY ORDINAL_POSITION
        """
        columns = conn.execute(text(columns_sql)).fetchall()
        
        # Drop existing table
        conn.execute(text(f"DROP TABLE IF EXISTS {config.name}"))
        
        # Create materialized table
        column_defs = []
        for col in columns:
            nullable = "NULL" if col.IS_NULLABLE == "YES" else "NOT NULL"
            default = f"DEFAULT {col.COLUMN_DEFAULT}" if col.COLUMN_DEFAULT else ""
            column_defs.append(f"{col.COLUMN_NAME} {col.DATA_TYPE} {nullable} {default}")
        
        create_sql = f"""
        CREATE TABLE {config.name} (
            {', '.join(column_defs)},
            _mv_created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            _mv_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )
        """
        conn.execute(text(create_sql))
        
        # Populate the table
        if config.refresh_strategy != RefreshStrategy.ON_DEMAND:
            await self._refresh_mysql_materialized_table(conn, config)
        
        # Create indexes
        for index_def in config.indexes:
            try:
                conn.execute(text(f"CREATE INDEX {index_def}"))
            except Exception as e:
                logger.warning(f"Index creation failed for {config.name}: {e}")
        
        self.views[config.name] = config
        
        if config.refresh_strategy == RefreshStrategy.SCHEDULED and config.refresh_interval:
            self.refresh_schedule[config.name] = datetime.now() + config.refresh_interval
        
        logger.info(f"Created MySQL materialized table: {config.name}")
        return True
    
    async def _create_sqlite_materialized_table(self, conn, config: MaterializedViewConfig) -> bool:
        """Create SQLite materialized table"""
        
        # Drop existing table
        conn.execute(text(f"DROP TABLE IF EXISTS {config.name}"))
        
        # Create and populate table
        create_sql = f"""
        CREATE TABLE {config.name} AS
        {config.base_query}
        """
        conn.execute(text(create_sql))
        
        # Add metadata columns
        conn.execute(text(f"""
        ALTER TABLE {config.name} 
        ADD COLUMN _mv_created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        """))
        
        # Create indexes
        for index_def in config.indexes:
            try:
                conn.execute(text(f"CREATE INDEX {index_def}"))
            except Exception as e:
                logger.warning(f"Index creation failed for {config.name}: {e}")
        
        self.views[config.name] = config
        
        if config.refresh_strategy == RefreshStrategy.SCHEDULED and config.refresh_interval:
            self.refresh_schedule[config.name] = datetime.now() + config.refresh_interval
        
        logger.info(f"Created SQLite materialized table: {config.name}")
        return True
    
    async def refresh_materialized_view(self, view_name: str, incremental: bool = False) -> bool:
        """
        Refresh a materialized view
        
        Args:
            view_name: Name of the view to refresh
            incremental: Whether to perform incremental refresh
            
        Returns:
            bool: Success status
        """
        if view_name not in self.views:
            logger.error(f"Materialized view {view_name} not found")
            return False
        
        config = self.views[view_name]
        start_time = time.time()
        
        try:
            with self.engine.connect() as conn:
                if self.engine.dialect.name == 'postgresql':
                    success = await self._refresh_postgres_view(conn, config, incremental)
                elif self.engine.dialect.name == 'mysql':
                    success = await self._refresh_mysql_materialized_table(conn, config, incremental)
                else:
                    success = await self._refresh_sqlite_table(conn, config, incremental)
                
                if success:
                    # Update performance stats
                    refresh_time = time.time() - start_time
                    if view_name not in self.performance_stats:
                        self.performance_stats[view_name] = {
                            "refresh_count": 0,
                            "total_refresh_time": 0,
                            "last_refresh": None,
                            "avg_refresh_time": 0
                        }
                    
                    stats = self.performance_stats[view_name]
                    stats["refresh_count"] += 1
                    stats["total_refresh_time"] += refresh_time
                    stats["last_refresh"] = datetime.now()
                    stats["avg_refresh_time"] = stats["total_refresh_time"] / stats["refresh_count"]
                    
                    # Update next scheduled refresh
                    if config.refresh_strategy == RefreshStrategy.SCHEDULED and config.refresh_interval:
                        self.refresh_schedule[view_name] = datetime.now() + config.refresh_interval
                    
                    logger.info(f"Refreshed materialized view {view_name} in {refresh_time:.2f}s")
                
                return success
                
        except Exception as e:
            logger.error(f"Failed to refresh materialized view {view_name}: {e}")
            return False
    
    async def _refresh_postgres_view(self, conn, config: MaterializedViewConfig, incremental: bool) -> bool:
        """Refresh PostgreSQL materialized view"""
        
        if incremental and config.incremental_column:
            # Get last refresh timestamp
            last_refresh_sql = f"""
            SELECT MAX({config.incremental_column}) as last_value
            FROM {config.name}
            """
            result = conn.execute(text(last_refresh_sql)).fetchone()
            last_value = result.last_value if result and result.last_value else datetime.min
            
            # Create incremental query
            incremental_query = f"""
            INSERT INTO {config.name}
            {config.base_query}
            WHERE {config.incremental_column} > '{last_value}'
            """
            conn.execute(text(incremental_query))
        else:
            # Full refresh
            refresh_sql = f"REFRESH MATERIALIZED VIEW CONCURRENTLY {config.name}"
            try:
                conn.execute(text(refresh_sql))
            except Exception:
                # Fall back to non-concurrent refresh
                refresh_sql = f"REFRESH MATERIALIZED VIEW {config.name}"
                conn.execute(text(refresh_sql))
        
        return True
    
    async def _refresh_mysql_materialized_table(self, conn, config: MaterializedViewConfig, incremental: bool = False) -> bool:
        """Refresh MySQL materialized table"""
        
        if incremental and config.incremental_column:
            # Get last refresh timestamp
            last_refresh_sql = f"""
            SELECT MAX({config.incremental_column}) as last_value
            FROM {config.name}
            """
            result = conn.execute(text(last_refresh_sql)).fetchone()
            last_value = result.last_value if result and result.last_value else datetime.min
            
            # Insert new records
            incremental_query = f"""
            INSERT INTO {config.name}
            {config.base_query}
            WHERE {config.incremental_column} > '{last_value}'
            """
            conn.execute(text(incremental_query))
        else:
            # Full refresh - truncate and repopulate
            conn.execute(text(f"TRUNCATE TABLE {config.name}"))
            
            insert_sql = f"""
            INSERT INTO {config.name}
            {config.base_query}
            """
            conn.execute(text(insert_sql))
        
        return True
    
    async def _refresh_sqlite_table(self, conn, config: MaterializedViewConfig, incremental: bool = False) -> bool:
        """Refresh SQLite materialized table"""
        
        if incremental and config.incremental_column:
            # Get last refresh timestamp
            last_refresh_sql = f"""
            SELECT MAX({config.incremental_column}) as last_value
            FROM {config.name}
            """
            result = conn.execute(text(last_refresh_sql)).fetchone()
            last_value = result.last_value if result and result.last_value else datetime.min
            
            # Insert new records
            incremental_query = f"""
            INSERT INTO {config.name}
            {config.base_query}
            WHERE {config.incremental_column} > '{last_value}'
            """
            conn.execute(text(incremental_query))
        else:
            # Full refresh - drop and recreate
            conn.execute(text(f"DROP TABLE IF EXISTS {config.name}"))
            
            create_sql = f"""
            CREATE TABLE {config.name} AS
            {config.base_query}
            """
            conn.execute(text(create_sql))
            
            # Recreate indexes
            for index_def in config.indexes:
                try:
                    conn.execute(text(f"CREATE INDEX {index_def}"))
                except Exception as e:
                    logger.warning(f"Index recreation failed: {e}")
        
        return True
    
    async def drop_materialized_view(self, view_name: str) -> bool:
        """
        Drop a materialized view
        
        Args:
            view_name: Name of the view to drop
            
        Returns:
            bool: Success status
        """
        try:
            with self.engine.connect() as conn:
                if self.engine.dialect.name == 'postgresql':
                    drop_sql = f"DROP MATERIALIZED VIEW IF EXISTS {view_name} CASCADE"
                else:
                    drop_sql = f"DROP TABLE IF EXISTS {view_name}"
                
                conn.execute(text(drop_sql))
                
                # Clean up configuration
                if view_name in self.views:
                    del self.views[view_name]
                if view_name in self.refresh_schedule:
                    del self.refresh_schedule[view_name]
                if view_name in self.performance_stats:
                    del self.performance_stats[view_name]
                
                logger.info(f"Dropped materialized view: {view_name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to drop materialized view {view_name}: {e}")
            return False
    
    def add_query_rewrite_rule(self, pattern: str, replacement: str, view_name: str):
        """
        Add a query rewrite rule to automatically use materialized views
        
        Args:
            pattern: Regex pattern to match in queries
            replacement: Replacement pattern
            view_name: Materialized view to use
        """
        rule = {
            "pattern": pattern,
            "replacement": replacement,
            "view_name": view_name,
            "compiled_pattern": re.compile(pattern, re.IGNORECASE)
        }
        self.query_rewrite_rules.append(rule)
        logger.info(f"Added query rewrite rule for view {view_name}")
    
    def rewrite_query(self, original_query: str) -> str:
        """
        Rewrite a query to use materialized views when possible
        
        Args:
            original_query: Original SQL query
            
        Returns:
            str: Rewritten query (or original if no rewrite possible)
        """
        rewritten_query = original_query
        
        for rule in self.query_rewrite_rules:
            if rule["compiled_pattern"].search(rewritten_query):
                # Check if the materialized view exists and is current
                view_name = rule["view_name"]
                if view_name in self.views:
                    rewritten_query = rule["compiled_pattern"].sub(
                        rule["replacement"], rewritten_query
                    )
                    logger.debug(f"Rewrote query to use materialized view: {view_name}")
                    break
        
        return rewritten_query
    
    async def auto_refresh_views(self):
        """
        Automatically refresh views based on their refresh strategy
        """
        current_time = datetime.now()
        
        for view_name, next_refresh in self.refresh_schedule.items():
            if current_time >= next_refresh:
                config = self.views[view_name]
                
                # Determine if incremental refresh is possible
                incremental = (
                    config.refresh_strategy == RefreshStrategy.INCREMENTAL and
                    config.incremental_column is not None
                )
                
                success = await self.refresh_materialized_view(view_name, incremental)
                if not success:
                    logger.error(f"Scheduled refresh failed for view: {view_name}")
    
    async def cascade_refresh(self, base_table: str):
        """
        Refresh all materialized views that depend on a base table
        
        Args:
            base_table: Name of the base table that changed
        """
        dependent_views = []
        
        for view_name, config in self.views.items():
            if base_table in config.dependencies:
                if config.refresh_strategy == RefreshStrategy.IMMEDIATE:
                    dependent_views.append(view_name)
        
        # Refresh dependent views
        for view_name in dependent_views:
            await self.refresh_materialized_view(view_name)
            logger.info(f"Cascaded refresh for view {view_name} due to change in {base_table}")
    
    async def optimize_view_performance(self, view_name: str) -> Dict[str, Any]:
        """
        Analyze and optimize performance of a materialized view
        
        Args:
            view_name: Name of the view to optimize
            
        Returns:
            Dict: Optimization recommendations
        """
        if view_name not in self.views:
            return {"error": f"View {view_name} not found"}
        
        recommendations = {
            "view_name": view_name,
            "recommendations": [],
            "current_stats": self.performance_stats.get(view_name, {}),
            "size_info": {},
            "index_usage": {}
        }
        
        try:
            with self.engine.connect() as conn:
                # Get view size information
                if self.engine.dialect.name == 'postgresql':
                    size_sql = f"""
                    SELECT 
                        pg_size_pretty(pg_total_relation_size('{view_name}')) as total_size,
                        pg_size_pretty(pg_relation_size('{view_name}')) as table_size,
                        (SELECT COUNT(*) FROM {view_name}) as row_count
                    """
                    size_result = conn.execute(text(size_sql)).fetchone()
                    recommendations["size_info"] = {
                        "total_size": size_result.total_size,
                        "table_size": size_result.table_size,
                        "row_count": size_result.row_count
                    }
                    
                    # Check index usage
                    index_usage_sql = f"""
                    SELECT 
                        indexname,
                        idx_scan,
                        idx_tup_read,
                        idx_tup_fetch
                    FROM pg_stat_user_indexes 
                    WHERE relname = '{view_name}'
                    """
                    index_results = conn.execute(text(index_usage_sql)).fetchall()
                    recommendations["index_usage"] = [
                        {
                            "name": row.indexname,
                            "scans": row.idx_scan,
                            "tuples_read": row.idx_tup_read,
                            "tuples_fetched": row.idx_tup_fetch
                        }
                        for row in index_results
                    ]
                
                # Performance recommendations
                config = self.views[view_name]
                stats = self.performance_stats.get(view_name, {})
                
                if stats.get("avg_refresh_time", 0) > 60:  # > 1 minute
                    recommendations["recommendations"].append(
                        "Consider incremental refresh to reduce refresh time"
                    )
                
                if config.refresh_strategy == RefreshStrategy.IMMEDIATE:
                    recommendations["recommendations"].append(
                        "Consider changing to scheduled refresh for better performance"
                    )
                
                if not config.indexes:
                    recommendations["recommendations"].append(
                        "Consider adding indexes to improve query performance"
                    )
                
                # Check for unused indexes
                for index_info in recommendations.get("index_usage", []):
                    if index_info["scans"] == 0:
                        recommendations["recommendations"].append(
                            f"Consider dropping unused index: {index_info['name']}"
                        )
                
                return recommendations
                
        except Exception as e:
            logger.error(f"Failed to analyze view performance for {view_name}: {e}")
            return {"error": str(e)}
    
    def get_view_dependencies(self, view_name: str) -> List[str]:
        """
        Get the dependency chain for a materialized view
        
        Args:
            view_name: Name of the view
            
        Returns:
            List[str]: List of dependent views
        """
        if view_name not in self.views:
            return []
        
        dependencies = []
        for other_view, config in self.views.items():
            if view_name in config.dependencies:
                dependencies.append(other_view)
                # Recursively find dependencies
                dependencies.extend(self.get_view_dependencies(other_view))
        
        return list(set(dependencies))  # Remove duplicates
    
    async def validate_view_freshness(self, view_name: str, max_age: timedelta) -> bool:
        """
        Check if a materialized view is fresh enough
        
        Args:
            view_name: Name of the view
            max_age: Maximum acceptable age
            
        Returns:
            bool: True if view is fresh, False otherwise
        """
        if view_name not in self.performance_stats:
            return False
        
        stats = self.performance_stats[view_name]
        last_refresh = stats.get("last_refresh")
        
        if not last_refresh:
            return False
        
        age = datetime.now() - last_refresh
        return age <= max_age


class QueryOptimizer:
    """
    Query optimizer that automatically uses materialized views
    """
    
    def __init__(self, view_manager: MaterializedViewManager):
        self.view_manager = view_manager
        
    async def optimize_query(self, query: str, context: Dict = None) -> Dict[str, Any]:
        """
        Optimize a query by potentially rewriting it to use materialized views
        
        Args:
            query: Original SQL query
            context: Additional context for optimization
            
        Returns:
            Dict: Optimization result with rewritten query and metadata
        """
        original_query = query
        rewritten_query = self.view_manager.rewrite_query(query)
        
        result = {
            "original_query": original_query,
            "optimized_query": rewritten_query,
            "used_materialized_view": rewritten_query != original_query,
            "recommendations": []
        }
        
        # Analyze query for potential materialized view opportunities
        if not result["used_materialized_view"]:
            recommendations = await self._analyze_query_for_materialization(query)
            result["recommendations"] = recommendations
        
        return result
    
    async def _analyze_query_for_materialization(self, query: str) -> List[str]:
        """
        Analyze a query to suggest materialized view opportunities
        
        Args:
            query: SQL query to analyze
            
        Returns:
            List[str]: List of recommendations
        """
        recommendations = []
        query_upper = query.upper()
        
        # Check for aggregation functions
        aggregation_functions = ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN("]
        has_aggregation = any(func in query_upper for func in aggregation_functions)
        
        # Check for GROUP BY
        has_group_by = "GROUP BY" in query_upper
        
        # Check for complex JOINs
        join_count = query_upper.count("JOIN")
        
        # Check for window functions
        has_window_functions = any(
            func in query_upper 
            for func in ["ROW_NUMBER(", "RANK(", "DENSE_RANK(", "LAG(", "LEAD("]
        )
        
        if has_aggregation and has_group_by:
            recommendations.append(
                "Consider creating a materialized view for this aggregation query"
            )
        
        if join_count >= 3:
            recommendations.append(
                "Consider creating a denormalized materialized view for complex joins"
            )
        
        if has_window_functions:
            recommendations.append(
                "Consider materializing window function results for better performance"
            )
        
        return recommendations


# Predefined materialized view configurations for common analytics
ANALYTICS_MATERIALIZED_VIEWS = {
    "document_analytics_daily": MaterializedViewConfig(
        name="mv_document_analytics_daily",
        base_query="""
        SELECT 
            DATE(uploaded_at) as analysis_date,
            client_id,
            COUNT(*) as document_count,
            SUM(file_size) as total_size,
            AVG(file_size) as avg_size,
            COUNT(CASE WHEN is_processed = TRUE THEN 1 END) as processed_count,
            COUNT(CASE WHEN content_type = 'application/pdf' THEN 1 END) as pdf_count,
            COUNT(CASE WHEN content_type = 'text/plain' THEN 1 END) as text_count
        FROM documents 
        WHERE uploaded_at >= CURRENT_DATE - INTERVAL '90 days'
        GROUP BY DATE(uploaded_at), client_id
        """,
        view_type=ViewType.ANALYTICS,
        refresh_strategy=RefreshStrategy.SCHEDULED,
        refresh_interval=timedelta(hours=1),
        dependencies=["documents"],
        indexes=[
            "idx_mv_doc_analytics_date ON mv_document_analytics_daily (analysis_date)",
            "idx_mv_doc_analytics_client ON mv_document_analytics_daily (client_id, analysis_date)"
        ],
        incremental_column="analysis_date"
    ),
    
    "arbitration_analysis_summary": MaterializedViewConfig(
        name="mv_arbitration_summary",
        base_query="""
        SELECT 
            d.client_id,
            d.content_type,
            COUNT(a.id) as total_analyses,
            AVG(a.confidence_score) as avg_confidence,
            COUNT(CASE WHEN a.has_arbitration_clause = TRUE THEN 1 END) as with_arbitration,
            COUNT(CASE WHEN a.has_arbitration_clause = FALSE THEN 1 END) as without_arbitration,
            AVG(a.processing_time_ms) as avg_processing_time,
            MAX(a.created_at) as last_analysis
        FROM documents d
        JOIN arbitration_analysis a ON d.id = a.document_id
        WHERE a.created_at >= CURRENT_DATE - INTERVAL '30 days'
        GROUP BY d.client_id, d.content_type
        """,
        view_type=ViewType.REPORTING,
        refresh_strategy=RefreshStrategy.SCHEDULED,
        refresh_interval=timedelta(hours=6),
        dependencies=["documents", "arbitration_analysis"],
        indexes=[
            "idx_mv_arb_summary_client ON mv_arbitration_summary (client_id)",
            "idx_mv_arb_summary_type ON mv_arbitration_summary (content_type)"
        ]
    ),
    
    "document_chunks_analytics": MaterializedViewConfig(
        name="mv_chunks_analytics", 
        base_query="""
        SELECT 
            d.client_id,
            DATE(d.uploaded_at) as chunk_date,
            COUNT(c.id) as total_chunks,
            AVG(c.token_count) as avg_tokens_per_chunk,
            SUM(c.token_count) as total_tokens,
            COUNT(CASE WHEN c.embedding_id IS NOT NULL THEN 1 END) as embedded_chunks
        FROM documents d
        JOIN document_chunks c ON d.id = c.document_id
        WHERE d.uploaded_at >= CURRENT_DATE - INTERVAL '60 days'
        GROUP BY d.client_id, DATE(d.uploaded_at)
        """,
        view_type=ViewType.ANALYTICS,
        refresh_strategy=RefreshStrategy.INCREMENTAL,
        refresh_interval=timedelta(hours=2),
        dependencies=["documents", "document_chunks"],
        indexes=[
            "idx_mv_chunks_client_date ON mv_chunks_analytics (client_id, chunk_date)"
        ],
        incremental_column="chunk_date"
    ),
    
    "user_activity_summary": MaterializedViewConfig(
        name="mv_user_activity",
        base_query="""
        SELECT 
            u.id as user_id,
            u.username,
            COUNT(d.id) as documents_uploaded,
            SUM(d.file_size) as total_data_uploaded,
            COUNT(a.id) as analyses_performed,
            AVG(a.confidence_score) as avg_analysis_confidence,
            MAX(d.uploaded_at) as last_activity
        FROM users u
        LEFT JOIN documents d ON u.id = d.uploaded_by
        LEFT JOIN arbitration_analysis a ON d.id = a.document_id
        WHERE d.uploaded_at >= CURRENT_DATE - INTERVAL '30 days'
        GROUP BY u.id, u.username
        """,
        view_type=ViewType.REPORTING,
        refresh_strategy=RefreshStrategy.SCHEDULED,
        refresh_interval=timedelta(hours=12),
        dependencies=["users", "documents", "arbitration_analysis"],
        indexes=[
            "idx_mv_user_activity_user ON mv_user_activity (user_id)",
            "idx_mv_user_activity_last ON mv_user_activity (last_activity)"
        ]
    )
}


async def setup_enterprise_materialized_views(engine, metadata: MetaData) -> MaterializedViewManager:
    """
    Set up enterprise-level materialized views for analytics
    
    Args:
        engine: SQLAlchemy engine
        metadata: SQLAlchemy metadata
        
    Returns:
        MaterializedViewManager: Configured view manager
    """
    
    view_manager = MaterializedViewManager(engine, metadata)
    
    # Create all predefined materialized views
    for view_name, config in ANALYTICS_MATERIALIZED_VIEWS.items():
        success = await view_manager.create_materialized_view(config)
        if success:
            logger.info(f"Successfully created materialized view: {config.name}")
        else:
            logger.error(f"Failed to create materialized view: {config.name}")
    
    # Set up query rewrite rules
    view_manager.add_query_rewrite_rule(
        r"SELECT.*FROM documents.*GROUP BY.*client_id",
        "SELECT * FROM mv_document_analytics_daily",
        "mv_document_analytics_daily"
    )
    
    view_manager.add_query_rewrite_rule(
        r"SELECT.*FROM documents.*JOIN arbitration_analysis.*GROUP BY",
        "SELECT * FROM mv_arbitration_summary", 
        "mv_arbitration_summary"
    )
    
    logger.info("Enterprise materialized views setup completed")
    return view_manager


# Background task for automatic view maintenance
async def materialized_view_maintenance(view_manager: MaterializedViewManager):
    """
    Background task for automatic materialized view maintenance
    """
    
    while True:
        try:
            # Auto-refresh views based on schedule
            await view_manager.auto_refresh_views()
            
            # Clean up old performance statistics (keep last 1000 entries)
            for view_name in view_manager.performance_stats:
                stats = view_manager.performance_stats[view_name]
                refresh_times = stats.get("refresh_times", [])
                if len(refresh_times) > 1000:
                    stats["refresh_times"] = refresh_times[-1000:]
            
            # Sleep for 5 minutes before next maintenance cycle
            await asyncio.sleep(300)
            
        except Exception as e:
            logger.error(f"Materialized view maintenance failed: {e}")
            await asyncio.sleep(60)  # Wait 1 minute on error