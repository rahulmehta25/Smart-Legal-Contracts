"""
Enterprise Database Partitioning Module

Implements table partitioning strategies for scaling to enterprise levels.
Supports date-based and client-based partitioning with automated management.

Target: Support 10,000+ concurrent users with <50ms query response time
"""

from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer, String, Text, DateTime, 
    Boolean, ForeignKey, text, inspect, func
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class PartitionStrategy(Enum):
    """Partitioning strategies supported"""
    DATE_RANGE = "date_range"
    HASH = "hash"
    LIST = "list"
    COMPOSITE = "composite"


@dataclass
class PartitionConfig:
    """Configuration for table partitioning"""
    table_name: str
    strategy: PartitionStrategy
    partition_column: str
    partition_interval: Optional[str] = None  # For date partitioning: 'month', 'week', 'day'
    hash_modulus: Optional[int] = None  # For hash partitioning
    list_values: Optional[List[str]] = None  # For list partitioning
    retention_period: Optional[timedelta] = None  # Auto-drop partitions older than this
    
    
class PartitionManager:
    """
    Advanced partition management for enterprise-scale operations
    
    Features:
    - Date-based partitioning for time-series data
    - Hash partitioning for horizontal scaling
    - Automated partition creation and cleanup
    - Partition pruning optimization
    - Cross-partition query optimization
    """
    
    def __init__(self, engine, metadata: MetaData):
        self.engine = engine
        self.metadata = metadata
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def create_partitioned_table(self, config: PartitionConfig) -> bool:
        """
        Create a partitioned table based on configuration
        
        Args:
            config: Partition configuration
            
        Returns:
            bool: Success status
        """
        try:
            with self.engine.connect() as conn:
                # PostgreSQL partitioning
                if self.engine.dialect.name == 'postgresql':
                    return await self._create_postgres_partitioned_table(conn, config)
                # MySQL partitioning
                elif self.engine.dialect.name == 'mysql':
                    return await self._create_mysql_partitioned_table(conn, config)
                # SQLite doesn't support native partitioning, use views
                else:
                    return await self._create_sqlite_partitioned_views(conn, config)
                    
        except Exception as e:
            logger.error(f"Failed to create partitioned table {config.table_name}: {e}")
            return False
    
    async def _create_postgres_partitioned_table(self, conn, config: PartitionConfig) -> bool:
        """Create PostgreSQL partitioned table"""
        
        if config.strategy == PartitionStrategy.DATE_RANGE:
            partition_by = f"RANGE ({config.partition_column})"
        elif config.strategy == PartitionStrategy.HASH:
            partition_by = f"HASH ({config.partition_column})"
        elif config.strategy == PartitionStrategy.LIST:
            partition_by = f"LIST ({config.partition_column})"
        else:
            raise ValueError(f"Unsupported partition strategy: {config.strategy}")
        
        # Create main partitioned table for documents
        if config.table_name == "documents":
            sql = f"""
            CREATE TABLE IF NOT EXISTS {config.table_name}_partitioned (
                id SERIAL,
                filename VARCHAR(255) NOT NULL,
                file_path VARCHAR(500),
                content TEXT NOT NULL,
                content_type VARCHAR(50) DEFAULT 'text/plain',
                file_size INTEGER,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed_at TIMESTAMP,
                is_processed BOOLEAN DEFAULT FALSE,
                client_id INTEGER NOT NULL,
                tenant_id VARCHAR(100) NOT NULL,
                created_date DATE NOT NULL DEFAULT CURRENT_DATE,
                PRIMARY KEY (id, {config.partition_column})
            ) PARTITION BY {partition_by};
            """
            
        elif config.table_name == "document_chunks":
            sql = f"""
            CREATE TABLE IF NOT EXISTS {config.table_name}_partitioned (
                id SERIAL,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                start_char INTEGER NOT NULL,
                end_char INTEGER NOT NULL,
                token_count INTEGER,
                embedding_id VARCHAR(100),
                client_id INTEGER NOT NULL,
                created_date DATE NOT NULL DEFAULT CURRENT_DATE,
                PRIMARY KEY (id, {config.partition_column}),
                FOREIGN KEY (document_id) REFERENCES documents_partitioned(id)
            ) PARTITION BY {partition_by};
            """
        
        conn.execute(text(sql))
        
        # Create initial partitions
        await self._create_initial_partitions(conn, config)
        
        # Create indexes on partitioned table
        await self._create_partition_indexes(conn, config)
        
        return True
    
    async def _create_mysql_partitioned_table(self, conn, config: PartitionConfig) -> bool:
        """Create MySQL partitioned table"""
        
        if config.strategy == PartitionStrategy.DATE_RANGE:
            partition_clause = f"PARTITION BY RANGE (TO_DAYS({config.partition_column}))"
        elif config.strategy == PartitionStrategy.HASH:
            partition_clause = f"PARTITION BY HASH({config.partition_column}) PARTITIONS {config.hash_modulus}"
        elif config.strategy == PartitionStrategy.LIST:
            partition_clause = f"PARTITION BY LIST COLUMNS({config.partition_column})"
        else:
            raise ValueError(f"Unsupported partition strategy: {config.strategy}")
        
        # Create main partitioned table
        if config.table_name == "documents":
            sql = f"""
            CREATE TABLE IF NOT EXISTS {config.table_name}_partitioned (
                id INT AUTO_INCREMENT,
                filename VARCHAR(255) NOT NULL,
                file_path VARCHAR(500),
                content LONGTEXT NOT NULL,
                content_type VARCHAR(50) DEFAULT 'text/plain',
                file_size INT,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed_at TIMESTAMP NULL,
                is_processed BOOLEAN DEFAULT FALSE,
                client_id INT NOT NULL,
                tenant_id VARCHAR(100) NOT NULL,
                created_date DATE NOT NULL DEFAULT (CURRENT_DATE),
                PRIMARY KEY (id, {config.partition_column}),
                INDEX idx_client_date (client_id, created_date),
                INDEX idx_tenant_date (tenant_id, created_date)
            ) {partition_clause}
            """
            
            # Add partition definitions for date range
            if config.strategy == PartitionStrategy.DATE_RANGE:
                sql += await self._generate_mysql_date_partitions(config)
                
        conn.execute(text(sql))
        return True
    
    async def _create_sqlite_partitioned_views(self, conn, config: PartitionConfig) -> bool:
        """
        Create SQLite partitioned views (since SQLite doesn't support native partitioning)
        Uses multiple tables with union views for partition-like behavior
        """
        
        # Create base tables for each partition
        partitions = await self._generate_sqlite_partition_tables(config)
        
        for partition_name, partition_sql in partitions.items():
            conn.execute(text(partition_sql))
        
        # Create union view
        view_sql = await self._create_sqlite_union_view(config, list(partitions.keys()))
        conn.execute(text(view_sql))
        
        return True
    
    async def _create_initial_partitions(self, conn, config: PartitionConfig):
        """Create initial partitions based on configuration"""
        
        if config.strategy == PartitionStrategy.DATE_RANGE:
            await self._create_date_partitions(conn, config)
        elif config.strategy == PartitionStrategy.HASH:
            await self._create_hash_partitions(conn, config)
        elif config.strategy == PartitionStrategy.LIST:
            await self._create_list_partitions(conn, config)
    
    async def _create_date_partitions(self, conn, config: PartitionConfig):
        """Create date-based partitions"""
        
        current_date = datetime.now().date()
        
        # Create partitions for past 6 months and future 6 months
        for i in range(-6, 7):
            if config.partition_interval == "month":
                partition_date = current_date.replace(day=1) + timedelta(days=32*i)
                partition_date = partition_date.replace(day=1)  # First day of month
                next_partition = (partition_date + timedelta(days=32)).replace(day=1)
                
                partition_name = f"{config.table_name}_y{partition_date.year}m{partition_date.month:02d}"
                
                sql = f"""
                CREATE TABLE IF NOT EXISTS {partition_name} 
                PARTITION OF {config.table_name}_partitioned
                FOR VALUES FROM ('{partition_date}') TO ('{next_partition}')
                """
                
            elif config.partition_interval == "week":
                # Weekly partitions
                week_start = current_date - timedelta(days=current_date.weekday()) + timedelta(weeks=i)
                week_end = week_start + timedelta(days=7)
                
                partition_name = f"{config.table_name}_w{week_start.strftime('%Y%U')}"
                
                sql = f"""
                CREATE TABLE IF NOT EXISTS {partition_name}
                PARTITION OF {config.table_name}_partitioned
                FOR VALUES FROM ('{week_start}') TO ('{week_end}')
                """
            
            conn.execute(text(sql))
    
    async def _create_hash_partitions(self, conn, config: PartitionConfig):
        """Create hash-based partitions"""
        
        for i in range(config.hash_modulus):
            partition_name = f"{config.table_name}_hash_{i}"
            
            sql = f"""
            CREATE TABLE IF NOT EXISTS {partition_name}
            PARTITION OF {config.table_name}_partitioned
            FOR VALUES WITH (MODULUS {config.hash_modulus}, REMAINDER {i})
            """
            
            conn.execute(text(sql))
    
    async def _create_list_partitions(self, conn, config: PartitionConfig):
        """Create list-based partitions"""
        
        for value in config.list_values:
            partition_name = f"{config.table_name}_list_{value}"
            
            sql = f"""
            CREATE TABLE IF NOT EXISTS {partition_name}
            PARTITION OF {config.table_name}_partitioned
            FOR VALUES IN ('{value}')
            """
            
            conn.execute(text(sql))
    
    async def _create_partition_indexes(self, conn, config: PartitionConfig):
        """Create optimized indexes on partitioned tables"""
        
        table_name = f"{config.table_name}_partitioned"
        
        # Common indexes for documents table
        if config.table_name == "documents":
            indexes = [
                f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_{config.table_name}_client_date ON {table_name} (client_id, created_date)",
                f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_{config.table_name}_tenant_date ON {table_name} (tenant_id, created_date)",
                f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_{config.table_name}_processed ON {table_name} (is_processed, uploaded_at)",
                f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_{config.table_name}_content_type ON {table_name} (content_type, created_date)",
                f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_{config.table_name}_filename ON {table_name} USING gin (filename gin_trgm_ops)",
            ]
        
        elif config.table_name == "document_chunks":
            indexes = [
                f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_{config.table_name}_doc_chunk ON {table_name} (document_id, chunk_index)",
                f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_{config.table_name}_embedding ON {table_name} (embedding_id) WHERE embedding_id IS NOT NULL",
                f"CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_{config.table_name}_content_search ON {table_name} USING gin (content gin_trgm_ops)",
            ]
        
        for index_sql in indexes:
            try:
                conn.execute(text(index_sql))
            except Exception as e:
                logger.warning(f"Index creation failed (may already exist): {e}")
    
    async def auto_manage_partitions(self, config: PartitionConfig):
        """
        Automatically manage partitions:
        - Create new partitions as needed
        - Drop old partitions based on retention policy
        - Update statistics
        """
        
        try:
            with self.engine.connect() as conn:
                # Create future partitions
                await self._create_future_partitions(conn, config)
                
                # Drop old partitions if retention policy is set
                if config.retention_period:
                    await self._drop_old_partitions(conn, config)
                
                # Update partition statistics
                await self._update_partition_statistics(conn, config)
                
        except Exception as e:
            logger.error(f"Auto partition management failed for {config.table_name}: {e}")
    
    async def _create_future_partitions(self, conn, config: PartitionConfig):
        """Create partitions for future dates"""
        
        if config.strategy != PartitionStrategy.DATE_RANGE:
            return
        
        current_date = datetime.now().date()
        
        # Create partitions for next 3 months
        for i in range(1, 4):
            if config.partition_interval == "month":
                future_date = current_date.replace(day=1) + timedelta(days=32*i)
                future_date = future_date.replace(day=1)
                next_month = (future_date + timedelta(days=32)).replace(day=1)
                
                partition_name = f"{config.table_name}_y{future_date.year}m{future_date.month:02d}"
                
                # Check if partition exists
                exists_sql = f"""
                SELECT COUNT(*) as count FROM information_schema.tables 
                WHERE table_name = '{partition_name}'
                """
                result = conn.execute(text(exists_sql)).fetchone()
                
                if result.count == 0:
                    sql = f"""
                    CREATE TABLE {partition_name} 
                    PARTITION OF {config.table_name}_partitioned
                    FOR VALUES FROM ('{future_date}') TO ('{next_month}')
                    """
                    conn.execute(text(sql))
                    logger.info(f"Created future partition: {partition_name}")
    
    async def _drop_old_partitions(self, conn, config: PartitionConfig):
        """Drop partitions older than retention period"""
        
        cutoff_date = datetime.now().date() - config.retention_period
        
        # Find old partitions
        old_partitions_sql = f"""
        SELECT schemaname, tablename 
        FROM pg_tables 
        WHERE tablename LIKE '{config.table_name}_%'
        AND schemaname = 'public'
        """
        
        results = conn.execute(text(old_partitions_sql)).fetchall()
        
        for row in results:
            table_name = row.tablename
            
            # Extract date from partition name
            try:
                if config.partition_interval == "month":
                    # Extract year and month from table name like "documents_y2023m01"
                    year_month = table_name.split('_')[-1]
                    if year_month.startswith('y') and 'm' in year_month:
                        year = int(year_month[1:5])
                        month = int(year_month[6:8])
                        partition_date = datetime(year, month, 1).date()
                        
                        if partition_date < cutoff_date:
                            drop_sql = f"DROP TABLE IF EXISTS {table_name}"
                            conn.execute(text(drop_sql))
                            logger.info(f"Dropped old partition: {table_name}")
                            
            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse partition date from {table_name}: {e}")
    
    async def _update_partition_statistics(self, conn, config: PartitionConfig):
        """Update statistics for partition pruning optimization"""
        
        if self.engine.dialect.name == 'postgresql':
            # Update statistics for all partitions
            sql = f"""
            SELECT schemaname, tablename 
            FROM pg_tables 
            WHERE tablename LIKE '{config.table_name}_%'
            AND schemaname = 'public'
            """
            
            results = conn.execute(text(sql)).fetchall()
            
            for row in results:
                analyze_sql = f"ANALYZE {row.tablename}"
                conn.execute(text(analyze_sql))
    
    async def get_partition_info(self, table_name: str) -> Dict:
        """Get detailed information about table partitions"""
        
        try:
            with self.engine.connect() as conn:
                if self.engine.dialect.name == 'postgresql':
                    sql = """
                    SELECT 
                        schemaname,
                        tablename,
                        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size,
                        (SELECT COUNT(*) FROM information_schema.columns 
                         WHERE table_name = t.tablename) as column_count
                    FROM pg_tables t
                    WHERE tablename LIKE %s
                    AND schemaname = 'public'
                    ORDER BY tablename
                    """
                    
                    results = conn.execute(text(sql), (f"{table_name}_%",)).fetchall()
                    
                    return {
                        "partitions": [
                            {
                                "name": row.tablename,
                                "size": row.size,
                                "columns": row.column_count
                            }
                            for row in results
                        ],
                        "total_partitions": len(results)
                    }
                    
        except Exception as e:
            logger.error(f"Failed to get partition info for {table_name}: {e}")
            return {"error": str(e)}
    
    def enable_partition_pruning(self, session: Session, table_name: str):
        """
        Enable partition pruning for optimal query performance
        """
        
        if self.engine.dialect.name == 'postgresql':
            # Enable constraint exclusion for partition pruning
            session.execute(text("SET constraint_exclusion = partition"))
            session.execute(text("SET enable_partition_pruning = on"))
            session.execute(text("SET enable_partitionwise_join = on"))
            session.execute(text("SET enable_partitionwise_aggregate = on"))
    
    async def optimize_partition_queries(self, query: str, partition_column: str, partition_value) -> str:
        """
        Optimize queries to take advantage of partition pruning
        
        Args:
            query: Original SQL query
            partition_column: Column used for partitioning
            partition_value: Value to filter on
            
        Returns:
            str: Optimized query with partition pruning hints
        """
        
        # Add partition key to WHERE clause if not present
        if f"WHERE" in query.upper():
            if partition_column not in query:
                optimized_query = query.replace(
                    "WHERE", 
                    f"WHERE {partition_column} = '{partition_value}' AND"
                )
            else:
                optimized_query = query
        else:
            # Add WHERE clause
            if "ORDER BY" in query.upper():
                optimized_query = query.replace(
                    "ORDER BY",
                    f"WHERE {partition_column} = '{partition_value}' ORDER BY"
                )
            else:
                optimized_query = f"{query} WHERE {partition_column} = '{partition_value}'"
        
        return optimized_query


class PartitionedQueryBuilder:
    """
    Query builder that automatically optimizes queries for partitioned tables
    """
    
    def __init__(self, partition_manager: PartitionManager):
        self.partition_manager = partition_manager
    
    def build_cross_partition_query(self, base_query: str, partitions: List[str]) -> str:
        """
        Build a query that efficiently spans multiple partitions
        
        Args:
            base_query: Base SELECT query
            partitions: List of partition names to query
            
        Returns:
            str: Optimized cross-partition query
        """
        
        union_queries = []
        for partition in partitions:
            partition_query = base_query.replace("FROM documents_partitioned", f"FROM {partition}")
            union_queries.append(f"({partition_query})")
        
        return " UNION ALL ".join(union_queries)
    
    def build_partition_aware_insert(self, table: str, data: Dict, partition_key: str) -> str:
        """
        Build an insert query that targets the correct partition
        
        Args:
            table: Target table name
            data: Data to insert
            partition_key: Partition key value
            
        Returns:
            str: Partition-aware insert query
        """
        
        columns = ", ".join(data.keys())
        values = ", ".join([f"'{v}'" if isinstance(v, str) else str(v) for v in data.values()])
        
        return f"""
        INSERT INTO {table}_partitioned ({columns})
        VALUES ({values})
        """


# Predefined partition configurations for common use cases
DOCUMENT_PARTITION_CONFIGS = {
    "documents_by_date": PartitionConfig(
        table_name="documents",
        strategy=PartitionStrategy.DATE_RANGE,
        partition_column="created_date",
        partition_interval="month",
        retention_period=timedelta(days=365*2)  # Keep 2 years of data
    ),
    
    "documents_by_client": PartitionConfig(
        table_name="documents",
        strategy=PartitionStrategy.HASH,
        partition_column="client_id",
        hash_modulus=16  # 16 hash partitions
    ),
    
    "chunks_by_date": PartitionConfig(
        table_name="document_chunks",
        strategy=PartitionStrategy.DATE_RANGE,
        partition_column="created_date",
        partition_interval="month",
        retention_period=timedelta(days=365*2)
    )
}


async def setup_enterprise_partitioning(engine, metadata: MetaData) -> PartitionManager:
    """
    Set up enterprise-level partitioning for the application
    
    Args:
        engine: SQLAlchemy engine
        metadata: SQLAlchemy metadata
        
    Returns:
        PartitionManager: Configured partition manager
    """
    
    manager = PartitionManager(engine, metadata)
    
    # Set up partitioning for main tables
    for config_name, config in DOCUMENT_PARTITION_CONFIGS.items():
        success = await manager.create_partitioned_table(config)
        if success:
            logger.info(f"Successfully created partitioned table: {config.table_name}")
        else:
            logger.error(f"Failed to create partitioned table: {config.table_name}")
    
    return manager


# Background task for automatic partition management
async def partition_maintenance_task(partition_manager: PartitionManager):
    """
    Background task for automatic partition maintenance
    Runs periodically to create new partitions and clean up old ones
    """
    
    while True:
        try:
            # Manage all configured partitions
            for config in DOCUMENT_PARTITION_CONFIGS.values():
                await partition_manager.auto_manage_partitions(config)
            
            # Sleep for 1 hour before next maintenance cycle
            await asyncio.sleep(3600)
            
        except Exception as e:
            logger.error(f"Partition maintenance task failed: {e}")
            await asyncio.sleep(300)  # Wait 5 minutes before retry