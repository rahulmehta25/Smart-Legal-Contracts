"""
Enterprise Database Sharding Module

Implements horizontal sharding strategies for multi-tenant architecture.
Provides automatic shard routing, cross-shard query optimization, and data distribution.

Target: Support 10,000+ concurrent users with <50ms query response time
"""

from sqlalchemy import create_engine, MetaData, text, inspect
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.pool import QueuePool
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import logging
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import time
import random

logger = logging.getLogger(__name__)


class ShardingStrategy(Enum):
    """Sharding strategies supported"""
    HASH = "hash"
    RANGE = "range"
    DIRECTORY = "directory"
    CONSISTENT_HASH = "consistent_hash"


class ShardStatus(Enum):
    """Shard status types"""
    ACTIVE = "active"
    READONLY = "readonly"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


@dataclass
class ShardConfig:
    """Configuration for a database shard"""
    shard_id: str
    database_url: str
    weight: float = 1.0  # For load balancing
    status: ShardStatus = ShardStatus.ACTIVE
    max_connections: int = 20
    connection_timeout: int = 30
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ShardingRule:
    """Rules for data distribution across shards"""
    table_name: str
    shard_key: str
    strategy: ShardingStrategy
    shard_count: int
    hash_function: str = "md5"  # md5, sha1, sha256
    range_boundaries: Optional[List[Any]] = None
    directory_mapping: Optional[Dict[str, str]] = None


class ConsistentHashRing:
    """
    Consistent hash ring for distributed sharding
    Provides minimal data movement when shards are added/removed
    """
    
    def __init__(self, virtual_nodes: int = 150):
        self.virtual_nodes = virtual_nodes
        self.ring = {}
        self.sorted_keys = []
        
    def add_shard(self, shard_id: str, weight: float = 1.0):
        """Add a shard to the hash ring"""
        nodes_count = int(self.virtual_nodes * weight)
        
        for i in range(nodes_count):
            virtual_key = f"{shard_id}:{i}"
            hash_value = self._hash(virtual_key)
            self.ring[hash_value] = shard_id
            
        self._update_sorted_keys()
        
    def remove_shard(self, shard_id: str):
        """Remove a shard from the hash ring"""
        keys_to_remove = [key for key, value in self.ring.items() if value == shard_id]
        
        for key in keys_to_remove:
            del self.ring[key]
            
        self._update_sorted_keys()
        
    def get_shard(self, key: str) -> str:
        """Get the shard for a given key"""
        if not self.ring:
            raise ValueError("No shards available in hash ring")
            
        hash_value = self._hash(key)
        
        # Find the first shard with hash >= key_hash
        for ring_hash in self.sorted_keys:
            if ring_hash >= hash_value:
                return self.ring[ring_hash]
                
        # Wrap around to the first shard
        return self.ring[self.sorted_keys[0]]
        
    def _hash(self, key: str) -> int:
        """Hash function for the ring"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
        
    def _update_sorted_keys(self):
        """Update the sorted keys list"""
        self.sorted_keys = sorted(self.ring.keys())


class ShardManager:
    """
    Advanced shard management for enterprise multi-tenant architecture
    
    Features:
    - Multiple sharding strategies (hash, range, directory, consistent hash)
    - Automatic shard routing and load balancing
    - Cross-shard query aggregation
    - Shard health monitoring and failover
    - Data rebalancing and migration
    """
    
    def __init__(self):
        self.shards: Dict[str, ShardConfig] = {}
        self.engines: Dict[str, Any] = {}
        self.sessions: Dict[str, sessionmaker] = {}
        self.sharding_rules: Dict[str, ShardingRule] = {}
        self.hash_ring = ConsistentHashRing()
        self.executor = ThreadPoolExecutor(max_workers=10)
        self.metrics = {
            "queries_by_shard": {},
            "response_times": {},
            "error_counts": {},
            "connection_pools": {}
        }
        
    async def add_shard(self, config: ShardConfig) -> bool:
        """
        Add a new shard to the cluster
        
        Args:
            config: Shard configuration
            
        Returns:
            bool: Success status
        """
        try:
            # Create database engine
            engine = create_engine(
                config.database_url,
                poolclass=QueuePool,
                pool_size=config.max_connections // 2,
                max_overflow=config.max_connections // 2,
                pool_timeout=config.connection_timeout,
                pool_recycle=3600,  # Recycle connections every hour
                echo=False
            )
            
            # Test connection
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
                
            # Store shard configuration
            self.shards[config.shard_id] = config
            self.engines[config.shard_id] = engine
            
            # Create session factory
            self.sessions[config.shard_id] = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=engine
            )
            
            # Add to consistent hash ring
            self.hash_ring.add_shard(config.shard_id, config.weight)
            
            # Initialize metrics
            self.metrics["queries_by_shard"][config.shard_id] = 0
            self.metrics["response_times"][config.shard_id] = []
            self.metrics["error_counts"][config.shard_id] = 0
            
            logger.info(f"Successfully added shard: {config.shard_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add shard {config.shard_id}: {e}")
            return False
    
    async def remove_shard(self, shard_id: str, migrate_data: bool = True) -> bool:
        """
        Remove a shard from the cluster
        
        Args:
            shard_id: ID of shard to remove
            migrate_data: Whether to migrate data to other shards
            
        Returns:
            bool: Success status
        """
        try:
            if shard_id not in self.shards:
                raise ValueError(f"Shard {shard_id} not found")
                
            # Set shard to readonly mode first
            self.shards[shard_id].status = ShardStatus.READONLY
            
            # Migrate data if requested
            if migrate_data:
                await self._migrate_shard_data(shard_id)
                
            # Remove from hash ring
            self.hash_ring.remove_shard(shard_id)
            
            # Close engine
            if shard_id in self.engines:
                self.engines[shard_id].dispose()
                del self.engines[shard_id]
                
            # Clean up
            del self.shards[shard_id]
            del self.sessions[shard_id]
            
            logger.info(f"Successfully removed shard: {shard_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove shard {shard_id}: {e}")
            return False
    
    def add_sharding_rule(self, rule: ShardingRule):
        """Add a sharding rule for a table"""
        self.sharding_rules[rule.table_name] = rule
        logger.info(f"Added sharding rule for table: {rule.table_name}")
    
    def get_shard_for_key(self, table_name: str, shard_key_value: Any) -> str:
        """
        Determine which shard a record belongs to
        
        Args:
            table_name: Name of the table
            shard_key_value: Value of the shard key
            
        Returns:
            str: Shard ID
        """
        if table_name not in self.sharding_rules:
            raise ValueError(f"No sharding rule found for table: {table_name}")
            
        rule = self.sharding_rules[table_name]
        
        if rule.strategy == ShardingStrategy.HASH:
            return self._hash_shard_selection(shard_key_value, rule)
        elif rule.strategy == ShardingStrategy.RANGE:
            return self._range_shard_selection(shard_key_value, rule)
        elif rule.strategy == ShardingStrategy.DIRECTORY:
            return self._directory_shard_selection(shard_key_value, rule)
        elif rule.strategy == ShardingStrategy.CONSISTENT_HASH:
            return self.hash_ring.get_shard(str(shard_key_value))
        else:
            raise ValueError(f"Unsupported sharding strategy: {rule.strategy}")
    
    def _hash_shard_selection(self, key_value: Any, rule: ShardingRule) -> str:
        """Select shard using hash function"""
        key_str = str(key_value)
        
        if rule.hash_function == "md5":
            hash_value = hashlib.md5(key_str.encode()).hexdigest()
        elif rule.hash_function == "sha1":
            hash_value = hashlib.sha1(key_str.encode()).hexdigest()
        elif rule.hash_function == "sha256":
            hash_value = hashlib.sha256(key_str.encode()).hexdigest()
        else:
            raise ValueError(f"Unsupported hash function: {rule.hash_function}")
            
        # Convert to integer and mod by shard count
        hash_int = int(hash_value, 16)
        shard_index = hash_int % rule.shard_count
        
        # Find active shard with this index
        active_shards = [sid for sid, config in self.shards.items() 
                        if config.status == ShardStatus.ACTIVE]
        
        if not active_shards:
            raise RuntimeError("No active shards available")
            
        return active_shards[shard_index % len(active_shards)]
    
    def _range_shard_selection(self, key_value: Any, rule: ShardingRule) -> str:
        """Select shard using range boundaries"""
        if not rule.range_boundaries:
            raise ValueError("Range boundaries not configured")
            
        for i, boundary in enumerate(rule.range_boundaries):
            if key_value <= boundary:
                shard_index = i
                break
        else:
            shard_index = len(rule.range_boundaries)
            
        active_shards = [sid for sid, config in self.shards.items() 
                        if config.status == ShardStatus.ACTIVE]
        
        return active_shards[shard_index % len(active_shards)]
    
    def _directory_shard_selection(self, key_value: Any, rule: ShardingRule) -> str:
        """Select shard using directory mapping"""
        if not rule.directory_mapping:
            raise ValueError("Directory mapping not configured")
            
        key_str = str(key_value)
        
        # Look for exact match first
        if key_str in rule.directory_mapping:
            return rule.directory_mapping[key_str]
            
        # Look for prefix match
        for prefix, shard_id in rule.directory_mapping.items():
            if key_str.startswith(prefix):
                return shard_id
                
        # Default to first active shard
        active_shards = [sid for sid, config in self.shards.items() 
                        if config.status == ShardStatus.ACTIVE]
        
        if not active_shards:
            raise RuntimeError("No active shards available")
            
        return active_shards[0]
    
    @contextmanager
    def get_shard_session(self, shard_id: str):
        """Get a database session for a specific shard"""
        if shard_id not in self.sessions:
            raise ValueError(f"Shard {shard_id} not found")
            
        if self.shards[shard_id].status == ShardStatus.OFFLINE:
            raise RuntimeError(f"Shard {shard_id} is offline")
            
        session = self.sessions[shard_id]()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    async def execute_on_shard(self, shard_id: str, query: str, params: Dict = None) -> List[Dict]:
        """
        Execute a query on a specific shard
        
        Args:
            shard_id: Target shard ID
            query: SQL query to execute
            params: Query parameters
            
        Returns:
            List[Dict]: Query results
        """
        start_time = time.time()
        
        try:
            with self.get_shard_session(shard_id) as session:
                if params:
                    result = session.execute(text(query), params)
                else:
                    result = session.execute(text(query))
                
                # Convert to list of dictionaries
                columns = result.keys()
                rows = [dict(zip(columns, row)) for row in result.fetchall()]
                
                # Update metrics
                self.metrics["queries_by_shard"][shard_id] += 1
                response_time = time.time() - start_time
                self.metrics["response_times"][shard_id].append(response_time)
                
                # Keep only last 100 response times
                if len(self.metrics["response_times"][shard_id]) > 100:
                    self.metrics["response_times"][shard_id] = \
                        self.metrics["response_times"][shard_id][-100:]
                
                return rows
                
        except Exception as e:
            self.metrics["error_counts"][shard_id] += 1
            logger.error(f"Query failed on shard {shard_id}: {e}")
            raise
    
    async def execute_cross_shard_query(self, query: str, params: Dict = None, 
                                      target_shards: Optional[List[str]] = None) -> List[Dict]:
        """
        Execute a query across multiple shards and aggregate results
        
        Args:
            query: SQL query to execute
            params: Query parameters
            target_shards: Specific shards to query (default: all active shards)
            
        Returns:
            List[Dict]: Aggregated results from all shards
        """
        if target_shards is None:
            target_shards = [sid for sid, config in self.shards.items() 
                           if config.status == ShardStatus.ACTIVE]
        
        # Execute queries in parallel
        tasks = []
        for shard_id in target_shards:
            task = self.execute_on_shard(shard_id, query, params)
            tasks.append((shard_id, task))
        
        # Collect results
        all_results = []
        
        for shard_id, task in tasks:
            try:
                shard_results = await task
                # Add shard ID to each result for tracking
                for row in shard_results:
                    row['_shard_id'] = shard_id
                all_results.extend(shard_results)
            except Exception as e:
                logger.error(f"Cross-shard query failed on shard {shard_id}: {e}")
                # Continue with other shards
                
        return all_results
    
    async def insert_record(self, table_name: str, data: Dict) -> bool:
        """
        Insert a record into the appropriate shard
        
        Args:
            table_name: Target table name
            data: Record data
            
        Returns:
            bool: Success status
        """
        try:
            # Get sharding rule
            if table_name not in self.sharding_rules:
                raise ValueError(f"No sharding rule for table: {table_name}")
                
            rule = self.sharding_rules[table_name]
            
            # Get shard key value
            if rule.shard_key not in data:
                raise ValueError(f"Shard key {rule.shard_key} not found in data")
                
            shard_key_value = data[rule.shard_key]
            
            # Determine target shard
            shard_id = self.get_shard_for_key(table_name, shard_key_value)
            
            # Build insert query
            columns = ", ".join(data.keys())
            placeholders = ", ".join([f":{key}" for key in data.keys()])
            query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
            
            # Execute on target shard
            await self.execute_on_shard(shard_id, query, data)
            
            logger.debug(f"Inserted record into shard {shard_id} for table {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert record into {table_name}: {e}")
            return False
    
    async def update_record(self, table_name: str, shard_key_value: Any, 
                          updates: Dict, conditions: Dict = None) -> bool:
        """
        Update a record in the appropriate shard
        
        Args:
            table_name: Target table name
            shard_key_value: Value of the shard key
            updates: Fields to update
            conditions: Additional WHERE conditions
            
        Returns:
            bool: Success status
        """
        try:
            # Determine target shard
            shard_id = self.get_shard_for_key(table_name, shard_key_value)
            
            # Build update query
            set_clause = ", ".join([f"{key} = :{key}" for key in updates.keys()])
            query = f"UPDATE {table_name} SET {set_clause}"
            
            # Add WHERE conditions
            rule = self.sharding_rules[table_name]
            where_conditions = [f"{rule.shard_key} = :shard_key_value"]
            params = {**updates, "shard_key_value": shard_key_value}
            
            if conditions:
                for key, value in conditions.items():
                    where_conditions.append(f"{key} = :{key}")
                    params[key] = value
            
            query += " WHERE " + " AND ".join(where_conditions)
            
            # Execute on target shard
            await self.execute_on_shard(shard_id, query, params)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update record in {table_name}: {e}")
            return False
    
    async def delete_record(self, table_name: str, shard_key_value: Any, 
                          conditions: Dict = None) -> bool:
        """
        Delete a record from the appropriate shard
        
        Args:
            table_name: Target table name
            shard_key_value: Value of the shard key
            conditions: Additional WHERE conditions
            
        Returns:
            bool: Success status
        """
        try:
            # Determine target shard
            shard_id = self.get_shard_for_key(table_name, shard_key_value)
            
            # Build delete query
            rule = self.sharding_rules[table_name]
            where_conditions = [f"{rule.shard_key} = :shard_key_value"]
            params = {"shard_key_value": shard_key_value}
            
            if conditions:
                for key, value in conditions.items():
                    where_conditions.append(f"{key} = :{key}")
                    params[key] = value
            
            query = f"DELETE FROM {table_name} WHERE " + " AND ".join(where_conditions)
            
            # Execute on target shard
            await self.execute_on_shard(shard_id, query, params)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete record from {table_name}: {e}")
            return False
    
    async def get_shard_metrics(self) -> Dict:
        """Get performance metrics for all shards"""
        metrics = {}
        
        for shard_id in self.shards.keys():
            response_times = self.metrics["response_times"][shard_id]
            
            metrics[shard_id] = {
                "status": self.shards[shard_id].status.value,
                "query_count": self.metrics["queries_by_shard"][shard_id],
                "error_count": self.metrics["error_counts"][shard_id],
                "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
                "max_response_time": max(response_times) if response_times else 0,
                "min_response_time": min(response_times) if response_times else 0,
                "connection_pool_size": self._get_pool_size(shard_id),
                "active_connections": self._get_active_connections(shard_id)
            }
            
        return metrics
    
    def _get_pool_size(self, shard_id: str) -> int:
        """Get connection pool size for a shard"""
        if shard_id in self.engines:
            engine = self.engines[shard_id]
            if hasattr(engine.pool, 'size'):
                return engine.pool.size()
        return 0
    
    def _get_active_connections(self, shard_id: str) -> int:
        """Get active connection count for a shard"""
        if shard_id in self.engines:
            engine = self.engines[shard_id]
            if hasattr(engine.pool, 'checkedout'):
                return engine.pool.checkedout()
        return 0
    
    async def rebalance_shards(self) -> bool:
        """
        Rebalance data across shards for optimal performance
        This is a complex operation that should be run during maintenance windows
        """
        try:
            # Get current data distribution
            distribution = await self._analyze_data_distribution()
            
            # Identify imbalanced shards
            imbalanced_shards = self._identify_imbalanced_shards(distribution)
            
            if not imbalanced_shards:
                logger.info("Shards are already balanced")
                return True
            
            # Create rebalancing plan
            rebalancing_plan = await self._create_rebalancing_plan(imbalanced_shards)
            
            # Execute rebalancing
            success = await self._execute_rebalancing_plan(rebalancing_plan)
            
            if success:
                logger.info("Shard rebalancing completed successfully")
            else:
                logger.error("Shard rebalancing failed")
                
            return success
            
        except Exception as e:
            logger.error(f"Shard rebalancing failed: {e}")
            return False
    
    async def _migrate_shard_data(self, source_shard_id: str):
        """Migrate data from a shard to other shards"""
        # This is a simplified implementation
        # In production, this would need careful planning and execution
        
        logger.info(f"Starting data migration from shard {source_shard_id}")
        
        # Get all tables with sharding rules
        for table_name, rule in self.sharding_rules.items():
            # Extract all data from source shard
            query = f"SELECT * FROM {table_name}"
            records = await self.execute_on_shard(source_shard_id, query)
            
            # Redistribute records to appropriate shards
            for record in records:
                # Remove shard metadata
                record.pop('_shard_id', None)
                
                # Insert into correct shard
                await self.insert_record(table_name, record)
                
            logger.info(f"Migrated {len(records)} records from {table_name}")


class ShardedQueryBuilder:
    """
    Query builder that automatically handles sharded queries
    """
    
    def __init__(self, shard_manager: ShardManager):
        self.shard_manager = shard_manager
    
    async def find_by_shard_key(self, table_name: str, shard_key_value: Any, 
                               conditions: Dict = None) -> List[Dict]:
        """
        Find records by shard key (single shard query)
        
        Args:
            table_name: Target table name
            shard_key_value: Value of the shard key
            conditions: Additional WHERE conditions
            
        Returns:
            List[Dict]: Query results
        """
        # Determine target shard
        shard_id = self.shard_manager.get_shard_for_key(table_name, shard_key_value)
        
        # Build query
        rule = self.shard_manager.sharding_rules[table_name]
        where_conditions = [f"{rule.shard_key} = :shard_key_value"]
        params = {"shard_key_value": shard_key_value}
        
        if conditions:
            for key, value in conditions.items():
                where_conditions.append(f"{key} = :{key}")
                params[key] = value
        
        query = f"SELECT * FROM {table_name} WHERE " + " AND ".join(where_conditions)
        
        return await self.shard_manager.execute_on_shard(shard_id, query, params)
    
    async def find_all(self, table_name: str, conditions: Dict = None, 
                      limit: int = None, order_by: str = None) -> List[Dict]:
        """
        Find records across all shards
        
        Args:
            table_name: Target table name
            conditions: WHERE conditions
            limit: Result limit
            order_by: ORDER BY clause
            
        Returns:
            List[Dict]: Aggregated results from all shards
        """
        # Build base query
        query = f"SELECT * FROM {table_name}"
        params = {}
        
        if conditions:
            where_conditions = []
            for key, value in conditions.items():
                where_conditions.append(f"{key} = :{key}")
                params[key] = value
            query += " WHERE " + " AND ".join(where_conditions)
        
        if order_by:
            query += f" ORDER BY {order_by}"
            
        if limit:
            query += f" LIMIT {limit}"
        
        # Execute across all shards
        results = await self.shard_manager.execute_cross_shard_query(query, params)
        
        # If we have an ORDER BY clause, we need to re-sort the combined results
        if order_by and len(results) > 1:
            # Parse order by clause (simplified)
            sort_column = order_by.split()[0]
            reverse = "DESC" in order_by.upper()
            
            results.sort(key=lambda x: x.get(sort_column, ""), reverse=reverse)
        
        # Apply global limit if specified
        if limit and len(results) > limit:
            results = results[:limit]
        
        return results
    
    async def aggregate_query(self, table_name: str, aggregation: str, 
                            group_by: str = None, conditions: Dict = None) -> List[Dict]:
        """
        Execute aggregation queries across shards
        
        Args:
            table_name: Target table name
            aggregation: Aggregation function (COUNT, SUM, AVG, etc.)
            group_by: GROUP BY clause
            conditions: WHERE conditions
            
        Returns:
            List[Dict]: Aggregated results
        """
        # Build query
        query = f"SELECT {aggregation}"
        if group_by:
            query += f", {group_by}"
        query += f" FROM {table_name}"
        
        params = {}
        if conditions:
            where_conditions = []
            for key, value in conditions.items():
                where_conditions.append(f"{key} = :{key}")
                params[key] = value
            query += " WHERE " + " AND ".join(where_conditions)
        
        if group_by:
            query += f" GROUP BY {group_by}"
        
        # Execute across all shards
        shard_results = await self.shard_manager.execute_cross_shard_query(query, params)
        
        # Aggregate results from different shards
        if "COUNT" in aggregation.upper():
            return self._aggregate_count_results(shard_results, group_by)
        elif "SUM" in aggregation.upper():
            return self._aggregate_sum_results(shard_results, group_by)
        elif "AVG" in aggregation.upper():
            return self._aggregate_avg_results(shard_results, group_by)
        else:
            return shard_results
    
    def _aggregate_count_results(self, results: List[Dict], group_by: str = None) -> List[Dict]:
        """Aggregate COUNT results from multiple shards"""
        if not group_by:
            # Simple count aggregation
            total_count = sum(row.get("count", 0) for row in results)
            return [{"count": total_count}]
        else:
            # Group by aggregation
            aggregated = {}
            for row in results:
                key = row.get(group_by)
                count = row.get("count", 0)
                aggregated[key] = aggregated.get(key, 0) + count
            
            return [{"count": count, group_by: key} for key, count in aggregated.items()]
    
    def _aggregate_sum_results(self, results: List[Dict], group_by: str = None) -> List[Dict]:
        """Aggregate SUM results from multiple shards"""
        if not group_by:
            # Simple sum aggregation
            total_sum = sum(row.get("sum", 0) for row in results)
            return [{"sum": total_sum}]
        else:
            # Group by aggregation
            aggregated = {}
            for row in results:
                key = row.get(group_by)
                sum_value = row.get("sum", 0)
                aggregated[key] = aggregated.get(key, 0) + sum_value
            
            return [{"sum": sum_value, group_by: key} for key, sum_value in aggregated.items()]
    
    def _aggregate_avg_results(self, results: List[Dict], group_by: str = None) -> List[Dict]:
        """Aggregate AVG results from multiple shards"""
        # For average, we need to recalculate based on sum and count
        # This is a simplified implementation
        if not group_by:
            total_sum = sum(row.get("avg", 0) * row.get("_count", 1) for row in results)
            total_count = sum(row.get("_count", 1) for row in results)
            avg = total_sum / total_count if total_count > 0 else 0
            return [{"avg": avg}]
        else:
            # Group by aggregation (simplified)
            return results


# Predefined sharding configurations for common use cases
ENTERPRISE_SHARD_CONFIGS = {
    "primary_cluster": [
        ShardConfig(
            shard_id="shard_01",
            database_url="postgresql://user:pass@shard01.example.com/db",
            weight=1.0,
            max_connections=50
        ),
        ShardConfig(
            shard_id="shard_02", 
            database_url="postgresql://user:pass@shard02.example.com/db",
            weight=1.0,
            max_connections=50
        ),
        ShardConfig(
            shard_id="shard_03",
            database_url="postgresql://user:pass@shard03.example.com/db",
            weight=1.0,
            max_connections=50
        ),
        ShardConfig(
            shard_id="shard_04",
            database_url="postgresql://user:pass@shard04.example.com/db",
            weight=1.0,
            max_connections=50
        )
    ]
}

SHARDING_RULES = {
    "documents": ShardingRule(
        table_name="documents",
        shard_key="client_id",
        strategy=ShardingStrategy.CONSISTENT_HASH,
        shard_count=4
    ),
    "document_chunks": ShardingRule(
        table_name="document_chunks", 
        shard_key="client_id",
        strategy=ShardingStrategy.CONSISTENT_HASH,
        shard_count=4
    ),
    "arbitration_analysis": ShardingRule(
        table_name="arbitration_analysis",
        shard_key="client_id", 
        strategy=ShardingStrategy.CONSISTENT_HASH,
        shard_count=4
    )
}


async def setup_enterprise_sharding() -> Tuple[ShardManager, ShardedQueryBuilder]:
    """
    Set up enterprise-level sharding for the application
    
    Returns:
        Tuple[ShardManager, ShardedQueryBuilder]: Configured shard manager and query builder
    """
    
    # Create shard manager
    shard_manager = ShardManager()
    
    # Add shards (in production, these would be real database URLs)
    for config in ENTERPRISE_SHARD_CONFIGS["primary_cluster"]:
        await shard_manager.add_shard(config)
    
    # Add sharding rules
    for rule in SHARDING_RULES.values():
        shard_manager.add_sharding_rule(rule)
    
    # Create query builder
    query_builder = ShardedQueryBuilder(shard_manager)
    
    logger.info("Enterprise sharding setup completed")
    
    return shard_manager, query_builder


# Background task for shard health monitoring
async def shard_health_monitor(shard_manager: ShardManager):
    """
    Background task for monitoring shard health and performance
    """
    
    while True:
        try:
            # Check health of each shard
            for shard_id in shard_manager.shards.keys():
                try:
                    # Simple health check query
                    await shard_manager.execute_on_shard(shard_id, "SELECT 1", {})
                    
                    # Update status to active if it was offline
                    if shard_manager.shards[shard_id].status == ShardStatus.OFFLINE:
                        shard_manager.shards[shard_id].status = ShardStatus.ACTIVE
                        logger.info(f"Shard {shard_id} is back online")
                        
                except Exception as e:
                    # Mark shard as offline
                    shard_manager.shards[shard_id].status = ShardStatus.OFFLINE
                    logger.error(f"Shard {shard_id} health check failed: {e}")
            
            # Sleep for 30 seconds before next check
            await asyncio.sleep(30)
            
        except Exception as e:
            logger.error(f"Shard health monitor failed: {e}")
            await asyncio.sleep(60)  # Wait longer on error