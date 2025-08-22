"""
Delta Lake Manager

Comprehensive Delta Lake management with advanced features:
- ACID transactions and concurrency control
- Time travel and versioning
- Schema evolution and enforcement
- Optimization (Z-ordering, compaction, vacuum)
- Change Data Feed (CDF)
- Liquid clustering
- Predictive optimization
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import *
from delta.tables import DeltaTable
from delta import configure_spark_with_delta_pip


class OptimizationStrategy(Enum):
    """Delta table optimization strategies"""
    AUTO = "auto"
    ZORDER = "zorder"
    COMPACT = "compact"
    VACUUM = "vacuum"
    LIQUID_CLUSTERING = "liquid_clustering"


@dataclass
class DeltaTableConfig:
    """Configuration for Delta tables"""
    table_name: str
    path: str
    partition_columns: Optional[List[str]] = None
    zorder_columns: Optional[List[str]] = None
    liquid_clustering_columns: Optional[List[str]] = None
    enable_change_data_feed: bool = False
    enable_deletion_vectors: bool = True
    auto_optimize: bool = True
    optimize_write: bool = True
    tune_file_sizes: bool = True
    target_file_size_mb: int = 1024
    log_retention_duration: str = "30 days"
    deleted_file_retention_duration: str = "7 days"
    checkpoint_interval: int = 10
    enable_predictive_optimization: bool = False
    data_skipping_enabled: bool = True
    column_mapping_mode: str = "none"  # none, id, name


@dataclass
class TableMetrics:
    """Metrics for Delta table operations"""
    num_files: int = 0
    size_in_bytes: int = 0
    num_records: int = 0
    num_versions: int = 0
    last_modified: Optional[datetime] = None
    avg_file_size_mb: float = 0.0
    min_file_size_mb: float = 0.0
    max_file_size_mb: float = 0.0
    partition_count: int = 0
    optimization_score: float = 0.0


class DeltaLakeManager:
    """
    Comprehensive Delta Lake management system.
    
    Features:
    - ACID transaction management
    - Schema evolution and enforcement
    - Advanced optimization strategies
    - Time travel and versioning
    - Change Data Feed (CDF)
    - Liquid clustering
    - Automated maintenance
    - Performance monitoring
    - Data lineage tracking
    """
    
    def __init__(self, spark: SparkSession, config: Dict[str, Any]):
        self.spark = self._configure_delta_spark(spark)
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Table registry
        self.registered_tables: Dict[str, DeltaTableConfig] = {}
        self.table_metrics: Dict[str, TableMetrics] = {}
        
        # Optimization tracking
        self.optimization_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Configure Delta Lake settings
        self._configure_delta_settings()
    
    def _configure_delta_spark(self, spark: SparkSession) -> SparkSession:
        """Configure Spark with Delta Lake"""
        try:
            # Configure Delta Lake extensions
            spark.conf.set("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            spark.conf.set("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            
            return spark
            
        except Exception as e:
            self.logger.error(f"Error configuring Delta Spark: {str(e)}")
            return spark
    
    def _configure_delta_settings(self):
        """Configure Delta Lake specific settings"""
        try:
            # Performance optimizations
            self.spark.conf.set("spark.databricks.delta.optimizeWrite.enabled", "true")
            self.spark.conf.set("spark.databricks.delta.autoCompact.enabled", "true")
            
            # Schema enforcement
            self.spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled", "false")
            
            # Change Data Feed
            self.spark.conf.set("spark.databricks.delta.properties.defaults.enableChangeDataFeed", "false")
            
            # Deletion Vectors
            self.spark.conf.set("spark.databricks.delta.properties.defaults.enableDeletionVectors", "true")
            
            # Predictive optimization
            self.spark.conf.set("spark.databricks.delta.predictiveOptimization.enabled", "true")
            
            # Checkpointing
            self.spark.conf.set("spark.databricks.delta.checkpoint.writeStatsAsJson", "true")
            
        except Exception as e:
            self.logger.warning(f"Could not set Delta configuration: {str(e)}")
    
    def create_table(
        self,
        config: DeltaTableConfig,
        df: Optional[DataFrame] = None,
        schema: Optional[StructType] = None,
        replace_if_exists: bool = False
    ) -> bool:
        """
        Create a new Delta table
        
        Args:
            config: Delta table configuration
            df: Optional DataFrame to create table from
            schema: Optional schema definition
            replace_if_exists: Whether to replace existing table
            
        Returns:
            bool: Success status
        """
        try:
            self.logger.info(f"Creating Delta table: {config.table_name}")
            
            # Check if table exists
            table_exists = self._table_exists(config.path)
            
            if table_exists and not replace_if_exists:
                self.logger.warning(f"Table {config.table_name} already exists")
                return False
            
            # Create table builder
            builder = self.spark.createDataFrame([], schema) if schema else df
            
            if builder is None:
                raise ValueError("Either DataFrame or schema must be provided")
            
            # Configure table properties
            table_properties = self._build_table_properties(config)
            
            # Write the table
            writer = builder.write.format("delta")
            
            # Add table properties
            for key, value in table_properties.items():
                writer = writer.option(key, value)
            
            # Partition if specified
            if config.partition_columns:
                writer = writer.partitionBy(*config.partition_columns)
            
            # Write mode
            mode = "overwrite" if replace_if_exists else "error"
            writer = writer.mode(mode)
            
            # Save the table
            writer.save(config.path)
            
            # Register table
            self.registered_tables[config.table_name] = config
            
            # Initialize metrics
            self._update_table_metrics(config.table_name)
            
            self.logger.info(f"Delta table {config.table_name} created successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating Delta table {config.table_name}: {str(e)}")
            return False
    
    def _build_table_properties(self, config: DeltaTableConfig) -> Dict[str, str]:
        """Build Delta table properties from configuration"""
        properties = {}
        
        # Change Data Feed
        if config.enable_change_data_feed:
            properties["delta.enableChangeDataFeed"] = "true"
        
        # Auto optimization
        if config.auto_optimize:
            properties["delta.autoOptimize.optimizeWrite"] = "true"
            properties["delta.autoOptimize.autoCompact"] = "true"
        
        # Deletion vectors
        if config.enable_deletion_vectors:
            properties["delta.enableDeletionVectors"] = "true"
        
        # Retention settings
        properties["delta.logRetentionDuration"] = config.log_retention_duration
        properties["delta.deletedFileRetentionDuration"] = config.deleted_file_retention_duration
        
        # Checkpoint interval
        properties["delta.checkpointInterval"] = str(config.checkpoint_interval)
        
        # Target file size
        properties["delta.tuneFileSizesForRewrites"] = str(config.tune_file_sizes).lower()
        properties["delta.targetFileSize"] = f"{config.target_file_size_mb}MB"
        
        # Column mapping
        if config.column_mapping_mode != "none":
            properties["delta.columnMapping.mode"] = config.column_mapping_mode
        
        # Data skipping
        properties["delta.dataSkippingEnabled"] = str(config.data_skipping_enabled).lower()
        
        return properties
    
    def write_to_table(
        self,
        table_name: str,
        df: DataFrame,
        mode: str = "append",
        merge_condition: Optional[str] = None,
        update_condition: Optional[str] = None,
        when_matched_update: Optional[Dict[str, str]] = None,
        when_not_matched_insert: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Write data to Delta table with various modes
        
        Args:
            table_name: Name of the table
            df: DataFrame to write
            mode: Write mode (append, overwrite, merge)
            merge_condition: Condition for merge operation
            update_condition: Additional update condition
            when_matched_update: Column mappings for matched records
            when_not_matched_insert: Column mappings for new records
            
        Returns:
            bool: Success status
        """
        try:
            if table_name not in self.registered_tables:
                raise ValueError(f"Table {table_name} not registered")
            
            config = self.registered_tables[table_name]
            
            if mode == "merge":
                return self._merge_into_table(
                    config, df, merge_condition, update_condition,
                    when_matched_update, when_not_matched_insert
                )
            else:
                # Standard write operations
                writer = df.write.format("delta").mode(mode)
                
                # Add optimizations
                if config.optimize_write:
                    writer = writer.option("optimizeWrite", "true")
                
                writer.save(config.path)
                
                # Update metrics
                self._update_table_metrics(table_name)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error writing to table {table_name}: {str(e)}")
            return False
    
    def _merge_into_table(
        self,
        config: DeltaTableConfig,
        source_df: DataFrame,
        merge_condition: str,
        update_condition: Optional[str],
        when_matched_update: Optional[Dict[str, str]],
        when_not_matched_insert: Optional[Dict[str, str]]
    ) -> bool:
        """Perform merge operation on Delta table"""
        try:
            delta_table = DeltaTable.forPath(self.spark, config.path)
            
            # Start merge operation
            merge_builder = delta_table.alias("target").merge(
                source_df.alias("source"),
                merge_condition
            )
            
            # Configure when matched clause
            if when_matched_update:
                if update_condition:
                    merge_builder = merge_builder.whenMatchedUpdate(
                        condition=update_condition,
                        set=when_matched_update
                    )
                else:
                    merge_builder = merge_builder.whenMatchedUpdateAll()
            
            # Configure when not matched clause
            if when_not_matched_insert:
                merge_builder = merge_builder.whenNotMatchedInsert(
                    values=when_not_matched_insert
                )
            else:
                merge_builder = merge_builder.whenNotMatchedInsertAll()
            
            # Execute merge
            merge_builder.execute()
            
            # Update metrics
            self._update_table_metrics(config.table_name)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in merge operation: {str(e)}")
            return False
    
    def read_table(
        self,
        table_name: str,
        version: Optional[int] = None,
        timestamp: Optional[str] = None,
        columns: Optional[List[str]] = None,
        filter_condition: Optional[str] = None
    ) -> Optional[DataFrame]:
        """
        Read from Delta table with time travel support
        
        Args:
            table_name: Name of the table
            version: Specific version to read
            timestamp: Specific timestamp to read
            columns: Columns to select
            filter_condition: Filter condition
            
        Returns:
            Optional[DataFrame]: DataFrame or None if error
        """
        try:
            if table_name not in self.registered_tables:
                raise ValueError(f"Table {table_name} not registered")
            
            config = self.registered_tables[table_name]
            
            # Build reader
            reader = self.spark.read.format("delta")
            
            # Add time travel options
            if version is not None:
                reader = reader.option("versionAsOf", version)
            elif timestamp is not None:
                reader = reader.option("timestampAsOf", timestamp)
            
            # Load the table
            df = reader.load(config.path)
            
            # Apply column selection
            if columns:
                df = df.select(*columns)
            
            # Apply filters
            if filter_condition:
                df = df.filter(filter_condition)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error reading table {table_name}: {str(e)}")
            return None
    
    def read_change_data_feed(
        self,
        table_name: str,
        start_version: Optional[int] = None,
        end_version: Optional[int] = None,
        start_timestamp: Optional[str] = None,
        end_timestamp: Optional[str] = None
    ) -> Optional[DataFrame]:
        """
        Read Change Data Feed from Delta table
        
        Args:
            table_name: Name of the table
            start_version: Start version for CDF
            end_version: End version for CDF
            start_timestamp: Start timestamp for CDF
            end_timestamp: End timestamp for CDF
            
        Returns:
            Optional[DataFrame]: CDF DataFrame
        """
        try:
            if table_name not in self.registered_tables:
                raise ValueError(f"Table {table_name} not registered")
            
            config = self.registered_tables[table_name]
            
            if not config.enable_change_data_feed:
                raise ValueError(f"Change Data Feed not enabled for table {table_name}")
            
            # Build CDF reader
            reader = self.spark.read.format("delta").option("readChangeFeed", "true")
            
            # Add version range
            if start_version is not None:
                reader = reader.option("startingVersion", start_version)
            if end_version is not None:
                reader = reader.option("endingVersion", end_version)
            
            # Add timestamp range
            if start_timestamp is not None:
                reader = reader.option("startingTimestamp", start_timestamp)
            if end_timestamp is not None:
                reader = reader.option("endingTimestamp", end_timestamp)
            
            return reader.load(config.path)
            
        except Exception as e:
            self.logger.error(f"Error reading CDF for table {table_name}: {str(e)}")
            return None
    
    def optimize_table(
        self,
        table_name: str,
        strategy: OptimizationStrategy = OptimizationStrategy.AUTO,
        zorder_columns: Optional[List[str]] = None,
        where_condition: Optional[str] = None
    ) -> bool:
        """
        Optimize Delta table using various strategies
        
        Args:
            table_name: Name of the table to optimize
            strategy: Optimization strategy to use
            zorder_columns: Columns for Z-ordering
            where_condition: Where condition for selective optimization
            
        Returns:
            bool: Success status
        """
        try:
            if table_name not in self.registered_tables:
                raise ValueError(f"Table {table_name} not registered")
            
            config = self.registered_tables[table_name]
            delta_table = DeltaTable.forPath(self.spark, config.path)
            
            optimization_start = datetime.now()
            
            if strategy == OptimizationStrategy.ZORDER:
                # Z-order optimization
                columns = zorder_columns or config.zorder_columns
                if columns:
                    if where_condition:
                        delta_table.optimize().where(where_condition).executeZOrderBy(*columns)
                    else:
                        delta_table.optimize().executeZOrderBy(*columns)
                    
                    self.logger.info(f"Z-order optimization completed for {table_name}")
                else:
                    self.logger.warning(f"No Z-order columns specified for {table_name}")
            
            elif strategy == OptimizationStrategy.COMPACT:
                # File compaction
                if where_condition:
                    delta_table.optimize().where(where_condition).executeCompaction()
                else:
                    delta_table.optimize().executeCompaction()
                
                self.logger.info(f"Compaction completed for {table_name}")
            
            elif strategy == OptimizationStrategy.AUTO:
                # Auto optimization (compaction + Z-order if configured)
                optimize_builder = delta_table.optimize()
                
                if where_condition:
                    optimize_builder = optimize_builder.where(where_condition)
                
                if config.zorder_columns:
                    optimize_builder.executeZOrderBy(*config.zorder_columns)
                else:
                    optimize_builder.executeCompaction()
                
                self.logger.info(f"Auto optimization completed for {table_name}")
            
            # Record optimization history
            optimization_end = datetime.now()
            self._record_optimization(
                table_name, strategy, optimization_start, optimization_end
            )
            
            # Update metrics
            self._update_table_metrics(table_name)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error optimizing table {table_name}: {str(e)}")
            return False
    
    def vacuum_table(
        self,
        table_name: str,
        retention_hours: Optional[int] = None,
        dry_run: bool = False
    ) -> bool:
        """
        Vacuum Delta table to remove old files
        
        Args:
            table_name: Name of the table
            retention_hours: Retention period in hours
            dry_run: Whether to perform dry run
            
        Returns:
            bool: Success status
        """
        try:
            if table_name not in self.registered_tables:
                raise ValueError(f"Table {table_name} not registered")
            
            config = self.registered_tables[table_name]
            delta_table = DeltaTable.forPath(self.spark, config.path)
            
            if retention_hours is not None:
                if dry_run:
                    result = delta_table.vacuum(retention_hours, dry_run=True)
                    self.logger.info(f"Vacuum dry run for {table_name}: {result}")
                else:
                    delta_table.vacuum(retention_hours)
                    self.logger.info(f"Vacuum completed for {table_name}")
            else:
                if dry_run:
                    result = delta_table.vacuum(dry_run=True)
                    self.logger.info(f"Vacuum dry run for {table_name}: {result}")
                else:
                    delta_table.vacuum()
                    self.logger.info(f"Vacuum completed for {table_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error vacuuming table {table_name}: {str(e)}")
            return False
    
    def get_table_history(self, table_name: str, limit: int = 100) -> Optional[DataFrame]:
        """Get version history of Delta table"""
        try:
            if table_name not in self.registered_tables:
                raise ValueError(f"Table {table_name} not registered")
            
            config = self.registered_tables[table_name]
            delta_table = DeltaTable.forPath(self.spark, config.path)
            
            return delta_table.history(limit)
            
        except Exception as e:
            self.logger.error(f"Error getting history for table {table_name}: {str(e)}")
            return None
    
    def _table_exists(self, path: str) -> bool:
        """Check if Delta table exists at path"""
        try:
            DeltaTable.forPath(self.spark, path)
            return True
        except Exception:
            return False
    
    def _update_table_metrics(self, table_name: str):
        """Update metrics for a Delta table"""
        try:
            config = self.registered_tables[table_name]
            delta_table = DeltaTable.forPath(self.spark, config.path)
            
            # Get table details
            details = delta_table.detail().collect()[0]
            
            metrics = TableMetrics(
                num_files=details.get("numFiles", 0),
                size_in_bytes=details.get("sizeInBytes", 0),
                num_records=details.get("numRecords", 0),
                last_modified=details.get("lastModified"),
                partition_count=len(details.get("partitionColumns", []))
            )
            
            # Calculate file size statistics
            if metrics.num_files > 0:
                metrics.avg_file_size_mb = metrics.size_in_bytes / metrics.num_files / (1024 * 1024)
            
            # Get version count from history
            history = delta_table.history(1000)  # Last 1000 versions
            metrics.num_versions = history.count()
            
            # Store metrics
            self.table_metrics[table_name] = metrics
            
        except Exception as e:
            self.logger.error(f"Error updating metrics for table {table_name}: {str(e)}")
    
    def _record_optimization(
        self,
        table_name: str,
        strategy: OptimizationStrategy,
        start_time: datetime,
        end_time: datetime
    ):
        """Record optimization history"""
        if table_name not in self.optimization_history:
            self.optimization_history[table_name] = []
        
        self.optimization_history[table_name].append({
            "strategy": strategy.value,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds()
        })
    
    def get_table_metrics(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific table"""
        if table_name not in self.table_metrics:
            self._update_table_metrics(table_name)
        
        metrics = self.table_metrics.get(table_name)
        if metrics:
            return {
                "table_name": table_name,
                "num_files": metrics.num_files,
                "size_in_bytes": metrics.size_in_bytes,
                "size_mb": metrics.size_in_bytes / (1024 * 1024),
                "num_records": metrics.num_records,
                "num_versions": metrics.num_versions,
                "last_modified": metrics.last_modified.isoformat() if metrics.last_modified else None,
                "avg_file_size_mb": metrics.avg_file_size_mb,
                "partition_count": metrics.partition_count,
                "optimization_score": metrics.optimization_score
            }
        return None
    
    def list_tables(self) -> List[Dict[str, Any]]:
        """List all registered tables with their metrics"""
        tables = []
        for table_name in self.registered_tables:
            table_info = {
                "name": table_name,
                "config": self.registered_tables[table_name],
                "metrics": self.get_table_metrics(table_name)
            }
            tables.append(table_info)
        return tables
    
    def cleanup(self):
        """Cleanup Delta Lake manager resources"""
        try:
            self.logger.info("Delta Lake manager cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")


# Utility functions

def create_delta_config(
    table_name: str,
    path: str,
    partition_columns: Optional[List[str]] = None,
    enable_cdf: bool = False,
    auto_optimize: bool = True
) -> DeltaTableConfig:
    """Create a basic Delta table configuration"""
    return DeltaTableConfig(
        table_name=table_name,
        path=path,
        partition_columns=partition_columns,
        enable_change_data_feed=enable_cdf,
        auto_optimize=auto_optimize
    )