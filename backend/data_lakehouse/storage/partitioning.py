"""
Advanced Partitioning Strategies

Intelligent partitioning system for optimal query performance:
- Multi-dimensional partitioning
- Dynamic partition pruning
- Partition size optimization
- Automated partition management
- Partition evolution strategies
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import math

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import *


class PartitionStrategy(Enum):
    """Partitioning strategy types"""
    RANGE = "range"
    HASH = "hash"
    LIST = "list"
    HYBRID = "hybrid"
    TIME_BASED = "time_based"
    SIZE_BASED = "size_based"


class PartitionGranularity(Enum):
    """Time-based partitioning granularities"""
    YEAR = "year"
    QUARTER = "quarter"
    MONTH = "month"
    WEEK = "week"
    DAY = "day"
    HOUR = "hour"


@dataclass
class PartitionConfig:
    """Configuration for table partitioning"""
    strategy: PartitionStrategy
    columns: List[str]
    granularity: Optional[PartitionGranularity] = None
    max_partition_size_mb: int = 1024
    min_partition_size_mb: int = 128
    max_partitions: int = 10000
    enable_dynamic_pruning: bool = True
    enable_partition_evolution: bool = True
    custom_partition_func: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PartitionInfo:
    """Information about a table partition"""
    partition_values: Dict[str, str]
    file_count: int
    size_bytes: int
    record_count: int
    last_modified: datetime
    min_values: Dict[str, Any]
    max_values: Dict[str, Any]
    is_optimal: bool = False


class PartitioningStrategy:
    """
    Base class for partitioning strategies
    """
    
    def __init__(self, config: PartitionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def generate_partition_expression(self, df: DataFrame) -> List[str]:
        """Generate partition expressions for the DataFrame"""
        raise NotImplementedError("Subclasses must implement generate_partition_expression")
    
    def estimate_partition_count(self, df: DataFrame) -> int:
        """Estimate the number of partitions that will be created"""
        raise NotImplementedError("Subclasses must implement estimate_partition_count")
    
    def validate_partitioning(self, df: DataFrame) -> List[str]:
        """Validate if the partitioning strategy is appropriate for the data"""
        errors = []
        
        # Check if partition columns exist
        for col in self.config.columns:
            if col not in df.columns:
                errors.append(f"Partition column '{col}' not found in DataFrame")
        
        # Estimate partition count
        try:
            estimated_partitions = self.estimate_partition_count(df)
            if estimated_partitions > self.config.max_partitions:
                errors.append(f"Estimated partition count ({estimated_partitions}) exceeds maximum ({self.config.max_partitions})")
        except Exception as e:
            errors.append(f"Error estimating partition count: {str(e)}")
        
        return errors


class TimeBasedPartitioning(PartitioningStrategy):
    """Time-based partitioning strategy"""
    
    def generate_partition_expression(self, df: DataFrame) -> List[str]:
        """Generate time-based partition expressions"""
        if not self.config.columns:
            raise ValueError("Time column must be specified for time-based partitioning")
        
        time_column = self.config.columns[0]
        expressions = []
        
        if self.config.granularity == PartitionGranularity.YEAR:
            expressions.append(f"year({time_column}) as partition_year")
        elif self.config.granularity == PartitionGranularity.QUARTER:
            expressions.extend([
                f"year({time_column}) as partition_year",
                f"quarter({time_column}) as partition_quarter"
            ])
        elif self.config.granularity == PartitionGranularity.MONTH:
            expressions.extend([
                f"year({time_column}) as partition_year",
                f"month({time_column}) as partition_month"
            ])
        elif self.config.granularity == PartitionGranularity.WEEK:
            expressions.extend([
                f"year({time_column}) as partition_year",
                f"weekofyear({time_column}) as partition_week"
            ])
        elif self.config.granularity == PartitionGranularity.DAY:
            expressions.extend([
                f"year({time_column}) as partition_year",
                f"month({time_column}) as partition_month", 
                f"dayofmonth({time_column}) as partition_day"
            ])
        elif self.config.granularity == PartitionGranularity.HOUR:
            expressions.extend([
                f"year({time_column}) as partition_year",
                f"month({time_column}) as partition_month",
                f"dayofmonth({time_column}) as partition_day",
                f"hour({time_column}) as partition_hour"
            ])
        
        return expressions
    
    def estimate_partition_count(self, df: DataFrame) -> int:
        """Estimate partition count for time-based partitioning"""
        time_column = self.config.columns[0]
        
        # Get time range
        time_stats = df.select(
            min(col(time_column)).alias("min_time"),
            max(col(time_column)).alias("max_time")
        ).collect()[0]
        
        if not time_stats["min_time"] or not time_stats["max_time"]:
            return 0
        
        min_time = time_stats["min_time"]
        max_time = time_stats["max_time"]
        
        # Calculate partition count based on granularity
        time_diff = max_time - min_time
        
        if self.config.granularity == PartitionGranularity.YEAR:
            return max(1, time_diff.days // 365)
        elif self.config.granularity == PartitionGranularity.QUARTER:
            return max(1, time_diff.days // 90)
        elif self.config.granularity == PartitionGranularity.MONTH:
            return max(1, time_diff.days // 30)
        elif self.config.granularity == PartitionGranularity.WEEK:
            return max(1, time_diff.days // 7)
        elif self.config.granularity == PartitionGranularity.DAY:
            return max(1, time_diff.days)
        elif self.config.granularity == PartitionGranularity.HOUR:
            return max(1, int(time_diff.total_seconds() // 3600))
        
        return 1


class HashBasedPartitioning(PartitioningStrategy):
    """Hash-based partitioning strategy"""
    
    def generate_partition_expression(self, df: DataFrame) -> List[str]:
        """Generate hash-based partition expressions"""
        if not self.config.columns:
            raise ValueError("Hash columns must be specified")
        
        # Calculate optimal number of hash buckets
        record_count = df.count()
        optimal_partitions = min(
            self.config.max_partitions,
            max(1, record_count // 1000000)  # ~1M records per partition
        )
        
        hash_expr = "hash(" + ", ".join(self.config.columns) + f") % {optimal_partitions}"
        return [f"({hash_expr}) as partition_hash"]
    
    def estimate_partition_count(self, df: DataFrame) -> int:
        """Estimate partition count for hash-based partitioning"""
        record_count = df.count()
        return min(self.config.max_partitions, max(1, record_count // 1000000))


class SizeBasedPartitioning(PartitioningStrategy):
    """Size-based dynamic partitioning"""
    
    def generate_partition_expression(self, df: DataFrame) -> List[str]:
        """Generate size-based partition expressions"""
        # This would typically be implemented with custom logic
        # For now, fall back to simple partitioning
        if self.config.columns:
            return [f"{self.config.columns[0]} as partition_key"]
        else:
            # Use monotonically increasing ID for size-based partitioning
            return ["monotonically_increasing_id() as partition_id"]
    
    def estimate_partition_count(self, df: DataFrame) -> int:
        """Estimate partition count based on data size"""
        # Estimate data size
        estimated_size_mb = df.count() * len(df.columns) * 100 / (1024 * 1024)  # Rough estimate
        
        target_partition_size = (self.config.max_partition_size_mb + self.config.min_partition_size_mb) / 2
        estimated_partitions = max(1, int(estimated_size_mb / target_partition_size))
        
        return min(self.config.max_partitions, estimated_partitions)


class PartitionManager:
    """
    Comprehensive partition management system
    
    Features:
    - Multi-strategy partitioning
    - Automatic partition optimization
    - Partition monitoring and analytics
    - Dynamic partition evolution
    - Partition pruning optimization
    """
    
    def __init__(self, spark: SparkSession, config: Dict[str, Any]):
        self.spark = spark
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Partition strategies registry
        self.strategies = {
            PartitionStrategy.TIME_BASED: TimeBasedPartitioning,
            PartitionStrategy.HASH: HashBasedPartitioning,
            PartitionStrategy.SIZE_BASED: SizeBasedPartitioning
        }
        
        # Table partition configurations
        self.table_configs: Dict[str, PartitionConfig] = {}
        
        # Partition analytics
        self.partition_metrics: Dict[str, Dict[str, PartitionInfo]] = {}
    
    def configure_partitioning(
        self,
        table_name: str,
        df: DataFrame,
        strategy: PartitionStrategy,
        columns: List[str],
        **kwargs
    ) -> PartitionConfig:
        """
        Configure partitioning for a table
        
        Args:
            table_name: Name of the table
            df: DataFrame to analyze for partitioning
            strategy: Partitioning strategy
            columns: Columns to partition by
            **kwargs: Additional configuration options
            
        Returns:
            PartitionConfig: Configured partitioning
        """
        try:
            # Create partition configuration
            config = PartitionConfig(
                strategy=strategy,
                columns=columns,
                granularity=kwargs.get("granularity"),
                max_partition_size_mb=kwargs.get("max_partition_size_mb", 1024),
                min_partition_size_mb=kwargs.get("min_partition_size_mb", 128),
                max_partitions=kwargs.get("max_partitions", 10000),
                enable_dynamic_pruning=kwargs.get("enable_dynamic_pruning", True),
                enable_partition_evolution=kwargs.get("enable_partition_evolution", True),
                custom_partition_func=kwargs.get("custom_partition_func"),
                metadata=kwargs.get("metadata", {})
            )
            
            # Validate configuration
            strategy_impl = self.strategies[strategy](config)
            validation_errors = strategy_impl.validate_partitioning(df)
            
            if validation_errors:
                raise ValueError(f"Partition validation failed: {validation_errors}")
            
            # Store configuration
            self.table_configs[table_name] = config
            
            self.logger.info(f"Configured {strategy.value} partitioning for table {table_name}")
            return config
            
        except Exception as e:
            self.logger.error(f"Error configuring partitioning for {table_name}: {str(e)}")
            raise
    
    def apply_partitioning(self, table_name: str, df: DataFrame) -> DataFrame:
        """
        Apply partitioning to a DataFrame
        
        Args:
            table_name: Name of the table
            df: DataFrame to partition
            
        Returns:
            DataFrame: Partitioned DataFrame
        """
        try:
            if table_name not in self.table_configs:
                raise ValueError(f"No partition configuration found for table {table_name}")
            
            config = self.table_configs[table_name]
            strategy_impl = self.strategies[config.strategy](config)
            
            # Generate partition expressions
            partition_expressions = strategy_impl.generate_partition_expression(df)
            
            # Add partition columns to DataFrame
            partitioned_df = df
            for expr in partition_expressions:
                partitioned_df = partitioned_df.withColumn(
                    expr.split(" as ")[-1], 
                    expr(expr.split(" as ")[0])
                )
            
            return partitioned_df
            
        except Exception as e:
            self.logger.error(f"Error applying partitioning to {table_name}: {str(e)}")
            raise
    
    def analyze_partitions(self, table_name: str, table_path: str) -> Dict[str, PartitionInfo]:
        """
        Analyze partition characteristics and performance
        
        Args:
            table_name: Name of the table
            table_path: Path to the table
            
        Returns:
            Dict[str, PartitionInfo]: Partition analysis results
        """
        try:
            # Read table metadata
            df = self.spark.read.format("delta").load(table_path)
            
            # Get partition information (this is simplified)
            partition_info = {}
            
            if table_name in self.table_configs:
                config = self.table_configs[table_name]
                
                # Analyze each partition
                for partition_col in config.columns:
                    if partition_col in df.columns:
                        partition_stats = df.groupBy(partition_col).agg(
                            count("*").alias("record_count"),
                            min("*").alias("min_values"),
                            max("*").alias("max_values")
                        ).collect()
                        
                        for stat in partition_stats:
                            partition_key = str(stat[partition_col])
                            
                            partition_info[partition_key] = PartitionInfo(
                                partition_values={partition_col: partition_key},
                                file_count=1,  # Simplified
                                size_bytes=stat["record_count"] * 1000,  # Estimated
                                record_count=stat["record_count"],
                                last_modified=datetime.now(),
                                min_values={"value": stat["min_values"]},
                                max_values={"value": stat["max_values"]},
                                is_optimal=self._is_partition_optimal(stat["record_count"] * 1000, config)
                            )
            
            # Store partition metrics
            self.partition_metrics[table_name] = partition_info
            
            return partition_info
            
        except Exception as e:
            self.logger.error(f"Error analyzing partitions for {table_name}: {str(e)}")
            return {}
    
    def _is_partition_optimal(self, size_bytes: int, config: PartitionConfig) -> bool:
        """Check if partition size is within optimal range"""
        size_mb = size_bytes / (1024 * 1024)
        return config.min_partition_size_mb <= size_mb <= config.max_partition_size_mb
    
    def recommend_partitioning(self, df: DataFrame, table_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Recommend optimal partitioning strategy for a DataFrame
        
        Args:
            df: DataFrame to analyze
            table_name: Optional table name for context
            
        Returns:
            Dict[str, Any]: Partitioning recommendations
        """
        try:
            recommendations = {
                "table_name": table_name,
                "analysis_timestamp": datetime.now().isoformat(),
                "data_characteristics": {},
                "recommendations": []
            }
            
            # Analyze data characteristics
            record_count = df.count()
            column_count = len(df.columns)
            estimated_size_mb = record_count * column_count * 100 / (1024 * 1024)
            
            recommendations["data_characteristics"] = {
                "record_count": record_count,
                "column_count": column_count,
                "estimated_size_mb": estimated_size_mb
            }
            
            # Analyze columns for partitioning suitability
            date_columns = []
            high_cardinality_columns = []
            low_cardinality_columns = []
            
            for column in df.columns:
                col_type = dict(df.dtypes)[column]
                
                if "timestamp" in col_type.lower() or "date" in col_type.lower():
                    date_columns.append(column)
                
                # Analyze cardinality (simplified)
                distinct_count = df.select(column).distinct().count()
                cardinality_ratio = distinct_count / record_count if record_count > 0 else 0
                
                if cardinality_ratio > 0.1:
                    high_cardinality_columns.append((column, cardinality_ratio))
                elif cardinality_ratio < 0.01:
                    low_cardinality_columns.append((column, cardinality_ratio))
            
            # Generate recommendations
            if date_columns and estimated_size_mb > 1024:  # > 1GB
                recommendations["recommendations"].append({
                    "strategy": PartitionStrategy.TIME_BASED.value,
                    "columns": date_columns[:1],  # Use first date column
                    "granularity": PartitionGranularity.MONTH.value if estimated_size_mb > 10240 else PartitionGranularity.DAY.value,
                    "reason": "Large dataset with date/timestamp columns suitable for time-based partitioning",
                    "priority": "high"
                })
            
            if high_cardinality_columns and record_count > 10000000:  # > 10M records
                recommendations["recommendations"].append({
                    "strategy": PartitionStrategy.HASH.value,
                    "columns": [col[0] for col in high_cardinality_columns[:2]],
                    "reason": "High-cardinality columns suitable for hash-based distribution",
                    "priority": "medium"
                })
            
            if estimated_size_mb > 5000:  # > 5GB
                recommendations["recommendations"].append({
                    "strategy": PartitionStrategy.SIZE_BASED.value,
                    "columns": [],
                    "reason": "Large dataset benefits from size-based partitioning",
                    "priority": "low"
                })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating partition recommendations: {str(e)}")
            return {"error": str(e)}
    
    def optimize_partitions(self, table_name: str, table_path: str) -> Dict[str, Any]:
        """
        Optimize existing partitions
        
        Args:
            table_name: Name of the table
            table_path: Path to the table
            
        Returns:
            Dict[str, Any]: Optimization results
        """
        try:
            # Analyze current partitions
            partition_info = self.analyze_partitions(table_name, table_path)
            
            optimization_results = {
                "table_name": table_name,
                "optimization_timestamp": datetime.now().isoformat(),
                "current_partitions": len(partition_info),
                "optimizations": []
            }
            
            # Identify optimization opportunities
            small_partitions = []
            large_partitions = []
            
            for partition_key, info in partition_info.items():
                if not info.is_optimal:
                    size_mb = info.size_bytes / (1024 * 1024)
                    config = self.table_configs.get(table_name)
                    
                    if config:
                        if size_mb < config.min_partition_size_mb:
                            small_partitions.append((partition_key, info))
                        elif size_mb > config.max_partition_size_mb:
                            large_partitions.append((partition_key, info))
            
            # Recommend optimizations
            if small_partitions:
                optimization_results["optimizations"].append({
                    "type": "merge_small_partitions",
                    "affected_partitions": len(small_partitions),
                    "description": f"Merge {len(small_partitions)} small partitions to improve performance"
                })
            
            if large_partitions:
                optimization_results["optimizations"].append({
                    "type": "split_large_partitions", 
                    "affected_partitions": len(large_partitions),
                    "description": f"Split {len(large_partitions)} large partitions for better parallelism"
                })
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Error optimizing partitions for {table_name}: {str(e)}")
            return {"error": str(e)}
    
    def get_partition_metrics(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get partition metrics for a table"""
        if table_name not in self.partition_metrics:
            return None
        
        metrics = self.partition_metrics[table_name]
        
        return {
            "table_name": table_name,
            "total_partitions": len(metrics),
            "optimal_partitions": sum(1 for info in metrics.values() if info.is_optimal),
            "total_size_mb": sum(info.size_bytes for info in metrics.values()) / (1024 * 1024),
            "total_records": sum(info.record_count for info in metrics.values()),
            "avg_partition_size_mb": sum(info.size_bytes for info in metrics.values()) / len(metrics) / (1024 * 1024) if metrics else 0,
            "partitions": {k: {
                "size_mb": v.size_bytes / (1024 * 1024),
                "record_count": v.record_count,
                "is_optimal": v.is_optimal
            } for k, v in metrics.items()}
        }
    
    def cleanup(self):
        """Cleanup partition manager resources"""
        try:
            self.table_configs.clear()
            self.partition_metrics.clear()
            self.logger.info("Partition manager cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during partition manager cleanup: {str(e)}")


# Utility functions

def auto_partition_table(
    spark: SparkSession,
    df: DataFrame,
    table_name: str,
    max_partition_size_mb: int = 1024
) -> Tuple[DataFrame, PartitionConfig]:
    """
    Automatically determine and apply optimal partitioning
    
    Args:
        spark: SparkSession
        df: DataFrame to partition
        table_name: Name of the table
        max_partition_size_mb: Maximum partition size in MB
        
    Returns:
        Tuple[DataFrame, PartitionConfig]: Partitioned DataFrame and config
    """
    manager = PartitionManager(spark, {})
    
    # Get recommendations
    recommendations = manager.recommend_partitioning(df, table_name)
    
    if recommendations.get("recommendations"):
        # Use the highest priority recommendation
        top_recommendation = recommendations["recommendations"][0]
        strategy = PartitionStrategy(top_recommendation["strategy"])
        columns = top_recommendation["columns"]
        
        # Configure partitioning
        config = manager.configure_partitioning(
            table_name, df, strategy, columns,
            max_partition_size_mb=max_partition_size_mb,
            granularity=top_recommendation.get("granularity")
        )
        
        # Apply partitioning
        partitioned_df = manager.apply_partitioning(table_name, df)
        
        return partitioned_df, config
    else:
        # Return original DataFrame with no partitioning
        return df, PartitionConfig(PartitionStrategy.RANGE, [])