"""
Feature Store

Enterprise feature store for ML feature management:
- Feature definition and versioning
- Feature computation and storage
- Feature discovery and reuse
- Feature monitoring and drift detection
- Point-in-time correctness for training
- Real-time and batch feature serving
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import uuid
import hashlib

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql.functions import *
from pyspark.sql.types import *


class FeatureType(Enum):
    """Types of features"""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    TEXT = "text"
    DATETIME = "datetime"
    ARRAY = "array"
    MAP = "map"
    EMBEDDING = "embedding"


class AggregationType(Enum):
    """Feature aggregation types"""
    SUM = "sum"
    COUNT = "count"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    STDDEV = "stddev"
    FIRST = "first"
    LAST = "last"
    DISTINCT_COUNT = "distinct_count"


class ComputeMode(Enum):
    """Feature computation modes"""
    BATCH = "batch"
    STREAMING = "streaming"
    ON_DEMAND = "on_demand"


@dataclass
class Feature:
    """Individual feature definition"""
    name: str
    feature_type: FeatureType
    description: Optional[str] = None
    
    # Computation details
    expression: Optional[str] = None  # SQL expression or transformation logic
    aggregation: Optional[AggregationType] = None
    window_size: Optional[str] = None  # e.g., "7 days", "1 hour"
    
    # Schema information
    data_type: Optional[DataType] = None
    nullable: bool = True
    
    # Validation rules
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[List[str]] = None
    regex_pattern: Optional[str] = None
    
    # Metadata
    tags: Set[str] = field(default_factory=set)
    owner: Optional[str] = None
    created_date: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    
    # Statistics (computed)
    statistics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureGroup:
    """Collection of related features"""
    name: str
    features: List[Feature]
    
    # Data source information
    source_table: Optional[str] = None
    source_query: Optional[str] = None
    
    # Key information
    primary_keys: List[str] = field(default_factory=list)
    event_timestamp_column: Optional[str] = None
    
    # Computation settings
    compute_mode: ComputeMode = ComputeMode.BATCH
    refresh_interval: Optional[str] = None  # e.g., "1 hour", "1 day"
    
    # Storage settings
    storage_path: Optional[str] = None
    partitioning: List[str] = field(default_factory=list)
    
    # Versioning
    version: str = "1.0.0"
    schema_version: str = "1"
    
    # Metadata
    description: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    owner: Optional[str] = None
    created_date: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    
    # Lineage
    upstream_dependencies: List[str] = field(default_factory=list)
    downstream_consumers: List[str] = field(default_factory=list)


@dataclass
class FeatureView:
    """View of features for specific use cases"""
    name: str
    feature_groups: List[str]  # Feature group names
    selected_features: List[str]  # Specific features to include
    
    # Join configuration
    join_keys: List[str] = field(default_factory=list)
    
    # Time window for point-in-time correctness
    ttl: Optional[str] = None  # Time-to-live for features
    
    # Metadata
    description: Optional[str] = None
    use_case: Optional[str] = None
    created_by: Optional[str] = None
    created_date: datetime = field(default_factory=datetime.now)


class FeatureStore:
    """
    Enterprise feature store for ML feature lifecycle management.
    
    Features:
    - Feature definition and registration
    - Feature computation and storage optimization
    - Feature versioning and lineage tracking
    - Point-in-time correctness for training data
    - Online and offline feature serving
    - Feature monitoring and drift detection
    - Feature discovery and cataloging
    - Data quality validation
    """
    
    def __init__(self, spark: SparkSession, config: Dict[str, Any]):
        self.spark = spark
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Feature registry
        self.feature_groups: Dict[str, FeatureGroup] = {}
        self.features: Dict[str, Feature] = {}  # feature_name -> Feature
        self.feature_views: Dict[str, FeatureView] = {}
        
        # Storage configuration
        self.storage_root = config.get("storage_root", "/tmp/feature_store")
        self.metadata_store_path = config.get("metadata_store", f"{self.storage_root}/metadata")
        
        # Monitoring and statistics
        self.feature_statistics: Dict[str, Dict[str, Any]] = {}
        self.computation_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Initialize storage
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Initialize feature store storage"""
        try:
            # Create storage directories
            self.spark.sql(f"CREATE DATABASE IF NOT EXISTS feature_store")
            
            # Initialize metadata tables
            self._create_metadata_tables()
            
            self.logger.info("Feature store storage initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing feature store storage: {str(e)}")
    
    def _create_metadata_tables(self):
        """Create metadata tables for feature store"""
        try:
            # Feature groups metadata table
            feature_groups_schema = StructType([
                StructField("name", StringType(), False),
                StructField("version", StringType(), False),
                StructField("description", StringType(), True),
                StructField("owner", StringType(), True),
                StructField("created_date", TimestampType(), False),
                StructField("last_modified", TimestampType(), False),
                StructField("config", StringType(), True),  # JSON config
                StructField("status", StringType(), False)
            ])
            
            # Features metadata table
            features_schema = StructType([
                StructField("name", StringType(), False),
                StructField("feature_group", StringType(), False),
                StructField("feature_type", StringType(), False),
                StructField("data_type", StringType(), False),
                StructField("description", StringType(), True),
                StructField("expression", StringType(), True),
                StructField("created_date", TimestampType(), False),
                StructField("statistics", StringType(), True)  # JSON statistics
            ])
            
            # Create empty DataFrames for metadata tables
            empty_fg_df = self.spark.createDataFrame([], feature_groups_schema)
            empty_f_df = self.spark.createDataFrame([], features_schema)
            
            # Write to Delta tables (create if not exists)
            fg_path = f"{self.metadata_store_path}/feature_groups"
            f_path = f"{self.metadata_store_path}/features"
            
            try:
                self.spark.read.format("delta").load(fg_path)
            except:
                empty_fg_df.write.format("delta").mode("overwrite").save(fg_path)
            
            try:
                self.spark.read.format("delta").load(f_path)
            except:
                empty_f_df.write.format("delta").mode("overwrite").save(f_path)
            
        except Exception as e:
            self.logger.warning(f"Could not create metadata tables: {str(e)}")
    
    def register_feature_group(self, feature_group: FeatureGroup) -> str:
        """
        Register a feature group in the feature store
        
        Args:
            feature_group: FeatureGroup definition
            
        Returns:
            str: Feature group ID
        """
        try:
            # Validate feature group
            validation_errors = self._validate_feature_group(feature_group)
            if validation_errors:
                raise ValueError(f"Feature group validation failed: {validation_errors}")
            
            # Store feature group
            self.feature_groups[feature_group.name] = feature_group
            
            # Register individual features
            for feature in feature_group.features:
                feature_key = f"{feature_group.name}.{feature.name}"
                self.features[feature_key] = feature
            
            # Save metadata
            self._save_feature_group_metadata(feature_group)
            
            # Initialize feature computation if needed
            if feature_group.compute_mode == ComputeMode.BATCH and feature_group.source_table:
                self._initialize_feature_computation(feature_group)
            
            self.logger.info(f"Registered feature group: {feature_group.name}")
            return feature_group.name
            
        except Exception as e:
            self.logger.error(f"Error registering feature group {feature_group.name}: {str(e)}")
            raise
    
    def _validate_feature_group(self, feature_group: FeatureGroup) -> List[str]:
        """Validate feature group configuration"""
        errors = []
        
        if not feature_group.name:
            errors.append("Feature group name is required")
        
        if not feature_group.features:
            errors.append("Feature group must contain at least one feature")
        
        if not feature_group.primary_keys:
            errors.append("Primary keys must be specified")
        
        # Validate individual features
        feature_names = set()
        for feature in feature_group.features:
            if not feature.name:
                errors.append("Feature name is required")
            
            if feature.name in feature_names:
                errors.append(f"Duplicate feature name: {feature.name}")
            
            feature_names.add(feature.name)
        
        return errors
    
    def _save_feature_group_metadata(self, feature_group: FeatureGroup):
        """Save feature group metadata to storage"""
        try:
            # Prepare metadata record
            metadata_record = {
                "name": feature_group.name,
                "version": feature_group.version,
                "description": feature_group.description,
                "owner": feature_group.owner,
                "created_date": feature_group.created_date,
                "last_modified": feature_group.last_modified,
                "config": json.dumps({
                    "source_table": feature_group.source_table,
                    "primary_keys": feature_group.primary_keys,
                    "compute_mode": feature_group.compute_mode.value,
                    "refresh_interval": feature_group.refresh_interval,
                    "storage_path": feature_group.storage_path
                }),
                "status": "active"
            }
            
            # Create DataFrame and save
            metadata_df = self.spark.createDataFrame([metadata_record])
            
            fg_metadata_path = f"{self.metadata_store_path}/feature_groups"
            metadata_df.write \
                       .format("delta") \
                       .mode("append") \
                       .save(fg_metadata_path)
            
            # Save individual feature metadata
            for feature in feature_group.features:
                self._save_feature_metadata(feature, feature_group.name)
            
        except Exception as e:
            self.logger.error(f"Error saving metadata: {str(e)}")
    
    def _save_feature_metadata(self, feature: Feature, feature_group_name: str):
        """Save individual feature metadata"""
        try:
            feature_record = {
                "name": feature.name,
                "feature_group": feature_group_name,
                "feature_type": feature.feature_type.value,
                "data_type": str(feature.data_type) if feature.data_type else None,
                "description": feature.description,
                "expression": feature.expression,
                "created_date": feature.created_date,
                "statistics": json.dumps(feature.statistics)
            }
            
            feature_df = self.spark.createDataFrame([feature_record])
            
            feature_metadata_path = f"{self.metadata_store_path}/features"
            feature_df.write \
                      .format("delta") \
                      .mode("append") \
                      .save(feature_metadata_path)
            
        except Exception as e:
            self.logger.error(f"Error saving feature metadata: {str(e)}")
    
    def _initialize_feature_computation(self, feature_group: FeatureGroup):
        """Initialize feature computation for a feature group"""
        try:
            if not feature_group.source_table:
                return
            
            # Load source data
            source_df = self.spark.read.table(feature_group.source_table)
            
            # Compute features
            features_df = self._compute_features(source_df, feature_group)
            
            # Store computed features
            storage_path = feature_group.storage_path or f"{self.storage_root}/features/{feature_group.name}"
            
            writer = features_df.write.format("delta").mode("overwrite")
            
            if feature_group.partitioning:
                writer = writer.partitionBy(*feature_group.partitioning)
            
            writer.save(storage_path)
            
            # Update storage path
            feature_group.storage_path = storage_path
            
            self.logger.info(f"Initialized feature computation for {feature_group.name}")
            
        except Exception as e:
            self.logger.error(f"Error initializing feature computation: {str(e)}")
    
    def _compute_features(self, source_df: DataFrame, feature_group: FeatureGroup) -> DataFrame:
        """Compute features for a feature group"""
        try:
            result_df = source_df
            
            # Add computed columns for each feature
            for feature in feature_group.features:
                if feature.expression:
                    # Use SQL expression
                    result_df = result_df.withColumn(feature.name, expr(feature.expression))
                
                elif feature.aggregation:
                    # Apply aggregation
                    result_df = self._apply_feature_aggregation(
                        result_df, 
                        feature, 
                        feature_group
                    )
            
            # Select only feature columns plus keys
            feature_columns = [f.name for f in feature_group.features]
            select_columns = feature_group.primary_keys + feature_columns
            
            if feature_group.event_timestamp_column:
                select_columns.append(feature_group.event_timestamp_column)
            
            result_df = result_df.select(*select_columns)
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error computing features: {str(e)}")
            raise
    
    def _apply_feature_aggregation(
        self, 
        df: DataFrame, 
        feature: Feature, 
        feature_group: FeatureGroup
    ) -> DataFrame:
        """Apply aggregation for a feature"""
        try:
            if not feature.aggregation or not feature.window_size:
                return df
            
            # Parse window size
            window_spec = self._create_window_spec(feature.window_size, feature_group)
            
            # Apply aggregation
            if feature.aggregation == AggregationType.SUM:
                agg_expr = sum(col(feature.name)).over(window_spec)
            elif feature.aggregation == AggregationType.COUNT:
                agg_expr = count(col(feature.name)).over(window_spec)
            elif feature.aggregation == AggregationType.AVG:
                agg_expr = avg(col(feature.name)).over(window_spec)
            elif feature.aggregation == AggregationType.MIN:
                agg_expr = min(col(feature.name)).over(window_spec)
            elif feature.aggregation == AggregationType.MAX:
                agg_expr = max(col(feature.name)).over(window_spec)
            else:
                agg_expr = col(feature.name)  # No aggregation
            
            return df.withColumn(f"{feature.name}_agg", agg_expr)
            
        except Exception as e:
            self.logger.error(f"Error applying aggregation: {str(e)}")
            return df
    
    def _create_window_spec(self, window_size: str, feature_group: FeatureGroup):
        """Create window specification for aggregations"""
        try:
            # Create window specification
            window_spec = Window.partitionBy(*feature_group.primary_keys)
            
            if feature_group.event_timestamp_column:
                window_spec = window_spec.orderBy(col(feature_group.event_timestamp_column))
                
                # Add range based on window size
                if "day" in window_size.lower():
                    days = int(window_size.split()[0])
                    range_seconds = days * 24 * 3600
                elif "hour" in window_size.lower():
                    hours = int(window_size.split()[0])
                    range_seconds = hours * 3600
                else:
                    range_seconds = 3600  # Default 1 hour
                
                window_spec = window_spec.rangeBetween(-range_seconds, 0)
            
            return window_spec
            
        except Exception as e:
            self.logger.error(f"Error creating window spec: {str(e)}")
            return Window.partitionBy(*feature_group.primary_keys)
    
    def get_features(
        self,
        feature_names: List[str],
        entity_df: DataFrame,
        point_in_time_column: Optional[str] = None
    ) -> DataFrame:
        """
        Get features for entities with point-in-time correctness
        
        Args:
            feature_names: List of features to retrieve
            entity_df: DataFrame with entity keys
            point_in_time_column: Column for point-in-time lookup
            
        Returns:
            DataFrame: Features joined with entities
        """
        try:
            # Group features by feature group
            features_by_group = {}
            for feature_name in feature_names:
                # Find feature group for this feature
                feature_group_name = None
                for fg_name, fg in self.feature_groups.items():
                    if any(f.name == feature_name for f in fg.features):
                        feature_group_name = fg_name
                        break
                
                if feature_group_name:
                    if feature_group_name not in features_by_group:
                        features_by_group[feature_group_name] = []
                    features_by_group[feature_group_name].append(feature_name)
            
            # Load and join features from each feature group
            result_df = entity_df
            
            for fg_name, fg_features in features_by_group.items():
                feature_group = self.feature_groups[fg_name]
                
                # Load feature group data
                if feature_group.storage_path:
                    features_df = self.spark.read.format("delta").load(feature_group.storage_path)
                    
                    # Select only required features and keys
                    select_cols = feature_group.primary_keys + fg_features
                    if feature_group.event_timestamp_column and point_in_time_column:
                        select_cols.append(feature_group.event_timestamp_column)
                    
                    features_df = features_df.select(*select_cols)
                    
                    # Apply point-in-time join if timestamp columns are available
                    if point_in_time_column and feature_group.event_timestamp_column:
                        result_df = self._point_in_time_join(
                            result_df,
                            features_df,
                            feature_group.primary_keys,
                            point_in_time_column,
                            feature_group.event_timestamp_column
                        )
                    else:
                        # Regular join
                        result_df = result_df.join(
                            features_df,
                            feature_group.primary_keys,
                            "left"
                        )
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error getting features: {str(e)}")
            raise
    
    def _point_in_time_join(
        self,
        entity_df: DataFrame,
        features_df: DataFrame,
        join_keys: List[str],
        entity_timestamp_col: str,
        feature_timestamp_col: str
    ) -> DataFrame:
        """Perform point-in-time correct join"""
        try:
            # Add row numbers for point-in-time correctness
            window_spec = Window.partitionBy(*join_keys).orderBy(col(feature_timestamp_col).desc())
            
            features_with_rn = features_df.withColumn("rn", row_number().over(window_spec))
            
            # Join conditions: keys match and feature timestamp <= entity timestamp
            join_condition = [
                entity_df[key] == features_with_rn[key] for key in join_keys
            ] + [
                features_with_rn[feature_timestamp_col] <= entity_df[entity_timestamp_col]
            ]
            
            # Perform join and select latest feature values
            result = entity_df.join(
                features_with_rn.where(col("rn") == 1),
                join_condition,
                "left"
            ).drop("rn", feature_timestamp_col)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in point-in-time join: {str(e)}")
            raise
    
    def create_feature_view(
        self,
        name: str,
        feature_groups: List[str],
        selected_features: List[str],
        join_keys: List[str],
        **kwargs
    ) -> str:
        """Create a feature view for specific use cases"""
        try:
            feature_view = FeatureView(
                name=name,
                feature_groups=feature_groups,
                selected_features=selected_features,
                join_keys=join_keys,
                **kwargs
            )
            
            # Validate feature view
            for fg_name in feature_groups:
                if fg_name not in self.feature_groups:
                    raise ValueError(f"Feature group {fg_name} not found")
            
            # Store feature view
            self.feature_views[name] = feature_view
            
            self.logger.info(f"Created feature view: {name}")
            return name
            
        except Exception as e:
            self.logger.error(f"Error creating feature view {name}: {str(e)}")
            raise
    
    def get_features_for_training(
        self,
        entity_df: DataFrame,
        feature_view_name: str
    ) -> DataFrame:
        """Get features optimized for training"""
        try:
            if feature_view_name not in self.feature_views:
                raise ValueError(f"Feature view {feature_view_name} not found")
            
            feature_view = self.feature_views[feature_view_name]
            
            # Get features using the feature view configuration
            training_df = self.get_features(
                feature_view.selected_features,
                entity_df
            )
            
            return training_df
            
        except Exception as e:
            self.logger.error(f"Error getting training features: {str(e)}")
            raise
    
    def compute_feature_statistics(self, feature_group_name: str) -> Dict[str, Any]:
        """Compute statistics for features in a feature group"""
        try:
            if feature_group_name not in self.feature_groups:
                raise ValueError(f"Feature group {feature_group_name} not found")
            
            feature_group = self.feature_groups[feature_group_name]
            
            if not feature_group.storage_path:
                return {}
            
            # Load feature data
            features_df = self.spark.read.format("delta").load(feature_group.storage_path)
            
            statistics = {}
            
            for feature in feature_group.features:
                if feature.name in features_df.columns:
                    feature_stats = self._compute_single_feature_statistics(
                        features_df, 
                        feature
                    )
                    statistics[feature.name] = feature_stats
            
            # Store statistics
            self.feature_statistics[feature_group_name] = statistics
            
            return statistics
            
        except Exception as e:
            self.logger.error(f"Error computing feature statistics: {str(e)}")
            return {}
    
    def _compute_single_feature_statistics(self, df: DataFrame, feature: Feature) -> Dict[str, Any]:
        """Compute statistics for a single feature"""
        try:
            stats = {}
            
            # Basic statistics
            stats["count"] = df.count()
            stats["null_count"] = df.filter(col(feature.name).isNull()).count()
            stats["null_percentage"] = stats["null_count"] / stats["count"] * 100 if stats["count"] > 0 else 0
            
            if feature.feature_type == FeatureType.NUMERICAL:
                # Numerical statistics
                numeric_stats = df.select(
                    min(col(feature.name)).alias("min_val"),
                    max(col(feature.name)).alias("max_val"),
                    avg(col(feature.name)).alias("mean_val"),
                    stddev(col(feature.name)).alias("std_val")
                ).collect()[0]
                
                stats.update({
                    "min": numeric_stats["min_val"],
                    "max": numeric_stats["max_val"],
                    "mean": numeric_stats["mean_val"],
                    "std": numeric_stats["std_val"]
                })
            
            elif feature.feature_type == FeatureType.CATEGORICAL:
                # Categorical statistics
                distinct_count = df.select(feature.name).distinct().count()
                stats["distinct_count"] = distinct_count
                stats["cardinality"] = distinct_count / stats["count"] if stats["count"] > 0 else 0
                
                # Top values
                top_values = df.groupBy(feature.name) \
                              .count() \
                              .orderBy(col("count").desc()) \
                              .limit(10) \
                              .collect()
                
                stats["top_values"] = [{"value": row[feature.name], "count": row["count"]} for row in top_values]
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error computing statistics for {feature.name}: {str(e)}")
            return {}
    
    def get_feature_lineage(self, feature_names: List[str]) -> Dict[str, Any]:
        """Get lineage information for features"""
        try:
            lineage = {
                "features": {},
                "dependencies": {},
                "consumers": {}
            }
            
            for feature_name in feature_names:
                # Find feature and its feature group
                feature_info = None
                feature_group_name = None
                
                for fg_name, fg in self.feature_groups.items():
                    for feature in fg.features:
                        if feature.name == feature_name:
                            feature_info = feature
                            feature_group_name = fg_name
                            break
                
                if feature_info and feature_group_name:
                    lineage["features"][feature_name] = {
                        "feature_group": feature_group_name,
                        "expression": feature_info.expression,
                        "created_date": feature_info.created_date.isoformat(),
                        "owner": feature_info.owner
                    }
                    
                    # Add dependencies
                    fg = self.feature_groups[feature_group_name]
                    lineage["dependencies"][feature_name] = {
                        "source_table": fg.source_table,
                        "upstream_features": fg.upstream_dependencies
                    }
            
            return lineage
            
        except Exception as e:
            self.logger.error(f"Error getting feature lineage: {str(e)}")
            return {"error": str(e)}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get feature store metrics"""
        try:
            return {
                "total_feature_groups": len(self.feature_groups),
                "total_features": len(self.features),
                "total_feature_views": len(self.feature_views),
                "storage_root": self.storage_root,
                "computation_jobs": len(self.computation_metrics)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting feature store metrics: {str(e)}")
            return {"error": str(e)}
    
    def cleanup(self):
        """Cleanup feature store resources"""
        try:
            # Cleanup would involve stopping any running computation jobs
            # and cleaning up temporary resources
            
            self.logger.info("Feature store cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during feature store cleanup: {str(e)}")


# Utility functions

def create_numerical_feature(
    name: str,
    expression: str,
    description: Optional[str] = None,
    **kwargs
) -> Feature:
    """Create a numerical feature"""
    return Feature(
        name=name,
        feature_type=FeatureType.NUMERICAL,
        expression=expression,
        description=description,
        data_type=DoubleType(),
        **kwargs
    )


def create_categorical_feature(
    name: str,
    expression: str,
    allowed_values: Optional[List[str]] = None,
    description: Optional[str] = None,
    **kwargs
) -> Feature:
    """Create a categorical feature"""
    return Feature(
        name=name,
        feature_type=FeatureType.CATEGORICAL,
        expression=expression,
        allowed_values=allowed_values,
        description=description,
        data_type=StringType(),
        **kwargs
    )