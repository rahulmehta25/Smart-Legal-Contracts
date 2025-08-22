"""
Batch Data Ingestion Engine

Handles batch ingestion from various sources including:
- Relational databases (PostgreSQL, MySQL, SQL Server, Oracle)
- NoSQL databases (MongoDB, Cassandra, DynamoDB)
- File systems (CSV, JSON, Parquet, Avro, ORC)
- Cloud storage (S3, GCS, Azure Blob)
- FTP/SFTP servers
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.utils import AnalysisException


@dataclass
class BatchJobConfig:
    """Configuration for batch ingestion jobs"""
    job_id: str
    source_type: str
    source_config: Dict[str, Any]
    target_path: str
    format: str = "delta"
    mode: str = "append"
    partition_by: Optional[List[str]] = None
    schema: Optional[StructType] = None
    transformations: Optional[List[str]] = None
    quality_checks: Optional[Dict[str, Any]] = None
    schedule: Optional[str] = None
    retry_config: Optional[Dict[str, int]] = None


class BatchIngestionEngine:
    """
    Batch data ingestion engine with comprehensive ETL capabilities.
    
    Features:
    - Multi-format file reading (CSV, JSON, Parquet, Avro, ORC)
    - Database connectivity with JDBC
    - Schema inference and validation
    - Data transformations and cleansing
    - Incremental loading strategies
    - Data quality validation
    - Error handling and retry logic
    - Performance optimization
    """
    
    def __init__(self, spark: SparkSession, config: Dict[str, Any]):
        self.spark = spark
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Job tracking
        self.active_jobs: Dict[str, BatchJobConfig] = {}
        self.job_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Performance settings
        self._configure_spark_settings()
    
    def _configure_spark_settings(self):
        """Configure Spark for optimal batch processing"""
        try:
            # Adaptive query execution
            self.spark.conf.set("spark.sql.adaptive.enabled", "true")
            self.spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
            
            # Dynamic partition pruning
            self.spark.conf.set("spark.sql.optimizer.dynamicPartitionPruning.enabled", "true")
            
            # Broadcast joins
            self.spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "10MB")
            
            # Serialization
            self.spark.conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            
        except Exception as e:
            self.logger.warning(f"Could not set Spark configuration: {str(e)}")
    
    def run_batch_ingestion(
        self,
        job_id: str,
        connector,
        target_path: str,
        schema: Optional[StructType] = None,
        transformations: Optional[List[str]] = None,
        quality_checks: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Run a batch ingestion job
        
        Args:
            job_id: Unique identifier for the job
            connector: Data connector instance
            target_path: Target path for writing data
            schema: Expected schema (optional)
            transformations: List of transformations to apply
            quality_checks: Data quality validation rules
            
        Returns:
            bool: Success status
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting batch ingestion job: {job_id}")
            
            # Initialize job metrics
            self.job_metrics[job_id] = {
                "status": "running",
                "start_time": start_time,
                "records_read": 0,
                "records_written": 0,
                "bytes_read": 0,
                "bytes_written": 0,
                "errors": []
            }
            
            # Read data from source
            df = self._read_from_connector(connector, schema)
            
            if df is None:
                raise Exception("Failed to read data from source")
            
            # Update read metrics
            self.job_metrics[job_id]["records_read"] = df.count()
            
            # Apply transformations
            if transformations:
                df = self._apply_transformations(df, transformations, job_id)
            
            # Validate data quality
            if quality_checks:
                df = self._validate_data_quality(df, quality_checks, job_id)
            
            # Write to target
            success = self._write_to_target(df, target_path, job_id)
            
            # Update final metrics
            end_time = datetime.now()
            self.job_metrics[job_id].update({
                "status": "completed" if success else "failed",
                "end_time": end_time,
                "duration_seconds": (end_time - start_time).total_seconds(),
                "records_written": df.count() if success else 0
            })
            
            if success:
                self.logger.info(f"Batch ingestion job {job_id} completed successfully")
            else:
                self.logger.error(f"Batch ingestion job {job_id} failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error in batch ingestion job {job_id}: {str(e)}")
            self.job_metrics[job_id]["status"] = "failed"
            self.job_metrics[job_id]["errors"].append(str(e))
            return False
    
    def _read_from_connector(self, connector, schema: Optional[StructType] = None) -> Optional[DataFrame]:
        """Read data from the provided connector"""
        try:
            if hasattr(connector, 'read_batch'):
                return connector.read_batch(self.spark, schema)
            else:
                raise NotImplementedError(f"Connector {type(connector)} does not support batch reading")
                
        except Exception as e:
            self.logger.error(f"Error reading from connector: {str(e)}")
            return None
    
    def _apply_transformations(self, df: DataFrame, transformations: List[str], job_id: str) -> DataFrame:
        """Apply transformation rules to the DataFrame"""
        try:
            for transformation in transformations:
                # Parse transformation rule
                if transformation.startswith("select:"):
                    # Select specific columns
                    columns = transformation.replace("select:", "").split(",")
                    df = df.select([col(c.strip()) for c in columns])
                
                elif transformation.startswith("filter:"):
                    # Apply filter condition
                    condition = transformation.replace("filter:", "")
                    df = df.filter(condition)
                
                elif transformation.startswith("rename:"):
                    # Rename columns (format: "rename:old_name->new_name")
                    rename_rule = transformation.replace("rename:", "")
                    old_name, new_name = rename_rule.split("->")
                    df = df.withColumnRenamed(old_name.strip(), new_name.strip())
                
                elif transformation.startswith("cast:"):
                    # Cast column types (format: "cast:column_name:data_type")
                    cast_rule = transformation.replace("cast:", "")
                    parts = cast_rule.split(":")
                    column_name, data_type = parts[0], parts[1]
                    df = df.withColumn(column_name, col(column_name).cast(data_type))
                
                elif transformation.startswith("add_column:"):
                    # Add computed column (format: "add_column:column_name:expression")
                    add_rule = transformation.replace("add_column:", "")
                    parts = add_rule.split(":", 1)
                    column_name, expression = parts[0], parts[1]
                    df = df.withColumn(column_name, expr(expression))
                
                elif transformation == "deduplicate":
                    # Remove duplicates
                    df = df.dropDuplicates()
                
                elif transformation.startswith("partition:"):
                    # Repartition DataFrame
                    num_partitions = int(transformation.replace("partition:", ""))
                    df = df.repartition(num_partitions)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error applying transformations for job {job_id}: {str(e)}")
            self.job_metrics[job_id]["errors"].append(f"Transformation error: {str(e)}")
            raise
    
    def _validate_data_quality(self, df: DataFrame, quality_checks: Dict[str, Any], job_id: str) -> DataFrame:
        """Validate data quality and apply rules"""
        try:
            # Null checks
            if quality_checks.get("null_checks"):
                for column, max_null_percentage in quality_checks["null_checks"].items():
                    if column in df.columns:
                        null_count = df.filter(col(column).isNull()).count()
                        total_count = df.count()
                        null_percentage = (null_count / total_count) * 100 if total_count > 0 else 0
                        
                        if null_percentage > max_null_percentage:
                            error_msg = f"Column {column} has {null_percentage:.2f}% null values, exceeds limit {max_null_percentage}%"
                            self.logger.error(error_msg)
                            self.job_metrics[job_id]["errors"].append(error_msg)
            
            # Range checks for numeric columns
            if quality_checks.get("range_checks"):
                for column, (min_val, max_val) in quality_checks["range_checks"].items():
                    if column in df.columns:
                        out_of_range_count = df.filter(
                            (col(column) < min_val) | (col(column) > max_val)
                        ).count()
                        
                        if out_of_range_count > 0:
                            error_msg = f"Column {column} has {out_of_range_count} values out of range [{min_val}, {max_val}]"
                            self.logger.warning(error_msg)
                            self.job_metrics[job_id]["errors"].append(error_msg)
            
            # Uniqueness checks
            if quality_checks.get("unique_checks"):
                for column in quality_checks["unique_checks"]:
                    if column in df.columns:
                        total_count = df.count()
                        distinct_count = df.select(column).distinct().count()
                        
                        if total_count != distinct_count:
                            error_msg = f"Column {column} has duplicate values: {total_count - distinct_count} duplicates"
                            self.logger.warning(error_msg)
                            self.job_metrics[job_id]["errors"].append(error_msg)
            
            # Custom validation rules
            if quality_checks.get("custom_rules"):
                for rule_name, rule_condition in quality_checks["custom_rules"].items():
                    violations = df.filter(expr(f"NOT ({rule_condition})")).count()
                    if violations > 0:
                        error_msg = f"Quality rule '{rule_name}' violated by {violations} records"
                        self.logger.warning(error_msg)
                        self.job_metrics[job_id]["errors"].append(error_msg)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error in data quality validation for job {job_id}: {str(e)}")
            self.job_metrics[job_id]["errors"].append(f"Quality validation error: {str(e)}")
            raise
    
    def _write_to_target(self, df: DataFrame, target_path: str, job_id: str) -> bool:
        """Write DataFrame to target location"""
        try:
            # Determine write mode and format from target path
            if target_path.endswith(".delta") or "delta" in self.config.get("default_format", "delta"):
                # Write as Delta table
                df.write \
                  .format("delta") \
                  .mode("append") \
                  .option("mergeSchema", "true") \
                  .save(target_path)
                
            elif target_path.endswith(".parquet"):
                # Write as Parquet
                df.write \
                  .format("parquet") \
                  .mode("append") \
                  .save(target_path)
            
            else:
                # Default to Delta format
                df.write \
                  .format("delta") \
                  .mode("append") \
                  .save(target_path)
            
            # Update write metrics
            self.job_metrics[job_id]["bytes_written"] = self._estimate_data_size(df)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing to target {target_path} for job {job_id}: {str(e)}")
            self.job_metrics[job_id]["errors"].append(f"Write error: {str(e)}")
            return False
    
    def _estimate_data_size(self, df: DataFrame) -> int:
        """Estimate the size of DataFrame in bytes"""
        try:
            # This is an approximation - in production you might want more accurate sizing
            record_count = df.count()
            column_count = len(df.columns)
            # Rough estimate: 100 bytes per column per record
            return record_count * column_count * 100
            
        except Exception:
            return 0
    
    def get_job_metrics(self, job_id: str) -> Dict[str, Any]:
        """Get metrics for a specific job"""
        return self.job_metrics.get(job_id, {})
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all jobs and their metrics"""
        return [
            {"job_id": job_id, **metrics}
            for job_id, metrics in self.job_metrics.items()
        ]
    
    def cleanup_completed_jobs(self, retention_hours: int = 24):
        """Clean up metrics for completed jobs older than retention period"""
        cutoff_time = datetime.now() - timedelta(hours=retention_hours)
        
        jobs_to_remove = []
        for job_id, metrics in self.job_metrics.items():
            if (metrics.get("status") in ["completed", "failed"] and 
                metrics.get("end_time", datetime.now()) < cutoff_time):
                jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self.job_metrics[job_id]
            self.logger.info(f"Cleaned up metrics for job: {job_id}")
    
    def run_incremental_load(
        self,
        job_id: str,
        connector,
        target_path: str,
        incremental_column: str,
        last_value: Any = None,
        schema: Optional[StructType] = None
    ) -> bool:
        """
        Run incremental data loading based on a timestamp or sequence column
        
        Args:
            job_id: Unique job identifier
            connector: Data connector
            target_path: Target path for data
            incremental_column: Column used for incremental loading
            last_value: Last processed value (for incremental loading)
            schema: Expected schema
            
        Returns:
            bool: Success status
        """
        try:
            # If no last value provided, get it from existing data
            if last_value is None:
                try:
                    existing_df = self.spark.read.format("delta").load(target_path)
                    last_value = existing_df.agg(max(col(incremental_column))).collect()[0][0]
                except (AnalysisException, Exception):
                    # Table doesn't exist or is empty, start fresh
                    last_value = None
            
            # Configure connector for incremental read
            if hasattr(connector, 'set_incremental_config'):
                connector.set_incremental_config(incremental_column, last_value)
            
            # Run the batch ingestion
            return self.run_batch_ingestion(
                job_id=job_id,
                connector=connector,
                target_path=target_path,
                schema=schema
            )
            
        except Exception as e:
            self.logger.error(f"Error in incremental load for job {job_id}: {str(e)}")
            return False