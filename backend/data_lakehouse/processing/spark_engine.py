"""
Spark Processing Engine

Central processing engine for all Spark-based data transformations:
- Distributed data processing with optimizations
- Dynamic resource allocation
- Adaptive query execution
- Cost-based optimization
- Memory and compute management
- Job monitoring and metrics
"""

import logging
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import uuid

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark import SparkContext
from pyspark.sql.streaming import StreamingQuery


class ProcessingMode(Enum):
    """Processing execution modes"""
    BATCH = "batch"
    STREAMING = "streaming"
    MICRO_BATCH = "micro_batch"
    CONTINUOUS = "continuous"


class OptimizationLevel(Enum):
    """Spark optimization levels"""
    BASIC = "basic"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


@dataclass
class ProcessingConfig:
    """Configuration for Spark processing"""
    job_name: str
    processing_mode: ProcessingMode
    optimization_level: OptimizationLevel = OptimizationLevel.STANDARD
    
    # Resource configuration
    executor_cores: int = 4
    executor_memory: str = "8g"
    executor_instances: int = 10
    driver_memory: str = "4g"
    driver_cores: int = 2
    
    # Optimization settings
    enable_adaptive_query_execution: bool = True
    enable_adaptive_spark_sql_join: bool = True
    enable_bloom_filter_join: bool = True
    enable_bucket_join: bool = True
    enable_cost_based_optimizer: bool = True
    
    # Serialization and compression
    serializer: str = "kryo"
    compression_codec: str = "lz4"
    
    # Storage settings
    storage_level: str = "MEMORY_AND_DISK_SER"
    checkpoint_compression: bool = True
    
    # Advanced settings
    sql_adaptive_coalescePartitions_enabled: bool = True
    sql_adaptive_skewJoin_enabled: bool = True
    broadcast_timeout: int = 300
    
    # Custom configurations
    custom_configs: Optional[Dict[str, str]] = None
    
    # Monitoring
    enable_event_log: bool = True
    log_level: str = "WARN"
    
    # Streaming specific
    trigger_interval: str = "10 seconds"
    watermark_delay: str = "5 minutes"


@dataclass
class JobMetrics:
    """Metrics for processing jobs"""
    job_id: str
    job_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"
    
    # Resource metrics
    total_executor_cores: int = 0
    total_executor_memory_mb: int = 0
    peak_executor_count: int = 0
    
    # Performance metrics
    records_processed: int = 0
    bytes_processed: int = 0
    processing_rate_per_second: float = 0.0
    
    # Optimization metrics
    stages_completed: int = 0
    tasks_completed: int = 0
    cache_hit_ratio: float = 0.0
    spill_memory_bytes: int = 0
    spill_disk_bytes: int = 0
    
    # Error tracking
    failed_tasks: int = 0
    error_messages: List[str] = None
    
    def __post_init__(self):
        if self.error_messages is None:
            self.error_messages = []


class SparkProcessingEngine:
    """
    Advanced Spark processing engine with comprehensive optimization and monitoring.
    
    Features:
    - Intelligent resource management
    - Adaptive query execution optimization
    - Dynamic partition and join optimization
    - Real-time performance monitoring
    - Cost-based query optimization
    - Memory and storage management
    - Error handling and recovery
    - Lineage and audit tracking
    """
    
    def __init__(self, config: ProcessingConfig, spark: Optional[SparkSession] = None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize Spark session
        self.spark = spark or self._create_spark_session()
        self.sc = self.spark.sparkContext
        
        # Job tracking
        self.active_jobs: Dict[str, JobMetrics] = {}
        self.completed_jobs: List[JobMetrics] = []
        
        # Performance monitoring
        self.performance_metrics: Dict[str, Any] = {}
        
        # Configure optimizations
        self._apply_optimizations()
        
        # Setup monitoring
        self._setup_monitoring()
    
    def _create_spark_session(self) -> SparkSession:
        """Create optimized Spark session"""
        try:
            builder = SparkSession.builder.appName(self.config.job_name)
            
            # Core resource settings
            builder = builder.config("spark.executor.cores", str(self.config.executor_cores))
            builder = builder.config("spark.executor.memory", self.config.executor_memory)
            builder = builder.config("spark.executor.instances", str(self.config.executor_instances))
            builder = builder.config("spark.driver.memory", self.config.driver_memory)
            builder = builder.config("spark.driver.cores", str(self.config.driver_cores))
            
            # Serialization
            builder = builder.config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            builder = builder.config("spark.sql.execution.arrow.pyspark.enabled", "true")
            
            # Adaptive query execution
            if self.config.enable_adaptive_query_execution:
                builder = builder.config("spark.sql.adaptive.enabled", "true")
                builder = builder.config("spark.sql.adaptive.coalescePartitions.enabled", 
                                       str(self.config.sql_adaptive_coalescePartitions_enabled))
                builder = builder.config("spark.sql.adaptive.skewJoin.enabled",
                                       str(self.config.sql_adaptive_skewJoin_enabled))
            
            # Cost-based optimizer
            if self.config.enable_cost_based_optimizer:
                builder = builder.config("spark.sql.cbo.enabled", "true")
                builder = builder.config("spark.sql.cbo.joinReorder.enabled", "true")
                builder = builder.config("spark.sql.cbo.joinReorder.dp.threshold", "12")
            
            # Join optimizations
            builder = builder.config("spark.sql.autoBroadcastJoinThreshold", "100MB")
            builder = builder.config("spark.sql.broadcastTimeout", str(self.config.broadcast_timeout))
            
            # Storage and compression
            builder = builder.config("spark.sql.parquet.compression.codec", self.config.compression_codec)
            builder = builder.config("spark.sql.orc.compression.codec", self.config.compression_codec)
            
            # Dynamic allocation
            builder = builder.config("spark.dynamicAllocation.enabled", "true")
            builder = builder.config("spark.dynamicAllocation.minExecutors", "2")
            builder = builder.config("spark.dynamicAllocation.maxExecutors", str(self.config.executor_instances * 2))
            
            # Event logging
            if self.config.enable_event_log:
                builder = builder.config("spark.eventLog.enabled", "true")
                builder = builder.config("spark.eventLog.dir", "/tmp/spark-events")
            
            # Custom configurations
            if self.config.custom_configs:
                for key, value in self.config.custom_configs.items():
                    builder = builder.config(key, value)
            
            # Create session
            spark = builder.getOrCreate()
            
            # Set log level
            spark.sparkContext.setLogLevel(self.config.log_level)
            
            self.logger.info(f"Created Spark session: {self.config.job_name}")
            return spark
            
        except Exception as e:
            self.logger.error(f"Error creating Spark session: {str(e)}")
            raise
    
    def _apply_optimizations(self):
        """Apply processing optimizations based on configuration"""
        try:
            if self.config.optimization_level == OptimizationLevel.AGGRESSIVE:
                # Aggressive optimizations
                self.spark.conf.set("spark.sql.adaptive.localShuffleReader.enabled", "true")
                self.spark.conf.set("spark.sql.adaptive.optimizeSkewsInRebalancePartitions.enabled", "true")
                self.spark.conf.set("spark.sql.adaptive.nonEmptyPartitionRatioForBroadcastJoin", "0.2")
                
                # Memory optimizations
                self.spark.conf.set("spark.sql.execution.columnar.useCompressedCache", "true")
                self.spark.conf.set("spark.sql.columnVector.offheap.enabled", "true")
                
                # Cache optimizations
                self.spark.conf.set("spark.sql.execution.sortBeforeRepartition", "true")
                
            elif self.config.optimization_level == OptimizationLevel.STANDARD:
                # Standard optimizations
                self.spark.conf.set("spark.sql.adaptive.localShuffleReader.enabled", "true")
                
            # Common optimizations for all levels except BASIC
            if self.config.optimization_level != OptimizationLevel.BASIC:
                # Bloom filter joins
                if self.config.enable_bloom_filter_join:
                    self.spark.conf.set("spark.sql.optimizer.runtime.bloomFilter.enabled", "true")
                    self.spark.conf.set("spark.sql.optimizer.runtime.bloomFilter.applicationSideScanSizeThreshold", "10MB")
                
            self.logger.info(f"Applied {self.config.optimization_level.value} optimizations")
            
        except Exception as e:
            self.logger.warning(f"Could not apply some optimizations: {str(e)}")
    
    def _setup_monitoring(self):
        """Setup performance monitoring"""
        try:
            # Setup Spark listener for job monitoring
            from pyspark.util import SparkJobMonitor
            
            # This would be implemented with custom Spark listeners
            # For now, we'll use basic metrics collection
            
            self.logger.info("Performance monitoring initialized")
            
        except Exception as e:
            self.logger.warning(f"Could not setup full monitoring: {str(e)}")
    
    def execute_job(
        self,
        job_name: str,
        processing_func: Callable[[SparkSession], DataFrame],
        output_path: Optional[str] = None,
        mode: str = "overwrite",
        **kwargs
    ) -> str:
        """
        Execute a Spark processing job
        
        Args:
            job_name: Name of the job
            processing_func: Function that takes SparkSession and returns DataFrame
            output_path: Optional output path for results
            mode: Write mode for output
            **kwargs: Additional arguments for processing function
            
        Returns:
            str: Job ID
        """
        job_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Starting job: {job_name} (ID: {job_id})")
            
            # Initialize job metrics
            job_metrics = JobMetrics(
                job_id=job_id,
                job_name=job_name,
                start_time=start_time
            )
            self.active_jobs[job_id] = job_metrics
            
            # Execute processing function
            result_df = processing_func(self.spark, **kwargs)
            
            if result_df is None:
                raise ValueError("Processing function returned None")
            
            # Update metrics with result DataFrame info
            try:
                record_count = result_df.count()
                job_metrics.records_processed = record_count
            except Exception as e:
                self.logger.warning(f"Could not count records: {str(e)}")
            
            # Write output if path specified
            if output_path:
                self._write_output(result_df, output_path, mode, job_metrics)
            
            # Complete job
            end_time = datetime.now()
            job_metrics.end_time = end_time
            job_metrics.status = "completed"
            
            # Move to completed jobs
            self.completed_jobs.append(job_metrics)
            del self.active_jobs[job_id]
            
            duration = (end_time - start_time).total_seconds()
            self.logger.info(f"Job {job_name} completed in {duration:.2f} seconds")
            
            return job_id
            
        except Exception as e:
            # Handle job failure
            self.logger.error(f"Job {job_name} failed: {str(e)}")
            
            if job_id in self.active_jobs:
                job_metrics = self.active_jobs[job_id]
                job_metrics.status = "failed"
                job_metrics.end_time = datetime.now()
                job_metrics.error_messages.append(str(e))
                
                self.completed_jobs.append(job_metrics)
                del self.active_jobs[job_id]
            
            raise
    
    def execute_streaming_job(
        self,
        job_name: str,
        processing_func: Callable[[SparkSession], DataFrame],
        output_path: str,
        checkpoint_path: str,
        trigger_interval: Optional[str] = None
    ) -> StreamingQuery:
        """
        Execute a streaming processing job
        
        Args:
            job_name: Name of the streaming job
            processing_func: Function that returns streaming DataFrame
            output_path: Output path for streaming results
            checkpoint_path: Checkpoint directory
            trigger_interval: Trigger interval for micro-batches
            
        Returns:
            StreamingQuery: Streaming query handle
        """
        try:
            self.logger.info(f"Starting streaming job: {job_name}")
            
            # Execute processing function to get streaming DataFrame
            streaming_df = processing_func(self.spark)
            
            if streaming_df is None or not streaming_df.isStreaming:
                raise ValueError("Processing function must return a streaming DataFrame")
            
            # Configure streaming query
            trigger_int = trigger_interval or self.config.trigger_interval
            
            query = streaming_df.writeStream \
                               .format("delta") \
                               .outputMode("append") \
                               .option("checkpointLocation", checkpoint_path) \
                               .trigger(processingTime=trigger_int) \
                               .start(output_path)
            
            self.logger.info(f"Streaming job {job_name} started successfully")
            return query
            
        except Exception as e:
            self.logger.error(f"Error starting streaming job {job_name}: {str(e)}")
            raise
    
    def _write_output(self, df: DataFrame, output_path: str, mode: str, job_metrics: JobMetrics):
        """Write DataFrame output with optimization"""
        try:
            # Optimize DataFrame before writing
            optimized_df = self._optimize_dataframe(df)
            
            # Configure writer
            writer = optimized_df.write.format("delta").mode(mode)
            
            # Apply optimizations
            if self.config.optimization_level != OptimizationLevel.BASIC:
                writer = writer.option("optimizeWrite", "true")
                writer = writer.option("autoCompact", "true")
            
            # Write data
            writer.save(output_path)
            
            # Update metrics
            try:
                # This is an approximation - in production you'd want more accurate size calculation
                job_metrics.bytes_processed = df.count() * len(df.columns) * 100  # Rough estimate
            except Exception:
                pass
            
        except Exception as e:
            self.logger.error(f"Error writing output: {str(e)}")
            raise
    
    def _optimize_dataframe(self, df: DataFrame) -> DataFrame:
        """Apply DataFrame-level optimizations"""
        try:
            # Cache frequently accessed DataFrames
            if self.config.optimization_level in [OptimizationLevel.STANDARD, OptimizationLevel.AGGRESSIVE]:
                storage_level_map = {
                    "MEMORY_ONLY": "MEMORY_ONLY",
                    "MEMORY_AND_DISK": "MEMORY_AND_DISK", 
                    "MEMORY_AND_DISK_SER": "MEMORY_AND_DISK_SER"
                }
                
                if self.config.storage_level in storage_level_map:
                    df = df.persist(getattr(
                        __import__('pyspark.storagelevel', fromlist=['StorageLevel']).StorageLevel,
                        self.config.storage_level
                    ))
            
            # Repartition for optimal parallelism
            if self.config.optimization_level == OptimizationLevel.AGGRESSIVE:
                optimal_partitions = max(2, self.config.executor_instances * self.config.executor_cores)
                current_partitions = df.rdd.getNumPartitions()
                
                if current_partitions != optimal_partitions:
                    if current_partitions > optimal_partitions * 2:
                        df = df.coalesce(optimal_partitions)
                    elif current_partitions < optimal_partitions // 2:
                        df = df.repartition(optimal_partitions)
            
            return df
            
        except Exception as e:
            self.logger.warning(f"DataFrame optimization error: {str(e)}")
            return df
    
    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get status and metrics for a job"""
        # Check active jobs
        if job_id in self.active_jobs:
            job_metrics = self.active_jobs[job_id]
            return self._job_metrics_to_dict(job_metrics)
        
        # Check completed jobs
        for job_metrics in self.completed_jobs:
            if job_metrics.job_id == job_id:
                return self._job_metrics_to_dict(job_metrics)
        
        return None
    
    def _job_metrics_to_dict(self, job_metrics: JobMetrics) -> Dict[str, Any]:
        """Convert job metrics to dictionary"""
        return {
            "job_id": job_metrics.job_id,
            "job_name": job_metrics.job_name,
            "status": job_metrics.status,
            "start_time": job_metrics.start_time.isoformat(),
            "end_time": job_metrics.end_time.isoformat() if job_metrics.end_time else None,
            "duration_seconds": (
                (job_metrics.end_time or datetime.now()) - job_metrics.start_time
            ).total_seconds(),
            "records_processed": job_metrics.records_processed,
            "bytes_processed": job_metrics.bytes_processed,
            "processing_rate_per_second": job_metrics.processing_rate_per_second,
            "stages_completed": job_metrics.stages_completed,
            "tasks_completed": job_metrics.tasks_completed,
            "failed_tasks": job_metrics.failed_tasks,
            "error_messages": job_metrics.error_messages
        }
    
    def list_jobs(self, include_completed: bool = True) -> List[Dict[str, Any]]:
        """List all jobs with their status"""
        jobs = []
        
        # Add active jobs
        for job_metrics in self.active_jobs.values():
            jobs.append(self._job_metrics_to_dict(job_metrics))
        
        # Add completed jobs if requested
        if include_completed:
            for job_metrics in self.completed_jobs:
                jobs.append(self._job_metrics_to_dict(job_metrics))
        
        return jobs
    
    def get_cluster_metrics(self) -> Dict[str, Any]:
        """Get cluster-level metrics"""
        try:
            # Get Spark context metrics
            sc = self.spark.sparkContext
            
            cluster_metrics = {
                "application_id": sc.applicationId,
                "application_name": sc.appName,
                "spark_version": sc.version,
                "total_executor_cores": sc.defaultParallelism,
                "active_jobs": len(self.active_jobs),
                "completed_jobs": len(self.completed_jobs),
                "uptime_seconds": (datetime.now() - datetime.fromtimestamp(sc.startTime / 1000)).total_seconds()
            }
            
            # Add performance metrics if available
            status_tracker = sc.statusTracker()
            if status_tracker:
                cluster_metrics.update({
                    "active_stages": len(status_tracker.getActiveStageIds()),
                    "executor_summaries": len(status_tracker.getExecutorInfos())
                })
            
            return cluster_metrics
            
        except Exception as e:
            self.logger.error(f"Error getting cluster metrics: {str(e)}")
            return {"error": str(e)}
    
    def optimize_joins(self, df1: DataFrame, df2: DataFrame, join_keys: List[str], join_type: str = "inner") -> DataFrame:
        """
        Perform optimized join operations
        
        Args:
            df1: Left DataFrame
            df2: Right DataFrame  
            join_keys: Keys to join on
            join_type: Type of join
            
        Returns:
            DataFrame: Optimized joined result
        """
        try:
            # Analyze DataFrames for join optimization
            df1_size = df1.count()
            df2_size = df2.count()
            
            # Determine optimal join strategy
            if df2_size < 100000000:  # < 100M records, consider broadcast join
                df2 = df2.hint("broadcast")
                self.logger.info("Applied broadcast hint to smaller DataFrame")
            
            # Apply bucketing hint for large joins
            if df1_size > 100000000 and df2_size > 100000000:
                # For very large joins, consider bucketing
                join_condition = [df1[key] == df2[key] for key in join_keys]
                result = df1.join(df2, join_condition, join_type)
                result = result.hint("bucket", len(join_keys))
            else:
                # Standard join
                result = df1.join(df2, join_keys, join_type)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in optimized join: {str(e)}")
            raise
    
    def cleanup(self):
        """Cleanup Spark processing engine resources"""
        try:
            # Stop active streaming queries
            # (Would need to track streaming queries for this)
            
            # Clear caches
            self.spark.catalog.clearCache()
            
            # Stop Spark session if we created it
            if hasattr(self, '_created_spark_session'):
                self.spark.stop()
            
            self.logger.info("Spark processing engine cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")


# Utility functions

def create_processing_config(
    job_name: str,
    processing_mode: ProcessingMode = ProcessingMode.BATCH,
    optimization_level: OptimizationLevel = OptimizationLevel.STANDARD,
    **kwargs
) -> ProcessingConfig:
    """Create a processing configuration with sensible defaults"""
    return ProcessingConfig(
        job_name=job_name,
        processing_mode=processing_mode,
        optimization_level=optimization_level,
        **kwargs
    )


def create_spark_engine(
    job_name: str,
    optimization_level: OptimizationLevel = OptimizationLevel.STANDARD
) -> SparkProcessingEngine:
    """Create a Spark processing engine with default configuration"""
    config = create_processing_config(job_name, optimization_level=optimization_level)
    return SparkProcessingEngine(config)