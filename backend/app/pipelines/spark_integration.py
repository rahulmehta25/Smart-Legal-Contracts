"""
Apache Spark Integration for Big Data Processing

Provides comprehensive Spark integration with optimization techniques,
partitioning strategies, and performance monitoring for legal document processing.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np

# Spark imports
try:
    from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
    from pyspark.sql.functions import *
    from pyspark.sql.types import *
    from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
    from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
    from pyspark.ml.clustering import KMeans
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml import Pipeline
    from pyspark.streaming import StreamingContext
    from pyspark.sql.streaming import StreamingQuery
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False


class SparkJobType(Enum):
    """Types of Spark jobs"""
    BATCH_ETL = "batch_etl"
    STREAMING_ETL = "streaming_etl"
    ML_TRAINING = "ml_training"
    ML_INFERENCE = "ml_inference"
    DATA_VALIDATION = "data_validation"
    AGGREGATION = "aggregation"
    DEDUPLICATION = "deduplication"
    TEXT_PROCESSING = "text_processing"


class OptimizationStrategy(Enum):
    """Spark optimization strategies"""
    PARTITION_TUNING = "partition_tuning"
    CACHE_OPTIMIZATION = "cache_optimization"
    BROADCAST_JOIN = "broadcast_join"
    BUCKETING = "bucketing"
    COLUMN_PRUNING = "column_pruning"
    PREDICATE_PUSHDOWN = "predicate_pushdown"
    ADAPTIVE_QUERY = "adaptive_query"


@dataclass
class SparkConfig:
    """Spark configuration"""
    app_name: str
    master: str = "local[*]"
    driver_memory: str = "4g"
    executor_memory: str = "4g"
    executor_cores: int = 2
    max_executors: int = 10
    enable_adaptive_query: bool = True
    enable_dynamic_allocation: bool = True
    serializer: str = "org.apache.spark.serializer.KryoSerializer"
    additional_configs: Dict[str, str] = None
    
    def __post_init__(self):
        if self.additional_configs is None:
            self.additional_configs = {}


@dataclass
class SparkJobResult:
    """Result of Spark job execution"""
    job_id: str
    job_type: SparkJobType
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None
    execution_time_seconds: Optional[float] = None
    rows_processed: Optional[int] = None
    output_path: Optional[str] = None
    metrics: Dict[str, Any] = None
    error_message: Optional[str] = None
    optimization_applied: List[str] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
        if self.optimization_applied is None:
            self.optimization_applied = []


class SparkProcessor:
    """
    Advanced Spark processor with optimization and monitoring
    """
    
    def __init__(self, config: SparkConfig):
        if not SPARK_AVAILABLE:
            raise ImportError("PySpark is not available. Please install pyspark.")
        
        self.config = config
        self.spark: Optional[SparkSession] = None
        self.streaming_context: Optional[StreamingContext] = None
        self.active_jobs: Dict[str, SparkJobResult] = {}
        self.job_history: List[SparkJobResult] = []
        
        # Performance metrics
        self.performance_metrics = {
            'total_jobs': 0,
            'successful_jobs': 0,
            'failed_jobs': 0,
            'avg_execution_time': 0.0,
            'total_rows_processed': 0
        }
    
    def initialize_spark_session(self) -> SparkSession:
        """
        Initialize Spark session with optimized configuration
        
        Returns:
            Configured Spark session
        """
        if self.spark is not None:
            return self.spark
        
        builder = SparkSession.builder.appName(self.config.app_name)
        
        # Core configurations
        builder = builder.config("spark.master", self.config.master)
        builder = builder.config("spark.driver.memory", self.config.driver_memory)
        builder = builder.config("spark.executor.memory", self.config.executor_memory)
        builder = builder.config("spark.executor.cores", str(self.config.executor_cores))
        builder = builder.config("spark.dynamicAllocation.maxExecutors", str(self.config.max_executors))
        
        # Performance optimizations
        if self.config.enable_adaptive_query:
            builder = builder.config("spark.sql.adaptive.enabled", "true")
            builder = builder.config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            builder = builder.config("spark.sql.adaptive.skewJoin.enabled", "true")
        
        if self.config.enable_dynamic_allocation:
            builder = builder.config("spark.dynamicAllocation.enabled", "true")
            builder = builder.config("spark.dynamicAllocation.minExecutors", "1")
            builder = builder.config("spark.dynamicAllocation.maxExecutors", str(self.config.max_executors))
        
        # Serialization
        builder = builder.config("spark.serializer", self.config.serializer)
        
        # Additional configurations
        for key, value in self.config.additional_configs.items():
            builder = builder.config(key, value)
        
        # Initialize session
        self.spark = builder.getOrCreate()
        
        # Set log level
        self.spark.sparkContext.setLogLevel("WARN")
        
        return self.spark
    
    async def process_legal_documents_batch(self, 
                                          input_path: str,
                                          output_path: str,
                                          processing_config: Dict[str, Any]) -> SparkJobResult:
        """
        Process legal documents in batch mode with Spark
        
        Args:
            input_path: Path to input documents
            output_path: Path for output results
            processing_config: Processing configuration
        
        Returns:
            Job execution result
        """
        job_id = str(uuid.uuid4())
        job_result = SparkJobResult(
            job_id=job_id,
            job_type=SparkJobType.TEXT_PROCESSING,
            status="running",
            start_time=datetime.utcnow()
        )
        
        self.active_jobs[job_id] = job_result
        
        try:
            spark = self.initialize_spark_session()
            
            # Read documents
            if processing_config.get('file_format') == 'parquet':
                df = spark.read.parquet(input_path)
            elif processing_config.get('file_format') == 'json':
                df = spark.read.json(input_path)
            else:
                df = spark.read.text(input_path)
            
            # Apply optimizations
            df = await self._apply_optimizations(df, processing_config)
            
            # Process documents based on configuration
            if processing_config.get('operation') == 'arbitration_detection':
                processed_df = await self._process_arbitration_detection(df, processing_config)
            elif processing_config.get('operation') == 'clause_extraction':
                processed_df = await self._process_clause_extraction(df, processing_config)
            elif processing_config.get('operation') == 'risk_assessment':
                processed_df = await self._process_risk_assessment(df, processing_config)
            else:
                processed_df = df
            
            # Write results
            self._write_output(processed_df, output_path, processing_config)
            
            # Update job result
            job_result.end_time = datetime.utcnow()
            job_result.execution_time_seconds = (job_result.end_time - job_result.start_time).total_seconds()
            job_result.rows_processed = processed_df.count()
            job_result.output_path = output_path
            job_result.status = "completed"
            
            # Update metrics
            self._update_performance_metrics(job_result)
            
            return job_result
            
        except Exception as e:
            job_result.end_time = datetime.utcnow()
            job_result.status = "failed"
            job_result.error_message = str(e)
            job_result.execution_time_seconds = (job_result.end_time - job_result.start_time).total_seconds()
            
            self._update_performance_metrics(job_result)
            raise
        
        finally:
            self.active_jobs.pop(job_id, None)
            self.job_history.append(job_result)
    
    async def create_streaming_pipeline(self, 
                                       source_config: Dict[str, Any],
                                       processing_config: Dict[str, Any],
                                       sink_config: Dict[str, Any]) -> str:
        """
        Create streaming data pipeline with Spark Structured Streaming
        
        Args:
            source_config: Source configuration (Kafka, file, etc.)
            processing_config: Processing configuration
            sink_config: Sink configuration
        
        Returns:
            Pipeline ID
        """
        pipeline_id = str(uuid.uuid4())
        
        spark = self.initialize_spark_session()
        
        # Create streaming DataFrame
        if source_config['type'] == 'kafka':
            streaming_df = spark.readStream \
                .format("kafka") \
                .option("kafka.bootstrap.servers", source_config['bootstrap_servers']) \
                .option("subscribe", source_config['topic']) \
                .load()
            
            # Parse Kafka messages
            streaming_df = streaming_df.select(
                col("key").cast("string"),
                col("value").cast("string"),
                col("timestamp")
            )
            
        elif source_config['type'] == 'file':
            streaming_df = spark.readStream \
                .format(source_config.get('format', 'json')) \
                .option("path", source_config['path']) \
                .load()
        
        else:
            raise ValueError(f"Unsupported source type: {source_config['type']}")
        
        # Apply processing
        processed_df = await self._apply_streaming_processing(streaming_df, processing_config)
        
        # Configure sink
        query_builder = processed_df.writeStream
        
        if sink_config['type'] == 'kafka':
            query = query_builder \
                .format("kafka") \
                .option("kafka.bootstrap.servers", sink_config['bootstrap_servers']) \
                .option("topic", sink_config['topic']) \
                .option("checkpointLocation", sink_config.get('checkpoint_location', f"/tmp/spark_checkpoints/{pipeline_id}")) \
                .start()
        
        elif sink_config['type'] == 'parquet':
            query = query_builder \
                .format("parquet") \
                .option("path", sink_config['path']) \
                .option("checkpointLocation", sink_config.get('checkpoint_location', f"/tmp/spark_checkpoints/{pipeline_id}")) \
                .trigger(processingTime=sink_config.get('trigger_interval', '10 seconds')) \
                .start()
        
        elif sink_config['type'] == 'console':
            query = query_builder \
                .format("console") \
                .option("truncate", "false") \
                .start()
        
        else:
            raise ValueError(f"Unsupported sink type: {sink_config['type']}")
        
        # Store pipeline reference
        self.active_jobs[pipeline_id] = SparkJobResult(
            job_id=pipeline_id,
            job_type=SparkJobType.STREAMING_ETL,
            status="running",
            start_time=datetime.utcnow()
        )
        
        return pipeline_id
    
    async def train_ml_model(self, 
                            training_data_path: str,
                            model_config: Dict[str, Any],
                            output_path: str) -> SparkJobResult:
        """
        Train machine learning model using Spark MLlib
        
        Args:
            training_data_path: Path to training data
            model_config: Model configuration
            output_path: Path to save trained model
        
        Returns:
            Training job result
        """
        job_id = str(uuid.uuid4())
        job_result = SparkJobResult(
            job_id=job_id,
            job_type=SparkJobType.ML_TRAINING,
            status="running",
            start_time=datetime.utcnow()
        )
        
        try:
            spark = self.initialize_spark_session()
            
            # Load training data
            df = spark.read.parquet(training_data_path)
            
            # Prepare features
            feature_columns = model_config.get('feature_columns', [])
            target_column = model_config.get('target_column', 'label')
            
            # Feature engineering pipeline
            stages = []
            
            # String indexing for categorical features
            categorical_columns = model_config.get('categorical_columns', [])
            for col_name in categorical_columns:
                indexer = StringIndexer(inputCol=col_name, outputCol=f"{col_name}_indexed")
                stages.append(indexer)
                
                # One-hot encoding
                encoder = OneHotEncoder(inputCol=f"{col_name}_indexed", outputCol=f"{col_name}_encoded")
                stages.append(encoder)
                
                # Update feature columns
                feature_columns = [f"{col_name}_encoded" if c == col_name else c for c in feature_columns]
            
            # Vector assembly
            assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
            stages.append(assembler)
            
            # Model selection
            model_type = model_config.get('model_type', 'logistic_regression')
            if model_type == 'logistic_regression':
                model = LogisticRegression(featuresCol="features", labelCol=target_column)
            elif model_type == 'random_forest':
                model = RandomForestClassifier(featuresCol="features", labelCol=target_column)
            elif model_type == 'kmeans':
                model = KMeans(featuresCol="features", k=model_config.get('k', 5))
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            stages.append(model)
            
            # Create pipeline
            pipeline = Pipeline(stages=stages)
            
            # Split data for training and testing
            if model_config.get('split_data', True):
                train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
            else:
                train_df = df
                test_df = None
            
            # Train model
            pipeline_model = pipeline.fit(train_df)
            
            # Evaluate model if test data is available
            metrics = {}
            if test_df is not None and model_type in ['logistic_regression', 'random_forest']:
                predictions = pipeline_model.transform(test_df)
                evaluator = BinaryClassificationEvaluator(labelCol=target_column)
                auc = evaluator.evaluate(predictions)
                metrics['auc'] = auc
            
            # Save model
            pipeline_model.write().overwrite().save(output_path)
            
            # Update job result
            job_result.end_time = datetime.utcnow()
            job_result.execution_time_seconds = (job_result.end_time - job_result.start_time).total_seconds()
            job_result.rows_processed = train_df.count()
            job_result.output_path = output_path
            job_result.status = "completed"
            job_result.metrics = metrics
            
            return job_result
            
        except Exception as e:
            job_result.end_time = datetime.utcnow()
            job_result.status = "failed"
            job_result.error_message = str(e)
            job_result.execution_time_seconds = (job_result.end_time - job_result.start_time).total_seconds()
            raise
        
        finally:
            self.job_history.append(job_result)
    
    async def _apply_optimizations(self, 
                                  df: SparkDataFrame,
                                  config: Dict[str, Any]) -> SparkDataFrame:
        """Apply Spark optimizations to DataFrame"""
        optimizations = config.get('optimizations', [])
        optimized_df = df
        
        for optimization in optimizations:
            if optimization == OptimizationStrategy.CACHE_OPTIMIZATION.value:
                if config.get('cache_strategy') == 'memory':
                    optimized_df = optimized_df.cache()
                elif config.get('cache_strategy') == 'disk':
                    optimized_df = optimized_df.persist(StorageLevel.DISK_ONLY)
            
            elif optimization == OptimizationStrategy.PARTITION_TUNING.value:
                partition_column = config.get('partition_column')
                num_partitions = config.get('num_partitions', 200)
                
                if partition_column:
                    optimized_df = optimized_df.repartition(num_partitions, col(partition_column))
                else:
                    optimized_df = optimized_df.repartition(num_partitions)
            
            elif optimization == OptimizationStrategy.COLUMN_PRUNING.value:
                required_columns = config.get('required_columns')
                if required_columns:
                    optimized_df = optimized_df.select(*required_columns)
            
            elif optimization == OptimizationStrategy.PREDICATE_PUSHDOWN.value:
                filters = config.get('filters', [])
                for filter_expr in filters:
                    optimized_df = optimized_df.filter(expr(filter_expr))
        
        return optimized_df
    
    async def _process_arbitration_detection(self, 
                                           df: SparkDataFrame,
                                           config: Dict[str, Any]) -> SparkDataFrame:
        """Process documents for arbitration clause detection"""
        text_column = config.get('text_column', 'text')
        
        # Define arbitration keywords
        arbitration_keywords = [
            'arbitration', 'arbitrator', 'arbitral', 'dispute resolution',
            'binding arbitration', 'final and binding', 'arbitration clause'
        ]
        
        # Create UDF for arbitration detection
        def detect_arbitration(text):
            if text is None:
                return False
            text_lower = text.lower()
            return any(keyword in text_lower for keyword in arbitration_keywords)
        
        from pyspark.sql.functions import udf
        from pyspark.sql.types import BooleanType
        
        arbitration_udf = udf(detect_arbitration, BooleanType())
        
        # Apply arbitration detection
        result_df = df.withColumn("contains_arbitration", arbitration_udf(col(text_column)))
        
        # Add confidence score
        def calculate_confidence(text):
            if text is None:
                return 0.0
            text_lower = text.lower()
            score = sum(1 for keyword in arbitration_keywords if keyword in text_lower)
            return min(score / len(arbitration_keywords), 1.0)
        
        confidence_udf = udf(calculate_confidence, DoubleType())
        result_df = result_df.withColumn("confidence_score", confidence_udf(col(text_column)))
        
        return result_df
    
    async def _process_clause_extraction(self, 
                                       df: SparkDataFrame,
                                       config: Dict[str, Any]) -> SparkDataFrame:
        """Extract specific clauses from legal documents"""
        text_column = config.get('text_column', 'text')
        clause_types = config.get('clause_types', ['liability', 'termination', 'payment'])
        
        # Define clause patterns
        clause_patterns = {
            'liability': ['liability', 'liable', 'damages', 'loss'],
            'termination': ['terminate', 'termination', 'end', 'expire'],
            'payment': ['payment', 'pay', 'fee', 'cost', 'price']
        }
        
        def extract_clauses(text):
            if text is None:
                return {}
            
            text_lower = text.lower()
            extracted = {}
            
            for clause_type in clause_types:
                if clause_type in clause_patterns:
                    patterns = clause_patterns[clause_type]
                    found = any(pattern in text_lower for pattern in patterns)
                    extracted[f"has_{clause_type}_clause"] = found
            
            return extracted
        
        from pyspark.sql.functions import udf
        from pyspark.sql.types import MapType, StringType, BooleanType
        
        extract_udf = udf(extract_clauses, MapType(StringType(), BooleanType()))
        
        # Apply clause extraction
        result_df = df.withColumn("extracted_clauses", extract_udf(col(text_column)))
        
        # Flatten the map into separate columns
        for clause_type in clause_types:
            result_df = result_df.withColumn(
                f"has_{clause_type}_clause",
                col("extracted_clauses").getItem(f"has_{clause_type}_clause")
            )
        
        return result_df
    
    async def _process_risk_assessment(self, 
                                     df: SparkDataFrame,
                                     config: Dict[str, Any]) -> SparkDataFrame:
        """Assess legal risks in documents"""
        text_column = config.get('text_column', 'text')
        
        # Define risk indicators
        high_risk_terms = ['penalty', 'forfeiture', 'breach', 'default', 'violation']
        medium_risk_terms = ['limitation', 'restriction', 'condition', 'requirement']
        low_risk_terms = ['standard', 'typical', 'common', 'usual']
        
        def assess_risk(text):
            if text is None:
                return 0.0
            
            text_lower = text.lower()
            
            high_risk_count = sum(1 for term in high_risk_terms if term in text_lower)
            medium_risk_count = sum(1 for term in medium_risk_terms if term in text_lower)
            low_risk_count = sum(1 for term in low_risk_terms if term in text_lower)
            
            # Calculate risk score (0-1 scale)
            risk_score = (high_risk_count * 0.8 + medium_risk_count * 0.5 + low_risk_count * 0.2) / 10
            return min(risk_score, 1.0)
        
        risk_udf = udf(assess_risk, DoubleType())
        
        # Apply risk assessment
        result_df = df.withColumn("risk_score", risk_udf(col(text_column)))
        
        # Categorize risk level
        result_df = result_df.withColumn(
            "risk_level",
            when(col("risk_score") >= 0.7, "high")
            .when(col("risk_score") >= 0.4, "medium")
            .otherwise("low")
        )
        
        return result_df
    
    async def _apply_streaming_processing(self, 
                                        streaming_df: SparkDataFrame,
                                        config: Dict[str, Any]) -> SparkDataFrame:
        """Apply processing to streaming DataFrame"""
        operation = config.get('operation', 'pass_through')
        
        if operation == 'arbitration_detection':
            return await self._process_arbitration_detection(streaming_df, config)
        elif operation == 'aggregation':
            return self._apply_streaming_aggregation(streaming_df, config)
        elif operation == 'windowing':
            return self._apply_windowing(streaming_df, config)
        else:
            return streaming_df
    
    def _apply_streaming_aggregation(self, 
                                   streaming_df: SparkDataFrame,
                                   config: Dict[str, Any]) -> SparkDataFrame:
        """Apply aggregations to streaming data"""
        group_by_columns = config.get('group_by_columns', [])
        aggregations = config.get('aggregations', {})
        
        if group_by_columns:
            grouped_df = streaming_df.groupBy(*group_by_columns)
        else:
            grouped_df = streaming_df
        
        # Apply aggregations
        agg_exprs = []
        for column, agg_type in aggregations.items():
            if agg_type == 'count':
                agg_exprs.append(count(column).alias(f"{column}_count"))
            elif agg_type == 'sum':
                agg_exprs.append(sum(column).alias(f"{column}_sum"))
            elif agg_type == 'avg':
                agg_exprs.append(avg(column).alias(f"{column}_avg"))
            elif agg_type == 'max':
                agg_exprs.append(max(column).alias(f"{column}_max"))
            elif agg_type == 'min':
                agg_exprs.append(min(column).alias(f"{column}_min"))
        
        if agg_exprs:
            return grouped_df.agg(*agg_exprs)
        else:
            return streaming_df
    
    def _apply_windowing(self, 
                        streaming_df: SparkDataFrame,
                        config: Dict[str, Any]) -> SparkDataFrame:
        """Apply time windows to streaming data"""
        timestamp_column = config.get('timestamp_column', 'timestamp')
        window_duration = config.get('window_duration', '10 minutes')
        slide_duration = config.get('slide_duration', '5 minutes')
        
        # Apply windowing
        windowed_df = streaming_df \
            .withWatermark(timestamp_column, "10 minutes") \
            .groupBy(
                window(col(timestamp_column), window_duration, slide_duration),
                *config.get('group_by_columns', [])
            )
        
        # Apply aggregations within windows
        aggregations = config.get('aggregations', {})
        agg_exprs = []
        for column, agg_type in aggregations.items():
            if agg_type == 'count':
                agg_exprs.append(count(column).alias(f"{column}_count"))
            elif agg_type == 'sum':
                agg_exprs.append(sum(column).alias(f"{column}_sum"))
            elif agg_type == 'avg':
                agg_exprs.append(avg(column).alias(f"{column}_avg"))
        
        if agg_exprs:
            return windowed_df.agg(*agg_exprs)
        else:
            return windowed_df.count()
    
    def _write_output(self, 
                     df: SparkDataFrame,
                     output_path: str,
                     config: Dict[str, Any]):
        """Write DataFrame to output path"""
        output_format = config.get('output_format', 'parquet')
        write_mode = config.get('write_mode', 'overwrite')
        
        writer = df.write.mode(write_mode)
        
        if output_format == 'parquet':
            partition_columns = config.get('partition_columns')
            if partition_columns:
                writer = writer.partitionBy(*partition_columns)
            writer.parquet(output_path)
        
        elif output_format == 'json':
            writer.json(output_path)
        
        elif output_format == 'csv':
            writer.option("header", "true").csv(output_path)
        
        elif output_format == 'delta':
            writer.format("delta").save(output_path)
        
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _update_performance_metrics(self, job_result: SparkJobResult):
        """Update performance metrics"""
        self.performance_metrics['total_jobs'] += 1
        
        if job_result.status == 'completed':
            self.performance_metrics['successful_jobs'] += 1
            
            if job_result.execution_time_seconds:
                current_avg = self.performance_metrics['avg_execution_time']
                total_successful = self.performance_metrics['successful_jobs']
                new_avg = ((current_avg * (total_successful - 1)) + job_result.execution_time_seconds) / total_successful
                self.performance_metrics['avg_execution_time'] = new_avg
        else:
            self.performance_metrics['failed_jobs'] += 1
        
        if job_result.rows_processed:
            self.performance_metrics['total_rows_processed'] += job_result.rows_processed
    
    def get_job_status(self, job_id: str) -> Optional[SparkJobResult]:
        """Get status of specific job"""
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]
        
        # Check job history
        for job in self.job_history:
            if job.job_id == job_id:
                return job
        
        return None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.performance_metrics.copy()
    
    def stop_streaming_pipeline(self, pipeline_id: str):
        """Stop streaming pipeline"""
        if pipeline_id in self.active_jobs:
            # In a real implementation, you would stop the StreamingQuery
            job_result = self.active_jobs[pipeline_id]
            job_result.status = "stopped"
            job_result.end_time = datetime.utcnow()
            
            self.active_jobs.pop(pipeline_id)
            self.job_history.append(job_result)
    
    def shutdown(self):
        """Shutdown Spark session and cleanup resources"""
        if self.spark:
            self.spark.stop()
            self.spark = None
        
        if self.streaming_context:
            self.streaming_context.stop()
            self.streaming_context = None