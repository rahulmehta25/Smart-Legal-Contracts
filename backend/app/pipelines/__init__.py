"""
Advanced Data Pipeline System

Provides enterprise-grade data pipeline capabilities including:
- Apache Spark integration for big data processing
- Kafka/Pulsar streaming pipelines
- Data lake architecture (S3/Delta Lake)
- ETL/ELT orchestration
- Data quality monitoring and validation
- Schema evolution handling
- Incremental processing
- Data lineage tracking
"""

from .spark_integration import SparkProcessor
from .streaming_pipelines import StreamingPipelineManager
from .data_lake import DataLakeManager
from .etl_orchestrator import ETLOrchestrator
from .data_quality import DataQualityMonitor
from .schema_evolution import SchemaEvolutionManager

__all__ = [
    'SparkProcessor',
    'StreamingPipelineManager',
    'DataLakeManager',
    'ETLOrchestrator',
    'DataQualityMonitor',
    'SchemaEvolutionManager'
]

__version__ = '1.0.0'