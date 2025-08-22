"""
Spark Processing Layer

Comprehensive data processing engine with:
- ETL/ELT pipeline orchestration
- Structured streaming processing
- Data quality validation and monitoring
- Schema evolution management
- Data lineage tracking
- Performance optimization
"""

from .spark_engine import SparkProcessingEngine
from .etl_pipeline import ETLPipeline, ETLStage
from .streaming_processor import StreamingProcessor
from .data_quality import DataQualityEngine, QualityCheck
from .lineage_tracker import LineageTracker
from .schema_manager import SchemaManager

__all__ = [
    "SparkProcessingEngine",
    "ETLPipeline",
    "ETLStage", 
    "StreamingProcessor",
    "DataQualityEngine",
    "QualityCheck",
    "LineageTracker",
    "SchemaManager"
]