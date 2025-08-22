"""
Data Ingestion Layer

Multi-source data ingestion capabilities supporting:
- Batch ingestion from databases and files
- Real-time streaming with Kafka/Kinesis
- Change Data Capture (CDC)
- API connectors and IoT streams
"""

from .batch_ingestion import BatchIngestionEngine
from .streaming_ingestion import StreamingIngestionEngine
from .cdc_ingestion import CDCIngestionEngine
from .connectors import (
    DatabaseConnector,
    FileConnector,
    APIConnector,
    IoTConnector,
    KafkaConnector
)
from .data_ingestion_engine import DataIngestionEngine

__all__ = [
    "BatchIngestionEngine",
    "StreamingIngestionEngine", 
    "CDCIngestionEngine",
    "DatabaseConnector",
    "FileConnector",
    "APIConnector",
    "IoTConnector",
    "KafkaConnector",
    "DataIngestionEngine"
]