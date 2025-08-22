"""
Modern Data Lakehouse Architecture

A comprehensive data lakehouse implementation using Apache Spark, Delta Lake,
and modern data stack technologies for unified batch and streaming analytics.
"""

__version__ = "1.0.0"
__author__ = "Data Engineering Team"

from .ingestion import DataIngestionEngine
from .storage import DeltaLakeManager
from .processing import SparkProcessingEngine
from .catalog import DataCatalogManager
from .query import QueryEngineManager
from .ml import MLPlatformManager

__all__ = [
    "DataIngestionEngine",
    "DeltaLakeManager", 
    "SparkProcessingEngine",
    "DataCatalogManager",
    "QueryEngineManager",
    "MLPlatformManager"
]