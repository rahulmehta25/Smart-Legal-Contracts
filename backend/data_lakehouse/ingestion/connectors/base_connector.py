"""
Base Connector Interface

Abstract base classes and configurations for all data connectors.
Provides standardized interface for batch and streaming data ingestion.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType


class ConnectorType(Enum):
    """Types of data connectors"""
    DATABASE = "database"
    FILE = "file"
    API = "api"
    STREAMING = "streaming"
    IOT = "iot"


class ConnectionStatus(Enum):
    """Connection status enumeration"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class ConnectorConfig:
    """Base configuration for all connectors"""
    connector_id: str
    connector_type: ConnectorType
    name: str
    description: Optional[str] = None
    connection_params: Optional[Dict[str, Any]] = None
    authentication: Optional[Dict[str, str]] = None
    timeout_seconds: int = 30
    retry_attempts: int = 3
    retry_delay_seconds: int = 5
    enable_ssl: bool = False
    ssl_config: Optional[Dict[str, str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class IngestionConfig:
    """Configuration for data ingestion operations"""
    batch_size: int = 10000
    max_records_per_partition: int = 1000000
    parallelism: int = 4
    enable_schema_inference: bool = True
    schema_validation: bool = True
    data_quality_checks: bool = True
    enable_compression: bool = True
    compression_codec: str = "snappy"
    incremental_column: Optional[str] = None
    incremental_strategy: str = "timestamp"  # timestamp, sequence, custom
    custom_transformations: Optional[List[str]] = None


class BaseConnector(ABC):
    """
    Abstract base class for all data connectors.
    
    Defines the standard interface that all connectors must implement
    for both batch and streaming data ingestion.
    """
    
    def __init__(self, config: ConnectorConfig, ingestion_config: IngestionConfig):
        self.config = config
        self.ingestion_config = ingestion_config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Connection state
        self.status = ConnectionStatus.DISCONNECTED
        self.last_connection_attempt: Optional[datetime] = None
        self.connection_error: Optional[str] = None
        
        # Metrics
        self.metrics = {
            "total_records_read": 0,
            "total_bytes_read": 0,
            "connection_attempts": 0,
            "successful_connections": 0,
            "failed_connections": 0,
            "last_activity": None
        }
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the data source.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        Close connection to the data source.
        
        Returns:
            bool: True if disconnection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def test_connection(self) -> Dict[str, Any]:
        """
        Test the connection to the data source.
        
        Returns:
            Dict[str, Any]: Connection test results with status and details
        """
        pass
    
    @abstractmethod
    def read_batch(
        self, 
        spark: SparkSession, 
        schema: Optional[StructType] = None
    ) -> Optional[DataFrame]:
        """
        Read data in batch mode.
        
        Args:
            spark: SparkSession instance
            schema: Optional predefined schema
            
        Returns:
            Optional[DataFrame]: DataFrame with data, None if failed
        """
        pass
    
    @abstractmethod
    def get_schema(self) -> Optional[StructType]:
        """
        Get the schema of the data source.
        
        Returns:
            Optional[StructType]: Schema of the data source
        """
        pass
    
    def read_stream(
        self, 
        spark: SparkSession, 
        schema: Optional[StructType] = None
    ) -> Optional[DataFrame]:
        """
        Read data in streaming mode.
        Default implementation raises NotImplementedError.
        Override in connectors that support streaming.
        
        Args:
            spark: SparkSession instance
            schema: Optional predefined schema
            
        Returns:
            Optional[DataFrame]: Streaming DataFrame, None if not supported
        """
        raise NotImplementedError(f"Streaming not supported by {self.__class__.__name__}")
    
    def read_cdc_stream(
        self, 
        spark: SparkSession, 
        schema: Optional[StructType] = None
    ) -> Optional[DataFrame]:
        """
        Read CDC (Change Data Capture) stream.
        Default implementation raises NotImplementedError.
        Override in connectors that support CDC.
        
        Args:
            spark: SparkSession instance
            schema: Optional predefined schema
            
        Returns:
            Optional[DataFrame]: CDC streaming DataFrame, None if not supported
        """
        raise NotImplementedError(f"CDC streaming not supported by {self.__class__.__name__}")
    
    def validate_config(self) -> List[str]:
        """
        Validate connector configuration.
        
        Returns:
            List[str]: List of validation errors (empty if valid)
        """
        errors = []
        
        if not self.config.connector_id:
            errors.append("connector_id is required")
        
        if not self.config.name:
            errors.append("name is required")
        
        if not self.config.connector_type:
            errors.append("connector_type is required")
        
        if self.config.timeout_seconds <= 0:
            errors.append("timeout_seconds must be positive")
        
        if self.config.retry_attempts < 0:
            errors.append("retry_attempts cannot be negative")
        
        return errors
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current connector status and metrics.
        
        Returns:
            Dict[str, Any]: Status information
        """
        return {
            "connector_id": self.config.connector_id,
            "name": self.config.name,
            "type": self.config.connector_type.value,
            "status": self.status.value,
            "last_connection_attempt": self.last_connection_attempt.isoformat() if self.last_connection_attempt else None,
            "connection_error": self.connection_error,
            "metrics": self.metrics.copy()
        }
    
    def update_metrics(self, records_read: int = 0, bytes_read: int = 0):
        """
        Update connector metrics.
        
        Args:
            records_read: Number of records read
            bytes_read: Number of bytes read
        """
        self.metrics["total_records_read"] += records_read
        self.metrics["total_bytes_read"] += bytes_read
        self.metrics["last_activity"] = datetime.now().isoformat()
    
    def _attempt_connection(self) -> bool:
        """
        Helper method to attempt connection with retry logic.
        
        Returns:
            bool: True if connection successful
        """
        self.metrics["connection_attempts"] += 1
        self.last_connection_attempt = datetime.now()
        
        for attempt in range(self.config.retry_attempts + 1):
            try:
                self.status = ConnectionStatus.CONNECTING
                
                if self.connect():
                    self.status = ConnectionStatus.CONNECTED
                    self.connection_error = None
                    self.metrics["successful_connections"] += 1
                    self.logger.info(f"Connected to {self.config.name} on attempt {attempt + 1}")
                    return True
                
            except Exception as e:
                error_msg = str(e)
                self.logger.warning(f"Connection attempt {attempt + 1} failed: {error_msg}")
                self.connection_error = error_msg
                
                if attempt < self.config.retry_attempts:
                    import time
                    time.sleep(self.config.retry_delay_seconds)
        
        self.status = ConnectionStatus.ERROR
        self.metrics["failed_connections"] += 1
        self.logger.error(f"Failed to connect to {self.config.name} after {self.config.retry_attempts + 1} attempts")
        return False
    
    def ensure_connected(self) -> bool:
        """
        Ensure the connector is connected, attempting to connect if necessary.
        
        Returns:
            bool: True if connected
        """
        if self.status != ConnectionStatus.CONNECTED:
            return self._attempt_connection()
        return True
    
    def supports_streaming(self) -> bool:
        """
        Check if connector supports streaming ingestion.
        
        Returns:
            bool: True if streaming is supported
        """
        try:
            # Try to call read_stream to see if it's implemented
            return hasattr(self, 'read_stream') and callable(getattr(self, 'read_stream'))
        except:
            return False
    
    def supports_cdc(self) -> bool:
        """
        Check if connector supports CDC ingestion.
        
        Returns:
            bool: True if CDC is supported
        """
        try:
            return hasattr(self, 'read_cdc_stream') and callable(getattr(self, 'read_cdc_stream'))
        except:
            return False
    
    def get_capabilities(self) -> Dict[str, bool]:
        """
        Get connector capabilities.
        
        Returns:
            Dict[str, bool]: Capability flags
        """
        return {
            "batch_read": True,  # All connectors must support batch read
            "streaming_read": self.supports_streaming(),
            "cdc_read": self.supports_cdc(),
            "schema_inference": self.ingestion_config.enable_schema_inference,
            "incremental_read": bool(self.ingestion_config.incremental_column)
        }
    
    def set_incremental_config(self, column: str, last_value: Any):
        """
        Configure incremental reading.
        
        Args:
            column: Column name for incremental reading
            last_value: Last processed value
        """
        self.ingestion_config.incremental_column = column
        self._last_incremental_value = last_value
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the connector.
        
        Returns:
            Dict[str, Any]: Health check results
        """
        try:
            # Test connection
            connection_test = self.test_connection()
            
            # Get current status
            status_info = self.get_status()
            
            # Determine overall health
            is_healthy = (
                self.status == ConnectionStatus.CONNECTED and
                connection_test.get("success", False)
            )
            
            return {
                "connector_id": self.config.connector_id,
                "healthy": is_healthy,
                "status": status_info,
                "connection_test": connection_test,
                "capabilities": self.get_capabilities(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "connector_id": self.config.connector_id,
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def cleanup(self):
        """
        Cleanup connector resources.
        """
        try:
            self.disconnect()
            self.logger.info(f"Cleaned up connector: {self.config.name}")
        except Exception as e:
            self.logger.error(f"Error during connector cleanup: {str(e)}")
    
    def __enter__(self):
        """Context manager entry"""
        self.ensure_connected()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(id={self.config.connector_id}, name={self.config.name}, status={self.status.value})"
    
    def __repr__(self) -> str:
        return self.__str__()