"""
Connector Factory

Central factory for creating and managing data connectors.
Provides registration, discovery, and lifecycle management of connectors.
"""

import logging
from typing import Dict, List, Optional, Any, Type, Union
from dataclasses import dataclass
from enum import Enum
import importlib
import inspect

from .base_connector import BaseConnector, ConnectorConfig, IngestionConfig, ConnectorType


@dataclass
class ConnectorRegistration:
    """Connector registration information"""
    connector_class: Type[BaseConnector]
    connector_type: ConnectorType
    name: str
    description: str
    required_params: List[str]
    optional_params: List[str]
    supports_streaming: bool
    supports_cdc: bool
    version: str = "1.0.0"


class ConnectorFactory:
    """
    Factory class for creating and managing data connectors.
    
    Features:
    - Dynamic connector registration and discovery
    - Connector lifecycle management
    - Configuration validation
    - Health monitoring
    - Plugin system for custom connectors
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Registered connectors
        self._registered_connectors: Dict[str, ConnectorRegistration] = {}
        
        # Active connector instances
        self._active_connectors: Dict[str, BaseConnector] = {}
        
        # Initialize built-in connectors
        self._register_builtin_connectors()
        
        # Load plugin connectors if configured
        self._load_plugin_connectors()
    
    def _register_builtin_connectors(self):
        """Register built-in connector types"""
        try:
            # Database connectors
            self._register_connector_type(
                "postgresql",
                "database_connector",
                "JDBCConnector",
                ConnectorType.DATABASE,
                "PostgreSQL Database Connector",
                ["host", "port", "database", "username", "password"],
                ["schema", "ssl_mode", "connection_properties"],
                supports_streaming=False,
                supports_cdc=True
            )
            
            self._register_connector_type(
                "mysql",
                "database_connector", 
                "JDBCConnector",
                ConnectorType.DATABASE,
                "MySQL Database Connector",
                ["host", "port", "database", "username", "password"],
                ["charset", "ssl_config"],
                supports_streaming=False,
                supports_cdc=True
            )
            
            # File connectors
            self._register_connector_type(
                "csv",
                "file_connector",
                "FileConnector",
                ConnectorType.FILE,
                "CSV File Connector",
                ["path"],
                ["delimiter", "header", "schema"],
                supports_streaming=True,
                supports_cdc=False
            )
            
            self._register_connector_type(
                "parquet",
                "file_connector",
                "FileConnector", 
                ConnectorType.FILE,
                "Parquet File Connector",
                ["path"],
                ["schema"],
                supports_streaming=True,
                supports_cdc=False
            )
            
            # Streaming connectors
            self._register_connector_type(
                "kafka",
                "kafka_connector",
                "KafkaConnector",
                ConnectorType.STREAMING,
                "Apache Kafka Connector",
                ["bootstrap_servers", "topic"],
                ["consumer_group", "security_protocol", "schema_registry_url"],
                supports_streaming=True,
                supports_cdc=True
            )
            
            # API connectors
            self._register_connector_type(
                "rest_api",
                "api_connector",
                "RESTConnector",
                ConnectorType.API,
                "REST API Connector",
                ["base_url"],
                ["auth_config", "headers", "rate_limit"],
                supports_streaming=False,
                supports_cdc=False
            )
            
            # IoT connectors
            self._register_connector_type(
                "mqtt",
                "iot_connector",
                "IoTConnector",
                ConnectorType.IOT,
                "MQTT IoT Connector",
                ["broker_host", "topic"],
                ["username", "password", "qos"],
                supports_streaming=True,
                supports_cdc=False
            )
            
        except Exception as e:
            self.logger.error(f"Error registering built-in connectors: {str(e)}")
    
    def _register_connector_type(
        self,
        connector_id: str,
        module_name: str,
        class_name: str,
        connector_type: ConnectorType,
        description: str,
        required_params: List[str],
        optional_params: List[str],
        supports_streaming: bool = False,
        supports_cdc: bool = False,
        version: str = "1.0.0"
    ):
        """Register a connector type"""
        try:
            # Dynamic import of connector class
            module = importlib.import_module(f".{module_name}", package=__package__)
            connector_class = getattr(module, class_name)
            
            # Validate it's a BaseConnector subclass
            if not issubclass(connector_class, BaseConnector):
                raise ValueError(f"{class_name} must inherit from BaseConnector")
            
            # Register the connector
            registration = ConnectorRegistration(
                connector_class=connector_class,
                connector_type=connector_type,
                name=connector_id,
                description=description,
                required_params=required_params,
                optional_params=optional_params,
                supports_streaming=supports_streaming,
                supports_cdc=supports_cdc,
                version=version
            )
            
            self._registered_connectors[connector_id] = registration
            self.logger.info(f"Registered connector: {connector_id}")
            
        except ImportError as e:
            self.logger.warning(f"Could not import connector {connector_id}: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error registering connector {connector_id}: {str(e)}")
    
    def _load_plugin_connectors(self):
        """Load plugin connectors from configuration"""
        plugins = self.config.get("plugins", {})
        
        for plugin_name, plugin_config in plugins.items():
            try:
                module_path = plugin_config.get("module")
                class_name = plugin_config.get("class")
                
                if module_path and class_name:
                    # Import plugin module
                    module = importlib.import_module(module_path)
                    connector_class = getattr(module, class_name)
                    
                    # Register plugin connector
                    self._register_plugin_connector(plugin_name, connector_class, plugin_config)
                    
            except Exception as e:
                self.logger.error(f"Error loading plugin connector {plugin_name}: {str(e)}")
    
    def _register_plugin_connector(self, name: str, connector_class: Type[BaseConnector], config: Dict[str, Any]):
        """Register a plugin connector"""
        try:
            if not issubclass(connector_class, BaseConnector):
                raise ValueError("Plugin connector must inherit from BaseConnector")
            
            registration = ConnectorRegistration(
                connector_class=connector_class,
                connector_type=ConnectorType(config.get("type", "api")),
                name=name,
                description=config.get("description", f"Plugin connector: {name}"),
                required_params=config.get("required_params", []),
                optional_params=config.get("optional_params", []),
                supports_streaming=config.get("supports_streaming", False),
                supports_cdc=config.get("supports_cdc", False),
                version=config.get("version", "1.0.0")
            )
            
            self._registered_connectors[name] = registration
            self.logger.info(f"Registered plugin connector: {name}")
            
        except Exception as e:
            self.logger.error(f"Error registering plugin connector {name}: {str(e)}")
            raise
    
    def create_connector(
        self,
        connector_type: str,
        connection_params: Dict[str, Any],
        connector_id: Optional[str] = None,
        ingestion_config: Optional[IngestionConfig] = None
    ) -> BaseConnector:
        """
        Create a connector instance
        
        Args:
            connector_type: Type of connector to create
            connection_params: Connection parameters
            connector_id: Optional unique identifier
            ingestion_config: Optional ingestion configuration
            
        Returns:
            BaseConnector: Configured connector instance
        """
        if connector_type not in self._registered_connectors:
            raise ValueError(f"Unknown connector type: {connector_type}")
        
        registration = self._registered_connectors[connector_type]
        
        # Validate required parameters
        missing_params = []
        for param in registration.required_params:
            if param not in connection_params:
                missing_params.append(param)
        
        if missing_params:
            raise ValueError(f"Missing required parameters for {connector_type}: {missing_params}")
        
        # Generate connector ID if not provided
        if connector_id is None:
            import uuid
            connector_id = f"{connector_type}_{uuid.uuid4().hex[:8]}"
        
        # Create connector configuration
        connector_config = ConnectorConfig(
            connector_id=connector_id,
            connector_type=registration.connector_type,
            name=f"{connector_type}_{connector_id}",
            description=registration.description,
            connection_params=connection_params,
            **self._extract_connector_config(connection_params)
        )
        
        # Use provided ingestion config or create default
        if ingestion_config is None:
            ingestion_config = IngestionConfig()
        
        try:
            # Instantiate the connector
            connector = registration.connector_class(connector_config, ingestion_config)
            
            # Store active connector
            self._active_connectors[connector_id] = connector
            
            self.logger.info(f"Created connector: {connector_id} (type: {connector_type})")
            return connector
            
        except Exception as e:
            self.logger.error(f"Error creating connector {connector_id}: {str(e)}")
            raise
    
    def _extract_connector_config(self, connection_params: Dict[str, Any]) -> Dict[str, Any]:
        """Extract connector configuration from connection parameters"""
        config = {}
        
        # Extract common configuration parameters
        if "timeout" in connection_params:
            config["timeout_seconds"] = connection_params["timeout"]
        
        if "retry_attempts" in connection_params:
            config["retry_attempts"] = connection_params["retry_attempts"]
        
        if "ssl_config" in connection_params:
            config["ssl_config"] = connection_params["ssl_config"]
            config["enable_ssl"] = True
        
        if "authentication" in connection_params:
            config["authentication"] = connection_params["authentication"]
        
        return config
    
    def get_connector(self, connector_id: str) -> Optional[BaseConnector]:
        """Get an active connector by ID"""
        return self._active_connectors.get(connector_id)
    
    def list_registered_connectors(self) -> Dict[str, Dict[str, Any]]:
        """List all registered connector types"""
        connectors = {}
        
        for name, registration in self._registered_connectors.items():
            connectors[name] = {
                "type": registration.connector_type.value,
                "description": registration.description,
                "required_params": registration.required_params,
                "optional_params": registration.optional_params,
                "supports_streaming": registration.supports_streaming,
                "supports_cdc": registration.supports_cdc,
                "version": registration.version
            }
        
        return connectors
    
    def list_active_connectors(self) -> Dict[str, Dict[str, Any]]:
        """List all active connector instances"""
        connectors = {}
        
        for connector_id, connector in self._active_connectors.items():
            connectors[connector_id] = connector.get_status()
        
        return connectors
    
    def get_required_fields(self, connector_type: str) -> List[str]:
        """Get required configuration fields for a connector type"""
        if connector_type not in self._registered_connectors:
            raise ValueError(f"Unknown connector type: {connector_type}")
        
        return self._registered_connectors[connector_type].required_params
    
    def validate_connector_config(self, connector_type: str, config: Dict[str, Any]) -> List[str]:
        """Validate connector configuration"""
        if connector_type not in self._registered_connectors:
            return [f"Unknown connector type: {connector_type}"]
        
        registration = self._registered_connectors[connector_type]
        errors = []
        
        # Check required parameters
        for param in registration.required_params:
            if param not in config:
                errors.append(f"Missing required parameter: {param}")
        
        # Type-specific validations could be added here
        
        return errors
    
    def remove_connector(self, connector_id: str) -> bool:
        """Remove and cleanup a connector"""
        try:
            if connector_id in self._active_connectors:
                connector = self._active_connectors[connector_id]
                connector.cleanup()
                del self._active_connectors[connector_id]
                
                self.logger.info(f"Removed connector: {connector_id}")
                return True
            else:
                self.logger.warning(f"Connector {connector_id} not found")
                return False
                
        except Exception as e:
            self.logger.error(f"Error removing connector {connector_id}: {str(e)}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all active connectors"""
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "total_connectors": len(self._active_connectors),
            "healthy_connectors": 0,
            "unhealthy_connectors": 0,
            "connectors": {}
        }
        
        for connector_id, connector in self._active_connectors.items():
            try:
                connector_health = await connector.health_check()
                health_report["connectors"][connector_id] = connector_health
                
                if connector_health.get("healthy", False):
                    health_report["healthy_connectors"] += 1
                else:
                    health_report["unhealthy_connectors"] += 1
                    
            except Exception as e:
                health_report["connectors"][connector_id] = {
                    "healthy": False,
                    "error": str(e)
                }
                health_report["unhealthy_connectors"] += 1
        
        return health_report
    
    def get_connector_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics from all connectors"""
        total_records = 0
        total_bytes = 0
        total_connections = 0
        failed_connections = 0
        
        connector_metrics = {}
        
        for connector_id, connector in self._active_connectors.items():
            status = connector.get_status()
            metrics = status.get("metrics", {})
            
            total_records += metrics.get("total_records_read", 0)
            total_bytes += metrics.get("total_bytes_read", 0)
            total_connections += metrics.get("connection_attempts", 0)
            failed_connections += metrics.get("failed_connections", 0)
            
            connector_metrics[connector_id] = metrics
        
        return {
            "summary": {
                "total_records_read": total_records,
                "total_bytes_read": total_bytes,
                "total_connections": total_connections,
                "failed_connections": failed_connections,
                "success_rate": (total_connections - failed_connections) / max(total_connections, 1) * 100
            },
            "connectors": connector_metrics
        }
    
    def cleanup_all(self):
        """Cleanup all active connectors"""
        try:
            for connector_id in list(self._active_connectors.keys()):
                self.remove_connector(connector_id)
            
            self.logger.info("All connectors cleaned up successfully")
            
        except Exception as e:
            self.logger.error(f"Error during connector cleanup: {str(e)}")
    
    def __len__(self) -> int:
        return len(self._active_connectors)
    
    def __contains__(self, connector_id: str) -> bool:
        return connector_id in self._active_connectors
    
    def __iter__(self):
        return iter(self._active_connectors.values())