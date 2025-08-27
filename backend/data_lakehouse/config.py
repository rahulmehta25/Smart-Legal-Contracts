"""
Data Lakehouse Configuration Management

Centralized configuration for all lakehouse components:
- Environment-specific configurations
- Service discovery and connection management
- Security and authentication settings
- Resource allocation and scaling parameters
- Feature toggles and operational settings
"""

import os
import json
import yaml
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging


class Environment(Enum):
    """Deployment environments"""
    LOCAL = "local"
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


@dataclass
class SparkConfig:
    """Spark configuration settings"""
    app_name: str = "DataLakehouse"
    master: str = "local[*]"
    executor_instances: int = 4
    executor_cores: int = 4
    executor_memory: str = "8g"
    driver_memory: str = "4g"
    driver_cores: int = 2
    
    # Optimization settings
    adaptive_query_execution: bool = True
    adaptive_coalescing: bool = True
    cost_based_optimizer: bool = True
    
    # Delta Lake settings
    delta_optimizations: bool = True
    enable_change_data_feed: bool = False
    
    # Custom configurations
    custom_configs: Dict[str, str] = field(default_factory=dict)


@dataclass
class StorageConfig:
    """Storage configuration settings"""
    # Base paths
    data_root: str = "/tmp/lakehouse/data"
    checkpoint_root: str = "/tmp/lakehouse/checkpoints"
    metadata_root: str = "/tmp/lakehouse/metadata"
    
    # Delta Lake settings
    log_retention_duration: str = "30 days"
    deleted_file_retention_duration: str = "7 days"
    checkpoint_interval: int = 10
    
    # Partitioning
    default_partition_size_mb: int = 1024
    max_partitions: int = 10000
    
    # Compression
    compression_codec: str = "snappy"
    enable_compression: bool = True


@dataclass
class IngestionConfig:
    """Data ingestion configuration"""
    batch_size: int = 10000
    max_records_per_partition: int = 1000000
    parallelism: int = 8
    
    # Retry settings
    max_retry_attempts: int = 3
    retry_delay_seconds: int = 5
    
    # Quality checks
    enable_data_quality_checks: bool = True
    quality_error_threshold: float = 0.05
    
    # Streaming settings
    trigger_interval: str = "10 seconds"
    watermark_delay: str = "5 minutes"


@dataclass
class QueryConfig:
    """Query engine configuration"""
    default_engine: str = "spark_sql"
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    
    # Engine-specific settings
    presto: Dict[str, Any] = field(default_factory=dict)
    trino: Dict[str, Any] = field(default_factory=dict)
    dremio: Dict[str, Any] = field(default_factory=dict)
    
    # Query optimization
    enable_query_optimization: bool = True
    cost_based_optimization: bool = True


@dataclass
class MLConfig:
    """Machine Learning platform configuration"""
    # MLflow settings
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "default"
    
    # Feature store
    feature_store_root: str = "/tmp/lakehouse/feature_store"
    enable_feature_monitoring: bool = True
    
    # Model serving
    model_registry_path: str = "/tmp/lakehouse/models"
    default_serving_port: int = 8080
    
    # AutoML settings
    enable_automl: bool = True
    max_training_time_hours: int = 6
    hyperparameter_tuning: bool = True


@dataclass
class SecurityConfig:
    """Security and authentication configuration"""
    enable_authentication: bool = False
    enable_authorization: bool = False
    
    # Authentication providers
    auth_provider: str = "none"  # none, oauth2, ldap, kerberos
    oauth2_config: Dict[str, str] = field(default_factory=dict)
    
    # Encryption
    enable_encryption_at_rest: bool = False
    enable_encryption_in_transit: bool = False
    
    # Access control
    enable_row_level_security: bool = False
    enable_column_masking: bool = False


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Health checks
    health_check_interval_seconds: int = 30
    
    # Alerting
    enable_alerting: bool = False
    alert_config: Dict[str, Any] = field(default_factory=dict)


class LakehouseConfig:
    """
    Central configuration manager for the data lakehouse platform.
    
    Manages environment-specific configurations, service discovery,
    and runtime settings for all lakehouse components.
    """
    
    def __init__(self, 
                 environment: Environment = Environment.LOCAL,
                 config_file: Optional[str] = None,
                 config_dict: Optional[Dict[str, Any]] = None):
        self.environment = environment
        self.logger = logging.getLogger(__name__)
        
        # Component configurations
        self.spark: SparkConfig = SparkConfig()
        self.storage: StorageConfig = StorageConfig()
        self.ingestion: IngestionConfig = IngestionConfig()
        self.query: QueryConfig = QueryConfig()
        self.ml: MLConfig = MLConfig()
        self.security: SecurityConfig = SecurityConfig()
        self.monitoring: MonitoringConfig = MonitoringConfig()
        
        # Load configuration
        if config_file:
            self.load_from_file(config_file)
        elif config_dict:
            self.load_from_dict(config_dict)
        else:
            self.load_from_environment()
        
        # Apply environment-specific overrides
        self._apply_environment_overrides()
    
    def load_from_file(self, config_file: str):
        """Load configuration from YAML or JSON file"""
        try:
            with open(config_file, 'r') as f:
                if config_file.endswith('.yaml') or config_file.endswith('.yml'):
                    config_data = yaml.safe_load(f)
                elif config_file.endswith('.json'):
                    config_data = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_file}")
            
            self.load_from_dict(config_data)
            self.logger.info(f"Loaded configuration from {config_file}")
            
        except Exception as e:
            self.logger.error(f"Error loading config file {config_file}: {str(e)}")
            raise
    
    def load_from_dict(self, config_dict: Dict[str, Any]):
        """Load configuration from dictionary"""
        try:
            # Update component configurations
            if 'spark' in config_dict:
                self._update_config(self.spark, config_dict['spark'])
            
            if 'storage' in config_dict:
                self._update_config(self.storage, config_dict['storage'])
            
            if 'ingestion' in config_dict:
                self._update_config(self.ingestion, config_dict['ingestion'])
            
            if 'query' in config_dict:
                self._update_config(self.query, config_dict['query'])
            
            if 'ml' in config_dict:
                self._update_config(self.ml, config_dict['ml'])
            
            if 'security' in config_dict:
                self._update_config(self.security, config_dict['security'])
            
            if 'monitoring' in config_dict:
                self._update_config(self.monitoring, config_dict['monitoring'])
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {str(e)}")
            raise
    
    def load_from_environment(self):
        """Load configuration from environment variables"""
        try:
            # Spark configuration from environment
            if os.getenv('SPARK_MASTER'):
                self.spark.master = os.getenv('SPARK_MASTER')
            
            if os.getenv('SPARK_EXECUTOR_INSTANCES'):
                self.spark.executor_instances = int(os.getenv('SPARK_EXECUTOR_INSTANCES'))
            
            if os.getenv('SPARK_EXECUTOR_MEMORY'):
                self.spark.executor_memory = os.getenv('SPARK_EXECUTOR_MEMORY')
            
            # Storage configuration from environment
            if os.getenv('LAKEHOUSE_DATA_ROOT'):
                self.storage.data_root = os.getenv('LAKEHOUSE_DATA_ROOT')
            
            if os.getenv('LAKEHOUSE_CHECKPOINT_ROOT'):
                self.storage.checkpoint_root = os.getenv('LAKEHOUSE_CHECKPOINT_ROOT')
            
            # ML configuration from environment
            if os.getenv('MLFLOW_TRACKING_URI'):
                self.ml.mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
            
            # Security configuration from environment
            if os.getenv('LAKEHOUSE_ENABLE_AUTH'):
                self.security.enable_authentication = os.getenv('LAKEHOUSE_ENABLE_AUTH').lower() == 'true'
            
            self.logger.info("Loaded configuration from environment variables")
            
        except Exception as e:
            self.logger.error(f"Error loading from environment: {str(e)}")
    
    def _update_config(self, config_obj, config_dict: Dict[str, Any]):
        """Update configuration object with dictionary values"""
        for key, value in config_dict.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
            else:
                self.logger.warning(f"Unknown configuration key: {key}")
    
    def _apply_environment_overrides(self):
        """Apply environment-specific configuration overrides"""
        if self.environment == Environment.PRODUCTION:
            # Production overrides
            self.spark.adaptive_query_execution = True
            self.spark.cost_based_optimizer = True
            self.storage.enable_compression = True
            self.security.enable_authentication = True
            self.security.enable_authorization = True
            self.monitoring.enable_metrics = True
            self.monitoring.enable_alerting = True
            
        elif self.environment == Environment.STAGING:
            # Staging overrides
            self.spark.adaptive_query_execution = True
            self.storage.enable_compression = True
            self.security.enable_authentication = True
            self.monitoring.enable_metrics = True
            
        elif self.environment == Environment.DEVELOPMENT:
            # Development overrides
            self.spark.master = "local[*]"
            self.security.enable_authentication = False
            self.monitoring.log_level = "DEBUG"
            
        elif self.environment == Environment.LOCAL:
            # Local development overrides
            self.spark.master = "local[2]"
            self.spark.executor_instances = 1
            self.spark.executor_memory = "2g"
            self.spark.driver_memory = "1g"
            self.storage.data_root = "/tmp/lakehouse/data"
            self.security.enable_authentication = False
            self.monitoring.log_level = "DEBUG"
    
    def get_spark_config(self) -> Dict[str, str]:
        """Get Spark configuration as dictionary"""
        config = {
            "spark.app.name": self.spark.app_name,
            "spark.master": self.spark.master,
            "spark.executor.instances": str(self.spark.executor_instances),
            "spark.executor.cores": str(self.spark.executor_cores),
            "spark.executor.memory": self.spark.executor_memory,
            "spark.driver.memory": self.spark.driver_memory,
            "spark.driver.cores": str(self.spark.driver_cores),
            
            # Optimizations
            "spark.sql.adaptive.enabled": str(self.spark.adaptive_query_execution).lower(),
            "spark.sql.adaptive.coalescePartitions.enabled": str(self.spark.adaptive_coalescing).lower(),
            "spark.sql.cbo.enabled": str(self.spark.cost_based_optimizer).lower(),
            
            # Delta Lake
            "spark.sql.extensions": "io.delta.sql.DeltaSparkSessionExtension",
            "spark.sql.catalog.spark_catalog": "org.apache.spark.sql.delta.catalog.DeltaCatalog"
        }
        
        # Add custom configurations
        config.update(self.spark.custom_configs)
        
        if self.spark.delta_optimizations:
            config.update({
                "spark.databricks.delta.optimizeWrite.enabled": "true",
                "spark.databricks.delta.autoCompact.enabled": "true"
            })
        
        return config
    
    def get_connection_string(self, service: str) -> Optional[str]:
        """Get connection string for a service"""
        connections = {
            "mlflow": self.ml.mlflow_tracking_uri,
            "feature_store": f"file://{self.ml.feature_store_root}",
            "data_root": self.storage.data_root,
            "checkpoint_root": self.storage.checkpoint_root,
            "metadata_root": self.storage.metadata_root
        }
        
        return connections.get(service)
    
    def validate(self) -> List[str]:
        """Validate configuration and return any errors"""
        errors = []
        
        # Validate Spark configuration
        if self.spark.executor_instances <= 0:
            errors.append("Spark executor instances must be positive")
        
        if self.spark.executor_cores <= 0:
            errors.append("Spark executor cores must be positive")
        
        # Validate storage paths
        if not self.storage.data_root:
            errors.append("Storage data root must be specified")
        
        # Validate ingestion settings
        if self.ingestion.batch_size <= 0:
            errors.append("Ingestion batch size must be positive")
        
        if self.ingestion.quality_error_threshold < 0 or self.ingestion.quality_error_threshold > 1:
            errors.append("Quality error threshold must be between 0 and 1")
        
        # Validate ML settings
        if not self.ml.mlflow_tracking_uri:
            errors.append("MLflow tracking URI must be specified")
        
        return errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "environment": self.environment.value,
            "spark": {
                "app_name": self.spark.app_name,
                "master": self.spark.master,
                "executor_instances": self.spark.executor_instances,
                "executor_cores": self.spark.executor_cores,
                "executor_memory": self.spark.executor_memory,
                "driver_memory": self.spark.driver_memory,
                "driver_cores": self.spark.driver_cores,
                "adaptive_query_execution": self.spark.adaptive_query_execution,
                "cost_based_optimizer": self.spark.cost_based_optimizer,
                "delta_optimizations": self.spark.delta_optimizations,
                "custom_configs": self.spark.custom_configs
            },
            "storage": {
                "data_root": self.storage.data_root,
                "checkpoint_root": self.storage.checkpoint_root,
                "metadata_root": self.storage.metadata_root,
                "log_retention_duration": self.storage.log_retention_duration,
                "compression_codec": self.storage.compression_codec,
                "enable_compression": self.storage.enable_compression
            },
            "ingestion": {
                "batch_size": self.ingestion.batch_size,
                "parallelism": self.ingestion.parallelism,
                "enable_data_quality_checks": self.ingestion.enable_data_quality_checks,
                "trigger_interval": self.ingestion.trigger_interval
            },
            "query": {
                "default_engine": self.query.default_engine,
                "enable_caching": self.query.enable_caching,
                "cache_ttl_hours": self.query.cache_ttl_hours
            },
            "ml": {
                "mlflow_tracking_uri": self.ml.mlflow_tracking_uri,
                "feature_store_root": self.ml.feature_store_root,
                "enable_automl": self.ml.enable_automl
            },
            "security": {
                "enable_authentication": self.security.enable_authentication,
                "enable_authorization": self.security.enable_authorization,
                "auth_provider": self.security.auth_provider
            },
            "monitoring": {
                "enable_metrics": self.monitoring.enable_metrics,
                "log_level": self.monitoring.log_level,
                "health_check_interval_seconds": self.monitoring.health_check_interval_seconds
            }
        }
    
    def save_to_file(self, file_path: str):
        """Save configuration to file"""
        try:
            config_dict = self.to_dict()
            
            with open(file_path, 'w') as f:
                if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif file_path.endswith('.json'):
                    json.dump(config_dict, f, indent=2)
                else:
                    raise ValueError(f"Unsupported file format: {file_path}")
            
            self.logger.info(f"Configuration saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {str(e)}")
            raise
    
    def get_service_config(self, service: str) -> Dict[str, Any]:
        """Get configuration for a specific service"""
        service_configs = {
            "ingestion": {
                "batch_size": self.ingestion.batch_size,
                "parallelism": self.ingestion.parallelism,
                "max_retry_attempts": self.ingestion.max_retry_attempts,
                "enable_data_quality_checks": self.ingestion.enable_data_quality_checks,
                "quality_error_threshold": self.ingestion.quality_error_threshold,
                "trigger_interval": self.ingestion.trigger_interval,
                "watermark_delay": self.ingestion.watermark_delay
            },
            "storage": {
                "data_root": self.storage.data_root,
                "checkpoint_root": self.storage.checkpoint_root,
                "metadata_root": self.storage.metadata_root,
                "log_retention_duration": self.storage.log_retention_duration,
                "compression_codec": self.storage.compression_codec,
                "default_partition_size_mb": self.storage.default_partition_size_mb,
                "max_partitions": self.storage.max_partitions
            },
            "query": {
                "default_engine": self.query.default_engine,
                "enable_caching": self.query.enable_caching,
                "cache_ttl_hours": self.query.cache_ttl_hours,
                "enable_query_optimization": self.query.enable_query_optimization,
                "cost_based_optimization": self.query.cost_based_optimization,
                "presto": self.query.presto,
                "trino": self.query.trino,
                "dremio": self.query.dremio
            },
            "ml": {
                "mlflow_tracking_uri": self.ml.mlflow_tracking_uri,
                "mlflow_experiment_name": self.ml.mlflow_experiment_name,
                "feature_store_root": self.ml.feature_store_root,
                "enable_feature_monitoring": self.ml.enable_feature_monitoring,
                "model_registry_path": self.ml.model_registry_path,
                "default_serving_port": self.ml.default_serving_port,
                "enable_automl": self.ml.enable_automl,
                "max_training_time_hours": self.ml.max_training_time_hours
            },
            "security": {
                "enable_authentication": self.security.enable_authentication,
                "enable_authorization": self.security.enable_authorization,
                "auth_provider": self.security.auth_provider,
                "oauth2_config": self.security.oauth2_config,
                "enable_encryption_at_rest": self.security.enable_encryption_at_rest,
                "enable_row_level_security": self.security.enable_row_level_security
            },
            "monitoring": {
                "enable_metrics": self.monitoring.enable_metrics,
                "metrics_port": self.monitoring.metrics_port,
                "log_level": self.monitoring.log_level,
                "log_format": self.monitoring.log_format,
                "health_check_interval_seconds": self.monitoring.health_check_interval_seconds,
                "enable_alerting": self.monitoring.enable_alerting
            }
        }
        
        return service_configs.get(service, {})


# Utility functions for common configuration scenarios

def create_local_config() -> LakehouseConfig:
    """Create configuration for local development"""
    return LakehouseConfig(Environment.LOCAL)


def create_production_config() -> LakehouseConfig:
    """Create configuration for production deployment"""
    return LakehouseConfig(Environment.PRODUCTION)


def load_config_from_file(config_file: str) -> LakehouseConfig:
    """Load configuration from file with automatic environment detection"""
    # Try to detect environment from filename or environment variable
    env = Environment.LOCAL
    
    if "prod" in config_file.lower() or os.getenv("ENV") == "production":
        env = Environment.PRODUCTION
    elif "staging" in config_file.lower() or os.getenv("ENV") == "staging":
        env = Environment.STAGING
    elif "dev" in config_file.lower() or os.getenv("ENV") == "development":
        env = Environment.DEVELOPMENT
    
    return LakehouseConfig(env, config_file=config_file)


def create_config_template(file_path: str, environment: Environment = Environment.LOCAL):
    """Create a configuration template file"""
    config = LakehouseConfig(environment)
    config.save_to_file(file_path)