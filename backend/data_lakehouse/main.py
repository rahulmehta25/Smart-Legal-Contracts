"""
Data Lakehouse Main Application

Main entry point and orchestrator for the data lakehouse platform:
- System initialization and configuration
- Component lifecycle management
- Service coordination and health monitoring
- CLI interface and administrative commands
- Graceful shutdown and cleanup
"""

import sys
import os
import argparse
import logging
import signal
import threading
import time
from typing import Dict, Any, Optional
from datetime import datetime

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspark.sql import SparkSession

from .config import LakehouseConfig, Environment, load_config_from_file, create_local_config
from .ingestion import DataIngestionEngine
from .storage import DeltaLakeManager
from .processing import SparkProcessingEngine
from .catalog import DataCatalogManager
from .query import QueryEngineManager
from .ml import MLPlatformManager


class LakehousePlatform:
    """
    Main data lakehouse platform orchestrator.
    
    Coordinates all lakehouse components and provides unified management
    interface for the entire platform.
    """
    
    def __init__(self, config: LakehouseConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Core components
        self.spark: Optional[SparkSession] = None
        self.ingestion_engine: Optional[DataIngestionEngine] = None
        self.storage_manager: Optional[DeltaLakeManager] = None
        self.processing_engine: Optional[SparkProcessingEngine] = None
        self.catalog_manager: Optional[DataCatalogManager] = None
        self.query_manager: Optional[QueryEngineManager] = None
        self.ml_platform: Optional[MLPlatformManager] = None
        
        # Platform state
        self.is_running = False
        self.start_time: Optional[datetime] = None
        self.shutdown_event = threading.Event()
        
        # Health monitoring
        self.health_status = {
            "status": "initializing",
            "components": {},
            "last_check": None
        }
        
        # Setup signal handlers
        self._setup_signal_handlers()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.monitoring.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s' 
                   if self.config.monitoring.log_format == 'standard'
                   else '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('/tmp/lakehouse.log') if self.config.environment != Environment.LOCAL else logging.StreamHandler(sys.stdout)
            ]
        )
        
        logger = logging.getLogger(__name__)
        logger.info(f"Logging initialized for environment: {self.config.environment.value}")
        return logger
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.shutdown()
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    def initialize(self) -> bool:
        """
        Initialize the lakehouse platform and all components
        
        Returns:
            bool: Success status
        """
        try:
            self.logger.info("Initializing Data Lakehouse Platform...")
            self.start_time = datetime.now()
            
            # Validate configuration
            config_errors = self.config.validate()
            if config_errors:
                self.logger.error(f"Configuration validation failed: {config_errors}")
                return False
            
            # Initialize Spark session
            if not self._initialize_spark():
                return False
            
            # Initialize core components
            if not self._initialize_components():
                return False
            
            # Start health monitoring
            self._start_health_monitoring()
            
            self.is_running = True
            self.health_status["status"] = "running"
            
            self.logger.info("Data Lakehouse Platform initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing platform: {str(e)}")
            self.health_status["status"] = "failed"
            return False
    
    def _initialize_spark(self) -> bool:
        """Initialize Spark session with lakehouse optimizations"""
        try:
            self.logger.info("Initializing Spark session...")
            
            # Get Spark configuration
            spark_config = self.config.get_spark_config()
            
            # Create Spark session builder
            builder = SparkSession.builder
            
            # Apply all configurations
            for key, value in spark_config.items():
                builder = builder.config(key, value)
            
            # Create session
            self.spark = builder.getOrCreate()
            
            # Log Spark configuration
            self.logger.info(f"Spark session created: {self.spark.sparkContext.applicationId}")
            self.logger.info(f"Spark version: {self.spark.version}")
            self.logger.info(f"Spark master: {self.spark.sparkContext.master}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing Spark: {str(e)}")
            return False
    
    def _initialize_components(self) -> bool:
        """Initialize all lakehouse components"""
        try:
            self.logger.info("Initializing lakehouse components...")
            
            # Initialize storage manager
            storage_config = self.config.get_service_config("storage")
            self.storage_manager = DeltaLakeManager(self.spark, storage_config)
            self.health_status["components"]["storage"] = "initialized"
            
            # Initialize ingestion engine
            ingestion_config = self.config.get_service_config("ingestion")
            self.ingestion_engine = DataIngestionEngine(self.spark, ingestion_config)
            self.health_status["components"]["ingestion"] = "initialized"
            
            # Initialize processing engine
            processing_config = self.config.get_service_config("processing")
            self.processing_engine = SparkProcessingEngine(
                self.config.get_service_config("processing"),
                self.spark
            )
            self.health_status["components"]["processing"] = "initialized"
            
            # Initialize catalog manager
            catalog_config = self.config.get_service_config("catalog")
            self.catalog_manager = DataCatalogManager(self.spark, catalog_config)
            self.health_status["components"]["catalog"] = "initialized"
            
            # Initialize query manager
            query_config = self.config.get_service_config("query")
            self.query_manager = QueryEngineManager(self.spark, query_config)
            self.health_status["components"]["query"] = "initialized"
            
            # Initialize ML platform
            ml_config = self.config.get_service_config("ml")
            self.ml_platform = MLPlatformManager(self.spark, ml_config)
            self.health_status["components"]["ml"] = "initialized"
            
            self.logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}")
            return False
    
    def _start_health_monitoring(self):
        """Start background health monitoring"""
        def health_monitor():
            while not self.shutdown_event.is_set():
                try:
                    self._perform_health_check()
                    time.sleep(self.config.monitoring.health_check_interval_seconds)
                except Exception as e:
                    self.logger.error(f"Error in health monitoring: {str(e)}")
        
        health_thread = threading.Thread(target=health_monitor, daemon=True)
        health_thread.start()
        self.logger.info("Health monitoring started")
    
    def _perform_health_check(self):
        """Perform comprehensive health check"""
        try:
            self.health_status["last_check"] = datetime.now()
            component_health = {}
            
            # Check Spark session
            if self.spark:
                try:
                    # Simple Spark operation to verify health
                    self.spark.sql("SELECT 1").collect()
                    component_health["spark"] = "healthy"
                except Exception as e:
                    component_health["spark"] = f"unhealthy: {str(e)}"
            
            # Check each component
            components = {
                "storage": self.storage_manager,
                "ingestion": self.ingestion_engine,
                "processing": self.processing_engine,
                "catalog": self.catalog_manager,
                "query": self.query_manager,
                "ml": self.ml_platform
            }
            
            for name, component in components.items():
                if component and hasattr(component, 'health_check'):
                    try:
                        health = component.health_check()
                        component_health[name] = "healthy" if health.get("healthy", False) else "unhealthy"
                    except Exception as e:
                        component_health[name] = f"unhealthy: {str(e)}"
                elif component:
                    component_health[name] = "healthy"
                else:
                    component_health[name] = "not_initialized"
            
            # Update overall health status
            unhealthy_components = [name for name, status in component_health.items() 
                                  if "unhealthy" in str(status)]
            
            if unhealthy_components:
                self.health_status["status"] = "degraded"
            else:
                self.health_status["status"] = "healthy"
            
            self.health_status["components"] = component_health
            
        except Exception as e:
            self.logger.error(f"Error performing health check: {str(e)}")
            self.health_status["status"] = "unhealthy"
    
    def get_platform_status(self) -> Dict[str, Any]:
        """Get comprehensive platform status"""
        try:
            status = {
                "platform": {
                    "environment": self.config.environment.value,
                    "running": self.is_running,
                    "start_time": self.start_time.isoformat() if self.start_time else None,
                    "uptime_seconds": (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
                },
                "health": self.health_status,
                "components": {}
            }
            
            # Get component-specific metrics
            if self.ingestion_engine:
                status["components"]["ingestion"] = self.ingestion_engine.get_global_metrics()
            
            if self.processing_engine:
                status["components"]["processing"] = self.processing_engine.get_cluster_metrics()
            
            if self.catalog_manager:
                status["components"]["catalog"] = self.catalog_manager.get_catalog_stats()
            
            if self.query_manager:
                status["components"]["query"] = self.query_manager.get_engine_metrics()
            
            if self.ml_platform:
                status["components"]["ml"] = self.ml_platform.get_platform_metrics()
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting platform status: {str(e)}")
            return {"error": str(e)}
    
    def run(self):
        """Run the platform (blocking call)"""
        if not self.is_running:
            if not self.initialize():
                sys.exit(1)
        
        self.logger.info("Data Lakehouse Platform is running...")
        self.logger.info("Press Ctrl+C to shutdown gracefully")
        
        try:
            # Keep the main thread alive
            while not self.shutdown_event.is_set():
                time.sleep(1)
        
        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received")
        
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Graceful shutdown of the platform"""
        if not self.is_running:
            return
        
        self.logger.info("Shutting down Data Lakehouse Platform...")
        self.is_running = False
        self.shutdown_event.set()
        
        try:
            # Shutdown components in reverse order
            components = [
                ("ML Platform", self.ml_platform),
                ("Query Manager", self.query_manager),
                ("Catalog Manager", self.catalog_manager),
                ("Processing Engine", self.processing_engine),
                ("Ingestion Engine", self.ingestion_engine),
                ("Storage Manager", self.storage_manager)
            ]
            
            for name, component in components:
                if component:
                    try:
                        if hasattr(component, 'cleanup'):
                            component.cleanup()
                        self.logger.info(f"{name} shutdown completed")
                    except Exception as e:
                        self.logger.error(f"Error shutting down {name}: {str(e)}")
            
            # Stop Spark session
            if self.spark:
                try:
                    self.spark.stop()
                    self.logger.info("Spark session stopped")
                except Exception as e:
                    self.logger.error(f"Error stopping Spark: {str(e)}")
            
            self.health_status["status"] = "stopped"
            self.logger.info("Data Lakehouse Platform shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {str(e)}")


def create_cli_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser"""
    parser = argparse.ArgumentParser(description="Data Lakehouse Platform")
    
    parser.add_argument(
        "--config",
        type=str,
        help="Configuration file path (YAML or JSON)"
    )
    
    parser.add_argument(
        "--environment",
        type=str,
        choices=["local", "development", "staging", "production"],
        default="local",
        help="Deployment environment"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as daemon (background process)"
    )
    
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show platform status and exit"
    )
    
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate configuration and exit"
    )
    
    parser.add_argument(
        "--create-config-template",
        type=str,
        help="Create configuration template file and exit"
    )
    
    return parser


def main():
    """Main entry point"""
    parser = create_cli_parser()
    args = parser.parse_args()
    
    try:
        # Handle special commands
        if args.create_config_template:
            from .config import create_config_template
            env = Environment(args.environment)
            create_config_template(args.create_config_template, env)
            print(f"Configuration template created: {args.create_config_template}")
            return
        
        # Load configuration
        if args.config:
            config = load_config_from_file(args.config)
        else:
            config = create_local_config()
        
        # Override log level if specified
        if args.log_level:
            config.monitoring.log_level = args.log_level
        
        # Validate configuration
        if args.validate_config:
            errors = config.validate()
            if errors:
                print("Configuration validation failed:")
                for error in errors:
                    print(f"  - {error}")
                sys.exit(1)
            else:
                print("Configuration is valid")
                return
        
        # Create and initialize platform
        platform = LakehousePlatform(config)
        
        # Show status if requested
        if args.status:
            if platform.initialize():
                status = platform.get_platform_status()
                import json
                print(json.dumps(status, indent=2, default=str))
                platform.shutdown()
            else:
                print("Platform failed to initialize")
                sys.exit(1)
            return
        
        # Run platform
        if args.daemon:
            # TODO: Implement proper daemon mode
            print("Daemon mode not yet implemented")
            sys.exit(1)
        else:
            platform.run()
    
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()