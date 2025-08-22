"""
Central Data Ingestion Engine

Orchestrates all data ingestion processes across different sources and formats.
Provides unified interface for batch, streaming, and CDC ingestion.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import *

from .batch_ingestion import BatchIngestionEngine
from .streaming_ingestion import StreamingIngestionEngine
from .cdc_ingestion import CDCIngestionEngine
from .connectors import ConnectorFactory, IngestionConfig


class IngestionMode(Enum):
    BATCH = "batch"
    STREAMING = "streaming"
    CDC = "cdc"
    HYBRID = "hybrid"


@dataclass
class IngestionJob:
    """Configuration for an ingestion job"""
    job_id: str
    source_type: str
    source_config: Dict[str, Any]
    target_path: str
    mode: IngestionMode
    schedule: Optional[str] = None
    schema: Optional[StructType] = None
    transformations: Optional[List[str]] = None
    quality_checks: Optional[Dict[str, Any]] = None
    retry_config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class DataIngestionEngine:
    """
    Central orchestrator for all data ingestion processes.
    
    Features:
    - Multi-source ingestion (databases, files, APIs, IoT)
    - Batch and streaming processing
    - Change Data Capture (CDC)
    - Schema evolution and validation
    - Data quality monitoring
    - Automatic retry and error handling
    - Lineage tracking
    """
    
    def __init__(self, spark: SparkSession, config: Dict[str, Any]):
        self.spark = spark
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize ingestion engines
        self.batch_engine = BatchIngestionEngine(spark, config.get("batch", {}))
        self.streaming_engine = StreamingIngestionEngine(spark, config.get("streaming", {}))
        self.cdc_engine = CDCIngestionEngine(spark, config.get("cdc", {}))
        
        # Connector factory
        self.connector_factory = ConnectorFactory(config.get("connectors", {}))
        
        # Active jobs tracking
        self.active_jobs: Dict[str, IngestionJob] = {}
        self.job_status: Dict[str, str] = {}
        
        # Metrics and monitoring
        self.metrics = {
            "jobs_completed": 0,
            "jobs_failed": 0,
            "total_records_ingested": 0,
            "total_bytes_ingested": 0
        }
    
    def register_job(self, job: IngestionJob) -> str:
        """Register a new ingestion job"""
        try:
            # Validate job configuration
            self._validate_job_config(job)
            
            # Store job configuration
            self.active_jobs[job.job_id] = job
            self.job_status[job.job_id] = "registered"
            
            self.logger.info(f"Registered ingestion job: {job.job_id}")
            return job.job_id
            
        except Exception as e:
            self.logger.error(f"Failed to register job {job.job_id}: {str(e)}")
            raise
    
    def start_job(self, job_id: str) -> bool:
        """Start an ingestion job"""
        try:
            if job_id not in self.active_jobs:
                raise ValueError(f"Job {job_id} not found")
            
            job = self.active_jobs[job_id]
            self.job_status[job_id] = "starting"
            
            # Create connector
            connector = self.connector_factory.create_connector(
                job.source_type, 
                job.source_config
            )
            
            # Route to appropriate engine based on mode
            if job.mode == IngestionMode.BATCH:
                result = self._start_batch_job(job, connector)
            elif job.mode == IngestionMode.STREAMING:
                result = self._start_streaming_job(job, connector)
            elif job.mode == IngestionMode.CDC:
                result = self._start_cdc_job(job, connector)
            elif job.mode == IngestionMode.HYBRID:
                result = self._start_hybrid_job(job, connector)
            else:
                raise ValueError(f"Unsupported ingestion mode: {job.mode}")
            
            if result:
                self.job_status[job_id] = "running"
                self.logger.info(f"Started ingestion job: {job_id}")
            else:
                self.job_status[job_id] = "failed"
                self.logger.error(f"Failed to start job: {job_id}")
            
            return result
            
        except Exception as e:
            self.job_status[job_id] = "failed"
            self.logger.error(f"Error starting job {job_id}: {str(e)}")
            return False
    
    def stop_job(self, job_id: str) -> bool:
        """Stop a running ingestion job"""
        try:
            if job_id not in self.active_jobs:
                raise ValueError(f"Job {job_id} not found")
            
            job = self.active_jobs[job_id]
            
            # Stop based on mode
            if job.mode == IngestionMode.STREAMING:
                self.streaming_engine.stop_stream(job_id)
            elif job.mode == IngestionMode.CDC:
                self.cdc_engine.stop_cdc_stream(job_id)
            
            self.job_status[job_id] = "stopped"
            self.logger.info(f"Stopped ingestion job: {job_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping job {job_id}: {str(e)}")
            return False
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status and metrics for a job"""
        if job_id not in self.active_jobs:
            return {"error": f"Job {job_id} not found"}
        
        job = self.active_jobs[job_id]
        status = self.job_status.get(job_id, "unknown")
        
        # Get detailed metrics based on job mode
        metrics = {}
        if job.mode == IngestionMode.BATCH:
            metrics = self.batch_engine.get_job_metrics(job_id)
        elif job.mode == IngestionMode.STREAMING:
            metrics = self.streaming_engine.get_stream_metrics(job_id)
        elif job.mode == IngestionMode.CDC:
            metrics = self.cdc_engine.get_cdc_metrics(job_id)
        
        return {
            "job_id": job_id,
            "status": status,
            "mode": job.mode.value,
            "source_type": job.source_type,
            "target_path": job.target_path,
            "metrics": metrics,
            "last_updated": datetime.now().isoformat()
        }
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all registered jobs with their status"""
        return [self.get_job_status(job_id) for job_id in self.active_jobs.keys()]
    
    def get_global_metrics(self) -> Dict[str, Any]:
        """Get global ingestion metrics"""
        return {
            "total_jobs": len(self.active_jobs),
            "running_jobs": len([s for s in self.job_status.values() if s == "running"]),
            "failed_jobs": len([s for s in self.job_status.values() if s == "failed"]),
            "completed_jobs": self.metrics["jobs_completed"],
            "total_records": self.metrics["total_records_ingested"],
            "total_bytes": self.metrics["total_bytes_ingested"],
            "uptime": (datetime.now() - getattr(self, 'start_time', datetime.now())).total_seconds()
        }
    
    def _validate_job_config(self, job: IngestionJob):
        """Validate job configuration"""
        if not job.job_id:
            raise ValueError("Job ID is required")
        
        if job.job_id in self.active_jobs:
            raise ValueError(f"Job {job.job_id} already exists")
        
        if not job.source_type:
            raise ValueError("Source type is required")
        
        if not job.target_path:
            raise ValueError("Target path is required")
        
        # Validate source configuration
        required_fields = self.connector_factory.get_required_fields(job.source_type)
        for field in required_fields:
            if field not in job.source_config:
                raise ValueError(f"Required field '{field}' missing from source config")
    
    def _start_batch_job(self, job: IngestionJob, connector) -> bool:
        """Start a batch ingestion job"""
        try:
            return self.batch_engine.run_batch_ingestion(
                job_id=job.job_id,
                connector=connector,
                target_path=job.target_path,
                schema=job.schema,
                transformations=job.transformations,
                quality_checks=job.quality_checks
            )
        except Exception as e:
            self.logger.error(f"Batch job failed: {str(e)}")
            return False
    
    def _start_streaming_job(self, job: IngestionJob, connector) -> bool:
        """Start a streaming ingestion job"""
        try:
            return self.streaming_engine.start_stream(
                job_id=job.job_id,
                connector=connector,
                target_path=job.target_path,
                schema=job.schema,
                transformations=job.transformations
            )
        except Exception as e:
            self.logger.error(f"Streaming job failed: {str(e)}")
            return False
    
    def _start_cdc_job(self, job: IngestionJob, connector) -> bool:
        """Start a CDC ingestion job"""
        try:
            return self.cdc_engine.start_cdc_stream(
                job_id=job.job_id,
                connector=connector,
                target_path=job.target_path,
                schema=job.schema
            )
        except Exception as e:
            self.logger.error(f"CDC job failed: {str(e)}")
            return False
    
    def _start_hybrid_job(self, job: IngestionJob, connector) -> bool:
        """Start a hybrid batch + streaming job"""
        try:
            # Start initial batch load
            batch_success = self._start_batch_job(job, connector)
            if not batch_success:
                return False
            
            # Start streaming for incremental updates
            streaming_job = IngestionJob(
                job_id=f"{job.job_id}_stream",
                source_type=job.source_type,
                source_config=job.source_config,
                target_path=job.target_path,
                mode=IngestionMode.STREAMING,
                schema=job.schema,
                transformations=job.transformations
            )
            
            return self._start_streaming_job(streaming_job, connector)
            
        except Exception as e:
            self.logger.error(f"Hybrid job failed: {str(e)}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all ingestion components"""
        health = {
            "status": "healthy",
            "components": {},
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Check Spark session
            health["components"]["spark"] = {
                "status": "healthy" if self.spark else "unhealthy",
                "version": self.spark.version if self.spark else None
            }
            
            # Check connectors
            health["components"]["connectors"] = await self.connector_factory.health_check()
            
            # Check running jobs
            failing_jobs = [job_id for job_id, status in self.job_status.items() 
                          if status == "failed"]
            
            if failing_jobs:
                health["status"] = "degraded"
                health["failing_jobs"] = failing_jobs
            
        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)
        
        return health
    
    def cleanup(self):
        """Cleanup resources and stop all jobs"""
        try:
            for job_id in list(self.active_jobs.keys()):
                self.stop_job(job_id)
            
            self.streaming_engine.cleanup()
            self.cdc_engine.cleanup()
            
            self.logger.info("Data ingestion engine cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")


# Factory function for easy initialization
def create_ingestion_engine(spark: SparkSession, config: Dict[str, Any]) -> DataIngestionEngine:
    """Factory function to create and configure ingestion engine"""
    return DataIngestionEngine(spark, config)