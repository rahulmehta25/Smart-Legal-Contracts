"""
Batch Processing Service with Celery

Provides asynchronous batch processing capabilities:
- Queue system with Celery and Redis
- Progress tracking and status updates
- Error recovery and retry logic
- Result caching and storage
- Batch job management
- Performance monitoring
"""

import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import traceback

# Celery imports
from celery import Celery, Task
from celery.result import AsyncResult
from celery.signals import task_prerun, task_postrun, task_failure, task_success
import redis

# Import our services
from .document_processor import DocumentProcessor, ProcessingResult
from .pdf_service import PDFProcessor
from .preprocessing import TextPreprocessor, PreprocessingResult
from .storage_service import StorageService, StorageConfig, StorageBackend

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job status enumeration"""
    PENDING = "pending"
    STARTED = "started"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILURE = "failure"
    REVOKED = "revoked"
    RETRY = "retry"


class JobType(Enum):
    """Job type enumeration"""
    DOCUMENT_PROCESSING = "document_processing"
    PDF_EXTRACTION = "pdf_extraction"
    TEXT_PREPROCESSING = "text_preprocessing"
    BATCH_UPLOAD = "batch_upload"
    BULK_ANALYSIS = "bulk_analysis"


@dataclass
class JobProgress:
    """Job progress tracking"""
    current: int = 0
    total: int = 0
    percentage: float = 0.0
    stage: str = "initializing"
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JobResult:
    """Job execution result"""
    job_id: str
    job_type: JobType
    status: JobStatus
    progress: JobProgress
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time: Optional[float] = None
    retries: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchJobConfig:
    """Configuration for batch jobs"""
    max_retries: int = 3
    retry_delay: int = 60  # seconds
    timeout: int = 3600    # 1 hour
    priority: int = 5      # 1-10, higher = more priority
    expires: int = 86400   # 24 hours
    store_results: bool = True
    notify_on_completion: bool = False
    callback_url: Optional[str] = None


class BatchProcessor:
    """Batch processing service using Celery"""
    
    def __init__(self, 
                 broker_url: str = 'redis://localhost:6379/0',
                 backend_url: str = 'redis://localhost:6379/0',
                 app_name: str = 'document_processor'):
        """
        Initialize batch processor
        
        Args:
            broker_url: Celery broker URL
            backend_url: Celery result backend URL  
            app_name: Application name
        """
        
        # Initialize Celery app
        self.celery_app = Celery(
            app_name,
            broker=broker_url,
            backend=backend_url
        )
        
        # Configure Celery
        self.celery_app.conf.update(
            task_serializer='json',
            accept_content=['json'],
            result_serializer='json',
            timezone='UTC',
            enable_utc=True,
            task_track_started=True,
            task_time_limit=3600,  # 1 hour hard limit
            task_soft_time_limit=3300,  # 55 minutes soft limit
            worker_prefetch_multiplier=1,
            result_expires=86400,  # 24 hours
            broker_transport_options={'master_name': 'mymaster'},
        )
        
        # Redis connection for additional features
        self.redis_client = redis.from_url(backend_url)
        
        # Service instances
        self.document_processor = DocumentProcessor()
        self.pdf_processor = PDFProcessor()
        self.text_preprocessor = TextPreprocessor()
        
        # Register tasks
        self._register_tasks()
    
    def _register_tasks(self):
        """Register Celery tasks"""
        
        @self.celery_app.task(bind=True, name='process_document')
        def process_document_task(self, file_path: str, filename: str, job_id: str, config: dict = None):
            """Process single document"""
            return self._execute_task(
                self, job_id, JobType.DOCUMENT_PROCESSING,
                self._process_document_impl, file_path, filename, config or {}
            )
        
        @self.celery_app.task(bind=True, name='extract_pdf_text')
        def extract_pdf_text_task(self, file_path: str, job_id: str, config: dict = None):
            """Extract text from PDF"""
            return self._execute_task(
                self, job_id, JobType.PDF_EXTRACTION,
                self._extract_pdf_text_impl, file_path, config or {}
            )
        
        @self.celery_app.task(bind=True, name='preprocess_text')
        def preprocess_text_task(self, text: str, job_id: str, config: dict = None):
            """Preprocess text"""
            return self._execute_task(
                self, job_id, JobType.TEXT_PREPROCESSING,
                self._preprocess_text_impl, text, config or {}
            )
        
        @self.celery_app.task(bind=True, name='batch_process_documents')
        def batch_process_documents_task(self, file_list: List[Dict], job_id: str, config: dict = None):
            """Process multiple documents in batch"""
            return self._execute_task(
                self, job_id, JobType.BATCH_UPLOAD,
                self._batch_process_documents_impl, file_list, config or {}
            )
        
        # Store task references
        self.tasks = {
            'process_document': process_document_task,
            'extract_pdf_text': extract_pdf_text_task,
            'preprocess_text': preprocess_text_task,
            'batch_process_documents': batch_process_documents_task,
        }
    
    def _execute_task(self, task_instance: Task, job_id: str, job_type: JobType, 
                     impl_func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Execute task with error handling and progress tracking"""
        
        start_time = time.time()
        
        try:
            # Update job status to started
            self._update_job_status(job_id, JobStatus.STARTED, 
                                  JobProgress(stage="starting", message="Task started"))
            
            # Execute implementation
            result = impl_func(task_instance, job_id, *args, **kwargs)
            
            # Update job status to success
            execution_time = time.time() - start_time
            self._update_job_status(job_id, JobStatus.SUCCESS,
                                  JobProgress(current=100, total=100, percentage=100.0,
                                            stage="completed", message="Task completed successfully"),
                                  result=result, execution_time=execution_time)
            
            return result
            
        except Exception as e:
            # Update job status to failure
            execution_time = time.time() - start_time
            error_msg = str(e)
            error_details = traceback.format_exc()
            
            logger.error(f"Task {job_id} failed: {error_msg}")
            logger.debug(f"Full traceback: {error_details}")
            
            self._update_job_status(job_id, JobStatus.FAILURE,
                                  JobProgress(stage="failed", message=f"Task failed: {error_msg}"),
                                  error=error_details, execution_time=execution_time)
            
            # Re-raise for Celery retry mechanism
            raise
    
    def _process_document_impl(self, task_instance: Task, job_id: str, 
                             file_path: str, filename: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation for document processing"""
        
        # Update progress
        self._update_progress(job_id, JobProgress(current=10, total=100, percentage=10.0,
                                                stage="processing", message="Processing document"))
        
        # Process document
        result = self.document_processor.process_document(file_path, filename)
        
        # Update progress
        self._update_progress(job_id, JobProgress(current=80, total=100, percentage=80.0,
                                                stage="finalizing", message="Finalizing results"))
        
        # Convert result to dictionary for JSON serialization
        result_dict = {
            'text': result.text,
            'word_count': result.metadata.word_count,
            'character_count': result.metadata.character_count,
            'file_type': result.metadata.file_type.value,
            'processing_time': result.processing_time,
            'extraction_method': result.extraction_method,
            'sections_count': len(result.structured_content),
            'tables_count': len(result.tables),
            'images_count': len(result.images),
            'errors': result.errors,
            'warnings': result.warnings
        }
        
        return result_dict
    
    def _extract_pdf_text_impl(self, task_instance: Task, job_id: str,
                              file_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation for PDF text extraction"""
        
        # Update progress
        self._update_progress(job_id, JobProgress(current=20, total=100, percentage=20.0,
                                                stage="extracting", message="Extracting PDF text"))
        
        # Extract text
        result = self.pdf_processor.extract_text(file_path)
        
        # Update progress
        self._update_progress(job_id, JobProgress(current=90, total=100, percentage=90.0,
                                                stage="finalizing", message="Processing complete"))
        
        # Convert result
        result_dict = {
            'text': result.text,
            'pages': len(result.pages),
            'processing_time': result.processing_time,
            'extraction_method': result.extraction_method,
            'metadata': {
                'title': result.metadata.title,
                'author': result.metadata.author,
                'pages': result.metadata.pages,
                'file_size': result.metadata.file_size
            },
            'errors': result.errors,
            'warnings': result.warnings
        }
        
        return result_dict
    
    def _preprocess_text_impl(self, task_instance: Task, job_id: str,
                             text: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation for text preprocessing"""
        
        # Update progress
        self._update_progress(job_id, JobProgress(current=30, total=100, percentage=30.0,
                                                stage="preprocessing", message="Preprocessing text"))
        
        # Preprocess text
        result = self.text_preprocessor.preprocess(text)
        
        # Update progress
        self._update_progress(job_id, JobProgress(current=90, total=100, percentage=90.0,
                                                stage="finalizing", message="Analysis complete"))
        
        # Convert result
        result_dict = {
            'cleaned_text': result.cleaned_text,
            'detected_language': result.detected_language,
            'content_type': result.content_type.value,
            'text_quality': result.text_quality.value,
            'statistics': {
                'word_count': result.statistics.word_count,
                'sentence_count': result.statistics.sentence_count,
                'paragraph_count': result.statistics.paragraph_count,
                'readability_score': result.statistics.readability_score,
                'lexical_diversity': result.statistics.lexical_diversity,
                'language_confidence': result.statistics.language_confidence
            },
            'sections_count': len(result.sections),
            'encoding_issues_fixed': result.encoding_issues_fixed,
            'warnings': result.warnings
        }
        
        return result_dict
    
    def _batch_process_documents_impl(self, task_instance: Task, job_id: str,
                                    file_list: List[Dict], config: Dict[str, Any]) -> Dict[str, Any]:
        """Implementation for batch document processing"""
        
        results = []
        total_files = len(file_list)
        
        for i, file_info in enumerate(file_list):
            # Update progress
            progress = JobProgress(
                current=i + 1,
                total=total_files,
                percentage=((i + 1) / total_files) * 100,
                stage="processing",
                message=f"Processing file {i + 1} of {total_files}: {file_info.get('filename', 'unknown')}"
            )
            self._update_progress(job_id, progress)
            
            try:
                # Process individual file
                result = self.document_processor.process_document(
                    file_info['file_path'], 
                    file_info['filename']
                )
                
                # Store result
                file_result = {
                    'filename': file_info['filename'],
                    'success': True,
                    'text_length': len(result.text),
                    'word_count': result.metadata.word_count,
                    'file_type': result.metadata.file_type.value,
                    'processing_time': result.processing_time,
                    'errors': result.errors,
                    'warnings': result.warnings
                }
                
                results.append(file_result)
                
            except Exception as e:
                logger.error(f"Failed to process file {file_info['filename']}: {e}")
                
                file_result = {
                    'filename': file_info['filename'],
                    'success': False,
                    'error': str(e)
                }
                
                results.append(file_result)
        
        # Final results
        successful = sum(1 for r in results if r.get('success', False))
        failed = total_files - successful
        
        batch_result = {
            'total_files': total_files,
            'successful': successful,
            'failed': failed,
            'success_rate': (successful / total_files) * 100 if total_files > 0 else 0,
            'results': results,
            'summary': {
                'total_words': sum(r.get('word_count', 0) for r in results if r.get('success')),
                'total_processing_time': sum(r.get('processing_time', 0) for r in results if r.get('success')),
                'file_types': list(set(r.get('file_type') for r in results if r.get('file_type')))
            }
        }
        
        return batch_result
    
    def submit_job(self, job_type: str, config: BatchJobConfig = None, **kwargs) -> str:
        """
        Submit job to processing queue
        
        Args:
            job_type: Type of job to run
            config: Job configuration
            **kwargs: Job-specific parameters
            
        Returns:
            Job ID for tracking
        """
        
        job_id = str(uuid.uuid4())
        config = config or BatchJobConfig()
        
        # Get task function
        task_func = self.tasks.get(job_type)
        if not task_func:
            raise ValueError(f"Unknown job type: {job_type}")
        
        # Submit task to Celery
        task_kwargs = dict(kwargs)
        task_kwargs['job_id'] = job_id
        task_kwargs['config'] = asdict(config)
        
        try:
            # Submit with configuration
            async_result = task_func.apply_async(
                kwargs=task_kwargs,
                retry=config.max_retries > 0,
                retry_policy={
                    'max_retries': config.max_retries,
                    'interval_start': config.retry_delay,
                    'interval_step': config.retry_delay,
                    'interval_max': config.retry_delay * 4,
                },
                expires=config.expires,
                priority=config.priority
            )
            
            # Store job metadata
            job_data = JobResult(
                job_id=job_id,
                job_type=JobType(job_type),
                status=JobStatus.PENDING,
                progress=JobProgress(stage="queued", message="Job queued for processing"),
                started_at=datetime.utcnow(),
                metadata={
                    'task_id': async_result.id,
                    'config': asdict(config),
                    'submitted_at': datetime.utcnow().isoformat()
                }
            )
            
            self._store_job_data(job_id, job_data)
            
            logger.info(f"Submitted job {job_id} of type {job_type}")
            
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to submit job {job_id}: {e}")
            raise
    
    def get_job_status(self, job_id: str) -> Optional[JobResult]:
        """Get job status and results"""
        
        return self._get_job_data(job_id)
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job"""
        
        try:
            job_data = self._get_job_data(job_id)
            if not job_data:
                return False
            
            # Revoke Celery task
            task_id = job_data.metadata.get('task_id')
            if task_id:
                self.celery_app.control.revoke(task_id, terminate=True)
            
            # Update job status
            self._update_job_status(job_id, JobStatus.REVOKED,
                                  JobProgress(stage="cancelled", message="Job cancelled by user"))
            
            logger.info(f"Cancelled job {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False
    
    def list_jobs(self, limit: int = 100, offset: int = 0, 
                  status_filter: Optional[JobStatus] = None) -> List[JobResult]:
        """List jobs with optional filtering"""
        
        try:
            # Get all job keys
            pattern = "job:*"
            keys = self.redis_client.keys(pattern)
            
            jobs = []
            for key in keys:
                try:
                    data = self.redis_client.get(key)
                    if data:
                        job_data = json.loads(data)
                        job_result = JobResult(**job_data)
                        
                        # Apply status filter
                        if status_filter and job_result.status != status_filter:
                            continue
                        
                        jobs.append(job_result)
                        
                except Exception as e:
                    logger.warning(f"Failed to deserialize job data for {key}: {e}")
            
            # Sort by creation time (newest first)
            jobs.sort(key=lambda x: x.started_at or datetime.min, reverse=True)
            
            # Apply pagination
            return jobs[offset:offset + limit]
            
        except Exception as e:
            logger.error(f"Failed to list jobs: {e}")
            return []
    
    def cleanup_completed_jobs(self, older_than_hours: int = 24) -> int:
        """Clean up old completed jobs"""
        
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
            
            pattern = "job:*"
            keys = self.redis_client.keys(pattern)
            
            deleted_count = 0
            for key in keys:
                try:
                    data = self.redis_client.get(key)
                    if data:
                        job_data = json.loads(data)
                        job_result = JobResult(**job_data)
                        
                        # Delete if completed and old enough
                        if (job_result.status in [JobStatus.SUCCESS, JobStatus.FAILURE] and
                            job_result.completed_at and job_result.completed_at < cutoff_time):
                            self.redis_client.delete(key)
                            deleted_count += 1
                            
                except Exception as e:
                    logger.warning(f"Failed to check job for cleanup {key}: {e}")
            
            logger.info(f"Cleaned up {deleted_count} old jobs")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup jobs: {e}")
            return 0
    
    def _update_job_status(self, job_id: str, status: JobStatus, progress: JobProgress,
                          result: Optional[Dict[str, Any]] = None, error: Optional[str] = None,
                          execution_time: Optional[float] = None):
        """Update job status in storage"""
        
        try:
            job_data = self._get_job_data(job_id)
            if not job_data:
                logger.warning(f"Job {job_id} not found for status update")
                return
            
            # Update fields
            job_data.status = status
            job_data.progress = progress
            
            if result is not None:
                job_data.result = result
            
            if error is not None:
                job_data.error = error
            
            if execution_time is not None:
                job_data.execution_time = execution_time
            
            if status in [JobStatus.SUCCESS, JobStatus.FAILURE]:
                job_data.completed_at = datetime.utcnow()
            
            self._store_job_data(job_id, job_data)
            
        except Exception as e:
            logger.error(f"Failed to update job status for {job_id}: {e}")
    
    def _update_progress(self, job_id: str, progress: JobProgress):
        """Update job progress"""
        
        try:
            job_data = self._get_job_data(job_id)
            if job_data:
                job_data.progress = progress
                self._store_job_data(job_id, job_data)
                
        except Exception as e:
            logger.error(f"Failed to update progress for {job_id}: {e}")
    
    def _store_job_data(self, job_id: str, job_data: JobResult):
        """Store job data in Redis"""
        
        try:
            key = f"job:{job_id}"
            
            # Convert to dict for JSON serialization
            data_dict = asdict(job_data)
            
            # Handle datetime serialization
            for field in ['started_at', 'completed_at']:
                if data_dict.get(field):
                    data_dict[field] = data_dict[field].isoformat()
            
            # Handle enum serialization
            data_dict['status'] = data_dict['status'].value
            data_dict['job_type'] = data_dict['job_type'].value
            
            self.redis_client.setex(key, 86400, json.dumps(data_dict))  # 24 hour expiry
            
        except Exception as e:
            logger.error(f"Failed to store job data for {job_id}: {e}")
    
    def _get_job_data(self, job_id: str) -> Optional[JobResult]:
        """Retrieve job data from Redis"""
        
        try:
            key = f"job:{job_id}"
            data = self.redis_client.get(key)
            
            if not data:
                return None
            
            data_dict = json.loads(data)
            
            # Handle datetime deserialization
            for field in ['started_at', 'completed_at']:
                if data_dict.get(field):
                    data_dict[field] = datetime.fromisoformat(data_dict[field])
            
            # Handle enum deserialization
            data_dict['status'] = JobStatus(data_dict['status'])
            data_dict['job_type'] = JobType(data_dict['job_type'])
            
            return JobResult(**data_dict)
            
        except Exception as e:
            logger.error(f"Failed to retrieve job data for {job_id}: {e}")
            return None


# Global instance (will be initialized in main app)
batch_processor: Optional[BatchProcessor] = None


def get_batch_processor() -> BatchProcessor:
    """Get global batch processor instance"""
    global batch_processor
    if not batch_processor:
        batch_processor = BatchProcessor()
    return batch_processor


# Convenience functions
def process_document_async(file_path: str, filename: str, config: BatchJobConfig = None) -> str:
    """Submit document processing job"""
    processor = get_batch_processor()
    return processor.submit_job('process_document', config, 
                              file_path=file_path, filename=filename)


def extract_pdf_text_async(file_path: str, config: BatchJobConfig = None) -> str:
    """Submit PDF text extraction job"""
    processor = get_batch_processor()
    return processor.submit_job('extract_pdf_text', config, file_path=file_path)


def preprocess_text_async(text: str, config: BatchJobConfig = None) -> str:
    """Submit text preprocessing job"""
    processor = get_batch_processor()
    return processor.submit_job('preprocess_text', config, text=text)


def batch_process_documents_async(file_list: List[Dict], config: BatchJobConfig = None) -> str:
    """Submit batch document processing job"""
    processor = get_batch_processor()
    return processor.submit_job('batch_process_documents', config, file_list=file_list)