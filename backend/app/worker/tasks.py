"""
Celery tasks for async document processing
"""
import time
import hashlib
from typing import Dict, Any, Optional, List
from celery import shared_task
from celery.utils.log import get_task_logger

from app.worker.celery_app import celery_app
from app.core.metrics import (
    CELERY_TASKS_TOTAL,
    CELERY_TASK_DURATION,
    CELERY_TASKS_IN_PROGRESS,
    DOCUMENTS_PROCESSED,
    DOCUMENT_PROCESSING_DURATION,
    AI_INFERENCE_DURATION,
    EMBEDDINGS_GENERATED,
)


logger = get_task_logger(__name__)


@celery_app.task(
    bind=True,
    name='app.worker.tasks.process_document_task',
    max_retries=3,
    default_retry_delay=60,
    soft_time_limit=300,
    time_limit=360,
)
def process_document_task(
    self,
    document_id: str,
    document_content: str,
    document_type: str = 'pdf',
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Process a document asynchronously.
    
    Args:
        document_id: Unique document identifier
        document_content: Raw document content or file path
        document_type: Type of document (pdf, txt, docx)
        options: Additional processing options
    
    Returns:
        Processing result with extracted data
    """
    task_name = 'process_document_task'
    CELERY_TASKS_IN_PROGRESS.labels(task_name=task_name).inc()
    start_time = time.time()
    
    try:
        logger.info(f"Processing document {document_id} of type {document_type}")
        
        # Simulate document processing (replace with actual logic)
        result = {
            'document_id': document_id,
            'status': 'processed',
            'chunks_created': 0,
            'processing_time': 0,
        }
        
        # Import actual processing logic
        try:
            from app.services.document_processor import DocumentProcessor
            processor = DocumentProcessor()
            result = processor.process(document_id, document_content, document_type, options or {})
        except ImportError:
            logger.warning("DocumentProcessor not available, using mock processing")
            # Mock processing for testing
            result['chunks_created'] = len(document_content) // 1000
        
        duration = time.time() - start_time
        result['processing_time'] = duration
        
        # Record metrics
        DOCUMENTS_PROCESSED.labels(status='success', document_type=document_type).inc()
        DOCUMENT_PROCESSING_DURATION.labels(document_type=document_type).observe(duration)
        CELERY_TASKS_TOTAL.labels(task_name=task_name, status='success').inc()
        
        logger.info(f"Document {document_id} processed successfully in {duration:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Error processing document {document_id}: {e}")
        DOCUMENTS_PROCESSED.labels(status='error', document_type=document_type).inc()
        CELERY_TASKS_TOTAL.labels(task_name=task_name, status='error').inc()
        
        # Retry on transient errors
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)
        raise
        
    finally:
        CELERY_TASKS_IN_PROGRESS.labels(task_name=task_name).dec()
        CELERY_TASK_DURATION.labels(task_name=task_name).observe(time.time() - start_time)


@celery_app.task(
    bind=True,
    name='app.worker.tasks.generate_embeddings_task',
    max_retries=3,
    default_retry_delay=30,
    soft_time_limit=120,
    time_limit=180,
)
def generate_embeddings_task(
    self,
    document_id: str,
    chunks: List[str],
    model: str = 'all-MiniLM-L6-v2'
) -> Dict[str, Any]:
    """
    Generate embeddings for document chunks.
    
    Args:
        document_id: Document identifier
        chunks: List of text chunks to embed
        model: Embedding model to use
    
    Returns:
        Embedding generation result
    """
    task_name = 'generate_embeddings_task'
    CELERY_TASKS_IN_PROGRESS.labels(task_name=task_name).inc()
    start_time = time.time()
    
    try:
        logger.info(f"Generating embeddings for {len(chunks)} chunks from document {document_id}")
        
        result = {
            'document_id': document_id,
            'embeddings_count': 0,
            'model': model,
            'processing_time': 0,
        }
        
        # Import actual embedding logic
        try:
            from app.ai.embeddings import EmbeddingGenerator
            generator = EmbeddingGenerator(model=model)
            embeddings = generator.generate(chunks)
            result['embeddings_count'] = len(embeddings)
        except ImportError:
            logger.warning("EmbeddingGenerator not available, using mock")
            result['embeddings_count'] = len(chunks)
        
        duration = time.time() - start_time
        result['processing_time'] = duration
        
        # Record metrics
        EMBEDDINGS_GENERATED.labels(model=model).inc(len(chunks))
        AI_INFERENCE_DURATION.labels(model=model, operation='embedding').observe(duration)
        CELERY_TASKS_TOTAL.labels(task_name=task_name, status='success').inc()
        
        logger.info(f"Generated {result['embeddings_count']} embeddings in {duration:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Error generating embeddings for {document_id}: {e}")
        CELERY_TASKS_TOTAL.labels(task_name=task_name, status='error').inc()
        
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)
        raise
        
    finally:
        CELERY_TASKS_IN_PROGRESS.labels(task_name=task_name).dec()
        CELERY_TASK_DURATION.labels(task_name=task_name).observe(time.time() - start_time)


@celery_app.task(
    bind=True,
    name='app.worker.tasks.analyze_arbitration_task',
    max_retries=3,
    default_retry_delay=60,
    soft_time_limit=600,
    time_limit=660,
)
def analyze_arbitration_task(
    self,
    document_id: str,
    content: str,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Analyze document for arbitration clauses.
    
    Args:
        document_id: Document identifier
        content: Document content to analyze
        options: Analysis options
    
    Returns:
        Analysis results with detected clauses
    """
    task_name = 'analyze_arbitration_task'
    CELERY_TASKS_IN_PROGRESS.labels(task_name=task_name).inc()
    start_time = time.time()
    
    try:
        logger.info(f"Analyzing document {document_id} for arbitration clauses")
        
        result = {
            'document_id': document_id,
            'clauses_found': [],
            'confidence_scores': [],
            'analysis_time': 0,
        }
        
        # Import actual analysis logic
        try:
            from app.services.arbitration_analyzer import ArbitrationAnalyzer
            analyzer = ArbitrationAnalyzer()
            analysis = analyzer.analyze(content, options or {})
            result['clauses_found'] = analysis.get('clauses', [])
            result['confidence_scores'] = analysis.get('scores', [])
        except ImportError:
            logger.warning("ArbitrationAnalyzer not available, using mock")
            result['clauses_found'] = []
        
        duration = time.time() - start_time
        result['analysis_time'] = duration
        
        # Record metrics
        AI_INFERENCE_DURATION.labels(model='arbitration', operation='analysis').observe(duration)
        CELERY_TASKS_TOTAL.labels(task_name=task_name, status='success').inc()
        
        logger.info(f"Analysis complete: found {len(result['clauses_found'])} clauses in {duration:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing document {document_id}: {e}")
        CELERY_TASKS_TOTAL.labels(task_name=task_name, status='error').inc()
        
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)
        raise
        
    finally:
        CELERY_TASKS_IN_PROGRESS.labels(task_name=task_name).dec()
        CELERY_TASK_DURATION.labels(task_name=task_name).observe(time.time() - start_time)


@celery_app.task(
    name='app.worker.tasks.cleanup_expired_cache_task',
    soft_time_limit=300,
    time_limit=360,
)
def cleanup_expired_cache_task() -> Dict[str, Any]:
    """
    Clean up expired cache entries (periodic task).
    
    Returns:
        Cleanup statistics
    """
    task_name = 'cleanup_expired_cache_task'
    start_time = time.time()
    
    try:
        logger.info("Starting cache cleanup")
        
        result = {
            'entries_cleaned': 0,
            'bytes_freed': 0,
            'cleanup_time': 0,
        }
        
        # Import actual cache cleanup logic
        try:
            from app.cache.redis_cache import RedisCache
            cache = RedisCache()
            stats = cache.cleanup_expired()
            result['entries_cleaned'] = stats.get('count', 0)
            result['bytes_freed'] = stats.get('bytes', 0)
        except ImportError:
            logger.warning("RedisCache not available, skipping cleanup")
        
        duration = time.time() - start_time
        result['cleanup_time'] = duration
        
        CELERY_TASKS_TOTAL.labels(task_name=task_name, status='success').inc()
        logger.info(f"Cache cleanup complete: {result['entries_cleaned']} entries in {duration:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Error during cache cleanup: {e}")
        CELERY_TASKS_TOTAL.labels(task_name=task_name, status='error').inc()
        raise
        
    finally:
        CELERY_TASK_DURATION.labels(task_name=task_name).observe(time.time() - start_time)
