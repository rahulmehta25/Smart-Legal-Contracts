"""
Prometheus metrics configuration for FastAPI
"""
import time
from typing import Callable
from functools import wraps
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    multiprocess,
    REGISTRY
)
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import os


# Check if running in multiprocess mode
def _get_registry():
    """Get appropriate registry for single/multi process mode"""
    if 'prometheus_multiproc_dir' in os.environ:
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return registry
    return REGISTRY


# Application info
APP_INFO = Info('slc_app', 'Smart Legal Contracts application info')
APP_INFO.info({
    'version': '1.0.0',
    'name': 'Smart Legal Contracts',
    'description': 'AI-powered legal document analysis'
})

# HTTP metrics
HTTP_REQUEST_TOTAL = Counter(
    'slc_http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

HTTP_REQUEST_DURATION = Histogram(
    'slc_http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

HTTP_REQUESTS_IN_PROGRESS = Gauge(
    'slc_http_requests_in_progress',
    'HTTP requests currently in progress',
    ['method', 'endpoint']
)

# Document processing metrics
DOCUMENTS_PROCESSED = Counter(
    'slc_documents_processed_total',
    'Total documents processed',
    ['status', 'document_type']
)

DOCUMENT_PROCESSING_DURATION = Histogram(
    'slc_document_processing_duration_seconds',
    'Document processing duration in seconds',
    ['document_type'],
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]
)

DOCUMENTS_IN_QUEUE = Gauge(
    'slc_documents_in_queue',
    'Number of documents currently in processing queue'
)

# AI/ML metrics
AI_INFERENCE_DURATION = Histogram(
    'slc_ai_inference_duration_seconds',
    'AI model inference duration in seconds',
    ['model', 'operation'],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]
)

AI_TOKENS_USED = Counter(
    'slc_ai_tokens_used_total',
    'Total AI tokens used',
    ['model', 'operation']
)

EMBEDDINGS_GENERATED = Counter(
    'slc_embeddings_generated_total',
    'Total embeddings generated',
    ['model']
)

# Vector DB metrics
VECTOR_SEARCH_DURATION = Histogram(
    'slc_vector_search_duration_seconds',
    'Vector similarity search duration in seconds',
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)

VECTOR_SEARCH_RESULTS = Histogram(
    'slc_vector_search_results_count',
    'Number of results returned from vector search',
    buckets=[1, 5, 10, 20, 50, 100]
)

# Cache metrics
CACHE_HITS = Counter(
    'slc_cache_hits_total',
    'Total cache hits',
    ['cache_type']
)

CACHE_MISSES = Counter(
    'slc_cache_misses_total',
    'Total cache misses',
    ['cache_type']
)

# Database metrics
DB_QUERY_DURATION = Histogram(
    'slc_db_query_duration_seconds',
    'Database query duration in seconds',
    ['operation'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
)

DB_CONNECTIONS_ACTIVE = Gauge(
    'slc_db_connections_active',
    'Active database connections'
)

# Celery task metrics
CELERY_TASKS_TOTAL = Counter(
    'slc_celery_tasks_total',
    'Total Celery tasks',
    ['task_name', 'status']
)

CELERY_TASK_DURATION = Histogram(
    'slc_celery_task_duration_seconds',
    'Celery task duration in seconds',
    ['task_name'],
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0]
)

CELERY_TASKS_IN_PROGRESS = Gauge(
    'slc_celery_tasks_in_progress',
    'Celery tasks currently in progress',
    ['task_name']
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware to collect HTTP metrics"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        method = request.method
        endpoint = request.url.path
        
        # Skip metrics endpoint to avoid recursion
        if endpoint == '/metrics':
            return await call_next(request)
        
        # Track in-progress requests
        HTTP_REQUESTS_IN_PROGRESS.labels(method=method, endpoint=endpoint).inc()
        
        start_time = time.time()
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            status_code = 500
            raise
        finally:
            # Record metrics
            duration = time.time() - start_time
            HTTP_REQUEST_TOTAL.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code
            ).inc()
            HTTP_REQUEST_DURATION.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            HTTP_REQUESTS_IN_PROGRESS.labels(method=method, endpoint=endpoint).dec()
        
        return response


def track_time(histogram: Histogram, **labels):
    """Decorator to track function execution time"""
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                histogram.labels(**labels).observe(time.time() - start_time)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                histogram.labels(**labels).observe(time.time() - start_time)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


def get_metrics() -> bytes:
    """Generate Prometheus metrics output"""
    registry = _get_registry()
    return generate_latest(registry)


def get_metrics_content_type() -> str:
    """Get the content type for Prometheus metrics"""
    return CONTENT_TYPE_LATEST


# Import asyncio for the decorator
import asyncio
