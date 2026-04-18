"""
Celery worker configuration for async document processing
"""
from .celery_app import celery_app
from .tasks import (
    process_document_task,
    generate_embeddings_task,
    analyze_arbitration_task,
    cleanup_expired_cache_task,
)

__all__ = [
    'celery_app',
    'process_document_task',
    'generate_embeddings_task',
    'analyze_arbitration_task',
    'cleanup_expired_cache_task',
]
