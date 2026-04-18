"""
Celery application configuration
"""
import os
from celery import Celery
from kombu import Exchange, Queue


# Get broker and backend URLs from environment
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/1')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/2')

# Create Celery app
celery_app = Celery(
    'slc_worker',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=['app.worker.tasks']
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Task execution settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_track_started=True,
    
    # Result settings
    result_expires=86400,  # 24 hours
    result_extended=True,
    
    # Worker settings
    worker_prefetch_multiplier=1,
    worker_concurrency=4,
    worker_max_tasks_per_child=1000,
    
    # Queue configuration
    task_queues=(
        Queue('default', Exchange('default'), routing_key='default'),
        Queue('documents', Exchange('documents'), routing_key='documents.#'),
        Queue('ai', Exchange('ai'), routing_key='ai.#'),
        Queue('maintenance', Exchange('maintenance'), routing_key='maintenance.#'),
    ),
    task_default_queue='default',
    task_default_exchange='default',
    task_default_routing_key='default',
    
    # Task routing
    task_routes={
        'app.worker.tasks.process_document_task': {'queue': 'documents'},
        'app.worker.tasks.generate_embeddings_task': {'queue': 'ai'},
        'app.worker.tasks.analyze_arbitration_task': {'queue': 'ai'},
        'app.worker.tasks.cleanup_expired_cache_task': {'queue': 'maintenance'},
    },
    
    # Beat schedule for periodic tasks
    beat_schedule={
        'cleanup-expired-cache': {
            'task': 'app.worker.tasks.cleanup_expired_cache_task',
            'schedule': 3600.0,  # Every hour
        },
    },
    
    # Retry settings
    task_annotations={
        '*': {
            'rate_limit': '100/m',
            'max_retries': 3,
            'default_retry_delay': 60,
        }
    },
)


# Optional: Configure for production
if os.getenv('ENVIRONMENT') == 'production':
    celery_app.conf.update(
        # More conservative settings for production
        worker_prefetch_multiplier=1,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        
        # Logging
        worker_hijack_root_logger=False,
        worker_log_format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        worker_task_log_format='%(asctime)s - %(name)s - %(levelname)s - %(task_name)s[%(task_id)s] - %(message)s',
    )
