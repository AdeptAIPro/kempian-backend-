"""
Celery configuration for Jobvite async job processing.

This module configures Celery for background job processing.
Uncomment and configure when ready to use Celery in production.

Setup:
1. Install: pip install celery redis
2. Start Redis: redis-server
3. Start Celery worker: celery -A app.jobvite.celery_config worker --loglevel=info
4. Start Celery beat (for scheduled tasks): celery -A app.jobvite.celery_config beat --loglevel=info
"""

import os
from celery import Celery
from celery.schedules import crontab

# Get Redis URL from environment
REDIS_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')

# Create Celery app
celery_app = Celery(
    'jobvite',
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=['app.jobvite.async_jobs']
)

# Celery configuration
celery_app.conf.update(
    # Task settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Task execution
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes max per task
    task_soft_time_limit=25 * 60,  # 25 minutes soft limit
    
    # Retry settings
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_default_retry_delay=60,  # 1 minute
    task_max_retries=3,
    
    # Result backend
    result_expires=3600,  # Results expire after 1 hour
    
    # Worker settings
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50,  # Restart worker after 50 tasks
    
    # Scheduled tasks (Celery Beat)
    beat_schedule={
        'sync-all-tenants-hourly': {
            'task': 'app.jobvite.async_jobs.sync_all_tenants_task',
            'schedule': crontab(minute=0),  # Every hour
        },
        'sync-all-tenants-daily': {
            'task': 'app.jobvite.async_jobs.sync_all_tenants_task',
            'schedule': crontab(hour=2, minute=0),  # Daily at 2 AM
        },
    },
)

# Task routing
celery_app.conf.task_routes = {
    'app.jobvite.async_jobs.*': {'queue': 'jobvite'},
}

if __name__ == '__main__':
    celery_app.start()

