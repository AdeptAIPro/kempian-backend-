"""
Async job queue structure for Jobvite sync operations.
Provides interface for background job processing.

Supports both:
- In-memory job tracking (default, for dev/staging)
- Celery-based async processing (for production, set USE_CELERY=true)

To enable Celery:
1. Install: pip install celery redis
2. Set USE_CELERY=true in environment
3. Uncomment Celery task definitions below
4. Start Celery workers
"""

import os
from typing import Dict, Any, Optional, Callable
from app.simple_logger import get_logger
from datetime import datetime
from app.models import JobviteSettings, db

logger = get_logger("jobvite_async_jobs")

# Job status tracking
_job_status: Dict[str, Dict[str, Any]] = {}

def _generate_job_id(tenant_id: int, job_type: str) -> str:
    """Generate unique job ID"""
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    return f"jobvite_{tenant_id}_{job_type}_{timestamp}"

def _update_job_status(job_id: str, status: str, progress: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
    """Update job status (in-memory for now, should use Redis/DB in production)"""
    _job_status[job_id] = {
        'status': status,  # 'pending', 'running', 'success', 'failed', 'partial'
        'progress': progress or {},
        'error': error,
        'updated_at': datetime.utcnow().isoformat()
    }

def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    """Get job status"""
    return _job_status.get(job_id)

def enqueue_sync_job(tenant_id: int, job_type: str, sync_function: Callable, *args, **kwargs) -> str:
    """
    Enqueue a sync job for async processing.
    
    Args:
        tenant_id: Tenant ID
        job_type: Type of sync ('jobs', 'candidates', 'onboarding', 'full')
        sync_function: Function to execute
        *args, **kwargs: Arguments for sync_function
    
    Returns:
        Job ID for tracking
    
    Note: In production, this should use Celery/RQ to queue the job.
    For now, it runs synchronously but returns a job ID for tracking.
    """
    job_id = _generate_job_id(tenant_id, job_type)
    
    # Update settings to indicate sync is running
    settings = JobviteSettings.query.filter_by(tenant_id=tenant_id, is_active=True).first()
    if settings:
        settings.last_sync_status = 'running'
        settings.last_error = None
        db.session.commit()
    
    _update_job_status(job_id, 'running', {'tenant_id': tenant_id, 'job_type': job_type})
    
    try:
        # Check if Celery is configured and enabled
        use_celery = os.getenv('USE_CELERY', 'false').lower() == 'true'
        
        if use_celery:
            # Use Celery for async processing
            try:
                from app.jobvite.celery_config import celery_app
                if job_type == 'jobs':
                    task = celery_app.send_task('jobvite.sync_jobs', args=[tenant_id])
                elif job_type == 'candidates':
                    task = celery_app.send_task('jobvite.sync_candidates', args=[tenant_id])
                elif job_type == 'onboarding':
                    task = celery_app.send_task('jobvite.sync_onboarding', args=[tenant_id])
                else:
                    # Fall back to synchronous for 'full' sync
                    use_celery = False
                
                if use_celery:
                    logger.info(f"Enqueued Celery task {task.id} for tenant {tenant_id}, type: {job_type}")
                    _update_job_status(job_id, 'pending', {'celery_task_id': task.id, 'tenant_id': tenant_id, 'job_type': job_type})
                    return job_id
            except ImportError:
                logger.warning("Celery not available, using synchronous execution")
                use_celery = False
            except Exception as e:
                logger.warning(f"Celery task enqueue failed: {e}, falling back to synchronous")
                use_celery = False
        
        # Synchronous execution (current default for dev/staging)
        logger.info(f"Starting sync job {job_id} for tenant {tenant_id}, type: {job_type}")
        
        # Execute sync function
        result = sync_function(*args, **kwargs)
        
        # Update status based on result
        if result.get('success'):
            synced_count = result.get('synced_count', 0)
            error_count = result.get('error_count', 0)
            
            if error_count == 0:
                status = 'success'
            elif synced_count > 0:
                status = 'partial'
            else:
                status = 'failed'
            
            _update_job_status(
                job_id,
                status,
                {
                    'synced_count': synced_count,
                    'error_count': error_count,
                    'total': synced_count + error_count
                },
                result.get('error')
            )
        else:
            _update_job_status(job_id, 'failed', error=result.get('error', 'Unknown error'))
        
        logger.info(f"Sync job {job_id} completed with status: {_job_status[job_id]['status']}")
        
        return job_id
        
    except Exception as e:
        logger.error(f"Sync job {job_id} failed: {e}", exc_info=True)
        _update_job_status(job_id, 'failed', error=str(e))
        
        # Update settings
        if settings:
            settings.last_sync_status = 'failed'
            settings.last_error = str(e)
            db.session.commit()
        
        return job_id

def cancel_job(job_id: str) -> bool:
    """
    Cancel a running job.
    
    Note: In production with Celery/RQ, this would revoke the task.
    For now, it just marks the job as cancelled.
    """
    if job_id in _job_status:
        _job_status[job_id]['status'] = 'cancelled'
        _job_status[job_id]['updated_at'] = datetime.utcnow().isoformat()
        return True
    return False

# Celery task definitions
# To enable Celery:
# 1. Install: pip install celery redis
# 2. Configure Redis: Set CELERY_BROKER_URL in .env
# 3. Uncomment the import and task definitions below
# 4. Start worker: celery -A app.jobvite.celery_config worker --loglevel=info
# 5. Start beat: celery -A app.jobvite.celery_config beat --loglevel=info

# Uncomment when Celery is configured:
"""
from app.jobvite.celery_config import celery_app
from app.jobvite.sync import sync_jobs_for_tenant, sync_candidates_for_tenant, sync_onboarding_for_tenant
from app.models import JobviteSettings, db
from app.simple_logger import get_logger

logger = get_logger("jobvite_celery_tasks")

@celery_app.task(bind=True, name='jobvite.sync_jobs')
def sync_jobs_task(self, tenant_id: int):
    \"\"\"Celery task to sync jobs for a tenant\"\"\"
    try:
        logger.info(f"Starting job sync task for tenant {tenant_id}")
        result = sync_jobs_for_tenant(tenant_id)
        logger.info(f"Job sync task completed for tenant {tenant_id}: {result}")
        return result
    except Exception as e:
        logger.error(f"Job sync task failed for tenant {tenant_id}: {e}")
        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=60, max_retries=3)

@celery_app.task(bind=True, name='jobvite.sync_candidates')
def sync_candidates_task(self, tenant_id: int):
    \"\"\"Celery task to sync candidates for a tenant\"\"\"
    try:
        logger.info(f"Starting candidate sync task for tenant {tenant_id}")
        result = sync_candidates_for_tenant(tenant_id)
        logger.info(f"Candidate sync task completed for tenant {tenant_id}: {result}")
        return result
    except Exception as e:
        logger.error(f"Candidate sync task failed for tenant {tenant_id}: {e}")
        raise self.retry(exc=e, countdown=60, max_retries=3)

@celery_app.task(bind=True, name='jobvite.sync_onboarding')
def sync_onboarding_task(self, tenant_id: int):
    \"\"\"Celery task to sync onboarding data for a tenant\"\"\"
    try:
        logger.info(f"Starting onboarding sync task for tenant {tenant_id}")
        result = sync_onboarding_for_tenant(tenant_id)
        logger.info(f"Onboarding sync task completed for tenant {tenant_id}: {result}")
        return result
    except Exception as e:
        logger.error(f"Onboarding sync task failed for tenant {tenant_id}: {e}")
        raise self.retry(exc=e, countdown=60, max_retries=3)

@celery_app.task(name='jobvite.sync_all_tenants')
def sync_all_tenants_task():
    \"\"\"Celery task to sync all active tenants (scheduled task)\"\"\"
    from app.jobvite.sync import run_sync_for_all_tenants
    try:
        logger.info("Starting sync for all tenants")
        run_sync_for_all_tenants()
        logger.info("Sync for all tenants completed")
    except Exception as e:
        logger.error(f"Sync for all tenants failed: {e}")
        raise
"""

# When Celery is NOT configured, use in-memory job tracking (current default)
# When Celery IS configured, uncomment above and update enqueue_sync_job() to use Celery

