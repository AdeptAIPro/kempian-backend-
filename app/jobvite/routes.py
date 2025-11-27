"""
Flask routes for Jobvite integration.
"""

from flask import Blueprint, request, jsonify, g
from app.models import (
    JobviteSettings,
    JobviteJob,
    JobviteCandidate,
    JobviteOnboardingProcess,
    JobviteOnboardingTask,
    db,
)
from app.utils import get_current_user
from app.jobvite.client_v2 import JobviteV2Client
from app.jobvite.client_onboarding import JobviteOnboardingClient
from app.jobvite.crypto import encrypt_at_rest, generate_rsa_key_pair
from app.jobvite.utils import normalize_environment, denormalize_environment, get_base_urls
from app.simple_logger import get_logger
from datetime import datetime
from sqlalchemy import or_
from typing import Tuple, Optional

logger = get_logger("jobvite_routes")

jobvite_bp = Blueprint('jobvite', __name__)

def get_jobvite_settings(tenant_id: int, environment: str = None) -> JobviteSettings:
    """Get Jobvite settings for tenant"""
    query = JobviteSettings.query.filter_by(tenant_id=tenant_id, is_active=True)
    if environment:
        # Normalize environment before query
        env_normalized = normalize_environment(environment)
        query = query.filter_by(environment=env_normalized)
    return query.first()

def _require_user_and_tenant() -> Tuple[Optional["User"], Optional[int], Optional[Tuple[dict, int]]]:
    """
    Ensure the request has an authenticated user and tenant context.
    Returns (user, tenant_id, error_response)
    """
    user_jwt = get_current_user()
    if not user_jwt:
        return None, None, ({"error": "Unauthorized"}, 401)
    
    from app.models import User
    user = User.query.filter_by(email=user_jwt.get('email')).first()
    if not user:
        return None, None, ({"error": "User not found"}, 404)
    
    tenant_id = g.tenant_id or user.tenant_id
    if not tenant_id:
        return user, None, ({"error": "Tenant not found"}, 404)
    
    return user, tenant_id, None

def _serialize_job(job: JobviteJob) -> dict:
    candidate_count = len(job.candidates) if job.candidates else 0
    return {
        'id': job.id,
        'jobviteJobId': job.jobvite_job_id,
        'requisitionId': job.requisition_id,
        'title': job.title,
        'status': job.status,
        'department': job.department,
        'category': job.category,
        'primaryRecruiterEmail': job.primary_recruiter_email,
        'primaryHiringManagerEmail': job.primary_hiring_manager_email,
        'locationMain': job.location_main,
        'region': job.region,
        'subsidiary': job.subsidiary,
        'salaryCurrency': job.salary_currency,
        'salaryMin': float(job.salary_min) if job.salary_min is not None else None,
        'salaryMax': float(job.salary_max) if job.salary_max is not None else None,
        'salaryFrequency': job.salary_frequency,
        'remoteType': job.remote_type,
        'candidateCount': candidate_count,
        'createdAt': job.created_at.isoformat() if job.created_at else None,
        'updatedAt': job.updated_at.isoformat() if job.updated_at else None,
        'rawJson': job.raw_json or {}
    }

def _serialize_candidate(candidate: JobviteCandidate) -> dict:
    return {
        'id': candidate.id,
        'jobviteCandidateId': candidate.jobvite_candidate_id,
        'jobviteApplicationId': candidate.jobvite_application_id,
        'jobviteJobId': candidate.jobvite_job_id,
        'jobId': candidate.job_id,
        'email': candidate.email,
        'firstName': candidate.first_name,
        'lastName': candidate.last_name,
        'workflowState': candidate.workflow_state,
        'personalDataProcessingStatus': candidate.personal_data_processing_status,
        'createdAt': candidate.created_at.isoformat() if candidate.created_at else None,
        'updatedAt': candidate.updated_at.isoformat() if candidate.updated_at else None,
        'rawJson': candidate.raw_json or {}
    }

def _serialize_onboarding_process(process: JobviteOnboardingProcess) -> dict:
    return {
        'id': process.id,
        'jobviteProcessId': process.jobvite_process_id,
        'jobviteNewHireId': process.jobvite_new_hire_id,
        'jobviteCandidateId': process.jobvite_candidate_id,
        'status': process.status,
        'hireDate': process.hire_date.isoformat() if process.hire_date else None,
        'kickoffDate': process.kickoff_date.isoformat() if process.kickoff_date else None,
        'milestoneStatus': process.milestone_status_json or {},
        'tasksSummary': process.tasks_summary_json or {},
        'createdAt': process.created_at.isoformat() if process.created_at else None,
        'updatedAt': process.updated_at.isoformat() if process.updated_at else None,
        'rawJson': process.raw_json or {},
    }

def _serialize_onboarding_task(task: JobviteOnboardingTask) -> dict:
    return {
        'id': task.id,
        'jobviteTaskId': task.jobvite_task_id,
        'name': task.name,
        'type': task.type,
        'status': task.status,
        'dueDate': task.due_date.isoformat() if task.due_date else None,
        'createdAt': task.created_at.isoformat() if task.created_at else None,
        'updatedAt': task.updated_at.isoformat() if task.updated_at else None,
        'rawJson': task.raw_json or {},
    }

# Configuration Endpoints
@jobvite_bp.route('/api/integrations/jobvite/config', methods=['POST'])
def save_config():
    """Save or update Jobvite configuration"""
    user_jwt = get_current_user()
    if not user_jwt or not user_jwt.get('email'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    from app.models import User
    user = User.query.filter_by(email=user_jwt['email']).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    tenant_id = g.tenant_id or user.tenant_id
    if not tenant_id:
        return jsonify({'error': 'Tenant not found'}), 404
    
    data = request.get_json()
    
    # Validate and normalize environment
    try:
        environment_normalized = normalize_environment(data.get('environment', 'Production'))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    
    # Validate required fields
    if not all([data.get('apiKey'), data.get('apiSecret'), data.get('companyId')]):
        return jsonify({'error': 'Missing required fields: apiKey, apiSecret, companyId'}), 400
    
    # Encrypt secrets
    api_secret_encrypted = encrypt_at_rest(data['apiSecret'])
    webhook_key_encrypted = None
    if data.get('webhookSigningKey'):
        webhook_key_encrypted = encrypt_at_rest(data['webhookSigningKey'])
    
    # Handle RSA keys
    our_public_key = data.get('ourPublicRsaKey')
    our_private_key_encrypted = None
    jobvite_public_key = data.get('jobvitePublicRsaKey')
    
    # If onboarding enabled but keys not provided, generate our key pair
    if data.get('syncConfig', {}).get('syncOnboarding') and not our_public_key:
        our_private_key_pem, our_public_key_pem = generate_rsa_key_pair()
        our_public_key = our_public_key_pem
        our_private_key_encrypted = encrypt_at_rest(our_private_key_pem)
    
    # Handle Service Account (for Onboarding API)
    service_account_username = data.get('serviceAccountUsername')
    service_account_password_encrypted = None
    if data.get('serviceAccountPassword'):
        service_account_password_encrypted = encrypt_at_rest(data['serviceAccountPassword'])
    
    # Save or update
    settings = get_jobvite_settings(tenant_id, environment_normalized)
    if settings:
        settings.api_key = data['apiKey']
        settings.api_secret_encrypted = api_secret_encrypted
        settings.company_id = data['companyId']
        if webhook_key_encrypted:
            settings.webhook_signing_key_encrypted = webhook_key_encrypted
        if our_public_key:
            settings.our_public_rsa_key = our_public_key
        if our_private_key_encrypted:
            settings.our_private_rsa_key_encrypted = our_private_key_encrypted
        if jobvite_public_key:
            settings.jobvite_public_rsa_key = jobvite_public_key
        if service_account_username:
            settings.service_account_username = service_account_username
        if service_account_password_encrypted:
            settings.service_account_password_encrypted = service_account_password_encrypted
        settings.sync_config = data.get('syncConfig', {})
        settings.updated_at = datetime.utcnow()
    else:
        settings = JobviteSettings(
            tenant_id=tenant_id,
            user_id=user.id,
            environment=environment_normalized,  # Store normalized value
            api_key=data['apiKey'],
            api_secret_encrypted=api_secret_encrypted,
            company_id=data['companyId'],
            webhook_signing_key_encrypted=webhook_key_encrypted,
            our_public_rsa_key=our_public_key,
            our_private_rsa_key_encrypted=our_private_key_encrypted,
            jobvite_public_rsa_key=jobvite_public_key,
            service_account_username=service_account_username,
            service_account_password_encrypted=service_account_password_encrypted,
            sync_config=data.get('syncConfig', {})
        )
        db.session.add(settings)
    
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error saving Jobvite config: {e}")
        return jsonify({'error': f'Failed to save configuration: {str(e)}'}), 500
    
    return jsonify({
        'success': True,
        'message': 'Configuration saved',
        'configId': settings.id
    }), 200

@jobvite_bp.route('/api/integrations/jobvite/config', methods=['GET'])
def get_config():
    """Get current Jobvite configuration"""
    user_jwt = get_current_user()
    if not user_jwt:
        return jsonify({'error': 'Unauthorized'}), 401
    
    from app.models import User
    user = User.query.filter_by(email=user_jwt['email']).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    tenant_id = g.tenant_id or user.tenant_id
    settings = get_jobvite_settings(tenant_id)
    
    if not settings:
        return jsonify({'connected': False}), 200
    
    # Return human-readable environment
    return jsonify({
        'connected': True,
        'environment': denormalize_environment(settings.environment),  # "Stage" or "Production"
        'companyId': settings.company_id,
        'syncConfig': settings.sync_config,
        'lastSyncAt': settings.last_sync_at.isoformat() if settings.last_sync_at else None,
        'lastSyncStatus': settings.last_sync_status,
        'lastError': settings.last_error
    }), 200

@jobvite_bp.route('/api/integrations/jobvite/test-connection', methods=['POST'])
def test_connection():
    """Test Jobvite connection"""
    user_jwt = get_current_user()
    if not user_jwt:
        return jsonify({'error': 'Unauthorized'}), 401
    
    data = request.get_json()
    
    # Validate and normalize environment
    try:
        environment_normalized = normalize_environment(data.get('environment', 'Production'))
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    
    # Validate required fields
    if not all([data.get('apiKey'), data.get('apiSecret'), data.get('companyId')]):
        return jsonify({'error': 'Missing required fields: apiKey, apiSecret, companyId'}), 400
    
    base_urls = get_base_urls(environment_normalized)
    
    try:
        # Test API v2 connection
        client = JobviteV2Client(
            api_key=data['apiKey'],
            api_secret=data['apiSecret'],
            company_id=data['companyId'],
            base_url=base_urls['v2']
        )
        
        # Try to fetch jobs (minimal filter)
        result = client.get_job(filters={'count': 1})
        
        return jsonify({
            'success': True,
            'message': 'Connection successful'
        }), 200
    except Exception as e:
        logger.error(f"Jobvite connection test failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

# Sync Endpoints (trigger background jobs)
@jobvite_bp.route('/api/integrations/jobvite/sync/jobs', methods=['POST'])
def sync_jobs():
    """Trigger job sync"""
    user_jwt = get_current_user()
    if not user_jwt:
        return jsonify({'error': 'Unauthorized'}), 401
    
    from app.models import User
    user = User.query.filter_by(email=user_jwt['email']).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    tenant_id = g.tenant_id or user.tenant_id
    if not tenant_id:
        return jsonify({'error': 'Tenant not found'}), 404
    
    # Trigger background sync job
    from app.jobvite.async_jobs import enqueue_sync_job
    from app.jobvite.sync import sync_jobs_for_tenant
    
    try:
        # Enqueue job (runs async if queue configured, otherwise synchronous)
        job_id = enqueue_sync_job(tenant_id, 'jobs', sync_jobs_for_tenant, tenant_id)
        
        return jsonify({
            'success': True,
            'message': 'Sync job queued',
            'jobId': job_id,
            'status': 'running'
        }), 202  # 202 Accepted - job is processing
    except Exception as e:
        logger.error(f"Error triggering job sync: {e}")
        return jsonify({'error': str(e)}), 500

@jobvite_bp.route('/api/integrations/jobvite/sync/candidates', methods=['POST'])
def sync_candidates():
    """Trigger candidate sync"""
    user_jwt = get_current_user()
    if not user_jwt:
        return jsonify({'error': 'Unauthorized'}), 401
    
    from app.models import User
    user = User.query.filter_by(email=user_jwt['email']).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    tenant_id = g.tenant_id or user.tenant_id
    if not tenant_id:
        return jsonify({'error': 'Tenant not found'}), 404
    
    from app.jobvite.async_jobs import enqueue_sync_job
    from app.jobvite.sync import sync_candidates_for_tenant
    
    try:
        job_id = enqueue_sync_job(tenant_id, 'candidates', sync_candidates_for_tenant, tenant_id)
        
        return jsonify({
            'success': True,
            'message': 'Sync job queued',
            'jobId': job_id,
            'status': 'running'
        }), 202
    except Exception as e:
        logger.error(f"Error triggering candidate sync: {e}")
        return jsonify({'error': str(e)}), 500

@jobvite_bp.route('/api/integrations/jobvite/sync/onboarding', methods=['POST'])
def sync_onboarding():
    """Trigger onboarding sync"""
    user_jwt = get_current_user()
    if not user_jwt:
        return jsonify({'error': 'Unauthorized'}), 401
    
    from app.models import User
    user = User.query.filter_by(email=user_jwt['email']).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    tenant_id = g.tenant_id or user.tenant_id
    if not tenant_id:
        return jsonify({'error': 'Tenant not found'}), 404
    
    from app.jobvite.async_jobs import enqueue_sync_job
    from app.jobvite.sync import sync_onboarding_for_tenant
    
    try:
        job_id = enqueue_sync_job(tenant_id, 'onboarding', sync_onboarding_for_tenant, tenant_id)
        
        return jsonify({
            'success': True,
            'message': 'Sync job queued',
            'jobId': job_id,
            'status': 'running'
        }), 202
    except Exception as e:
        logger.error(f"Error triggering onboarding sync: {e}")
        return jsonify({'error': str(e)}), 500

@jobvite_bp.route('/api/integrations/jobvite/sync/full', methods=['POST'])
def sync_full():
    """Trigger full sync (jobs + candidates + onboarding)"""
    user_jwt = get_current_user()
    if not user_jwt:
        return jsonify({'error': 'Unauthorized'}), 401
    
    from app.models import User
    user = User.query.filter_by(email=user_jwt['email']).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    tenant_id = g.tenant_id or user.tenant_id
    if not tenant_id:
        return jsonify({'error': 'Tenant not found'}), 404
    
    from app.jobvite.async_jobs import enqueue_sync_job
    from app.jobvite.sync import sync_jobs_for_tenant, sync_candidates_for_tenant, sync_onboarding_for_tenant
    
    try:
        # Enqueue all sync jobs
        job_ids = {
            'jobs': enqueue_sync_job(tenant_id, 'jobs', sync_jobs_for_tenant, tenant_id),
            'candidates': enqueue_sync_job(tenant_id, 'candidates', sync_candidates_for_tenant, tenant_id),
            'onboarding': enqueue_sync_job(tenant_id, 'onboarding', sync_onboarding_for_tenant, tenant_id)
        }
        
        return jsonify({
            'success': True,
            'message': 'Full sync jobs queued',
            'jobIds': job_ids,
            'status': 'running'
        }), 202
    except Exception as e:
        logger.error(f"Error triggering full sync: {e}")
        return jsonify({'error': str(e)}), 500

# Job status endpoint
@jobvite_bp.route('/api/integrations/jobvite/jobs/<job_id>/status', methods=['GET'])
def get_job_status(job_id: str):
    """Get status of a sync job"""
    user_jwt = get_current_user()
    if not user_jwt:
        return jsonify({'error': 'Unauthorized'}), 401
    
    from app.jobvite.async_jobs import get_job_status
    status = get_job_status(job_id)
    
    if not status:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(status), 200

# Webhook logs endpoint
@jobvite_bp.route('/api/integrations/jobvite/webhooks', methods=['GET'])
def get_webhook_logs():
    """Get webhook logs for the tenant"""
    user_jwt = get_current_user()
    if not user_jwt:
        return jsonify({'error': 'Unauthorized'}), 401
    
    from app.models import User, JobviteWebhookLog
    user = User.query.filter_by(email=user_jwt['email']).first()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    tenant_id = g.tenant_id or user.tenant_id
    if not tenant_id:
        return jsonify({'error': 'Tenant not found'}), 404
    
    # Get query parameters
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('pageSize', 25))
    event_type = request.args.get('eventType')
    signature_valid = request.args.get('signatureValid')
    processed = request.args.get('processed')
    date_from = request.args.get('dateFrom')
    date_to = request.args.get('dateTo')
    
    # Build query
    query = JobviteWebhookLog.query.filter_by(tenant_id=tenant_id)
    
    if event_type:
        query = query.filter_by(event_type=event_type)
    if signature_valid is not None:
        query = query.filter_by(signature_valid=signature_valid.lower() == 'true')
    if processed is not None:
        query = query.filter_by(processed=processed.lower() == 'true')
    if date_from:
        query = query.filter(JobviteWebhookLog.created_at >= datetime.fromisoformat(date_from))
    if date_to:
        query = query.filter(JobviteWebhookLog.created_at <= datetime.fromisoformat(date_to))
    
    # Pagination
    total = query.count()
    webhooks = query.order_by(JobviteWebhookLog.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()
    
    return jsonify({
        'webhooks': [{
            'id': w.id,
            'timestamp': w.created_at.isoformat() if w.created_at else None,
            'eventType': w.event_type,
            'source': w.source,
            'jobviteEntityId': w.jobvite_entity_id,
            'signatureValid': w.signature_valid,
            'processed': w.processed,
            'errorMessage': w.error_message
        } for w in webhooks],
        'pagination': {
            'page': page,
            'pageSize': page_size,
            'total': total,
            'totalPages': (total + page_size - 1) // page_size
        }
    }), 200

@jobvite_bp.route('/api/integrations/jobvite/jobs', methods=['GET'])
def list_jobvite_jobs():
    """List Jobvite jobs for the current tenant"""
    user, tenant_id, error = _require_user_and_tenant()
    if error:
        return jsonify(error[0]), error[1]
    
    page = max(1, int(request.args.get('page', 1)))
    page_size = min(100, max(1, int(request.args.get('pageSize', 25))))
    status = request.args.get('status')
    location = request.args.get('location')
    department = request.args.get('department')
    recruiter = request.args.get('recruiter')
    search = request.args.get('search')
    
    query = JobviteJob.query.filter_by(tenant_id=tenant_id)
    
    if status:
        query = query.filter(JobviteJob.status.ilike(status))
    if location:
        query = query.filter(JobviteJob.location_main.ilike(f"%{location}%"))
    if department:
        query = query.filter(JobviteJob.department.ilike(f"%{department}%"))
    if recruiter:
        query = query.filter(JobviteJob.primary_recruiter_email.ilike(f"%{recruiter}%"))
    if search:
        like = f"%{search}%"
        query = query.filter(
            or_(
                JobviteJob.title.ilike(like),
                JobviteJob.requisition_id.ilike(like),
                JobviteJob.jobvite_job_id.ilike(like)
            )
        )
    
    total = query.count()
    items = (
        query
        .order_by(JobviteJob.updated_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    
    return jsonify({
        'items': [_serialize_job(job) for job in items],
        'pagination': {
            'page': page,
            'pageSize': page_size,
            'total': total,
            'totalPages': (total + page_size - 1) // page_size
        }
    }), 200

@jobvite_bp.route('/api/integrations/jobvite/jobs/<int:job_id>', methods=['GET'])
def get_jobvite_job(job_id: int):
    """Get Jobvite job by ID"""
    user, tenant_id, error = _require_user_and_tenant()
    if error:
        return jsonify(error[0]), error[1]
    
    job = JobviteJob.query.filter_by(id=job_id, tenant_id=tenant_id).first()
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(_serialize_job(job)), 200

@jobvite_bp.route('/api/integrations/jobvite/jobs/<int:job_id>/candidates', methods=['GET'])
def get_jobvite_job_candidates(job_id: int):
    """Get candidates linked to a specific Jobvite job"""
    user, tenant_id, error = _require_user_and_tenant()
    if error:
        return jsonify(error[0]), error[1]
    
    job = JobviteJob.query.filter_by(id=job_id, tenant_id=tenant_id).first()
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    candidates = JobviteCandidate.query.filter_by(job_id=job.id, tenant_id=tenant_id).all()
    return jsonify({
        'items': [_serialize_candidate(candidate) for candidate in candidates],
        'pagination': {
            'page': 1,
            'pageSize': len(candidates),
            'total': len(candidates),
            'totalPages': 1
        }
    }), 200

@jobvite_bp.route('/api/integrations/jobvite/candidates', methods=['GET'])
def list_jobvite_candidates():
    """List Jobvite candidates for the current tenant"""
    user, tenant_id, error = _require_user_and_tenant()
    if error:
        return jsonify(error[0]), error[1]
    
    page = max(1, int(request.args.get('page', 1)))
    page_size = min(100, max(1, int(request.args.get('pageSize', 25))))
    workflow_state = request.args.get('workflowState')
    email = request.args.get('email')
    job_id = request.args.get('jobId')
    search = request.args.get('search')
    
    query = JobviteCandidate.query.filter_by(tenant_id=tenant_id)
    
    if workflow_state:
        query = query.filter(JobviteCandidate.workflow_state.ilike(workflow_state))
    if email:
        query = query.filter(JobviteCandidate.email.ilike(f"%{email}%"))
    if job_id:
        try:
            job_id_int = int(job_id)
            query = query.filter(JobviteCandidate.job_id == job_id_int)
        except ValueError:
            return jsonify({'error': 'Invalid job ID'}), 400
    if search:
        like = f"%{search}%"
        query = query.filter(
            or_(
                JobviteCandidate.first_name.ilike(like),
                JobviteCandidate.last_name.ilike(like),
                JobviteCandidate.email.ilike(like),
                JobviteCandidate.jobvite_candidate_id.ilike(like)
            )
        )
    
    total = query.count()
    items = (
        query
        .order_by(JobviteCandidate.updated_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    
    return jsonify({
        'items': [_serialize_candidate(candidate) for candidate in items],
        'pagination': {
            'page': page,
            'pageSize': page_size,
            'total': total,
            'totalPages': (total + page_size - 1) // page_size
        }
    }), 200

@jobvite_bp.route('/api/integrations/jobvite/candidates/<int:candidate_id>', methods=['GET'])
def get_jobvite_candidate(candidate_id: int):
    """Get Jobvite candidate by ID"""
    user, tenant_id, error = _require_user_and_tenant()
    if error:
        return jsonify(error[0]), error[1]
    
    candidate = JobviteCandidate.query.filter_by(id=candidate_id, tenant_id=tenant_id).first()
    if not candidate:
        return jsonify({'error': 'Candidate not found'}), 404
    
    return jsonify(_serialize_candidate(candidate)), 200

@jobvite_bp.route('/api/onboarding/processes', methods=['GET'])
def list_onboarding_processes():
    """List Jobvite onboarding processes for the current tenant"""
    user, tenant_id, error = _require_user_and_tenant()
    if error:
        return jsonify(error[0]), error[1]
    
    page = max(1, int(request.args.get('page', 1)))
    page_size = min(100, max(1, int(request.args.get('pageSize', 25))))
    status = request.args.get('status')
    start_date_param = request.args.get('startDate')
    end_date_param = request.args.get('endDate')
    
    query = JobviteOnboardingProcess.query.filter_by(tenant_id=tenant_id)
    
    if status:
        query = query.filter(JobviteOnboardingProcess.status.ilike(status))
    
    if start_date_param:
        try:
            start_date = datetime.strptime(start_date_param, '%Y-%m-%d').date()
        except ValueError:
            return jsonify({'error': 'Invalid startDate. Expected YYYY-MM-DD.'}), 400
        query = query.filter(
            or_(
                JobviteOnboardingProcess.kickoff_date >= start_date,
                JobviteOnboardingProcess.hire_date >= start_date,
            )
        )
    
    if end_date_param:
        try:
            end_date = datetime.strptime(end_date_param, '%Y-%m-%d').date()
        except ValueError:
            return jsonify({'error': 'Invalid endDate. Expected YYYY-MM-DD.'}), 400
        query = query.filter(
            or_(
                JobviteOnboardingProcess.kickoff_date <= end_date,
                JobviteOnboardingProcess.hire_date <= end_date,
            )
        )
    
    total = query.count()
    items = (
        query
        .order_by(JobviteOnboardingProcess.updated_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    
    return jsonify({
        'items': [_serialize_onboarding_process(process) for process in items],
        'pagination': {
            'page': page,
            'pageSize': page_size,
            'total': total,
            'totalPages': (total + page_size - 1) // page_size
        }
    }), 200

@jobvite_bp.route('/api/onboarding/processes/<int:process_id>', methods=['GET'])
def get_onboarding_process(process_id: int):
    """Get a single Jobvite onboarding process"""
    user, tenant_id, error = _require_user_and_tenant()
    if error:
        return jsonify(error[0]), error[1]
    
    process = JobviteOnboardingProcess.query.filter_by(id=process_id, tenant_id=tenant_id).first()
    if not process:
        return jsonify({'error': 'Onboarding process not found'}), 404
    
    return jsonify(_serialize_onboarding_process(process)), 200

@jobvite_bp.route('/api/onboarding/processes/<int:process_id>/tasks', methods=['GET'])
def get_onboarding_process_tasks(process_id: int):
    """Get tasks linked to a Jobvite onboarding process"""
    user, tenant_id, error = _require_user_and_tenant()
    if error:
        return jsonify(error[0]), error[1]
    
    process = JobviteOnboardingProcess.query.filter_by(id=process_id, tenant_id=tenant_id).first()
    if not process:
        return jsonify({'error': 'Onboarding process not found'}), 404
    
    tasks = (
        JobviteOnboardingTask.query
        .filter_by(process_id=process.id)
        .order_by(JobviteOnboardingTask.due_date.asc())
        .all()
    )
    
    return jsonify([_serialize_onboarding_task(task) for task in tasks]), 200

