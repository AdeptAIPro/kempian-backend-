"""
Flask routes for Jobvite integration.
"""

from flask import Blueprint, request, jsonify, g
import requests
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


def _validate_jobvite_credentials(api_key: str, api_secret: str, company_id: str, environment: str) -> Optional[Tuple[dict, int]]:
    """
    Ensure provided Jobvite credentials can reach the API.
    Returns error tuple when invalid, otherwise None.
    """
    # Validate input parameters
    if not api_key or not api_key.strip():
        return (
            {
                'error': 'API key cannot be empty',
                'details': 'Please provide a valid API key'
            },
            400,
        )
    if not api_secret or not api_secret.strip():
        return (
            {
                'error': 'API secret cannot be empty',
                'details': 'Please provide a valid API secret'
            },
            400,
        )
    if not company_id or not company_id.strip():
        return (
            {
                'error': 'Company ID cannot be empty',
                'details': 'Please provide a valid Company ID'
            },
            400,
        )
    
    base_urls = get_base_urls(environment)
    environment_display = denormalize_environment(environment)
    
    try:
        client = JobviteV2Client(
            api_key=api_key.strip(),
            api_secret=api_secret.strip(),
            company_id=company_id.strip(),
            base_url=base_urls['v2'],
        )
        # Make actual API call to validate credentials
        # Use /job?start=0&count=1 endpoint (Jobvite requires start/count params)
        # This will raise an exception if credentials are invalid
        result = client.get_job(start=0, count=1)
        
        # Verify we got a valid response (even if empty)
        if result is None:
            raise ValueError("API returned null response")
        
        # Validate response structure - must have jobs or requisitions key
        # Use fallback parsing: jobs OR requisitions OR []
        if isinstance(result, dict):
            jobs = result.get("jobs") or result.get("requisitions") or []
            # If neither key exists and it's not a single job response, log warning
            if 'jobs' not in result and 'requisitions' not in result and 'job' not in result:
                logger.warning(f"Jobvite test connection response missing expected keys. Available: {list(result.keys())}")
        
        # If we get here, credentials are valid
        logger.info(f"Jobvite credential validation successful for {environment_display} environment")
        return None
        
    except ValueError as exc:
        # This catches authentication errors, invalid endpoints, etc.
        error_msg = str(exc)
        logger.error(f"Jobvite credential validation failed ({environment_display}): {error_msg}")
        
        # Check if it's an authentication error
        if '401' in error_msg or 'Unauthorized' in error_msg or 'authentication' in error_msg.lower():
            return (
                {
                    'error': f'Invalid Jobvite credentials for {environment_display} environment',
                    'details': 'The API key, secret, or company ID is incorrect. Please verify your credentials match the selected environment (Production vs Stage).',
                    'troubleshooting': [
                        'Verify API key is correct (no extra spaces, full key copied)',
                        'Verify API secret is correct (matches the API key)',
                        'Ensure you are using Production credentials with Production environment (or Stage with Stage)',
                        'Check that API key has not been revoked or expired',
                        'Verify API key has required permissions'
                    ]
                },
                400,
            )
        else:
            return (
                {
                    'error': f'Jobvite API connection failed for {environment_display} environment',
                    'details': error_msg,
                },
                400,
            )
    except requests.exceptions.ConnectionError as exc:
        # Network/DNS errors - hostname cannot be resolved or connection refused
        error_msg = str(exc)
        logger.error(f"Jobvite API network error ({environment_display}): {error_msg}")
        
        # Check if it's a DNS resolution error
        if 'Failed to resolve' in error_msg or 'getaddrinfo failed' in error_msg or 'NameResolutionError' in error_msg:
            # Special handling for Stage environment DNS issues
            if environment == "stage":
                return (
                    {
                        'error': f'Cannot reach Jobvite {environment_display} API server',
                        'details': (
                            f'DNS resolution failed for {base_urls["v2"]}. '
                            f'The Stage environment (api-stg.jobvite.com) may not be accessible from your network. '
                            f'This is a common issue if your network blocks staging servers or if Jobvite Stage is not publicly accessible.'
                        ),
                        'troubleshooting': [
                            f'Verify you can access {base_urls["v2"]} from your network (try in a browser)',
                            'Check your internet connection and DNS settings',
                            'If Stage is not accessible, try using Production environment instead',
                            'Contact your network administrator if behind a firewall/proxy',
                            'Verify with Jobvite support that Stage environment is available for your account',
                            'Note: Stage credentials may only work with Production URL in some cases'
                        ],
                        'networkError': True,
                        'suggestedAction': 'Try using Production environment if Stage is not accessible'
                    },
                    400,
                )
            else:
                return (
                    {
                        'error': f'Cannot reach Jobvite {environment_display} API server',
                        'details': f'DNS resolution failed for {base_urls["v2"]}. This could be a network issue.',
                        'troubleshooting': [
                            f'Verify you can access {base_urls["v2"]} from your network',
                            'Check your internet connection and DNS settings',
                            'Contact your network administrator if behind a firewall/proxy',
                            'Verify the Jobvite API server is operational'
                        ],
                        'networkError': True
                    },
                    400,
                )
        else:
            return (
                {
                    'error': f'Network connection failed for {environment_display} environment',
                    'details': f'Cannot connect to {base_urls["v2"]}. Please check your network connection.',
                    'troubleshooting': [
                        'Check your internet connection',
                        'Verify firewall/proxy settings allow connections to Jobvite API',
                        f'Try accessing {base_urls["v2"]} in a browser to test connectivity',
                        'Check if your network blocks outbound HTTPS connections',
                        'Verify the Jobvite API server is operational'
                    ],
                    'networkError': True
                },
                400,
            )
    except requests.exceptions.Timeout as exc:
        # Request timeout
        error_msg = str(exc)
        logger.error(f"Jobvite API timeout ({environment_display}): {error_msg}")
        return (
            {
                'error': f'Request timeout for {environment_display} environment',
                'details': f'The request to {base_urls["v2"]} timed out. The server may be slow or unreachable.',
                'troubleshooting': [
                    'Check your internet connection speed',
                    'Verify the Jobvite API server is responding',
                    'Try again in a few moments',
                    'Contact Jobvite support if the issue persists'
                ],
                'networkError': True
            },
            400,
        )
    except Exception as exc:
        # Catch any other unexpected errors
        error_msg = str(exc)
        logger.error(f"Jobvite credential validation failed ({environment_display}): {error_msg}")
        
        # Check if it's a network-related error
        if 'Connection' in str(type(exc).__name__) or 'network' in error_msg.lower() or 'resolve' in error_msg.lower():
            return (
                {
                    'error': f'Network error connecting to {environment_display} environment',
                    'details': error_msg,
                    'troubleshooting': [
                        'Check your internet connection',
                        'Verify firewall/proxy settings',
                        'Try again in a few moments'
                    ],
                    'networkError': True
                },
                400,
            )
        else:
            return (
                {
                    'error': f'Failed to validate Jobvite credentials for {environment_display} environment',
                    'details': error_msg,
                },
                400,
            )

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
    raw_json = candidate.raw_json or {}
    
    # Extract additional fields from raw_json
    application_data = raw_json.get('application', {})
    job_data = application_data.get('job', {}) if isinstance(application_data, dict) else {}
    
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
        # Address fields
        'address': raw_json.get('address'),
        'address2': raw_json.get('address2'),
        'city': raw_json.get('city'),
        'state': raw_json.get('state'),
        'postalCode': raw_json.get('postalCode'),
        'country': raw_json.get('country'),
        'location': raw_json.get('location'),
        # Phone fields
        'homePhone': raw_json.get('homePhone'),
        'mobile': raw_json.get('mobile'),
        'workPhone': raw_json.get('workPhone'),
        # Additional fields
        'title': raw_json.get('title'),
        'companyName': raw_json.get('companyName'),
        'workStatus': raw_json.get('workStatus'),
        'smsConsent': raw_json.get('smsConsent'),
        # Application/Job details from nested structure
        'application': {
            'eId': application_data.get('eId') if isinstance(application_data, dict) else None,
            'workflowState': application_data.get('workflowState') if isinstance(application_data, dict) else None,
            'source': application_data.get('source') if isinstance(application_data, dict) else None,
            'sourceType': application_data.get('sourceType') if isinstance(application_data, dict) else None,
            'jobviteChannel': application_data.get('jobviteChannel') if isinstance(application_data, dict) else None,
            'sentDate': application_data.get('sentDate') if isinstance(application_data, dict) else None,
            'lastUpdatedDate': application_data.get('lastUpdatedDate') if isinstance(application_data, dict) else None,
            'consentStatus': application_data.get('consentStatus') if isinstance(application_data, dict) else None,
            'hasArtifacts': application_data.get('hasArtifacts') if isinstance(application_data, dict) else None,
        } if application_data else None,
        'job': {
            'eId': job_data.get('eId') if isinstance(job_data, dict) else None,
            'title': job_data.get('title') if isinstance(job_data, dict) else None,
            'location': job_data.get('location') if isinstance(job_data, dict) else None,
            'requisitionId': job_data.get('requisitionId') if isinstance(job_data, dict) else None,
            'categoryName': job_data.get('categoryName') if isinstance(job_data, dict) else None,
            'company': job_data.get('company') if isinstance(job_data, dict) else None,
            'jobType': job_data.get('jobType') if isinstance(job_data, dict) else None,
            'postingType': job_data.get('postingType') if isinstance(job_data, dict) else None,
            'primaryRecruiter': job_data.get('primaryRecruiter') if isinstance(job_data, dict) else None,
            'recruiters': job_data.get('recruiters') if isinstance(job_data, dict) else None,
        } if job_data else None,
        'rawJson': raw_json
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
    
    # Validate credentials for BOTH Production AND Stage environments
    # This ensures invalid credentials are caught before saving
    # Network errors are allowed (with warning) since they don't indicate invalid credentials
    network_warning = None
    validation_error = _validate_jobvite_credentials(
        api_key=data['apiKey'],
        api_secret=data['apiSecret'],
        company_id=data['companyId'],
        environment=environment_normalized,
    )
    if validation_error:
        error_data, status_code = validation_error
        # Allow saving credentials if it's a network error (DNS, connection, timeout)
        # These errors don't indicate invalid credentials, just connectivity issues
        if error_data.get('networkError', False):
            # Log warning but allow save
            logger.warning(
                f"Jobvite credentials saved with network warning for {environment_normalized}: "
                f"{error_data.get('error', 'Network error')}"
            )
            # Continue to save, but we'll include a warning in the response
            network_warning = error_data
        else:
            # Authentication or other validation errors - reject save
            return jsonify(validation_error[0]), validation_error[1]

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
        # Merge sync_config with defaults to ensure syncCandidates is enabled by default
        sync_config = data.get('syncConfig', {})
        # Ensure syncCandidates defaults to True if not explicitly set
        if 'syncCandidates' not in sync_config:
            sync_config['syncCandidates'] = True
        # Ensure syncJobs defaults to True if not explicitly set
        if 'syncJobs' not in sync_config:
            sync_config['syncJobs'] = True
        settings.sync_config = sync_config
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
            sync_config={}  # Will be set below with defaults
        )
        db.session.add(settings)
        # Merge sync_config with defaults to ensure syncCandidates is enabled by default
        sync_config = data.get('syncConfig', {})
        # Ensure syncCandidates defaults to True if not explicitly set
        if 'syncCandidates' not in sync_config:
            sync_config['syncCandidates'] = True
        # Ensure syncJobs defaults to True if not explicitly set
        if 'syncJobs' not in sync_config:
            sync_config['syncJobs'] = True
        settings.sync_config = sync_config
    
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error saving Jobvite config: {e}")
        return jsonify({'error': f'Failed to save configuration: {str(e)}'}), 500
    
    response_data = {
        'success': True,
        'message': 'Configuration saved',
        'configId': settings.id
    }
    
    # Include network warning if present (credentials saved but couldn't verify due to network)
    if network_warning:
        response_data['warning'] = {
            'message': network_warning.get('error', 'Network connectivity issue'),
            'details': network_warning.get('details', ''),
            'troubleshooting': network_warning.get('troubleshooting', [])
        }
        response_data['message'] = 'Configuration saved, but could not verify connectivity to Jobvite API'
    
    return jsonify(response_data), 200

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

@jobvite_bp.route('/api/integrations/jobvite/disconnect', methods=['POST'])
def disconnect_jobvite():
    """Disconnect Jobvite integration for the current tenant."""
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
    
    settings = JobviteSettings.query.filter_by(tenant_id=tenant_id).all()
    if not settings:
        return jsonify({'success': True, 'message': 'Jobvite integration already disconnected'}), 200
    
    try:
        for config in settings:
            db.session.delete(config)
        db.session.commit()
        logger.info(f"Jobvite integration disconnected for tenant {tenant_id}")
        return jsonify({'success': True, 'message': 'Jobvite integration disconnected successfully'}), 200
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error disconnecting Jobvite integration: {e}")
        return jsonify({'error': 'Failed to disconnect Jobvite integration'}), 500


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
    
    # Use the same validation function for consistency
    validation_error = _validate_jobvite_credentials(
            api_key=data['apiKey'],
            api_secret=data['apiSecret'],
            company_id=data['companyId'],
        environment=environment_normalized,
    )
    
    if validation_error:
        error_data, status_code = validation_error
        response_data = {
            'success': False,
            'error': error_data.get('error', 'Connection failed'),
            'details': error_data.get('details', ''),
            'troubleshooting': error_data.get('troubleshooting', [])
        }
        # Include networkError flag so frontend can handle network vs auth errors differently
        if error_data.get('networkError', False):
            response_data['networkError'] = True
            # For network errors, provide a more helpful message
            response_data['message'] = f'Cannot reach Jobvite {denormalize_environment(environment_normalized)} API server. This may be a network connectivity issue rather than invalid credentials.'
        else:
            response_data['networkError'] = False
            response_data['message'] = 'Connection test failed. Please verify your credentials.'
        
        return jsonify(response_data), status_code
    
    # If validation passed, return success
    environment_display = denormalize_environment(environment_normalized)
    return jsonify({
        'success': True,
        'message': f'Connection successful for {environment_display} environment'
    }), 200

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
    
    # Check if settings exist and syncCandidates is enabled
    settings = get_jobvite_settings(tenant_id)
    if not settings:
        return jsonify({
            'error': 'Jobvite integration not configured',
            'details': 'Please configure Jobvite integration first'
        }), 400
    
    if not settings.sync_config.get('syncCandidates'):
        return jsonify({
            'error': 'Candidate sync is disabled',
            'details': 'Please enable "syncCandidates" in the Jobvite integration settings',
            'syncConfig': settings.sync_config
        }), 400
    
    # Check for optional limit parameter in request body
    data = request.get_json(silent=True) or {}
    limit = data.get('limit')
    
    logger.info(f"[ROUTES DEBUG] Received sync request with data: {data}")
    logger.info(f"[ROUTES DEBUG] Extracted limit from request: {limit} (type: {type(limit)})")
    logger.info(f"[ROUTES DEBUG] Current sync_config before update: {settings.sync_config}")
    
    # Ensure sync_config exists
    if not settings.sync_config:
        settings.sync_config = {}
        logger.info(f"[ROUTES DEBUG] Initialized empty sync_config")
    
    # If limit is provided, update sync_config (persisted for future syncs)
    if limit is not None:
        try:
            # Handle empty string or null as "no limit"
            if limit == '' or (isinstance(limit, str) and limit.strip() == ''):
                # Clear the limit to sync all
                if 'candidateSyncLimit' in settings.sync_config:
                    del settings.sync_config['candidateSyncLimit']
                    db.session.commit()
                    db.session.refresh(settings)
                    logger.info(f"[INFO] Candidate sync limit cleared for tenant {tenant_id}. Will sync all candidates.")
            else:
                limit = int(limit)
                logger.info(f"[ROUTES DEBUG] Converted limit to int: {limit}")
                if limit > 0:
                    logger.info(f"[ROUTES DEBUG] Setting candidateSyncLimit to {limit} in sync_config")
                    logger.info(f"[ROUTES DEBUG] sync_config before setting limit: {settings.sync_config}")
                    settings.sync_config['candidateSyncLimit'] = limit
                    logger.info(f"[ROUTES DEBUG] sync_config after setting limit: {settings.sync_config}")
                    logger.info(f"[ROUTES DEBUG] Verifying limit was set: candidateSyncLimit = {settings.sync_config.get('candidateSyncLimit')}")
                    
                    # Persist the limit to database
                    try:
                        # Flush to ensure changes are visible in the same session
                        logger.info(f"[ROUTES DEBUG] Flushing database session...")
                        db.session.flush()
                        logger.info(f"[ROUTES DEBUG] Committing database session...")
                        db.session.commit()
                        logger.info(f"[ROUTES DEBUG] Database commit successful")
                        
                        # Refresh the settings object to ensure we have the latest data
                        logger.info(f"[ROUTES DEBUG] Refreshing settings object from database...")
                        db.session.refresh(settings)
                        logger.info(f"[ROUTES DEBUG] Settings refreshed. sync_config from DB: {settings.sync_config}")
                        logger.info(f"[ROUTES DEBUG] candidateSyncLimit from refreshed settings: {settings.sync_config.get('candidateSyncLimit')}")
                        
                        logger.info(f"[SUCCESS] Sync limit of {limit} candidates set and persisted for tenant {tenant_id}")
                        logger.info(f"[ROUTES DEBUG] Final sync_config after all operations: {settings.sync_config}")
                    except Exception as db_error:
                        db.session.rollback()
                        logger.error(f"[ROUTES ERROR] Failed to persist sync limit: {db_error}")
                        logger.error(f"[ROUTES ERROR] Exception type: {type(db_error)}")
                        import traceback
                        logger.error(f"[ROUTES ERROR] Traceback: {traceback.format_exc()}")
                else:
                    # Limit is 0 or negative, clear it
                    if 'candidateSyncLimit' in settings.sync_config:
                        del settings.sync_config['candidateSyncLimit']
                        db.session.commit()
                        db.session.refresh(settings)
                        logger.info(f"[INFO] Candidate sync limit cleared (limit was {limit}). Will sync all candidates.")
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid limit parameter. Must be a positive integer or empty to sync all.'}), 400
    
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

@jobvite_bp.route('/api/integrations/jobvite/sync/diagnostics', methods=['GET'])
def sync_diagnostics():
    """Get diagnostic information about sync configuration and status"""
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
    
    settings = get_jobvite_settings(tenant_id)
    if not settings:
        return jsonify({
            'configured': False,
            'message': 'Jobvite integration not configured'
        }), 200
    
    # Count candidates in database
    candidate_count = JobviteCandidate.query.filter_by(tenant_id=tenant_id).count()
    
    # Get sync config
    sync_config = settings.sync_config or {}
    
    diagnostics = {
        'configured': True,
        'environment': denormalize_environment(settings.environment),
        'companyId': settings.company_id,
        'syncConfig': sync_config,
        'syncCandidatesEnabled': sync_config.get('syncCandidates', False),
        'syncJobsEnabled': sync_config.get('syncJobs', False),
        'candidateSyncLimit': sync_config.get('candidateSyncLimit'),
        'candidateCount': candidate_count,
        'lastSyncAt': settings.last_sync_at.isoformat() if settings.last_sync_at else None,
        'lastSyncStatus': settings.last_sync_status,
        'lastError': settings.last_error,
        'isActive': settings.is_active
    }
    
    # Add warnings if needed
    warnings = []
    if not sync_config.get('syncCandidates'):
        warnings.append('Candidate sync is disabled. Enable syncCandidates in settings to sync candidates.')
    if sync_config.get('candidateSyncLimit'):
        warnings.append(f'Candidate sync is LIMITED to {sync_config.get("candidateSyncLimit")} candidates. Remove candidateSyncLimit from sync config to sync all candidates.')
    if not settings.is_active:
        warnings.append('Jobvite integration is inactive.')
    if settings.last_sync_status == 'failed':
        warnings.append(f'Last sync failed: {settings.last_error or "Unknown error"}')
    
    if warnings:
        diagnostics['warnings'] = warnings
    
    return jsonify(diagnostics), 200

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

@jobvite_bp.route('/api/integrations/jobvite/jobs/<int:job_id>/suggested-candidates', methods=['GET'])
@jobvite_bp.route('/integrations/jobvite/jobs/<int:job_id>/suggested-candidates', methods=['GET'])

def get_jobvite_job_suggested_candidates(job_id: int):
    """Get suggested candidates for a Jobvite job based on job requirements"""
    import json
    from datetime import datetime
    
    try:
        user, tenant_id, error = _require_user_and_tenant()
        if error:
            return jsonify(error[0]), error[1]
        
        # Get the Jobvite job
        job = JobviteJob.query.filter_by(id=job_id, tenant_id=tenant_id).first()
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        
        # Get query parameters
        refresh = request.args.get('refresh', 'false').lower() == 'true'
        limit = 3  # Return top 3 candidates for Jobvite
        
        # Build comprehensive job description for semantic matching
        job_description_parts = []
        
        # Add job title
        if job.title:
            job_description_parts.append(f"Job Title: {job.title}")
        
        # Add department/company
        if job.department:
            job_description_parts.append(f"Department: {job.department}")
        
        # Add location
        if job.locationMain:
            job_description_parts.append(f"Location: {job.locationMain}")
        
        # Add region
        if job.region:
            job_description_parts.append(f"Region: {job.region}")
        
        # Add category
        if job.category:
            job_description_parts.append(f"Category: {job.category}")
        
        # Add remote type
        if job.remoteType:
            job_description_parts.append(f"Remote Type: {job.remoteType}")
        
        # Add salary information
        if job.salaryMin or job.salaryMax:
            salary_str = f"Salary: {job.salaryCurrency or '$'} {job.salaryMin or 'N/A'} - {job.salaryMax or 'N/A'}"
            if job.salaryFrequency:
                salary_str += f" {job.salaryFrequency}"
            job_description_parts.append(salary_str)
        
        # Add raw JSON data if available (may contain description)
        if job.rawJson:
            try:
                raw_data = job.rawJson if isinstance(job.rawJson, dict) else json.loads(job.rawJson)
                if isinstance(raw_data, dict):
                    # Extract description from various possible fields
                    description = (
                        raw_data.get('description') or 
                        raw_data.get('jobDescription') or 
                        raw_data.get('summary') or 
                        raw_data.get('requirements') or
                        ''
                    )
                    if description:
                        job_description_parts.append(f"Description: {description}")
            except:
                pass
        
        # Combine into full job description
        full_job_description = "\n".join(job_description_parts)
        
        # Use semantic matching algorithm from service.py
        try:
            from app.search.service import semantic_match
            
            logger.info(f"Using semantic matching algorithm for Jobvite job {job_id}: {job.title}")
            
            # Call semantic match with the job description (fetch more to ensure we have good top 3)
            result = semantic_match(full_job_description, top_k=15)  # Fetch 15, then take top 3
            
            if not result or not result.get('results'):
                logger.warning(f"Semantic match returned no results for Jobvite job {job_id}")
                return jsonify({
                    'candidates': [],
                    'suggested_candidates': [],
                    'message': 'No matching candidates found',
                    'job_id': job_id,
                    'job_title': job.title
                }), 200
            
            # Format candidates from semantic match results
            formatted_candidates = []
            for candidate in result.get('results', []):
                try:
                    # Extract candidate data (handle different field name variations)
                    candidate_email = candidate.get('email') or candidate.get('Email') or ''
                    candidate_name = (
                        candidate.get('full_name') or 
                        candidate.get('FullName') or 
                        candidate.get('name') or 
                        candidate.get('Name') or 
                        'Unknown'
                    )
                    
                    # Extract skills
                    candidate_skills = candidate.get('skills') or candidate.get('Skills') or []
                    if isinstance(candidate_skills, str):
                        candidate_skills = [s.strip() for s in candidate_skills.split(',')]
                    elif not isinstance(candidate_skills, list):
                        candidate_skills = []
                    
                    # Extract experience years
                    experience_years = 0
                    exp_sources = [
                        candidate.get('experience_years'),
                        candidate.get('total_experience_years'),
                        candidate.get('Experience'),
                        candidate.get('experience')
                    ]
                    for exp_source in exp_sources:
                        if exp_source is not None:
                            try:
                                if isinstance(exp_source, (int, float)):
                                    experience_years = int(exp_source)
                                    break
                                elif isinstance(exp_source, str):
                                    # Try to extract number from string
                                    import re
                                    match = re.search(r'\d+', exp_source)
                                    if match:
                                        experience_years = int(match.group())
                                        break
                            except:
                                pass
                    
                    # Get match score/percentage
                    match_score = (
                        candidate.get('match_percentage') or 
                        candidate.get('Score') or 
                        candidate.get('score') or 
                        0
                    )
                    if isinstance(match_score, str):
                        try:
                            match_score = float(match_score.replace('%', ''))
                        except:
                            match_score = 0
                    
                    # Format candidate data for frontend
                    formatted_candidate = {
                        'id': candidate_email or f"candidate_{len(formatted_candidates)}",
                        'full_name': candidate_name,
                        'name': candidate_name,  # Also include 'name' for compatibility
                        'email': candidate_email,
                        'phone': candidate.get('phone') or candidate.get('Phone') or '',
                        'location': candidate.get('location') or candidate.get('Location') or '',
                        'experience_years': experience_years,
                        'skills': candidate_skills,
                        'summary': (
                            candidate.get('summary') or 
                            candidate.get('description') or 
                            candidate.get('resumeText') or 
                            candidate.get('resume_text') or 
                            ''
                        ),
                        'match_score': match_score / 100.0 if match_score > 1 else match_score,  # Normalize to 0-1
                        'matching_skills_count': len(candidate_skills)
                    }
                    
                    formatted_candidates.append(formatted_candidate)
                    
                except Exception as e:
                    logger.warning(f"Error formatting candidate from semantic match: {str(e)}")
                    continue
            
            # Sort by match score and take only top 3
            formatted_candidates.sort(key=lambda x: x.get('match_score', 0), reverse=True)
            top_candidates = formatted_candidates[:limit]
            
            logger.info(f"Found {len(top_candidates)} suggested candidates (top {limit}) for Jobvite job {job_id} using semantic matching")
            
            return jsonify({
                'candidates': top_candidates,
                'suggested_candidates': top_candidates,  # Also include for compatibility
                'total_matched': len(formatted_candidates),
                'job_id': job_id,
                'job_title': job.title,
                'algorithm_used': result.get('algorithm_used', 'Semantic Matching Algorithm'),
                'cached': False
            }), 200
            
        except ImportError as e:
            logger.error(f"Failed to import semantic_match from service: {str(e)}")
            return jsonify({
                'candidates': [],
                'suggested_candidates': [],
                'error': 'Semantic matching algorithm not available',
                'message': 'Please ensure the search service is properly configured'
            }), 500
        except Exception as e:
            logger.error(f"Error using semantic matching algorithm: {str(e)}", exc_info=True)
            return jsonify({
                'candidates': [],
                'suggested_candidates': [],
                'error': 'Failed to get suggested candidates using semantic matching',
                'message': str(e)
            }), 500
        
    except Exception as e:
        logger.error(f"Error getting suggested candidates for Jobvite job {job_id}: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to get suggested candidates'}), 500

