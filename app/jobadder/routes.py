from flask import Blueprint, request, jsonify, redirect, url_for, current_app, Response
from app.simple_logger import get_logger
from datetime import datetime
from app.models import JobAdderIntegration, User, db
from app.utils import get_current_user
import base64
import os
import uuid
from urllib.parse import urlencode
from sqlalchemy.exc import SQLAlchemyError
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

from .auth import JobAdderAuth
from .client import JobAdderClient, JobAdderAPIError

logger = get_logger("jobadder")
jobadder_bp = Blueprint('jobadder', __name__)


def _get_user():
    """Get authenticated user."""
    user_jwt = get_current_user()
    if not user_jwt or not user_jwt.get('email'):
        return None, jsonify({'error': 'Unauthorized'}), 401
    
    user = User.query.filter_by(email=user_jwt['email']).first()
    if not user:
        return None, jsonify({'error': 'User not found'}), 404
    
    return user, None, None


def _get_integration(user):
    """Get JobAdder integration for user."""
    return JobAdderIntegration.query.filter_by(user_id=user.id).first()


def _get_client(user):
    """Get JobAdder client for user. Returns (client, error_response, status_code)."""
    integration = _get_integration(user)
    if not integration:
        return None, jsonify({'error': 'JobAdder account not connected'}), 404
    
    try:
        return JobAdderClient(integration), None, None
    except JobAdderAPIError as e:
        logger.error('Failed to initialize client: %s', e)
        return None, jsonify({'error': str(e)}), 400


def _build_params(args, allowed_keys=None):
    """Build query parameters with pagination."""
    allowed_keys = allowed_keys or []
    params = {}
    
    # Pagination
    try:
        params['page'] = max(1, int(args.get('page', 1)))
        params['pageSize'] = max(1, min(int(args.get('pageSize', 20)), 100))
    except (TypeError, ValueError):
        params['page'] = 1
        params['pageSize'] = 20
    
    # Additional filters
    for key in allowed_keys:
        value = args.get(key)
        if value:
            params[key] = value
    
    return params


def _get_serializer():
    secret = current_app.config.get('SECRET_KEY') or os.getenv('SECRET_KEY') or 'jobadder-secret'
    return URLSafeTimedSerializer(secret, salt='jobadder-oauth')


def _generate_state(user_id):
    serializer = _get_serializer()
    payload = {'user_id': user_id, 'nonce': str(uuid.uuid4())}
    return serializer.dumps(payload)


def _load_state(state_token):
    serializer = _get_serializer()
    return serializer.loads(state_token, max_age=600)


def _get_redirect_uri():
    """
    Get the redirect URI for OAuth callback.
    Must exactly match what's registered in JobAdder Developer Portal.
    """
    # Priority 1: Explicit environment variable (recommended)
    custom = os.getenv('JOBADDER_OAUTH_REDIRECT_URI')
    if custom:
        # Ensure no trailing slash and proper format
        return custom.rstrip('/')
    
    # Priority 2: Use BACKEND_URL or API_URL environment variable
    backend_url = os.getenv('BACKEND_URL') or os.getenv('API_URL')
    if backend_url:
        # Ensure no trailing slash
        backend_url = backend_url.rstrip('/')
        # Build the callback URL
        callback_path = url_for('jobadder.jobadder_oauth_callback', _external=False)
        return f"{backend_url}{callback_path}"
    
    # Priority 3: Use request host header (works when behind proxy with proper headers)
    try:
        if request and request.host:
            scheme = 'https' if request.is_secure or request.headers.get('X-Forwarded-Proto') == 'https' else 'http'
            host = request.host
            callback_path = url_for('jobadder.jobadder_oauth_callback', _external=False)
            return f"{scheme}://{host}{callback_path}"
    except RuntimeError:
        # No request context (e.g., during testing)
        pass
    
    # Priority 4: Fallback to url_for with _external=True (may use wrong domain)
    # This is a last resort and may cause issues if SERVER_NAME is not set correctly
    uri = url_for('jobadder.jobadder_oauth_callback', _external=True)
    logger.warning("Using url_for(_external=True) for redirect URI. Consider setting JOBADDER_OAUTH_REDIRECT_URI or BACKEND_URL environment variable.")
    return uri.rstrip('/')


def _get_frontend_redirect_base():
    return os.getenv('FRONTEND_URL', 'http://localhost:5173').rstrip('/')


@jobadder_bp.route('/integrations/jobadder/health', methods=['GET'])
def health():
    """Health check endpoint to verify the route is accessible."""
    return jsonify({
        'ok': True,
        'service': 'jobadder',
        'endpoints': {
            'connect': '/integrations/jobadder/connect',
            'callback': '/integrations/jobadder/oauth/callback',
            'status': '/integrations/jobadder/status',
            'refresh': '/integrations/jobadder/oauth/refresh',
            'disconnect': '/integrations/jobadder/disconnect',
            'jobs': '/integrations/jobadder/jobs (GET, POST)',
            'candidates': '/integrations/jobadder/candidates (GET, POST)',
            'applications': '/integrations/jobadder/applications (GET, POST)',
            'companies': '/integrations/jobadder/companies (GET, POST)',
            'contacts': '/integrations/jobadder/contacts (GET, POST)',
            'placements': '/integrations/jobadder/placements (GET, POST)',
            'notes': '/integrations/jobadder/notes (GET, POST)',
            'activities': '/integrations/jobadder/activities (GET, POST)',
            'tasks': '/integrations/jobadder/tasks (GET, POST)',
            'users': '/integrations/jobadder/users (GET)',
            'workflows': '/integrations/jobadder/workflows (GET)',
            'customfields': '/integrations/jobadder/customfields (GET)',
            'requisitions': '/integrations/jobadder/requisitions (GET)',
            'jobboards': '/integrations/jobadder/jobboards (GET)',
            'webhooks': '/integrations/jobadder/webhooks (GET, POST, PUT, DELETE)',
            'partneractionbuttons': '/integrations/jobadder/partneractionbuttons (GET, POST, PUT, DELETE)',
            'attachments': '/integrations/jobadder/<resource>/<id>/attachments (GET, POST, DELETE)'
        },
        'features': {
            'oauth2': True,
            'rate_limiting': True,
            'write_operations': True,
            'webhooks': True,
            'partner_action_buttons': True,
            'file_operations': True
        }
    }), 200


@jobadder_bp.route('/integrations/jobadder/connect', methods=['POST'])
def connect():
    """
    Initiate JobAdder OAuth2 connection.
    Stores client credentials and returns an authorization URL for the user to visit.
    """
    user, error, status = _get_user()
    if error:
        return error, status
    
    data = request.get_json() or {}
    client_id = data.get('clientId') or data.get('client_id')
    client_secret = data.get('clientSecret') or data.get('client_secret')
    scope = data.get('scope') or os.getenv('JOBADDER_DEFAULT_SCOPE', 'read write offline_access')
    
    # Validate scope - never use jobadder.api, always use read write offline_access
    if 'jobadder.api' in scope.lower():
        logger.warning("Invalid scope detected: %s. Rejecting and using default 'read write offline_access'", scope)
        scope = 'read write offline_access'
    
    if not scope or scope.strip() == '':
        scope = 'read write offline_access'
    
    if not client_id or not client_secret:
        return jsonify({'error': 'clientId and clientSecret are required'}), 400
    
    enc_client_secret = base64.b64encode(client_secret.encode()).decode()
    
    try:
        integration = _get_integration(user)
        
        if integration:
            integration.client_id = client_id
            integration.client_secret = enc_client_secret
            integration.access_token = None
            integration.token_expires_at = None
            integration.refresh_token = None
            integration.account_name = None
            integration.account_email = None
            integration.account_user_id = None
            integration.account_company_id = None
            integration.updated_at = datetime.utcnow()
        else:
            integration = JobAdderIntegration(
                user_id=user.id,
                client_id=client_id,
                client_secret=enc_client_secret,
                access_token=None,
                token_expires_at=None,
                refresh_token=None,
                account_name=None,
                account_email=None,
                account_user_id=None,
                account_company_id=None,
            )
            db.session.add(integration)
        
        db.session.commit()
        
    except Exception as e:
        logger.error('Error saving integration: %s', e)
        db.session.rollback()
        return jsonify({'error': 'Failed to save integration'}), 500

    state = _generate_state(user.id)
    redirect_uri = _get_redirect_uri()
    auth = JobAdderAuth(client_id, client_secret, scope=scope)
    
    # Build authorization URL with properly encoded parameters
    # redirect_uri must be URL-encoded to match exactly what's registered
    # Scope must be space-separated and URL-encoded (%20)
    auth_params = {
        'response_type': 'code',
        'client_id': client_id,
        'redirect_uri': redirect_uri,  # urlencode will handle encoding
        'scope': scope,  # Will be URL-encoded (spaces become %20)
        'state': state,
        'prompt': 'consent',  # Required: always prompt for consent
    }
    auth_url = f"{auth.authorize_url}?{urlencode(auth_params)}"
    
    # Log detailed information for debugging domain issues
    logger.info("Generated JobAdder OAuth URL: authorize_url=%s, redirect_uri=%s, scope=%s", 
                auth.authorize_url, redirect_uri, scope)
    logger.info("Redirect URI source: JOBADDER_OAUTH_REDIRECT_URI=%s, BACKEND_URL=%s, API_URL=%s, request.host=%s",
                'set' if os.getenv('JOBADDER_OAUTH_REDIRECT_URI') else 'not set',
                os.getenv('BACKEND_URL') or 'not set',
                os.getenv('API_URL') or 'not set',
                request.host if request else 'no request context')
    
    return jsonify({
        'success': True,
        'authUrl': auth_url,
        'redirectUri': redirect_uri,
    }), 200


@jobadder_bp.route('/integrations/jobadder/status', methods=['GET'])
def status():
    """Check JobAdder connection status."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    integration = _get_integration(user)
    # If integration row is missing OR we don't have valid OAuth tokens yet,
    # treat the connection as not established. This prevents the frontend
    # dashboard from calling data endpoints that require a refresh token and
    # avoids noisy "Missing JobAdder refresh token" 502 errors.
    if not integration or not integration.refresh_token or not integration.access_token:
        return jsonify({'connected': False}), 200
    
    is_expired = False
    if integration.token_expires_at:
        is_expired = datetime.utcnow() >= integration.token_expires_at
    
    return jsonify({
        'connected': True,
        'account': {
            'name': integration.account_name,
            'email': integration.account_email,
            'userId': integration.account_user_id,
            'companyId': integration.account_company_id
        },
        'tokenExpired': is_expired
    }), 200


@jobadder_bp.route('/integrations/jobadder/oauth/refresh', methods=['POST'])
def refresh_token():
    """Manually refresh JobAdder access token."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    integration = _get_integration(user)
    if not integration:
        return jsonify({'error': 'JobAdder account not connected'}), 404
    
    if not integration.refresh_token:
        return jsonify({'error': 'No refresh token available. Please reconnect.'}), 400
    
    try:
        decoded_secret = base64.b64decode(integration.client_secret.encode()).decode()
    except Exception as e:
        logger.error('Failed to decode client secret: %s', e)
        return jsonify({'error': 'Invalid JobAdder credentials'}), 400
    
    auth = JobAdderAuth(integration.client_id, decoded_secret)
    success, token_data = auth.refresh_access_token(integration.refresh_token)
    
    if not success:
        logger.error('Token refresh failed: %s', token_data)
        return jsonify({'error': token_data or 'Failed to refresh token'}), 400
    
    try:
        integration.access_token = auth.access_token
        integration.refresh_token = auth.refresh_token
        integration.token_expires_at = auth.token_expires_at
        integration.updated_at = datetime.utcnow()
        
        db.session.add(integration)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Token refreshed successfully',
            'expires_at': integration.token_expires_at.isoformat() if integration.token_expires_at else None
        }), 200
    except Exception as e:
        logger.error('Error saving refreshed token: %s', e)
        db.session.rollback()
        return jsonify({'error': 'Failed to save refreshed token'}), 500


@jobadder_bp.route('/integrations/jobadder/disconnect', methods=['POST'])
def disconnect():
    """Disconnect JobAdder account and clean up webhooks/action buttons."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    integration = _get_integration(user)
    if not integration:
        return jsonify({'success': True, 'message': 'Not connected'}), 200
    
    # Clean up webhooks and partner action buttons before disconnecting
    try:
        client, error, status_code = _get_client(user)
        if not error:
            # Delete all webhooks
            try:
                webhooks = client.get_webhooks()
                if webhooks and 'items' in webhooks:
                    for webhook in webhooks['items']:
                        webhook_id = webhook.get('webhookId') or webhook.get('id')
                        if webhook_id:
                            try:
                                client.delete_webhook(str(webhook_id))
                                logger.info('Deleted webhook %s during disconnect', webhook_id)
                            except JobAdderAPIError as e:
                                logger.warning('Failed to delete webhook %s: %s', webhook_id, e)
            except JobAdderAPIError as e:
                logger.warning('Failed to fetch webhooks during disconnect: %s', e)
            
            # Delete all partner action buttons
            try:
                buttons = client.get_partner_action_buttons()
                if buttons and 'items' in buttons:
                    for button in buttons['items']:
                        button_id = button.get('buttonId') or button.get('id')
                        if button_id:
                            try:
                                client.delete_partner_action_button(str(button_id))
                                logger.info('Deleted partner action button %s during disconnect', button_id)
                            except JobAdderAPIError as e:
                                logger.warning('Failed to delete partner action button %s: %s', button_id, e)
            except JobAdderAPIError as e:
                logger.warning('Failed to fetch partner action buttons during disconnect: %s', e)
    except Exception as e:
        logger.warning('Error cleaning up webhooks/buttons during disconnect: %s', e)
        # Continue with disconnect even if cleanup fails
    
    try:
        db.session.delete(integration)
        db.session.commit()
        return jsonify({'success': True, 'message': 'Disconnected successfully'}), 200
    except Exception as e:
        logger.error('Error disconnecting: %s', e)
        db.session.rollback()
        return jsonify({'error': 'Failed to disconnect'}), 500


@jobadder_bp.route('/integrations/jobadder/jobs', methods=['GET'])
def jobs():
    """Get JobAdder jobs list."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    allowed_keys = ['keywords', 'status', 'ownerId', 'requisitionId', 'companyId', 
                    'updatedFrom', 'updatedTo', 'createdFrom', 'createdTo', 'location']
    params = _build_params(request.args, allowed_keys)
    
    try:
        data = client.get_jobs(params=params)
        return jsonify({'success': True, 'results': data}), 200
    except JobAdderAPIError as e:
        logger.error('Jobs fetch failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/jobs/<job_id>', methods=['GET'])
def job_detail(job_id):
    """Get single JobAdder job."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    try:
        job = client.get_job(job_id)
        return jsonify({'success': True, 'job': job}), 200
    except JobAdderAPIError as e:
        logger.error('Job fetch failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/jobs', methods=['POST'])
def create_job():
    """Create a new JobAdder job."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400
    
    try:
        job = client.create_job(data)
        return jsonify({'success': True, 'job': job}), 201
    except JobAdderAPIError as e:
        logger.error('Job creation failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/jobs/<job_id>', methods=['PUT'])
def update_job(job_id):
    """Update a JobAdder job."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400
    
    try:
        job = client.update_job(job_id, data)
        return jsonify({'success': True, 'job': job}), 200
    except JobAdderAPIError as e:
        logger.error('Job update failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/jobs/<job_id>', methods=['DELETE'])
def delete_job(job_id):
    """Delete a JobAdder job."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    try:
        result = client.delete_job(job_id)
        return jsonify({'success': True, 'message': 'Job deleted successfully'}), 200
    except JobAdderAPIError as e:
        logger.error('Job deletion failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/candidates', methods=['GET'])
def candidates():
    """Get JobAdder candidates list."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    allowed_keys = ['status', 'workflowStatus', 'jobId', 'email', 'name', 
                    'updatedFrom', 'updatedTo']
    params = _build_params(request.args, allowed_keys)
    
    try:
        data = client.get_candidates(params=params)
        return jsonify({'success': True, 'results': data}), 200
    except JobAdderAPIError as e:
        logger.error('Candidates fetch failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/candidates/<candidate_id>', methods=['GET'])
def candidate_detail(candidate_id):
    """Get single JobAdder candidate."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    try:
        candidate = client.get_candidate(candidate_id)
        return jsonify({'success': True, 'candidate': candidate}), 200
    except JobAdderAPIError as e:
        logger.error('Candidate fetch failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/candidates', methods=['POST'])
def create_candidate():
    """Create a new JobAdder candidate."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400
    
    try:
        candidate = client.create_candidate(data)
        return jsonify({'success': True, 'candidate': candidate}), 201
    except JobAdderAPIError as e:
        logger.error('Candidate creation failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/candidates/<candidate_id>', methods=['PUT'])
def update_candidate(candidate_id):
    """Update a JobAdder candidate."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400
    
    try:
        candidate = client.update_candidate(candidate_id, data)
        return jsonify({'success': True, 'candidate': candidate}), 200
    except JobAdderAPIError as e:
        logger.error('Candidate update failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/candidates/<candidate_id>', methods=['DELETE'])
def delete_candidate(candidate_id):
    """Delete a JobAdder candidate."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    try:
        result = client.delete_candidate(candidate_id)
        return jsonify({'success': True, 'message': 'Candidate deleted successfully'}), 200
    except JobAdderAPIError as e:
        logger.error('Candidate deletion failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/applications', methods=['GET'])
def applications():
    """Get JobAdder applications list."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    allowed_keys = ['status', 'jobId', 'candidateId', 'updatedFrom', 'updatedTo']
    params = _build_params(request.args, allowed_keys)
    
    try:
        data = client.get_applications(params=params)
        return jsonify({'success': True, 'results': data}), 200
    except JobAdderAPIError as e:
        logger.error('Applications fetch failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/applications/<application_id>', methods=['GET'])
def application_detail(application_id):
    """Get single JobAdder application."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    try:
        application = client.get_application(application_id)
        return jsonify({'success': True, 'application': application}), 200
    except JobAdderAPIError as e:
        logger.error('Application fetch failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/applications', methods=['POST'])
def create_application():
    """Create a new JobAdder application."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400
    
    try:
        application = client.create_application(data)
        return jsonify({'success': True, 'application': application}), 201
    except JobAdderAPIError as e:
        logger.error('Application creation failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/applications/<application_id>', methods=['PUT'])
def update_application(application_id):
    """Update a JobAdder application."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400
    
    try:
        application = client.update_application(application_id, data)
        return jsonify({'success': True, 'application': application}), 200
    except JobAdderAPIError as e:
        logger.error('Application update failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/applications/<application_id>', methods=['DELETE'])
def delete_application(application_id):
    """Delete a JobAdder application."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    try:
        result = client.delete_application(application_id)
        return jsonify({'success': True, 'message': 'Application deleted successfully'}), 200
    except JobAdderAPIError as e:
        logger.error('Application deletion failed: %s', e)
        return jsonify({'error': str(e)}), 502


# Companies endpoints
@jobadder_bp.route('/integrations/jobadder/companies', methods=['GET'])
def companies():
    """Get JobAdder companies list."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    allowed_keys = ['keywords', 'updatedFrom', 'updatedTo', 'createdFrom', 'createdTo']
    params = _build_params(request.args, allowed_keys)
    
    try:
        data = client.get_companies(params=params)
        return jsonify({'success': True, 'results': data}), 200
    except JobAdderAPIError as e:
        logger.error('Companies fetch failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/companies/<company_id>', methods=['GET'])
def company_detail(company_id):
    """Get single JobAdder company."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    try:
        company = client.get_company(company_id)
        return jsonify({'success': True, 'company': company}), 200
    except JobAdderAPIError as e:
        logger.error('Company fetch failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/companies', methods=['POST'])
def create_company():
    """Create a new JobAdder company."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400
    
    try:
        company = client.create_company(data)
        return jsonify({'success': True, 'company': company}), 201
    except JobAdderAPIError as e:
        logger.error('Company creation failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/companies/<company_id>', methods=['PUT'])
def update_company(company_id):
    """Update a JobAdder company."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400
    
    try:
        company = client.update_company(company_id, data)
        return jsonify({'success': True, 'company': company}), 200
    except JobAdderAPIError as e:
        logger.error('Company update failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/companies/<company_id>', methods=['DELETE'])
def delete_company(company_id):
    """Delete a JobAdder company."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    try:
        result = client.delete_company(company_id)
        return jsonify({'success': True, 'message': 'Company deleted successfully'}), 200
    except JobAdderAPIError as e:
        logger.error('Company deletion failed: %s', e)
        return jsonify({'error': str(e)}), 502


# Contacts endpoints
@jobadder_bp.route('/integrations/jobadder/contacts', methods=['GET'])
def contacts():
    """Get JobAdder contacts list."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    allowed_keys = ['companyId', 'email', 'name', 'updatedFrom', 'updatedTo']
    params = _build_params(request.args, allowed_keys)
    
    try:
        data = client.get_contacts(params=params)
        return jsonify({'success': True, 'results': data}), 200
    except JobAdderAPIError as e:
        logger.error('Contacts fetch failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/contacts/<contact_id>', methods=['GET'])
def contact_detail(contact_id):
    """Get single JobAdder contact."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    try:
        contact = client.get_contact(contact_id)
        return jsonify({'success': True, 'contact': contact}), 200
    except JobAdderAPIError as e:
        logger.error('Contact fetch failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/contacts', methods=['POST'])
def create_contact():
    """Create a new JobAdder contact."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400
    
    try:
        contact = client.create_contact(data)
        return jsonify({'success': True, 'contact': contact}), 201
    except JobAdderAPIError as e:
        logger.error('Contact creation failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/contacts/<contact_id>', methods=['PUT'])
def update_contact(contact_id):
    """Update a JobAdder contact."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400
    
    try:
        contact = client.update_contact(contact_id, data)
        return jsonify({'success': True, 'contact': contact}), 200
    except JobAdderAPIError as e:
        logger.error('Contact update failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/contacts/<contact_id>', methods=['DELETE'])
def delete_contact(contact_id):
    """Delete a JobAdder contact."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    try:
        result = client.delete_contact(contact_id)
        return jsonify({'success': True, 'message': 'Contact deleted successfully'}), 200
    except JobAdderAPIError as e:
        logger.error('Contact deletion failed: %s', e)
        return jsonify({'error': str(e)}), 502


# Placements endpoints
@jobadder_bp.route('/integrations/jobadder/placements', methods=['GET'])
def placements():
    """Get JobAdder placements list."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    allowed_keys = ['status', 'jobId', 'candidateId', 'companyId', 'updatedFrom', 'updatedTo']
    params = _build_params(request.args, allowed_keys)
    
    try:
        data = client.get_placements(params=params)
        return jsonify({'success': True, 'results': data}), 200
    except JobAdderAPIError as e:
        logger.error('Placements fetch failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/placements/<placement_id>', methods=['GET'])
def placement_detail(placement_id):
    """Get single JobAdder placement."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    try:
        placement = client.get_placement(placement_id)
        return jsonify({'success': True, 'placement': placement}), 200
    except JobAdderAPIError as e:
        logger.error('Placement fetch failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/placements', methods=['POST'])
def create_placement():
    """Create a new JobAdder placement."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400
    
    try:
        placement = client.create_placement(data)
        return jsonify({'success': True, 'placement': placement}), 201
    except JobAdderAPIError as e:
        logger.error('Placement creation failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/placements/<placement_id>', methods=['PUT'])
def update_placement(placement_id):
    """Update a JobAdder placement."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400
    
    try:
        placement = client.update_placement(placement_id, data)
        return jsonify({'success': True, 'placement': placement}), 200
    except JobAdderAPIError as e:
        logger.error('Placement update failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/placements/<placement_id>', methods=['DELETE'])
def delete_placement(placement_id):
    """Delete a JobAdder placement."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    try:
        result = client.delete_placement(placement_id)
        return jsonify({'success': True, 'message': 'Placement deleted successfully'}), 200
    except JobAdderAPIError as e:
        logger.error('Placement deletion failed: %s', e)
        return jsonify({'error': str(e)}), 502


# Notes endpoints
@jobadder_bp.route('/integrations/jobadder/notes', methods=['GET'])
def notes():
    """Get JobAdder notes list."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    allowed_keys = ['jobId', 'candidateId', 'companyId', 'contactId', 'updatedFrom', 'updatedTo']
    params = _build_params(request.args, allowed_keys)
    
    try:
        data = client.get_notes(params=params)
        return jsonify({'success': True, 'results': data}), 200
    except JobAdderAPIError as e:
        logger.error('Notes fetch failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/notes/<note_id>', methods=['GET'])
def note_detail(note_id):
    """Get single JobAdder note."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    try:
        note = client.get_note(note_id)
        return jsonify({'success': True, 'note': note}), 200
    except JobAdderAPIError as e:
        logger.error('Note fetch failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/notes', methods=['POST'])
def create_note():
    """Create a new JobAdder note."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400
    
    try:
        note = client.create_note(data)
        return jsonify({'success': True, 'note': note}), 201
    except JobAdderAPIError as e:
        logger.error('Note creation failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/notes/<note_id>', methods=['PUT'])
def update_note(note_id):
    """Update a JobAdder note."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400
    
    try:
        note = client.update_note(note_id, data)
        return jsonify({'success': True, 'note': note}), 200
    except JobAdderAPIError as e:
        logger.error('Note update failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/notes/<note_id>', methods=['DELETE'])
def delete_note(note_id):
    """Delete a JobAdder note."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    try:
        result = client.delete_note(note_id)
        return jsonify({'success': True, 'message': 'Note deleted successfully'}), 200
    except JobAdderAPIError as e:
        logger.error('Note deletion failed: %s', e)
        return jsonify({'error': str(e)}), 502


# Activities endpoints
@jobadder_bp.route('/integrations/jobadder/activities', methods=['GET'])
def activities():
    """Get JobAdder activities list."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    allowed_keys = ['jobId', 'candidateId', 'companyId', 'contactId', 'updatedFrom', 'updatedTo']
    params = _build_params(request.args, allowed_keys)
    
    try:
        data = client.get_activities(params=params)
        return jsonify({'success': True, 'results': data}), 200
    except JobAdderAPIError as e:
        logger.error('Activities fetch failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/activities/<activity_id>', methods=['GET'])
def activity_detail(activity_id):
    """Get single JobAdder activity."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    try:
        activity = client.get_activity(activity_id)
        return jsonify({'success': True, 'activity': activity}), 200
    except JobAdderAPIError as e:
        logger.error('Activity fetch failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/activities', methods=['POST'])
def create_activity():
    """Create a new JobAdder activity."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400
    
    try:
        activity = client.create_activity(data)
        return jsonify({'success': True, 'activity': activity}), 201
    except JobAdderAPIError as e:
        logger.error('Activity creation failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/activities/<activity_id>', methods=['PUT'])
def update_activity(activity_id):
    """Update a JobAdder activity."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400
    
    try:
        activity = client.update_activity(activity_id, data)
        return jsonify({'success': True, 'activity': activity}), 200
    except JobAdderAPIError as e:
        logger.error('Activity update failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/activities/<activity_id>', methods=['DELETE'])
def delete_activity(activity_id):
    """Delete a JobAdder activity."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    try:
        result = client.delete_activity(activity_id)
        return jsonify({'success': True, 'message': 'Activity deleted successfully'}), 200
    except JobAdderAPIError as e:
        logger.error('Activity deletion failed: %s', e)
        return jsonify({'error': str(e)}), 502


# Tasks endpoints
@jobadder_bp.route('/integrations/jobadder/tasks', methods=['GET'])
def tasks():
    """Get JobAdder tasks list."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    allowed_keys = ['status', 'assignedTo', 'jobId', 'candidateId', 'companyId', 'updatedFrom', 'updatedTo']
    params = _build_params(request.args, allowed_keys)
    
    try:
        data = client.get_tasks(params=params)
        return jsonify({'success': True, 'results': data}), 200
    except JobAdderAPIError as e:
        logger.error('Tasks fetch failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/tasks/<task_id>', methods=['GET'])
def task_detail(task_id):
    """Get single JobAdder task."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    try:
        task = client.get_task(task_id)
        return jsonify({'success': True, 'task': task}), 200
    except JobAdderAPIError as e:
        logger.error('Task fetch failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/tasks', methods=['POST'])
def create_task():
    """Create a new JobAdder task."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400
    
    try:
        task = client.create_task(data)
        return jsonify({'success': True, 'task': task}), 201
    except JobAdderAPIError as e:
        logger.error('Task creation failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/tasks/<task_id>', methods=['PUT'])
def update_task(task_id):
    """Update a JobAdder task."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400
    
    try:
        task = client.update_task(task_id, data)
        return jsonify({'success': True, 'task': task}), 200
    except JobAdderAPIError as e:
        logger.error('Task update failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/tasks/<task_id>', methods=['DELETE'])
def delete_task(task_id):
    """Delete a JobAdder task."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    try:
        result = client.delete_task(task_id)
        return jsonify({'success': True, 'message': 'Task deleted successfully'}), 200
    except JobAdderAPIError as e:
        logger.error('Task deletion failed: %s', e)
        return jsonify({'error': str(e)}), 502


# Users endpoints
@jobadder_bp.route('/integrations/jobadder/users', methods=['GET'])
def users():
    """Get JobAdder users list."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    params = _build_params(request.args, [])
    
    try:
        data = client.get_users(params=params)
        return jsonify({'success': True, 'results': data}), 200
    except JobAdderAPIError as e:
        logger.error('Users fetch failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/users/<user_id>', methods=['GET'])
def user_detail(user_id):
    """Get single JobAdder user."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    try:
        user_data = client.get_user(user_id)
        return jsonify({'success': True, 'user': user_data}), 200
    except JobAdderAPIError as e:
        logger.error('User fetch failed: %s', e)
        return jsonify({'error': str(e)}), 502


# Workflows endpoints
@jobadder_bp.route('/integrations/jobadder/workflows', methods=['GET'])
def workflows():
    """Get JobAdder workflows list."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    params = _build_params(request.args, [])
    
    try:
        data = client.get_workflows(params=params)
        return jsonify({'success': True, 'results': data}), 200
    except JobAdderAPIError as e:
        logger.error('Workflows fetch failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/workflows/<workflow_id>', methods=['GET'])
def workflow_detail(workflow_id):
    """Get single JobAdder workflow."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    try:
        workflow = client.get_workflow(workflow_id)
        return jsonify({'success': True, 'workflow': workflow}), 200
    except JobAdderAPIError as e:
        logger.error('Workflow fetch failed: %s', e)
        return jsonify({'error': str(e)}), 502


# Custom Fields endpoints
@jobadder_bp.route('/integrations/jobadder/customfields', methods=['GET'])
def custom_fields():
    """Get JobAdder custom fields list."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    allowed_keys = ['entityType']
    params = _build_params(request.args, allowed_keys)
    
    try:
        data = client.get_custom_fields(params=params)
        return jsonify({'success': True, 'results': data}), 200
    except JobAdderAPIError as e:
        logger.error('Custom fields fetch failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/customfields/<custom_field_id>', methods=['GET'])
def custom_field_detail(custom_field_id):
    """Get single JobAdder custom field."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    try:
        custom_field = client.get_custom_field(custom_field_id)
        return jsonify({'success': True, 'customField': custom_field}), 200
    except JobAdderAPIError as e:
        logger.error('Custom field fetch failed: %s', e)
        return jsonify({'error': str(e)}), 502


# Requisitions endpoints
@jobadder_bp.route('/integrations/jobadder/requisitions', methods=['GET'])
def requisitions():
    """Get JobAdder requisitions list."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    allowed_keys = ['status', 'companyId', 'updatedFrom', 'updatedTo']
    params = _build_params(request.args, allowed_keys)
    
    try:
        data = client.get_requisitions(params=params)
        return jsonify({'success': True, 'results': data}), 200
    except JobAdderAPIError as e:
        logger.error('Requisitions fetch failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/requisitions/<requisition_id>', methods=['GET'])
def requisition_detail(requisition_id):
    """Get single JobAdder requisition."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    try:
        requisition = client.get_requisition(requisition_id)
        return jsonify({'success': True, 'requisition': requisition}), 200
    except JobAdderAPIError as e:
        logger.error('Requisition fetch failed: %s', e)
        return jsonify({'error': str(e)}), 502


# Job Boards endpoints
@jobadder_bp.route('/integrations/jobadder/jobboards', methods=['GET'])
def job_boards():
    """Get JobAdder job boards list."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    params = _build_params(request.args, [])
    
    try:
        data = client.get_job_boards(params=params)
        return jsonify({'success': True, 'results': data}), 200
    except JobAdderAPIError as e:
        logger.error('Job boards fetch failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/jobboards/<job_board_id>', methods=['GET'])
def job_board_detail(job_board_id):
    """Get single JobAdder job board."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    try:
        job_board = client.get_job_board(job_board_id)
        return jsonify({'success': True, 'jobBoard': job_board}), 200
    except JobAdderAPIError as e:
        logger.error('Job board fetch failed: %s', e)
        return jsonify({'error': str(e)}), 502


# Webhooks endpoints
@jobadder_bp.route('/integrations/jobadder/webhooks', methods=['GET'])
def webhooks():
    """Get JobAdder webhooks list."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    params = _build_params(request.args, [])
    
    try:
        data = client.get_webhooks(params=params)
        return jsonify({'success': True, 'results': data}), 200
    except JobAdderAPIError as e:
        logger.error('Webhooks fetch failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/webhooks/<webhook_id>', methods=['GET'])
def webhook_detail(webhook_id):
    """Get single JobAdder webhook."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    try:
        webhook = client.get_webhook(webhook_id)
        return jsonify({'success': True, 'webhook': webhook}), 200
    except JobAdderAPIError as e:
        logger.error('Webhook fetch failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/webhooks', methods=['POST'])
def create_webhook():
    """Create a new JobAdder webhook."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400
    
    try:
        webhook = client.create_webhook(data)
        return jsonify({'success': True, 'webhook': webhook}), 201
    except JobAdderAPIError as e:
        logger.error('Webhook creation failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/webhooks/<webhook_id>', methods=['PUT'])
def update_webhook(webhook_id):
    """Update a JobAdder webhook."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400
    
    try:
        webhook = client.update_webhook(webhook_id, data)
        return jsonify({'success': True, 'webhook': webhook}), 200
    except JobAdderAPIError as e:
        logger.error('Webhook update failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/webhooks/<webhook_id>', methods=['DELETE'])
def delete_webhook(webhook_id):
    """Delete a JobAdder webhook."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    try:
        result = client.delete_webhook(webhook_id)
        return jsonify({'success': True, 'message': 'Webhook deleted successfully'}), 200
    except JobAdderAPIError as e:
        logger.error('Webhook deletion failed: %s', e)
        return jsonify({'error': str(e)}), 502


# Partner Action Buttons endpoints
@jobadder_bp.route('/integrations/jobadder/partneractionbuttons', methods=['GET'])
def partner_action_buttons():
    """Get JobAdder partner action buttons list."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    params = _build_params(request.args, [])
    
    try:
        data = client.get_partner_action_buttons(params=params)
        return jsonify({'success': True, 'results': data}), 200
    except JobAdderAPIError as e:
        logger.error('Partner action buttons fetch failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/partneractionbuttons/<button_id>', methods=['GET'])
def partner_action_button_detail(button_id):
    """Get single JobAdder partner action button."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    try:
        button = client.get_partner_action_button(button_id)
        return jsonify({'success': True, 'button': button}), 200
    except JobAdderAPIError as e:
        logger.error('Partner action button fetch failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/partneractionbuttons', methods=['POST'])
def create_partner_action_button():
    """Create a new JobAdder partner action button."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400
    
    try:
        button = client.create_partner_action_button(data)
        return jsonify({'success': True, 'button': button}), 201
    except JobAdderAPIError as e:
        logger.error('Partner action button creation failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/partneractionbuttons/<button_id>', methods=['PUT'])
def update_partner_action_button(button_id):
    """Update a JobAdder partner action button."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Request body is required'}), 400
    
    try:
        button = client.update_partner_action_button(button_id, data)
        return jsonify({'success': True, 'button': button}), 200
    except JobAdderAPIError as e:
        logger.error('Partner action button update failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/partneractionbuttons/<button_id>', methods=['DELETE'])
def delete_partner_action_button(button_id):
    """Delete a JobAdder partner action button."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    try:
        result = client.delete_partner_action_button(button_id)
        return jsonify({'success': True, 'message': 'Partner action button deleted successfully'}), 200
    except JobAdderAPIError as e:
        logger.error('Partner action button deletion failed: %s', e)
        return jsonify({'error': str(e)}), 502


# File/Attachment operations
@jobadder_bp.route('/integrations/jobadder/<resource_type>/<resource_id>/attachments', methods=['POST'])
def upload_file(resource_type, resource_id):
    """Upload a file attachment to a JobAdder resource."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        file_data = file.read()
        content_type = file.content_type or 'application/octet-stream'
        result = client.upload_file(resource_type, resource_id, file.filename, file_data, content_type)
        return jsonify({'success': True, 'attachment': result}), 201
    except JobAdderAPIError as e:
        logger.error('File upload failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/<resource_type>/<resource_id>/attachments/<attachment_id>', methods=['GET'])
def download_file(resource_type, resource_id, attachment_id):
    """Download a file attachment from a JobAdder resource."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    try:
        file_data = client.get_file(resource_type, resource_id, attachment_id)
        return Response(
            file_data,
            mimetype='application/octet-stream',
            headers={'Content-Disposition': f'attachment; filename=attachment_{attachment_id}'}
        )
    except JobAdderAPIError as e:
        logger.error('File download failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/<resource_type>/<resource_id>/attachments/<attachment_id>', methods=['DELETE'])
def delete_file(resource_type, resource_id, attachment_id):
    """Delete a file attachment from a JobAdder resource."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    client, error, status_code = _get_client(user)
    if error:
        return error, status_code
    
    try:
        result = client.delete_file(resource_type, resource_id, attachment_id)
        return jsonify({'success': True, 'message': 'File deleted successfully'}), 200
    except JobAdderAPIError as e:
        logger.error('File deletion failed: %s', e)
        return jsonify({'error': str(e)}), 502


@jobadder_bp.route('/integrations/jobadder/oauth/callback', methods=['GET'])
def jobadder_oauth_callback():
    """Handle JobAdder OAuth2 callback."""
    code = request.args.get('code')
    state_token = request.args.get('state')
    oauth_error = request.args.get('error')
    error_description = request.args.get('error_description')

    frontend_path = os.getenv('JOBADDER_OAUTH_CALLBACK_REDIRECT', '/integrations/jobadder/connect')

    def redirect_with_status(status, message=None):
        params = {'status': status}
        if message:
            params['message'] = message
        return redirect(f"{_get_frontend_redirect_base()}{frontend_path}?{urlencode(params)}")

    # Log all callback parameters for debugging
    logger.info("JobAdder OAuth callback received: code=%s, state=%s, error=%s", 
                'present' if code else 'missing', 
                'present' if state_token else 'missing',
                oauth_error or 'none')

    if oauth_error:
        # Map common OAuth errors to more helpful, user‑friendly messages
        friendly_message = error_description or oauth_error

        if oauth_error == 'invalid_scope':
            # Required: Log full details for invalid_scope errors
            # Try to reconstruct the authorization URL from available data
            attempted_scope = os.getenv('JOBADDER_DEFAULT_SCOPE', 'read write offline_access')
            redirect_uri = _get_redirect_uri()
            timestamp = datetime.utcnow().isoformat()
            
            # Try to get client_id and state from the callback
            client_id = None
            state_value = state_token if state_token else 'unknown'
            
            # Try to load state to get user info and reconstruct URL
            try:
                if state_token:
                    state_payload = _load_state(state_token)
                    user_id = state_payload.get('user_id')
                    if user_id:
                        user = User.query.get(user_id)
                        if user:
                            integration = _get_integration(user)
                            if integration:
                                client_id = integration.client_id
                                # Reconstruct the full authorization URL
                                auth = JobAdderAuth(integration.client_id, 'dummy', scope=attempted_scope)
                                auth_params = {
                                    'response_type': 'code',
                                    'client_id': client_id,
                                    'redirect_uri': redirect_uri,
                                    'scope': attempted_scope,
                                    'state': state_value,
                                    'prompt': 'consent',
                                }
                                full_auth_url = f"{auth.authorize_url}?{urlencode(auth_params)}"
                            else:
                                full_auth_url = f"https://id.jobadder.com/oauth2/authorize?response_type=code&client_id={client_id or 'unknown'}&redirect_uri={redirect_uri}&scope={attempted_scope}&state={state_value}&prompt=consent"
                        else:
                            full_auth_url = f"https://id.jobadder.com/oauth2/authorize?response_type=code&client_id={client_id or 'unknown'}&redirect_uri={redirect_uri}&scope={attempted_scope}&state={state_value}&prompt=consent"
                    else:
                        full_auth_url = f"https://id.jobadder.com/oauth2/authorize?response_type=code&client_id={client_id or 'unknown'}&redirect_uri={redirect_uri}&scope={attempted_scope}&state={state_value}&prompt=consent"
                else:
                    full_auth_url = f"https://id.jobadder.com/oauth2/authorize?response_type=code&client_id={client_id or 'unknown'}&redirect_uri={redirect_uri}&scope={attempted_scope}&state={state_value or 'unknown'}&prompt=consent"
            except Exception as e:
                logger.exception("Error reconstructing auth URL for invalid_scope logging: %s", e)
                full_auth_url = f"https://id.jobadder.com/oauth2/authorize?response_type=code&client_id={client_id or 'unknown'}&redirect_uri={redirect_uri}&scope={attempted_scope}&state={state_value or 'unknown'}&prompt=consent"
            
            # Required logging for invalid_scope error
            logger.error(
                "JobAdder OAuth invalid_scope error - Full details:\n"
                "  Full Authorize URL: %s\n"
                "  Requested Scope: %s\n"
                "  State Value: %s\n"
                "  Timestamp: %s\n"
                "  Redirect URI: %s\n"
                "  Client ID: %s\n"
                "  Error Description: %s",
                full_auth_url,
                attempted_scope,
                state_value,
                timestamp,
                redirect_uri,
                client_id or 'unknown',
                error_description or 'none'
            )
            
            friendly_message = (
                f"JobAdder rejected the requested API scopes (invalid_scope). "
                f"Attempted scope: '{attempted_scope}'\n\n"
                "This usually means your JobAdder app is not configured to allow these scopes. "
                "Please ensure your JobAdder app allows 'read write offline_access' scopes."
            )

        logger.warning("JobAdder OAuth error: %s - %s", oauth_error, error_description)
        return redirect_with_status('error', friendly_message)

    if not code or not state_token:
        logger.error("JobAdder OAuth callback missing code or state: code=%s, state=%s", 
                     'present' if code else 'missing',
                     'present' if state_token else 'missing')
        return redirect_with_status('error', 'Missing code or state in callback')

    try:
        state_payload = _load_state(state_token)
        user_id = state_payload.get('user_id')
    except (BadSignature, SignatureExpired):
        logger.error("Invalid or expired JobAdder OAuth state token")
        return redirect_with_status('error', 'Invalid or expired authorization state')

    if not user_id:
        logger.error("JobAdder OAuth state missing user_id")
        return redirect_with_status('error', 'Invalid authorization state')

    user = User.query.get(user_id)
    if not user:
        logger.error("JobAdder OAuth user not found for id=%s", user_id)
        return redirect_with_status('error', 'User not found')

    integration = _get_integration(user)
    if not integration or not integration.client_secret:
        logger.error("JobAdder OAuth integration missing for user_id=%s", user_id)
        return redirect_with_status('error', 'JobAdder integration not initialized')

    try:
        decoded_secret = base64.b64decode(integration.client_secret.encode()).decode()
    except Exception:
        logger.exception("Failed to decode JobAdder client secret for user_id=%s", user_id)
        return redirect_with_status('error', 'Invalid JobAdder credentials')

    redirect_uri = _get_redirect_uri()
    logger.info("Exchanging authorization code: redirect_uri=%s, token_url=%s", 
                redirect_uri, os.getenv("JOBADDER_TOKEN_URL", "https://id.jobadder.com/oauth2/token"))
    
    auth = JobAdderAuth(integration.client_id, decoded_secret)
    success, token_data = auth.exchange_authorization_code(code, redirect_uri)

    if not success:
        logger.error("JobAdder authorization code exchange failed: %s (redirect_uri=%s)", 
                     token_data, redirect_uri)
        return redirect_with_status('error', token_data or 'Failed to exchange authorization code')

    access_token = token_data.get('access_token')
    refresh_token = token_data.get('refresh_token')

    account_success, account_info = auth.get_account_info(access_token)
    if not account_success:
        logger.warning("JobAdder account info fetch failed: %s", account_info)
        account_info = {}

    try:
        integration.access_token = access_token
        integration.refresh_token = refresh_token
        integration.token_expires_at = auth.token_expires_at
        integration.account_name = account_info.get('name') or account_info.get('displayName') or 'JobAdder Account'
        integration.account_email = account_info.get('email')
        integration.account_user_id = str(account_info.get('userId') or account_info.get('id') or '')
        integration.account_company_id = str(account_info.get('companyId') or account_info.get('company_id') or '')
        integration.updated_at = datetime.utcnow()

        if not integration.created_at:
            integration.created_at = datetime.utcnow()

        db.session.add(integration)
        db.session.commit()
    except Exception as exc:
        logger.exception("Failed to persist JobAdder tokens: %s", exc)
        db.session.rollback()
        return redirect_with_status('error', 'Failed to store JobAdder tokens')

    return redirect_with_status('success')
