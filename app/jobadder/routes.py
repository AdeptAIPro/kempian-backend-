from flask import Blueprint, request, jsonify, redirect, url_for, current_app
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
    custom = os.getenv('JOBADDER_OAUTH_REDIRECT_URI')
    if custom:
        return custom
    return url_for('jobadder.jobadder_oauth_callback', _external=True)


def _get_frontend_redirect_base():
    return os.getenv('FRONTEND_URL', 'http://localhost:5173').rstrip('/')


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
    scope = data.get('scope') or os.getenv('JOBADDER_DEFAULT_SCOPE', 'jobadder.api offline_access')
    
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
    auth_params = {
        'response_type': 'code',
        'client_id': client_id,
        'redirect_uri': redirect_uri,
        'scope': scope,
        'state': state,
    }
    auth_url = f"{auth.authorize_url}?{urlencode(auth_params)}"
    
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
    if not integration:
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


@jobadder_bp.route('/integrations/jobadder/disconnect', methods=['POST'])
def disconnect():
    """Disconnect JobAdder account."""
    user, error, status_code = _get_user()
    if error:
        return error, status_code
    
    integration = _get_integration(user)
    if not integration:
        return jsonify({'success': True, 'message': 'Not connected'}), 200
    
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

    if oauth_error:
        logger.warning("JobAdder OAuth error: %s - %s", oauth_error, error_description)
        return redirect_with_status('error', error_description or oauth_error)

    if not code or not state_token:
        logger.error("JobAdder OAuth callback missing code or state")
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
    auth = JobAdderAuth(integration.client_id, decoded_secret)
    success, token_data = auth.exchange_authorization_code(code, redirect_uri)

    if not success:
        logger.error("JobAdder authorization code exchange failed: %s", token_data)
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
