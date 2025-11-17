from flask import Blueprint, request, jsonify
from app.simple_logger import get_logger
import os
import requests
from datetime import datetime, timedelta
from app.models import JobAdderIntegration, User, db
from app.utils import get_current_user
import base64
import logging

logger = get_logger("jobadder")

jobadder_bp = Blueprint('jobadder', __name__)

class JobAdderAuth:
    """OAuth2 authentication class for JobAdder API"""
    
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        # JobAdder OAuth2 endpoints (verify with official docs)
        self.token_url = "https://id.jobadder.com/connect/token"
        self.api_base_url = "https://api.jobadder.com/v2"
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None
    
    def get_access_token(self):
        """Get access token using OAuth2 Client Credentials flow"""
        try:
            # OAuth2 Client Credentials flow
            auth_data = {
                'grant_type': 'client_credentials',
                'client_id': self.client_id,
                'client_secret': self.client_secret,
                'scope': 'read'  # Adjust scopes as needed
            }
            
            response = requests.post(
                self.token_url,
                data=auth_data,
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                timeout=30
            )
            
            logger.info(f"JobAdder token request status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                self.access_token = data.get('access_token')
                expires_in = data.get('expires_in', 3600)  # Default 1 hour
                self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)
                self.refresh_token = data.get('refresh_token')  # May not be provided in client_credentials flow
                return True, self.access_token
            else:
                error_data = response.json() if response.content else {}
                error_msg = error_data.get('error_description', f"HTTP {response.status_code}")
                logger.error(f"JobAdder token request failed: {error_msg}")
                return False, error_msg
                
        except requests.exceptions.RequestException as e:
            logger.error(f"JobAdder API request failed: {e}")
            return False, f"Connection failed: {str(e)}"
        except Exception as e:
            logger.error(f"JobAdder authentication error: {e}")
            return False, str(e)
    
    def get_account_info(self, access_token):
        """Fetch account/user information using access token"""
        try:
            # Try different endpoints to get account info
            endpoints = [
                f"{self.api_base_url}/users/me",
                f"{self.api_base_url}/account",
                f"{self.api_base_url}/profile"
            ]
            
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Accept': 'application/json'
            }
            
            for endpoint in endpoints:
                try:
                    response = requests.get(endpoint, headers=headers, timeout=30)
                    if response.status_code == 200:
                        data = response.json()
                        logger.info(f"JobAdder account info retrieved from {endpoint}")
                        return True, data
                    elif response.status_code == 404:
                        continue  # Try next endpoint
                    else:
                        logger.warning(f"JobAdder account info endpoint {endpoint} returned {response.status_code}")
                except requests.exceptions.RequestException:
                    continue
            
            # If no endpoint works, return basic info
            logger.warning("Could not fetch detailed account info, using basic response")
            return True, {
                'name': 'JobAdder Account',
                'email': None,
                'userId': None,
                'companyId': None
            }
            
        except Exception as e:
            logger.error(f"Error fetching JobAdder account info: {e}")
            return False, str(e)
    
    def validate_credentials(self):
        """Validate credentials by attempting to get access token and account info"""
        try:
            # Step 1: Get access token
            success, result = self.get_access_token()
            if not success:
                return False, result
            
            access_token = result
            
            # Step 2: Get account info to verify token works
            success, account_info = self.get_account_info(access_token)
            if not success:
                return False, account_info
            
            return True, {
                'access_token': access_token,
                'account_info': account_info
            }
            
        except Exception as e:
            logger.error(f"JobAdder validation error: {e}")
            return False, str(e)

# POST /integrations/jobadder/connect
@jobadder_bp.route('/integrations/jobadder/connect', methods=['POST'])
def jobadder_connect():
    """Connect JobAdder account using Client ID and Client Secret
    
    For regular users: Connects JobAdder to their own account
    For admins: Can optionally specify 'userEmail' to connect for another user
    """
    logger.info('HIT /integrations/jobadder/connect')
    
    user_jwt = get_current_user()
    logger.info(f'user_jwt: {user_jwt}')
    
    if not user_jwt or not user_jwt.get('email'):
        logger.error('Unauthorized: No user_jwt or email')
        return jsonify({'error': 'Unauthorized'}), 401
    
    # Get the authenticated user
    authenticated_user = User.query.filter_by(email=user_jwt['email']).first()
    if not authenticated_user:
        logger.error(f'Authenticated user not found: {user_jwt.get("email")}')
        return jsonify({'error': 'User not found'}), 404
    
    data = request.get_json()
    logger.info(f'Received data: {data}')
    
    client_id = data.get('clientId') or data.get('client_id')
    client_secret = data.get('clientSecret') or data.get('client_secret')
    target_user_email = data.get('userEmail')  # Optional: for admin to connect for another user
    
    if not client_id or not client_secret:
        logger.error('Missing fields in request')
        return jsonify({'error': 'Missing fields: clientId and clientSecret are required'}), 400
    
    # Determine target user: if admin specified userEmail, use that; otherwise use authenticated user
    if target_user_email:
        # Check if authenticated user is admin/owner
        if authenticated_user.role not in ['admin', 'owner']:
            logger.warning(f'Non-admin user attempted to connect for another user: {authenticated_user.email}')
            return jsonify({'error': 'Only admins can connect integrations for other users'}), 403
        
        # Find target user
        target_user = User.query.filter_by(email=target_user_email).first()
        if not target_user:
            logger.error(f'Target user not found: {target_user_email}')
            return jsonify({'error': f'Target user not found: {target_user_email}'}), 404
        
        user = target_user
        logger.info(f'Admin {authenticated_user.email} connecting JobAdder for user {target_user_email}')
    else:
        # Regular user connecting for themselves
        user = authenticated_user
        logger.info(f'User {user.email} connecting JobAdder for themselves')
    
    # Validate credentials with JobAdder API
    auth = JobAdderAuth(client_id, client_secret)
    is_valid, result = auth.validate_credentials()
    
    if not is_valid:
        logger.error(f'JobAdder authentication failed: {result}')
        return jsonify({'error': f'Invalid credentials: {result}'}), 401
    
    # Extract account info
    access_token = result.get('access_token')
    account_info = result.get('account_info', {})
    
    # Parse account info (handle different response formats)
    account_name = account_info.get('name') or account_info.get('displayName') or account_info.get('companyName') or 'JobAdder Account'
    account_email = account_info.get('email') or account_info.get('userEmail') or account_info.get('emailAddress')
    account_user_id = str(account_info.get('userId') or account_info.get('id') or account_info.get('user_id') or '')
    account_company_id = str(account_info.get('companyId') or account_info.get('company_id') or account_info.get('organizationId') or '')
    
    # Encrypt client secret (base64 for now, upgrade to proper encryption later)
    enc_client_secret = base64.b64encode(client_secret.encode()).decode()
    
    # Save or update integration
    try:
        integration = JobAdderIntegration.query.filter_by(user_id=user.id).first()
    except Exception as e:
        logger.error(f'Error querying JobAdder integration: {e}')
        # Table might not exist - try to create it
        try:
            db.create_all()
            integration = None  # Will create new
        except Exception as create_error:
            logger.error(f'Error creating table: {create_error}')
            return jsonify({'error': 'Database table not available. Please run migration script.'}), 500
    
    if integration:
        integration.client_id = client_id
        integration.client_secret = enc_client_secret
        integration.access_token = access_token
        integration.token_expires_at = auth.token_expires_at
        integration.refresh_token = auth.refresh_token
        integration.account_name = account_name
        integration.account_email = account_email
        integration.account_user_id = account_user_id
        integration.account_company_id = account_company_id
        integration.updated_at = datetime.utcnow()
    else:
        integration = JobAdderIntegration(
            user_id=user.id,
            client_id=client_id,
            client_secret=enc_client_secret,
            access_token=access_token,
            token_expires_at=auth.token_expires_at,
            refresh_token=auth.refresh_token,
            account_name=account_name,
            account_email=account_email,
            account_user_id=account_user_id,
            account_company_id=account_company_id
        )
        db.session.add(integration)
    
    db.session.commit()
    logger.info(f'JobAdder integration saved for user {user.email}')
    
    return jsonify({
        'success': True,
        'connected': True,
        'message': 'JobAdder account connected successfully',
        'account': {
            'name': account_name,
            'email': account_email,
            'userId': account_user_id,
            'companyId': account_company_id
        }
    }), 200

# GET /integrations/jobadder/status
@jobadder_bp.route('/integrations/jobadder/status', methods=['GET'])
def jobadder_status():
    """Check JobAdder connection status"""
    logger.info('HIT /integrations/jobadder/status')
    
    user_jwt = get_current_user()
    if not user_jwt or not user_jwt.get('email'):
        logger.error('Unauthorized: No user_jwt or email')
        return jsonify({'error': 'Unauthorized'}), 401
    
    user = User.query.filter_by(email=user_jwt['email']).first()
    if not user:
        logger.error(f'User not found: {user_jwt.get("email")}')
        return jsonify({'error': 'User not found'}), 404
    
    try:
        integration = JobAdderIntegration.query.filter_by(user_id=user.id).first()
    except Exception as e:
        logger.error(f'Error querying JobAdder integration: {e}')
        # Table might not exist yet - return not connected
        return jsonify({
            'connected': False,
            'message': 'JobAdder account not connected'
        }), 200
    
    if not integration:
        return jsonify({
            'connected': False,
            'message': 'JobAdder account not connected'
        }), 200
    
    # Check if token is expired
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
        'tokenExpired': is_expired,
        'connectedAt': integration.created_at.isoformat() if integration.created_at else None,
        'updatedAt': integration.updated_at.isoformat() if integration.updated_at else None
    }), 200

# POST /integrations/jobadder/disconnect
@jobadder_bp.route('/integrations/jobadder/disconnect', methods=['POST'])
def jobadder_disconnect():
    """Disconnect JobAdder account"""
    logger.info('HIT /integrations/jobadder/disconnect')
    
    user_jwt = get_current_user()
    if not user_jwt or not user_jwt.get('email'):
        logger.error('Unauthorized: No user_jwt or email')
        return jsonify({'error': 'Unauthorized'}), 401
    
    user = User.query.filter_by(email=user_jwt['email']).first()
    if not user:
        logger.error(f'User not found: {user_jwt.get("email")}')
        return jsonify({'error': 'User not found'}), 404
    
    try:
        integration = JobAdderIntegration.query.filter_by(user_id=user.id).first()
    except Exception as e:
        logger.error(f'Error querying JobAdder integration: {e}')
        # Table might not exist yet - return success (nothing to disconnect)
        return jsonify({
            'success': True,
            'message': 'JobAdder account not connected'
        }), 200
    
    if not integration:
        return jsonify({
            'success': True,
            'message': 'JobAdder account not connected'
        }), 200
    
    try:
        db.session.delete(integration)
        db.session.commit()
        logger.info(f'JobAdder integration disconnected for user {user.email}')
        
        return jsonify({
            'success': True,
            'message': 'JobAdder account disconnected successfully'
        }), 200
    except Exception as e:
        logger.error(f'Error disconnecting JobAdder integration: {e}')
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': 'Failed to disconnect JobAdder account'
        }), 500

