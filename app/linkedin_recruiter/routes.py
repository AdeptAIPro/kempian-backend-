from flask import Blueprint, request, jsonify
from app.simple_logger import get_logger
import os
import requests
from datetime import datetime, timedelta
from app.models import LinkedInRecruiterIntegration, User, db
from app.utils import get_current_user
import base64
import logging

logger = get_logger("linkedin_recruiter")

linkedin_recruiter_bp = Blueprint('linkedin_recruiter', __name__)

class LinkedInRecruiterAuth:
    """OAuth2 authentication class for LinkedIn Recruiter System Connect API"""
    
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        # LinkedIn OAuth2 endpoints for Recruiter System Connect
        self.token_url = "https://www.linkedin.com/oauth/v2/accessToken"
        self.api_base_url = "https://api.linkedin.com/v2"
        self.access_token = None
        self.refresh_token = None
        self.token_expires_at = None
    
    def get_access_token(self):
        """Get access token using OAuth2 Client Credentials flow"""
        try:
            # OAuth2 Client Credentials flow for LinkedIn Recruiter
            auth_data = {
                'grant_type': 'client_credentials',
                'client_id': self.client_id,
                'client_secret': self.client_secret
            }
            
            response = requests.post(
                self.token_url,
                data=auth_data,
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                timeout=30
            )
            
            logger.info(f"LinkedIn Recruiter token request status: {response.status_code}")
            
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
                logger.error(f"LinkedIn Recruiter token request failed: {error_msg}")
                return False, error_msg
                
        except requests.exceptions.RequestException as e:
            logger.error(f"LinkedIn Recruiter API request failed: {e}")
            return False, f"Connection failed: {str(e)}"
        except Exception as e:
            logger.error(f"LinkedIn Recruiter authentication error: {e}")
            return False, str(e)
    
    def get_account_info(self, access_token, company_id=None):
        """Fetch account/organization information using access token"""
        try:
            # Try different endpoints to get account info
            endpoints = []
            
            # Try to get organization info if company_id is provided
            if company_id:
                endpoints.append(f"{self.api_base_url}/organizations/{company_id}")
            
            # Try user profile endpoint
            endpoints.extend([
                f"{self.api_base_url}/userinfo",
                f"{self.api_base_url}/me",
                f"{self.api_base_url}/people/~"
            ])
            
            headers = {
                'Authorization': f'Bearer {access_token}',
                'Accept': 'application/json'
            }
            
            account_info = {}
            
            for endpoint in endpoints:
                try:
                    response = requests.get(endpoint, headers=headers, timeout=30)
                    if response.status_code == 200:
                        data = response.json()
                        logger.info(f"LinkedIn Recruiter account info retrieved from {endpoint}")
                        # Merge data into account_info
                        if isinstance(data, dict):
                            account_info.update(data)
                    elif response.status_code == 404:
                        continue  # Try next endpoint
                    else:
                        logger.warning(f"LinkedIn Recruiter account info endpoint {endpoint} returned {response.status_code}")
                except requests.exceptions.RequestException:
                    continue
            
            # If no endpoint works, return basic info
            if not account_info:
                logger.warning("Could not fetch detailed account info, using basic response")
                return True, {
                    'name': 'LinkedIn Recruiter Account',
                    'email': None,
                    'userId': None,
                    'organizationId': None
                }
            
            return True, account_info
            
        except Exception as e:
            logger.error(f"Error fetching LinkedIn Recruiter account info: {e}")
            return False, str(e)
    
    def validate_credentials(self, company_id=None):
        """Validate credentials by attempting to get access token and account info"""
        try:
            # Step 1: Get access token
            success, result = self.get_access_token()
            if not success:
                return False, result
            
            access_token = result
            
            # Step 2: Get account info to verify token works
            success, account_info = self.get_account_info(access_token, company_id)
            if not success:
                return False, account_info
            
            return True, {
                'access_token': access_token,
                'account_info': account_info
            }
            
        except Exception as e:
            logger.error(f"LinkedIn Recruiter validation error: {e}")
            return False, str(e)

# POST /integrations/linkedin-recruiter/connect
@linkedin_recruiter_bp.route('/integrations/linkedin-recruiter/connect', methods=['POST'])
def linkedin_recruiter_connect():
    """Connect LinkedIn Recruiter account using Client ID, Client Secret, Company ID, and Contract ID"""
    logger.info('HIT /integrations/linkedin-recruiter/connect')
    
    user_jwt = get_current_user()
    logger.info(f'user_jwt: {user_jwt}')
    
    if not user_jwt or not user_jwt.get('email'):
        logger.error('Unauthorized: No user_jwt or email')
        return jsonify({'error': 'Unauthorized'}), 401
    
    user = User.query.filter_by(email=user_jwt['email']).first()
    if not user:
        logger.error(f'User not found: {user_jwt.get("email")}')
        return jsonify({'error': 'User not found'}), 404
    
    data = request.get_json()
    logger.info(f'Received data: {data}')
    
    client_id = data.get('clientId') or data.get('client_id')
    client_secret = data.get('clientSecret') or data.get('client_secret')
    company_id = data.get('companyId') or data.get('company_id')
    contract_id = data.get('contractId') or data.get('contract_id')
    
    if not client_id or not client_secret:
        logger.error('Missing fields in request')
        return jsonify({'error': 'Missing fields: clientId and clientSecret are required'}), 400
    
    if not company_id or not contract_id:
        logger.error('Missing fields in request')
        return jsonify({'error': 'Missing fields: companyId and contractId are required'}), 400
    
    # Validate credentials with LinkedIn Recruiter API
    auth = LinkedInRecruiterAuth(client_id, client_secret)
    is_valid, result = auth.validate_credentials(company_id)
    
    if not is_valid:
        logger.error(f'LinkedIn Recruiter authentication failed: {result}')
        return jsonify({'error': f'Invalid credentials: {result}'}), 401
    
    # Extract account info
    access_token = result.get('access_token')
    account_info = result.get('account_info', {})
    
    # Parse account info (handle different response formats)
    account_name = account_info.get('name') or account_info.get('localizedName') or account_info.get('displayName') or account_info.get('companyName') or 'LinkedIn Recruiter Account'
    account_email = account_info.get('email') or account_info.get('emailAddress') or account_info.get('primaryContactEmail')
    account_user_id = str(account_info.get('userId') or account_info.get('id') or account_info.get('user_id') or account_info.get('sub') or '')
    account_organization_id = str(account_info.get('organizationId') or account_info.get('organization_id') or account_info.get('orgId') or company_id or '')
    
    # Encrypt client secret (base64 for now, upgrade to proper encryption later)
    enc_client_secret = base64.b64encode(client_secret.encode()).decode()
    
    # Save or update integration
    try:
        integration = LinkedInRecruiterIntegration.query.filter_by(user_id=user.id).first()
    except Exception as e:
        logger.error(f'Error querying LinkedIn Recruiter integration: {e}')
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
        integration.company_id = company_id
        integration.contract_id = contract_id
        integration.access_token = access_token
        integration.token_expires_at = auth.token_expires_at
        integration.refresh_token = auth.refresh_token
        integration.account_name = account_name
        integration.account_email = account_email
        integration.account_user_id = account_user_id
        integration.account_organization_id = account_organization_id
        integration.updated_at = datetime.utcnow()
    else:
        integration = LinkedInRecruiterIntegration(
            user_id=user.id,
            client_id=client_id,
            client_secret=enc_client_secret,
            company_id=company_id,
            contract_id=contract_id,
            access_token=access_token,
            token_expires_at=auth.token_expires_at,
            refresh_token=auth.refresh_token,
            account_name=account_name,
            account_email=account_email,
            account_user_id=account_user_id,
            account_organization_id=account_organization_id
        )
        db.session.add(integration)
    
    db.session.commit()
    logger.info(f'LinkedIn Recruiter integration saved for user {user.email}')
    
    return jsonify({
        'success': True,
        'connected': True,
        'message': 'LinkedIn Recruiter account connected successfully',
        'account': {
            'name': account_name,
            'email': account_email,
            'userId': account_user_id,
            'organizationId': account_organization_id,
            'companyId': company_id,
            'contractId': contract_id
        }
    }), 200

# GET /integrations/linkedin-recruiter/status
@linkedin_recruiter_bp.route('/integrations/linkedin-recruiter/status', methods=['GET'])
def linkedin_recruiter_status():
    """Check LinkedIn Recruiter connection status"""
    logger.info('HIT /integrations/linkedin-recruiter/status')
    
    user_jwt = get_current_user()
    if not user_jwt or not user_jwt.get('email'):
        logger.error('Unauthorized: No user_jwt or email')
        return jsonify({'error': 'Unauthorized'}), 401
    
    user = User.query.filter_by(email=user_jwt['email']).first()
    if not user:
        logger.error(f'User not found: {user_jwt.get("email")}')
        return jsonify({'error': 'User not found'}), 404
    
    try:
        integration = LinkedInRecruiterIntegration.query.filter_by(user_id=user.id).first()
    except Exception as e:
        logger.error(f'Error querying LinkedIn Recruiter integration: {e}')
        # Table might not exist yet - return not connected
        return jsonify({
            'connected': False,
            'message': 'LinkedIn Recruiter account not connected'
        }), 200
    
    if not integration:
        return jsonify({
            'connected': False,
            'message': 'LinkedIn Recruiter account not connected'
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
            'organizationId': integration.account_organization_id,
            'companyId': integration.company_id,
            'contractId': integration.contract_id
        },
        'tokenExpired': is_expired,
        'connectedAt': integration.created_at.isoformat() if integration.created_at else None,
        'updatedAt': integration.updated_at.isoformat() if integration.updated_at else None
    }), 200

# POST /integrations/linkedin-recruiter/disconnect
@linkedin_recruiter_bp.route('/integrations/linkedin-recruiter/disconnect', methods=['POST'])
def linkedin_recruiter_disconnect():
    """Disconnect LinkedIn Recruiter account"""
    logger.info('HIT /integrations/linkedin-recruiter/disconnect')
    
    user_jwt = get_current_user()
    if not user_jwt or not user_jwt.get('email'):
        logger.error('Unauthorized: No user_jwt or email')
        return jsonify({'error': 'Unauthorized'}), 401
    
    user = User.query.filter_by(email=user_jwt['email']).first()
    if not user:
        logger.error(f'User not found: {user_jwt.get("email")}')
        return jsonify({'error': 'User not found'}), 404
    
    try:
        integration = LinkedInRecruiterIntegration.query.filter_by(user_id=user.id).first()
    except Exception as e:
        logger.error(f'Error querying LinkedIn Recruiter integration: {e}')
        # Table might not exist yet - return success (nothing to disconnect)
        return jsonify({
            'success': True,
            'message': 'LinkedIn Recruiter account not connected'
        }), 200
    
    if not integration:
        return jsonify({
            'success': True,
            'message': 'LinkedIn Recruiter account not connected'
        }), 200
    
    try:
        db.session.delete(integration)
        db.session.commit()
        logger.info(f'LinkedIn Recruiter integration disconnected for user {user.email}')
        
        return jsonify({
            'success': True,
            'message': 'LinkedIn Recruiter account disconnected successfully'
        }), 200
    except Exception as e:
        logger.error(f'Error disconnecting LinkedIn Recruiter integration: {e}')
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': 'Failed to disconnect LinkedIn Recruiter account'
        }), 500

