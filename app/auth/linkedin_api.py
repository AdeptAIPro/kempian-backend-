"""
LinkedIn Verified-on-LinkedIn API Endpoints
Provides endpoints to fetch LinkedIn profile and verification data
"""

from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
from typing import Tuple, Optional
from app.simple_logger import get_logger
from app.models import db, User, UserLinkedIn
from app.utils import get_current_user
from app.auth.linkedin_oidc import (
    get_identity_me, get_verification_report,
    decrypt_token
)

logger = get_logger("linkedin_api")

linkedin_api_bp = Blueprint('linkedin_api', __name__)


def refresh_access_token(user_linkedin: UserLinkedIn) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Attempt to refresh access token using refresh token
    
    Returns:
        (success, new_access_token, error_message) tuple
    """
    from app.auth.linkedin_oidc import decrypt_token, encrypt_token, LINKEDIN_OAUTH_TOKEN, LINKEDIN_CLIENT_ID, LINKEDIN_CLIENT_SECRET
    import requests
    import time
    
    if not user_linkedin.refresh_token_encrypted:
        return False, None, "No refresh token available"
    
    try:
        refresh_token = decrypt_token(user_linkedin.refresh_token_encrypted)
        if not refresh_token:
            return False, None, "Failed to decrypt refresh token"
        
        # Attempt token refresh
        response = requests.post(
            LINKEDIN_OAUTH_TOKEN,
            data={
                'grant_type': 'refresh_token',
                'refresh_token': refresh_token,
                'client_id': LINKEDIN_CLIENT_ID,
                'client_secret': LINKEDIN_CLIENT_SECRET
            },
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            timeout=10
        )
        
        if not response.ok:
            error_data = response.json() if response.content else {}
            error_msg = error_data.get('error_description', response.text)
            logger.warning(f"Token refresh failed: {error_msg}")
            return False, None, error_msg
        
        token_data = response.json()
        new_access_token = token_data.get('access_token')
        expires_in = token_data.get('expires_in', 5184000)
        
        if not new_access_token:
            return False, None, "No access token in refresh response"
        
        # Update stored tokens
        user_linkedin.access_token_encrypted = encrypt_token(new_access_token)
        if token_data.get('refresh_token'):
            user_linkedin.refresh_token_encrypted = encrypt_token(token_data['refresh_token'])
        user_linkedin.token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
        user_linkedin.updated_at = datetime.utcnow()
        db.session.commit()
        
        logger.info("Access token refreshed successfully")
        return True, new_access_token, None
        
    except Exception as e:
        logger.error(f"Token refresh error: {e}", exc_info=True)
        return False, None, str(e)


def get_user_linkedin_token(user_id: int) -> tuple:
    """
    Get decrypted LinkedIn access token for user, attempting refresh if expired
    
    Returns:
        (success, access_token, error_message) tuple
    """
    try:
        user_linkedin = UserLinkedIn.query.filter_by(user_id=user_id).first()
        
        if not user_linkedin:
            return False, None, "LinkedIn not connected for this user"
        
        # Check if token is expired
        is_expired = user_linkedin.token_expires_at and user_linkedin.token_expires_at < datetime.utcnow()
        
        if is_expired:
            # Attempt to refresh
            logger.info("Access token expired, attempting refresh")
            success, new_token, error_msg = refresh_access_token(user_linkedin)
            if success:
                return True, new_token, None
            else:
                # Refresh failed, require re-auth
                return False, None, "LinkedIn access token expired and refresh failed - re-authentication required"
        
        # Decrypt access token
        try:
            access_token = decrypt_token(user_linkedin.access_token_encrypted)
            if not access_token:
                return False, None, "Failed to decrypt access token"
            
            return True, access_token, None
        except Exception as e:
            logger.error(f"Token decryption error: {e}")
            return False, None, "Failed to decrypt access token"
            
    except Exception as e:
        logger.error(f"Error getting LinkedIn token: {e}")
        return False, None, str(e)


@linkedin_api_bp.route('/', methods=['GET'])
def get_linkedin_profile():
    """
    Get LinkedIn profile and verification summary for current user
    Returns normalized profile data with verifiedCategories
    """
    try:
        # Get current user
        current_user = get_current_user()
        if not current_user:
            return jsonify({
                'error': 'ERR_UNAUTHORIZED',
                'message': 'Authentication required'
            }), 401
        
        # Get user ID from token
        user_id = current_user.get('id') or current_user.get('sub')
        if not user_id:
            # Try to get from email
            email = current_user.get('email')
            if email:
                user = User.query.filter_by(email=email).first()
                if user:
                    user_id = user.id
                else:
                    return jsonify({
                        'error': 'ERR_USER_NOT_FOUND',
                        'message': 'User not found'
                    }), 404
            else:
                return jsonify({
                    'error': 'ERR_INVALID_TOKEN',
                    'message': 'Invalid authentication token'
                }), 401
        
        # Get LinkedIn token
        success, access_token, error_msg = get_user_linkedin_token(user_id)
        if not success:
            return jsonify({
                'error': 'ERR_LINKEDIN_NOT_CONNECTED' if 'not connected' in error_msg.lower() else 'ERR_LINKEDIN_TOKEN_EXPIRED',
                'message': error_msg
            }), 401 if 'expired' in error_msg.lower() or 're-auth' in error_msg.lower() else 404
        
        # Fetch identityMe
        success, identity_data, error_msg = get_identity_me(access_token)
        if not success:
            logger.error(f"identityMe fetch failed: {error_msg}")
            return jsonify({
                'error': 'ERR_LINKEDIN_API_ERROR',
                'message': error_msg or 'Failed to fetch LinkedIn profile'
            }), 502
        
        # Normalize response
        verified_categories = []
        if 'verifiedCategories' in identity_data:
            verified_categories = identity_data['verifiedCategories']
        elif 'verificationStatus' in identity_data:
            # Handle different response formats
            status = identity_data.get('verificationStatus', {})
            if status.get('identityVerified'):
                verified_categories.append('IDENTITY')
            if status.get('workplaceVerified'):
                verified_categories.append('WORKPLACE')
        
        normalized = {
            'linkedInId': identity_data.get('id', ''),
            'displayName': identity_data.get('displayName', ''),
            'email': identity_data.get('email', {}).get('emailAddress', '') if isinstance(identity_data.get('email'), dict) else identity_data.get('email', ''),
            'profileUrl': identity_data.get('profileUrl', ''),
            'verifiedCategories': verified_categories,
            'raw': identity_data  # Include raw for debugging
        }
        
        logger.info(f"LinkedIn profile fetched for user_id: {user_id}")
        return jsonify(normalized), 200
        
    except Exception as e:
        logger.error(f"Error fetching LinkedIn profile: {e}", exc_info=True)
        return jsonify({
            'error': 'ERR_INTERNAL_ERROR',
            'message': 'An error occurred while fetching LinkedIn profile'
        }), 500


@linkedin_api_bp.route('/report', methods=['GET'])
def get_verification_report():
    """
    Get detailed LinkedIn verification report for current user
    Returns normalized verification data with criteria details
    """
    try:
        # Get current user
        current_user = get_current_user()
        if not current_user:
            return jsonify({
                'error': 'ERR_UNAUTHORIZED',
                'message': 'Authentication required'
            }), 401
        
        # Get user ID from token
        user_id = current_user.get('id') or current_user.get('sub')
        if not user_id:
            email = current_user.get('email')
            if email:
                user = User.query.filter_by(email=email).first()
                if user:
                    user_id = user.id
                else:
                    return jsonify({
                        'error': 'ERR_USER_NOT_FOUND',
                        'message': 'User not found'
                    }), 404
            else:
                return jsonify({
                    'error': 'ERR_INVALID_TOKEN',
                    'message': 'Invalid authentication token'
                }), 401
        
        # Get LinkedIn token
        success, access_token, error_msg = get_user_linkedin_token(user_id)
        if not success:
            return jsonify({
                'error': 'ERR_LINKEDIN_NOT_CONNECTED' if 'not connected' in error_msg.lower() else 'ERR_LINKEDIN_TOKEN_EXPIRED',
                'message': error_msg
            }), 401 if 'expired' in error_msg.lower() or 're-auth' in error_msg.lower() else 404
        
        # Fetch verification report
        criteria = request.args.getlist('criteria') or ['IDENTITY', 'WORKPLACE']
        success, report_data, error_msg = get_verification_report(access_token, criteria)
        
        if not success:
            # Check if it's a tier limitation
            if '403' in error_msg or 'insufficient' in error_msg.lower():
                return jsonify({
                    'verified': False,
                    'tier': 'LITE',
                    'message': 'Verification report not available for current tier',
                    'criteria': {}
                }), 200
            
            logger.error(f"verificationReport fetch failed: {error_msg}")
            return jsonify({
                'error': 'ERR_LINKEDIN_API_ERROR',
                'message': error_msg or 'Failed to fetch verification report'
            }), 502
        
        # Normalize response
        criteria_data = {}
        verified = False
        
        if 'verificationCriteria' in report_data:
            for criterion in report_data['verificationCriteria']:
                criterion_type = criterion.get('type', '')
                status = criterion.get('status', 'NOT_VERIFIED')
                timestamp = criterion.get('timestamp')
                evidence = criterion.get('evidence', [])
                
                criteria_data[criterion_type] = {
                    'status': status,
                    'timestamp': timestamp,
                    'evidence': evidence
                }
                
                if status == 'VERIFIED':
                    verified = True
        
        normalized = {
            'verified': verified,
            'criteria': criteria_data,
            'raw': report_data  # Include raw for debugging
        }
        
        logger.info(f"LinkedIn verification report fetched for user_id: {user_id}")
        return jsonify(normalized), 200
        
    except Exception as e:
        logger.error(f"Error fetching LinkedIn verification report: {e}", exc_info=True)
        return jsonify({
            'error': 'ERR_INTERNAL_ERROR',
            'message': 'An error occurred while fetching verification report'
        }), 500


@linkedin_api_bp.route('/status', methods=['GET'])
def get_token_status():
    """
    Admin endpoint to check LinkedIn token status
    Shows expiry times and connection status (admin-only)
    """
    try:
        # Get current user
        current_user = get_current_user()
        if not current_user:
            return jsonify({
                'error': 'ERR_UNAUTHORIZED',
                'message': 'Authentication required'
            }), 401
        
        # Check admin role
        role = current_user.get('role') or current_user.get('custom:role')
        if role not in ['admin', 'owner']:
            return jsonify({
                'error': 'ERR_FORBIDDEN',
                'message': 'Admin access required'
            }), 403
        
        # Get user ID
        user_id = current_user.get('id') or current_user.get('sub')
        if not user_id:
            email = current_user.get('email')
            if email:
                user = User.query.filter_by(email=email).first()
                if user:
                    user_id = user.id
        
        if not user_id:
            return jsonify({
                'error': 'ERR_USER_NOT_FOUND',
                'message': 'User not found'
            }), 404
        
        user_linkedin = UserLinkedIn.query.filter_by(user_id=user_id).first()
        
        if not user_linkedin:
            return jsonify({
                'connected': False,
                'message': 'LinkedIn not connected'
            }), 200
        
        is_expired = user_linkedin.token_expires_at and user_linkedin.token_expires_at < datetime.utcnow()
        
        return jsonify({
            'connected': True,
            'linkedin_id': user_linkedin.linkedin_id,
            'token_expires_at': user_linkedin.token_expires_at.isoformat() if user_linkedin.token_expires_at else None,
            'is_expired': is_expired,
            'last_synced_at': user_linkedin.last_synced_at.isoformat() if user_linkedin.last_synced_at else None,
            'has_refresh_token': bool(user_linkedin.refresh_token_encrypted),
            'has_id_token': bool(user_linkedin.id_token_encrypted)
        }), 200
        
    except Exception as e:
        logger.error(f"Error checking token status: {e}", exc_info=True)
        return jsonify({
            'error': 'ERR_INTERNAL_ERROR',
            'message': 'An error occurred while checking token status'
        }), 500

