"""
LinkedIn Token Revocation Endpoint
Handles token revocation and account unlinking
"""

from flask import Blueprint, request, jsonify
from app.simple_logger import get_logger
from app.models import db, UserLinkedIn
from app.utils import get_current_user
import requests
import os

logger = get_logger("linkedin_revoke")

linkedin_revoke_bp = Blueprint('linkedin_revoke', __name__)


@linkedin_revoke_bp.route('/revoke', methods=['POST'])
def revoke_linkedin_token():
    """
    Revoke LinkedIn tokens and unlink account
    Optionally revokes tokens on LinkedIn's side as well
    """
    try:
        # Get current user
        current_user = get_current_user()
        if not current_user:
            return jsonify({
                'error': 'ERR_UNAUTHORIZED',
                'message': 'Authentication required'
            }), 401
        
        # Get user ID
        user_id = current_user.get('id') or current_user.get('sub')
        if not user_id:
            email = current_user.get('email')
            if email:
                from app.models import User
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
        
        # Find LinkedIn connection
        user_linkedin = UserLinkedIn.query.filter_by(user_id=user_id).first()
        if not user_linkedin:
            return jsonify({
                'error': 'ERR_LINKEDIN_NOT_CONNECTED',
                'message': 'LinkedIn not connected for this user'
            }), 404
        
        # Optionally revoke on LinkedIn's side
        revoke_on_linkedin = request.json.get('revoke_on_linkedin', False) if request.is_json else False
        
        if revoke_on_linkedin:
            try:
                from app.auth.linkedin_oidc import decrypt_token, LINKEDIN_OAUTH_TOKEN, LINKEDIN_CLIENT_ID, LINKEDIN_CLIENT_SECRET
                access_token = decrypt_token(user_linkedin.access_token_encrypted)
                
                if access_token:
                    # Revoke token on LinkedIn
                    revoke_response = requests.post(
                        'https://www.linkedin.com/oauth/v2/revoke',
                        data={
                            'token': access_token,
                            'client_id': LINKEDIN_CLIENT_ID,
                            'client_secret': LINKEDIN_CLIENT_SECRET
                        },
                        headers={'Content-Type': 'application/x-www-form-urlencoded'},
                        timeout=10
                    )
                    
                    if revoke_response.ok:
                        logger.info(f"Token revoked on LinkedIn for user_id: {user_id}")
                    else:
                        logger.warning(f"Failed to revoke token on LinkedIn: {revoke_response.status_code}")
            except Exception as e:
                logger.warning(f"Error revoking token on LinkedIn (continuing with local deletion): {e}")
        
        # Delete local tokens
        linkedin_id = user_linkedin.linkedin_id
        db.session.delete(user_linkedin)
        
        # Also clear linkedin_id from user record
        from app.models import User
        user = User.query.get(user_id)
        if user and user.linkedin_id == linkedin_id:
            user.linkedin_id = None
        
        db.session.commit()
        
        logger.info(f"LinkedIn tokens revoked and account unlinked for user_id: {user_id}, linkedin_id: {linkedin_id[:16] if linkedin_id else 'unknown'}...")
        
        return jsonify({
            'success': True,
            'message': 'LinkedIn account unlinked successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error revoking LinkedIn tokens: {e}", exc_info=True)
        db.session.rollback()
        return jsonify({
            'error': 'ERR_REVOKE_FAILED',
            'message': 'Failed to revoke LinkedIn tokens'
        }), 500

