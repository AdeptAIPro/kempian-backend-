"""
LinkedIn OIDC Authentication Routes
Handles OAuth 2.0 / OIDC flow endpoints for LinkedIn Sign-In
"""

from flask import Blueprint, request, jsonify, redirect, session, url_for
from datetime import datetime, timedelta
from app.simple_logger import get_logger
from app.models import db, User, UserLinkedIn
from app.utils import get_current_user
from app.auth.linkedin_oidc import (
    generate_state, store_state, validate_state,
    exchange_code_for_token, validate_id_token,
    get_identity_me, get_verification_report,
    build_authorization_url, get_nonce,
    encrypt_token, decrypt_token
)

logger = get_logger("linkedin_auth")

linkedin_auth_bp = Blueprint('linkedin_auth', __name__)


@linkedin_auth_bp.route('/', methods=['GET'])
def initiate_auth():
    """
    Start LinkedIn OAuth flow
    Generates secure state and redirects to LinkedIn authorization page
    """
    try:
        # Generate secure state
        state = generate_state()
        
        # Store state (in production, use Redis or database)
        session_id = session.get('session_id') or request.remote_addr
        store_state(state, session_id)
        
        # Build authorization URL
        auth_url = build_authorization_url(state)
        
        logger.info(f"LinkedIn auth initiated - state: {state[:16]}...")
        
        # Redirect to LinkedIn
        return redirect(auth_url), 302
        
    except Exception as e:
        logger.error(f"Error initiating LinkedIn auth: {e}", exc_info=True)
        return jsonify({
            'error': 'ERR_LINKEDIN_AUTH_INIT_FAILED',
            'message': 'Failed to initiate LinkedIn authentication'
        }), 500


@linkedin_auth_bp.route('/callback', methods=['GET'])
def callback():
    """
    Handle LinkedIn OAuth callback
    Validates state, exchanges code for tokens, validates ID token,
    and creates/updates user record
    """
    try:
        code = request.args.get('code')
        state = request.args.get('state')
        error = request.args.get('error')
        error_description = request.args.get('error_description')
        
        # Handle OAuth errors
        if error:
            logger.error(f"LinkedIn OAuth error: {error} - {error_description}")
            return jsonify({
                'error': f'ERR_LINKEDIN_{error.upper()}',
                'message': error_description or error
            }), 400
        
        # Validate required parameters
        if not code:
            logger.error("LinkedIn callback missing authorization code")
            return jsonify({
                'error': 'ERR_LINKEDIN_MISSING_CODE',
                'message': 'Authorization code is required'
            }), 400
        
        if not state:
            logger.error("LinkedIn callback missing state")
            return jsonify({
                'error': 'ERR_LINKEDIN_MISSING_STATE',
                'message': 'State parameter is required'
            }), 400
        
        # Validate state
        if not validate_state(state):
            logger.error(f"LinkedIn callback state validation failed: {state[:16]}...")
            return jsonify({
                'error': 'ERR_LINKEDIN_STATE_MISMATCH',
                'message': 'Invalid or expired state parameter'
            }), 403
        
        # Exchange code for tokens
        import os
        redirect_uri = os.getenv('LINKEDIN_REDIRECT_URI')
        if not redirect_uri:
            return jsonify({
                'error': 'ERR_LINKEDIN_CONFIG',
                'message': 'LinkedIn redirect URI not configured'
            }), 500
        
        success, token_data, error_msg = exchange_code_for_token(code, redirect_uri)
        
        if not success:
            logger.error(f"Token exchange failed: {error_msg}")
            return jsonify({
                'error': 'ERR_LINKEDIN_TOKEN_EXCHANGE',
                'message': error_msg or 'Failed to exchange code for token'
            }), 502
        
        access_token = token_data.get('access_token')
        expires_in = token_data.get('expires_in', 5184000)  # Default 60 days
        id_token = token_data.get('id_token')
        refresh_token = token_data.get('refresh_token')
        
        # Get nonce for validation
        nonce = get_nonce(state)
        
        # Validate ID token if present
        linkedin_member_id = None
        email = None
        
        if id_token:
            is_valid, payload = validate_id_token(id_token, nonce=nonce)
            if not is_valid:
                logger.error("ID token validation failed")
                return jsonify({
                    'error': 'ERR_LINKEDIN_ID_TOKEN_INVALID',
                    'message': 'ID token validation failed'
                }), 401
            
            linkedin_member_id = payload.get('sub')
            email = payload.get('email')
        
        # If no ID token or no sub in token, fetch from identityMe
        if not linkedin_member_id:
            success, identity_data, error_msg = get_identity_me(access_token)
            if not success:
                logger.error(f"Failed to fetch identityMe: {error_msg}")
                return jsonify({
                    'error': 'ERR_LINKEDIN_IDENTITY_FETCH_FAILED',
                    'message': error_msg or 'Failed to fetch user identity'
                }), 502
            
            linkedin_member_id = identity_data.get('id')
            email = email or identity_data.get('email', {}).get('emailAddress')
        
        if not linkedin_member_id:
            logger.error("Could not determine LinkedIn member ID")
            return jsonify({
                'error': 'ERR_LINKEDIN_NO_MEMBER_ID',
                'message': 'Could not determine LinkedIn member ID'
            }), 400
        
        # Calculate token expiry
        token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in) if expires_in else None
        
        # Find or create user
        user_linkedin = UserLinkedIn.query.filter_by(linkedin_id=linkedin_member_id).first()
        
        if user_linkedin:
            # Update existing record
            user_linkedin.access_token_encrypted = encrypt_token(access_token)
            if refresh_token:
                user_linkedin.refresh_token_encrypted = encrypt_token(refresh_token)
            if id_token:
                user_linkedin.id_token_encrypted = encrypt_token(id_token)
            user_linkedin.token_expires_at = token_expires_at
            user_linkedin.last_synced_at = datetime.utcnow()
            user_linkedin.updated_at = datetime.utcnow()
            user = user_linkedin.user
        else:
            # Check if user exists by email
            user = None
            if email:
                user = User.query.filter_by(email=email).first()
            
            # Create new user if doesn't exist
            if not user:
                from app.models import Tenant
                # Get or create default tenant (you may need to adjust this)
                default_tenant = Tenant.query.first()
                if not default_tenant:
                    logger.error("No tenant found - cannot create user")
                    return jsonify({
                        'error': 'ERR_NO_TENANT',
                        'message': 'No tenant available for user creation'
                    }), 500
                
                user = User(
                    email=email or f"linkedin_{linkedin_member_id}@linkedin.local",
                    tenant_id=default_tenant.id,
                    role='job_seeker',  # Default role
                    linkedin_id=linkedin_member_id
                )
                db.session.add(user)
                db.session.flush()  # Get user.id
            
            # Create UserLinkedIn record
            user_linkedin = UserLinkedIn(
                user_id=user.id,
                linkedin_id=linkedin_member_id,
                access_token_encrypted=encrypt_token(access_token),
                refresh_token_encrypted=encrypt_token(refresh_token) if refresh_token else None,
                id_token_encrypted=encrypt_token(id_token) if id_token else None,
                token_expires_at=token_expires_at,
                last_synced_at=datetime.utcnow()
            )
            db.session.add(user_linkedin)
            
            # Update user's linkedin_id if not set
            if not user.linkedin_id:
                user.linkedin_id = linkedin_member_id
        
        db.session.commit()
        
        logger.info(f"LinkedIn auth successful - user_id: {user.id}, linkedin_id: {linkedin_member_id}")
        
        # Redirect to frontend success page
        # In production, you'd generate a session/JWT token here
        frontend_url = os.getenv('FRONTEND_URL') or request.headers.get('Origin') or 'http://localhost:8081'
        # Remove trailing slash if present
        frontend_url = frontend_url.rstrip('/')
        return redirect(f"{frontend_url}/auth/linkedin/success?user_id={user.id}"), 302
        
    except Exception as e:
        logger.error(f"Error in LinkedIn callback: {e}", exc_info=True)
        db.session.rollback()
        return jsonify({
            'error': 'ERR_LINKEDIN_CALLBACK_FAILED',
            'message': 'An error occurred during authentication'
        }), 500

