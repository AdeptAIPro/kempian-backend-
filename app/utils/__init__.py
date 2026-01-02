import jwt
from flask import request, jsonify
from app.simple_logger import get_logger
from functools import wraps
import os
import json

COGNITO_JWT_ISSUER = f"https://cognito-idp.{os.getenv('COGNITO_REGION')}.amazonaws.com/{os.getenv('COGNITO_USER_POOL_ID')}"


def _normalize_user_payload(payload):
    """Ensure common fields (like email) are available on decoded tokens."""
    if not payload or not isinstance(payload, dict):
        return None

    email = (
        payload.get('email')
        or payload.get('Email')
        or payload.get('user_email')
    )

    if not email:
        username = (
            payload.get('username')
            or payload.get('cognito:username')
            or payload.get('preferred_username')
        )
        if username and '@' in username:
            email = username

    if email:
        payload['email'] = email.lower()

    return payload


# For demo: skip signature verification (for prod, fetch and verify JWKS)
def decode_jwt(token):
    try:
        payload = jwt.decode(token, options={"verify_signature": False, "verify_aud": False})
        return _normalize_user_payload(payload)
    except Exception:
        return None


def get_current_user():
    auth = request.headers.get('Authorization', None)
    if not auth:
        return None
    # Handle malformed auth headers (e.g., "Bearer " without token)
    parts = auth.split(' ')
    if len(parts) < 2 or not parts[1]:
        return None
    token = parts[1]
    return decode_jwt(token)


def get_current_user_flexible():
    """Get current user from either Cognito JWT or custom auth token"""
    auth = request.headers.get('Authorization', None)
    if not auth:
        return None
    
    # Handle malformed auth headers (e.g., "Bearer " without token)
    parts = auth.split(' ')
    if len(parts) < 2 or not parts[1]:
        return None
    token = parts[1]
    
    # First try to decode as Cognito JWT
    try:
        payload = decode_jwt(token)
        if payload:
            return payload
    except Exception:
        pass
    
    # If not a valid JWT, try to decode as custom token
    try:
        # Custom tokens are usually base64 encoded user data
        import base64
        decoded = base64.b64decode(token + '==').decode('utf-8')  # Add padding if needed
        user_data = json.loads(decoded)
        return _normalize_user_payload(user_data)
    except Exception:
        pass
    
    return None

def require_role(role):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            user = get_current_user()
            if not user or user.get('custom:role') != role:
                return jsonify({'error': 'Forbidden'}), 403
            return f(*args, **kwargs)
        return wrapper
    return decorator

def require_tenant(tenant_id):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            user = get_current_user()
            if not user or int(user.get('custom:tenant_id', 0)) != int(tenant_id):
                return jsonify({'error': 'Forbidden'}), 403
            return f(*args, **kwargs)
        return wrapper
    return decorator 

# Multitenancy helpers
from flask import g

def get_current_tenant_id():
    """Return tenant_id from request-scoped context if available, else from JWT."""
    try:
        if hasattr(g, 'tenant_id') and g.tenant_id:
            return g.tenant_id
    except Exception:
        pass
    user = get_current_user_flexible() or get_current_user()
    if user:
        try:
            tenant_id_claim = user.get('custom:tenant_id')
            if tenant_id_claim:
                return int(tenant_id_claim)
        except Exception:
            pass

        # Fallback: look up user by email if available
        email = user.get('email')
        if email:
            try:
                from app.models import User as UserModel  # Local import to avoid circular dependency
                db_user = UserModel.query.filter_by(email=email).first()
                if db_user and db_user.tenant_id:
                    return db_user.tenant_id
            except Exception:
                pass
    return 0

def require_tenant_context(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        tenant_id = get_current_tenant_id()
        if not tenant_id:
            return jsonify({'error': 'Tenant context missing'}), 403
        return f(*args, **kwargs)
    return wrapper

def notify_admins_new_user(email: str, role: str | None = None, name: str | None = None):
    """Log an admin activity event that a new user signed up and send email notification.

    This writes into the existing admin activity logs so the admin dashboard
    can surface the notification without extra infrastructure.
    Also sends an email notification to vinit@adeptaipro.com
    """
    try:
        logger_utils = get_logger("utils")
        from app.services.admin_activity_logger import AdminActivityLogger
        AdminActivityLogger.log_admin_action(
            admin_email='system@kempian.ai',
            admin_id='system-signup-notification',  # Use a system ID instead of None
            admin_role='admin',
            action=f"New user signup: {email}{f' (role: {role})' if role else ''}",
            endpoint='system.signup',
            method='POST',
            request_data={'email': email, 'role': role} if role else {'email': email},
            status_code=201,
            tenant_id=None
        )
        
        # Send email notification to admin
        try:
            from app.emails.smtp import send_admin_notification_email
            send_admin_notification_email(email, role or 'user', name)
        except Exception as email_err:
            logger_utils.warning(f"Admin email notification failed: {email_err}")
            
    except Exception as e:
        logger_utils = get_logger("utils")
        logger_utils.warning(f"notify_admins_new_user failed: {e}")


