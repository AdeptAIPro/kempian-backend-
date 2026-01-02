# import jwt
# from flask import request, jsonify
# from app.simple_logger import get_logger
# from functools import wraps
# import os
# from datetime import datetime, timedelta
# from app.auth_utils import decode_jwt  # Use hardened decoder

# COGNITO_JWT_ISSUER = f"https://cognito-idp.{os.getenv('COGNITO_REGION')}.amazonaws.com/{os.getenv('COGNITO_USER_POOL_ID')}"

# def get_current_user():
#     auth = request.headers.get('Authorization', None)
#     if not auth or not isinstance(auth, str) or not auth.startswith('Bearer '):
#         return None
#     token = auth.split(' ')[1]
#     return decode_jwt(token)

# def require_role(role):
#     def decorator(f):
#         @wraps(f)
#         def wrapper(*args, **kwargs):
#             user = get_current_user()
#             if not user or user.get('custom:role') != role:
#                 return jsonify({'error': 'Forbidden'}), 403
#             return f(*args, **kwargs)
#         return wrapper
#     return decorator

# def require_tenant(tenant_id):
#     def decorator(f):
#         @wraps(f)
#         def wrapper(*args, **kwargs):
#             user = get_current_user()
#             if not user or int(user.get('custom:tenant_id', 0)) != int(tenant_id):
#                 return jsonify({'error': 'Forbidden'}), 403
#             return f(*args, **kwargs)
#         return wrapper
#     return decorator

# def is_token_expired(token):
#     """Check if a JWT token is expired"""
#     try:
#         payload = jwt.decode(token, options={"verify_signature": False, "verify_aud": False})
#         if 'exp' in payload:
#             exp_timestamp = payload['exp']
#             current_timestamp = datetime.utcnow().timestamp()
#             return current_timestamp >= exp_timestamp
#         return False
#     except Exception:
#         return True

# def get_token_expiry(token):
#     """Get the expiration time of a JWT token"""
#     try:
#         payload = jwt.decode(token, options={"verify_signature": False, "verify_aud": False})
#         if 'exp' in payload:
#             return datetime.fromtimestamp(payload['exp'])
#         return None
#     except Exception:
#         return None

# def get_token_remaining_time(token):
#     """Get the remaining time until token expires in seconds"""
#     try:
#         payload = jwt.decode(token, options={"verify_signature": False, "verify_aud": False})
#         if 'exp' in payload:
#             exp_timestamp = payload['exp']
#             current_timestamp = datetime.utcnow().timestamp()
#             remaining = exp_timestamp - current_timestamp
#             return max(0, int(remaining))
#         return 0
#     except Exception:
#         return 0 
import jwt
from flask import request, jsonify
from app.simple_logger import get_logger
from functools import wraps
import os
from datetime import datetime, timedelta

logger = get_logger("utils")

COGNITO_JWT_ISSUER = f"https://cognito-idp.{os.getenv('COGNITO_REGION')}.amazonaws.com/{os.getenv('COGNITO_USER_POOL_ID')}"

def decode_jwt(token):
    """
    Decode JWT token with proper signature verification
    """
    try:
        from app.auth.jwt_utils import verify_cognito_jwt
        
        # Use proper JWT verification
        payload = verify_cognito_jwt(token)
        return payload
    except Exception as e:
        logger.error(f"JWT decode error: {e}")
        return None

def get_current_user():
    auth = request.headers.get('Authorization', None)
    if not auth:
        return None
    
    if not auth.startswith('Bearer '):
        return None
    
    try:
        token = auth.split(' ')[1]
        if not token or len(token.split('.')) != 3:
            return None
        
        # Try proper JWT verification first
        from app.auth.jwt_utils import verify_cognito_jwt
        payload = verify_cognito_jwt(token)
        
        if payload:
            logger.info(f"JWT verification successful for user: {payload.get('email', 'unknown')}")
            return payload
        
        # Fallback: use unsafe decoding for development/testing
        logger.warning("JWT verification failed, using unsafe decoding as fallback")
        payload = decode_jwt(token)
        
        if payload:
            logger.info(f"Unsafe JWT decoding successful for user: {payload.get('email', 'unknown')}")
            return payload
        
        return None
        
    except Exception as e:
        logger.error(f"Error in get_current_user: {e}")
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

def is_token_expired(token):
    """Check if a JWT token is expired"""
    try:
        payload = jwt.decode(token, options={"verify_signature": False, "verify_aud": False})
        if 'exp' in payload:
            exp_timestamp = payload['exp']
            current_timestamp = datetime.utcnow().timestamp()
            return current_timestamp >= exp_timestamp
        return False
    except Exception:
        return True

def get_token_expiry(token):
    """Get the expiration time of a JWT token"""
    try:
        payload = jwt.decode(token, options={"verify_signature": False, "verify_aud": False})
        if 'exp' in payload:
            return datetime.fromtimestamp(payload['exp'])
        return None
    except Exception:
        return None

def get_token_remaining_time(token):
    """Get the remaining time until token expires in seconds"""
    try:
        payload = jwt.decode(token, options={"verify_signature": False, "verify_aud": False})
        if 'exp' in payload:
            exp_timestamp = payload['exp']
            current_timestamp = datetime.utcnow().timestamp()
            remaining = exp_timestamp - current_timestamp
            return max(0, int(remaining))
        return 0
    except Exception:
        return 0 

def notify_admins_new_user(email: str, role: str | None = None, name: str | None = None):
    """Log an admin activity event that a new user signed up and send email notification.

    This writes into the existing admin activity logs so the admin dashboard
    can surface the notification without extra infrastructure.
    Also sends an email notification to vinit@adeptaipro.com
    """
    try:
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
            logger.warning(f"Admin email notification failed: {email_err}")
            
    except Exception as e:
        logger.warning(f"notify_admins_new_user failed: {e}")