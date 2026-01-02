# import jwt
# from flask import request, jsonify
# from app.simple_logger import get_logger
# from functools import wraps
# import os
# from datetime import datetime, timedelta

# COGNITO_JWT_ISSUER = f"https://cognito-idp.{os.getenv('COGNITO_REGION')}.amazonaws.com/{os.getenv('COGNITO_USER_POOL_ID')}"

# # For demo: skip signature verification (for prod, fetch and verify JWKS)
# def decode_jwt(token):
#     try:
#         payload = jwt.decode(token, options={"verify_signature": False, "verify_aud": False})
        
#         # Check if token is expired
#         if 'exp' in payload:
#             exp_timestamp = payload['exp']
#             current_timestamp = datetime.utcnow().timestamp()
            
#             if current_timestamp >= exp_timestamp:
#                 return None  # Token expired
        
#         return payload
#     except Exception:
#         return None

# def get_current_user():
#     auth = request.headers.get('Authorization', None)
#     if not auth:
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
import json
from flask import request, jsonify, g
from app.simple_logger import get_logger
from functools import wraps
import os
from datetime import datetime, timedelta

COGNITO_JWT_ISSUER = f"https://cognito-idp.{os.getenv('COGNITO_REGION')}.amazonaws.com/{os.getenv('COGNITO_USER_POOL_ID')}"

# For demo: skip signature verification (for prod, fetch and verify JWKS)
def decode_jwt(token):
    try:
        payload = jwt.decode(token, options={"verify_signature": False, "verify_aud": False})
        
        # Check if token is expired
        if 'exp' in payload:
            exp_timestamp = payload['exp']
            current_timestamp = datetime.utcnow().timestamp()
            
            if current_timestamp >= exp_timestamp:
                return None  # Token expired
        
        return payload
    except Exception:
        return None

def get_current_user():
    auth = request.headers.get('Authorization', None)
    if not auth:
        return None
    token = auth.split(' ')[1]
    return decode_jwt(token)

def get_current_user_flexible():
    """Get current user from either Cognito JWT or custom auth token"""
    auth = request.headers.get('Authorization', None)
    if not auth:
        return None
    
    token = auth.split(' ')[1]
    
    # First try to decode as Cognito JWT
    try:
        payload = decode_jwt(token)
        if payload and 'email' in payload:
            return payload
    except:
        pass
    
    # If not a valid JWT, try to decode as custom token
    try:
        # Custom tokens are usually base64 encoded user data
        import base64
        decoded = base64.b64decode(token + '==').decode('utf-8')  # Add padding if needed
        user_data = json.loads(decoded)
        if 'email' in user_data:
            return user_data
    except:
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

def get_current_tenant_id():
    """Get current tenant ID from Flask g object (set by before_request handler)"""
    try:
        # Try to get from Flask g object (set by _inject_tenant_context in __init__.py)
        if hasattr(g, 'tenant_id') and g.tenant_id:
            return g.tenant_id
        
        # Fallback: get from user token
        user = get_current_user_flexible() or get_current_user()
        if user:
            # Try to get tenant_id from JWT token
            tenant_id = user.get('custom:tenant_id')
            if tenant_id:
                return int(tenant_id)
            
            # Fallback: get tenant_id from database using email
            from app.models import User as UserModel
            email = user.get('email')
            if email:
                db_user = UserModel.query.filter_by(email=email).first()
                if db_user:
                    return db_user.tenant_id
        
        return None
    except Exception:
        return None 