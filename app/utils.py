import jwt
from flask import request, jsonify
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