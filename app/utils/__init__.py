import jwt
from flask import request, jsonify
from functools import wraps
import os
import json

COGNITO_JWT_ISSUER = f"https://cognito-idp.{os.getenv('COGNITO_REGION')}.amazonaws.com/{os.getenv('COGNITO_USER_POOL_ID')}"

# For demo: skip signature verification (for prod, fetch and verify JWKS)
def decode_jwt(token):
    try:
        payload = jwt.decode(token, options={"verify_signature": False, "verify_aud": False})
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