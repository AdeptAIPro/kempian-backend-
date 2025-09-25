import jwt
from flask import request, jsonify
from app.simple_logger import get_logger
from functools import wraps
import os
import json
from app.auth_utils import decode_jwt  # Use hardened decoder

COGNITO_JWT_ISSUER = f"https://cognito-idp.{os.getenv('COGNITO_REGION')}.amazonaws.com/{os.getenv('COGNITO_USER_POOL_ID')}"

# // Use decode_jwt from app.auth_utils

def get_current_user():
    auth = request.headers.get('Authorization', None)
    if not auth:
        return None
    token = auth.split(' ')[1]
    return decode_jwt(token)

def get_current_user_flexible():
    """Get current user strictly from a Bearer JWT token.

    Security hardening:
    - Require Authorization header to start with 'Bearer '
    - Do NOT accept arbitrary base64 tokens as identity
    - Decode JWT (signature verification may be disabled in dev, but structure and email are required)
    """
    auth = request.headers.get('Authorization', None)
    if not auth or not isinstance(auth, str):
        return None

    if not auth.startswith('Bearer '):
        return None

    parts = auth.split(' ')
    if len(parts) != 2 or not parts[1]:
        return None

    token = parts[1]

    try:
        payload = decode_jwt(token)
        if payload and isinstance(payload, dict) and payload.get('email'):
            return payload
    except Exception:
        return None

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


