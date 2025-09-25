import jwt
import json
from flask import request, jsonify, current_app
from app.simple_logger import get_logger
from functools import wraps
import os
from datetime import datetime, timedelta
import requests
from jose import jwt as jose_jwt

COGNITO_JWT_ISSUER = f"https://cognito-idp.{os.getenv('COGNITO_REGION')}.amazonaws.com/{os.getenv('COGNITO_USER_POOL_ID')}"

_jwks_cache = None

def _get_cognito_jwks():
    global _jwks_cache
    if _jwks_cache is None:
        issuer = f"https://cognito-idp.{os.getenv('COGNITO_REGION')}.amazonaws.com/{os.getenv('COGNITO_USER_POOL_ID')}"
        jwks_url = f"{issuer}/.well-known/jwks.json"
        resp = requests.get(jwks_url, timeout=5)
        resp.raise_for_status()
        _jwks_cache = resp.json()
    return _jwks_cache

def decode_jwt(token):
    """Decode and verify JWTs.

    - RS256 (Cognito): verify with JWKS, audience and issuer
    - HS256 (internal): verify with SECRET_KEY
    Returns payload dict or None if invalid/expired.
    """
    try:
        header = jose_jwt.get_unverified_header(token)
        alg = header.get('alg')
    except Exception:
        return None

    try:
        if alg == 'RS256':
            jwks = _get_cognito_jwks()
            kid = header.get('kid')
            key = None
            for k in jwks.get('keys', []):
                if k.get('kid') == kid:
                    key = k
                    break
            if not key:
                return None
            audience = os.getenv('COGNITO_CLIENT_ID')
            issuer = f"https://cognito-idp.{os.getenv('COGNITO_REGION')}.amazonaws.com/{os.getenv('COGNITO_USER_POOL_ID')}"
            payload = jose_jwt.decode(
                token,
                key,
                algorithms=['RS256'],
                audience=audience,
                issuer=issuer
            )
        elif alg == 'HS256':
            secret = current_app.config.get('SECRET_KEY') or os.getenv('SECRET_KEY')
            if not secret:
                return None
            payload = jwt.decode(token, secret, algorithms=['HS256'])
        else:
            return None

        # Expiry check (PyJWT and jose verify exp by default when present)
        if 'exp' in payload:
            exp_timestamp = payload['exp']
            current_timestamp = datetime.utcnow().timestamp()
            if current_timestamp >= exp_timestamp:
                return None

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
    """Get current user strictly from a Bearer JWT token.

    Security hardening:
    - Require Authorization header to start with 'Bearer '
    - Do NOT accept arbitrary base64 tokens as identity
    - Decode JWT and require 'email' claim
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