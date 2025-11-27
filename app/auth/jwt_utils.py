"""
JWT Utilities for Cognito Token Verification
Provides secure JWT token validation with proper signature verification
"""

import jwt
import requests
import json
from datetime import datetime
from app.simple_logger import get_logger
import os

logger = get_logger("jwt_utils")

COGNITO_REGION = os.getenv('COGNITO_REGION')
COGNITO_USER_POOL_ID = os.getenv('COGNITO_USER_POOL_ID')
COGNITO_CLIENT_ID = os.getenv('COGNITO_CLIENT_ID')

# Cache for JWKS to avoid repeated requests
_jwks_cache = None
_jwks_cache_time = None
JWKS_CACHE_DURATION = 3600  # 1 hour

def get_cognito_jwk():
    """
    Get Cognito JWKS (JSON Web Key Set) with caching
    """
    global _jwks_cache, _jwks_cache_time
    
    current_time = datetime.utcnow().timestamp()
    
    # Return cached JWKS if still valid
    if _jwks_cache and _jwks_cache_time and (current_time - _jwks_cache_time) < JWKS_CACHE_DURATION:
        return _jwks_cache
    
    try:
        jwks_url = f"https://cognito-idp.{COGNITO_REGION}.amazonaws.com/{COGNITO_USER_POOL_ID}/.well-known/jwks.json"
        logger.info(f"[JWT] Fetching JWKS from: {jwks_url}")
        
        response = requests.get(jwks_url, timeout=10)
        response.raise_for_status()
        
        _jwks_cache = response.json()
        _jwks_cache_time = current_time
        
        logger.info(f"[JWT] Successfully cached JWKS with {len(_jwks_cache.get('keys', []))} keys")
        return _jwks_cache
        
    except Exception as e:
        logger.error(f"[JWT] Error fetching JWKS: {e}")
        raise Exception(f"Failed to fetch JWKS: {str(e)}")

def get_public_key(token):
    """
    Get the public key for JWT verification from JWKS
    """
    try:
        # Decode header without verification to get KID
        header = jwt.get_unverified_header(token)
        kid = header.get('kid')
        
        if not kid:
            raise Exception("No KID found in JWT header")
        
        # Get JWKS
        jwks = get_cognito_jwk()
        
        # Find the matching key
        for key in jwks.get('keys', []):
            if key.get('kid') == kid:
                return key
        
        raise Exception(f"No matching key found for KID: {kid}")
        
    except Exception as e:
        logger.error(f"[JWT] Error getting public key: {e}")
        raise

def verify_cognito_jwt(token, access_token=None):
    """
    Verify a Cognito JWT token with proper signature verification
    
    Args:
        token: The JWT token to verify
        access_token: Optional access token for at_hash validation
    
    Returns:
        dict: Decoded token payload if valid, None if invalid
    """
    try:
        logger.info(f"[JWT] Verifying Cognito JWT token")
        
        # Get the public key
        key = get_public_key(token)
        
        # Decode and verify the token
        payload = jwt.decode(
            token,
            key,
            algorithms=['RS256'],
            audience=COGNITO_CLIENT_ID,
            issuer=f'https://cognito-idp.{COGNITO_REGION}.amazonaws.com/{COGNITO_USER_POOL_ID}',
            access_token=access_token,  # For at_hash validation
            options={
                'verify_signature': True,
                'verify_aud': True,
                'verify_iss': True,
                'verify_exp': True,
                'verify_iat': True,
                'verify_nbf': True
            }
        )
        
        logger.info(f"[JWT] Token verification successful for user: {payload.get('email', 'unknown')}")
        return payload
        
    except jwt.ExpiredSignatureError:
        logger.warning(f"[JWT] Token has expired")
        return None
    except jwt.InvalidAudienceError:
        logger.warning(f"[JWT] Invalid audience in token")
        return None
    except jwt.InvalidIssuerError:
        logger.warning(f"[JWT] Invalid issuer in token")
        return None
    except jwt.InvalidSignatureError:
        logger.warning(f"[JWT] Invalid signature in token")
        return None
    except Exception as e:
        logger.error(f"[JWT] Token verification failed: {e}")
        return None

def is_token_expired(token):
    """
    Check if a JWT token is expired without full verification
    """
    try:
        payload = jwt.decode(token, options={"verify_signature": False, "verify_aud": False})
        if 'exp' in payload:
            exp_timestamp = payload['exp']
            current_timestamp = datetime.utcnow().timestamp()
            return current_timestamp >= exp_timestamp
        return False
    except Exception:
        return True

def get_token_remaining_time(token):
    """
    Get the remaining time until token expires in seconds
    """
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

def decode_jwt_unsafe(token):
    """
    Decode JWT without verification (for debugging only)
    """
    try:
        payload = jwt.decode(token, options={"verify_signature": False, "verify_aud": False})
        return payload
    except Exception:
        return None