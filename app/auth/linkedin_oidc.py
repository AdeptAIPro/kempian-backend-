"""
LinkedIn OIDC Authentication Service
Handles OAuth 2.0 / OIDC flow for LinkedIn Sign-In with Verified-on-LinkedIn integration
"""

import os
import secrets
import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from jose import jwt, jwk
from jose.utils import base64url_decode
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
import base64
from flask import session
from app.simple_logger import get_logger

logger = get_logger("linkedin_oidc")

# LinkedIn OAuth Configuration
LINKEDIN_CLIENT_ID = os.getenv('LINKEDIN_CLIENT_ID', '228358385')
LINKEDIN_CLIENT_SECRET = os.getenv('LINKEDIN_CLIENT_SECRET')
LINKEDIN_REDIRECT_URI = os.getenv('LINKEDIN_REDIRECT_URI')
LINKEDIN_OAUTH_AUTHORIZE = os.getenv('LINKEDIN_OAUTH_AUTHORIZE', 'https://www.linkedin.com/oauth/v2/authorization')
LINKEDIN_OAUTH_TOKEN = os.getenv('LINKEDIN_OAUTH_TOKEN', 'https://www.linkedin.com/oauth/v2/accessToken')
LINKEDIN_API_BASE = os.getenv('LINKEDIN_API_BASE', 'https://api.linkedin.com/rest')
LINKEDIN_API_VERSION = os.getenv('LINKEDIN_API_VERSION', '202504')
LINKEDIN_JWKS_URL = os.getenv('LINKEDIN_JWKS_URL', 'https://www.linkedin.com/oauth/openid/jwks')
LINKEDIN_SCOPES = os.getenv('LINKEDIN_SCOPES', 'openid profile email r_profile_basicinfo r_verify')

# State storage - use Redis if available, fallback to in-memory for dev
_state_store: Dict[str, dict] = {}
_redis_client = None
_redis_available = False

# JWKS cache
_jwks_cache: Optional[dict] = None
_jwks_cache_time: Optional[float] = None
JWKS_CACHE_TTL = 86400  # 24 hours

# Initialize Redis client for state storage
def _init_redis():
    """Initialize Redis client for state storage"""
    global _redis_client, _redis_available
    if _redis_client is not None:
        return
    
    try:
        import redis
        redis_url = os.getenv('REDIS_URL', '')
        if redis_url:
            _redis_client = redis.from_url(redis_url, decode_responses=True)
        else:
            # Try ElastiCache config
            elasticache_host = os.getenv('ELASTICACHE_ENDPOINT', '')
            if elasticache_host:
                elasticache_port = int(os.getenv('ELASTICACHE_PORT', '6379'))
                elasticache_auth = os.getenv('ELASTICACHE_AUTH_TOKEN', '')
                ssl_enabled = os.getenv('REDIS_SSL', 'false').lower() in ('true', '1', 'yes')
                _redis_client = redis.Redis(
                    host=elasticache_host,
                    port=elasticache_port,
                    password=elasticache_auth if elasticache_auth else None,
                    ssl=ssl_enabled,
                    decode_responses=True
                )
            else:
                # Try local Redis
                _redis_client = redis.Redis(
                    host=os.getenv('REDIS_HOST', 'localhost'),
                    port=int(os.getenv('REDIS_PORT', '6379')),
                    db=int(os.getenv('REDIS_DB', '0')),
                    password=os.getenv('REDIS_PASSWORD') or None,
                    decode_responses=True
                )
        
        # Test connection
        _redis_client.ping()
        _redis_available = True
        logger.info("Redis connected for LinkedIn OAuth state storage")
    except Exception as e:
        logger.warning(f"Redis not available for state storage, using in-memory fallback: {e}")
        _redis_client = None
        _redis_available = False

# Initialize on import
_init_redis()


def get_encryption_key() -> bytes:
    """Get encryption key from environment or derive from JWT secret"""
    key = os.getenv('LINKEDIN_ENCRYPTION_KEY')
    if not key:
        # Fallback: derive from JWT secret
        jwt_secret = os.getenv('JWT_SECRET_KEY', 'default-secret')
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'linkedin_salt',
            iterations=100000,
            backend=default_backend()
        )
        return kdf.derive(jwt_secret.encode())
    try:
        return base64.b64decode(key)
    except:
        key_bytes = key.encode() if isinstance(key, str) else key
        return key_bytes.ljust(32, b'0')[:32]


def encrypt_token(plaintext: str) -> str:
    """Encrypt token for storage (AES-256-CBC)"""
    if not plaintext:
        return ''
    key = get_encryption_key()
    iv = os.urandom(16)
    
    cipher = Cipher(
        algorithms.AES(key),
        modes.CBC(iv),
        backend=default_backend()
    )
    encryptor = cipher.encryptor()
    
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(plaintext.encode()) + padder.finalize()
    
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()
    return base64.b64encode(iv + ciphertext).decode()


def decrypt_token(ciphertext: str) -> str:
    """Decrypt stored token"""
    if not ciphertext:
        return ''
    key = get_encryption_key()
    data = base64.b64decode(ciphertext)
    
    iv = data[:16]
    encrypted = data[16:]
    
    cipher = Cipher(
        algorithms.AES(key),
        modes.CBC(iv),
        backend=default_backend()
    )
    decryptor = cipher.decryptor()
    
    padded_plaintext = decryptor.update(encrypted) + decryptor.finalize()
    
    unpadder = padding.PKCS7(128).unpadder()
    plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()
    
    return plaintext.decode()


def generate_state() -> str:
    """Generate cryptographically secure state for OAuth flow"""
    return secrets.token_urlsafe(32)


def store_state(state: str, session_id: Optional[str] = None) -> None:
    """Store state with TTL (5 minutes) - uses Redis if available"""
    global _redis_client, _redis_available
    
    # Re-initialize Redis if needed
    if _redis_client is None:
        _init_redis()
    
    session_id = session_id or (session.get('session_id') if hasattr(session, 'get') else None) or 'default'
    state_data = {
        'created_at': time.time(),
        'session_id': session_id
    }
    
    if _redis_available and _redis_client:
        try:
            # Store in Redis with 5 minute TTL
            key = f"linkedin:state:{state}"
            _redis_client.setex(key, 300, json.dumps(state_data))
            logger.debug(f"State stored in Redis: {state[:16]}...")
        except Exception as e:
            logger.warning(f"Redis state storage failed, using in-memory: {e}")
            _redis_available = False
            _state_store[state] = state_data
    else:
        # Fallback to in-memory
        _state_store[state] = state_data
        # Clean up expired states
        current_time = time.time()
        expired = [s for s, data in _state_store.items() 
                   if current_time - data.get('created_at', 0) > 300]
        for s in expired:
            del _state_store[s]


def validate_state(state: str) -> bool:
    """Validate state and remove it if valid - uses Redis if available"""
    global _redis_client, _redis_available
    
    # Re-initialize Redis if needed
    if _redis_client is None:
        _init_redis()
    
    if _redis_available and _redis_client:
        try:
            key = f"linkedin:state:{state}"
            state_json = _redis_client.get(key)
            if not state_json:
                logger.warning(f"State not found in Redis: {state[:16]}...")
                return False
            
            # Delete state (one-time use)
            _redis_client.delete(key)
            logger.debug(f"State validated and removed from Redis: {state[:16]}...")
            return True
        except Exception as e:
            logger.warning(f"Redis state validation failed, checking in-memory: {e}")
            _redis_available = False
    
    # Fallback to in-memory
    if state not in _state_store:
        return False
    
    data = _state_store[state]
    current_time = time.time()
    
    if current_time - data.get('created_at', 0) > 300:
        del _state_store[state]
        return False
    
    # Remove state after validation (one-time use)
    del _state_store[state]
    return True


def get_linkedin_jwks(force_refresh: bool = False) -> dict:
    """Fetch LinkedIn JWKS with caching and immediate refresh on failure"""
    global _jwks_cache, _jwks_cache_time
    
    current_time = time.time()
    
    # Return cached JWKS if still valid and not forcing refresh
    if not force_refresh and _jwks_cache and _jwks_cache_time and (current_time - _jwks_cache_time) < JWKS_CACHE_TTL:
        return _jwks_cache
    
    try:
        response = requests.get(LINKEDIN_JWKS_URL, timeout=10)
        response.raise_for_status()
        _jwks_cache = response.json()
        _jwks_cache_time = current_time
        logger.info("LinkedIn JWKS fetched and cached")
        return _jwks_cache
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch LinkedIn JWKS: {e}", exc_info=True)
        # If we have a cached JWKS, return it as fallback
        if _jwks_cache:
            logger.warning("Using expired JWKS cache as fallback - signature validation may fail")
            return _jwks_cache
        # If no cache and this is not a forced refresh, raise
        raise
    except Exception as e:
        logger.error(f"Unexpected error fetching LinkedIn JWKS: {e}", exc_info=True)
        if _jwks_cache:
            logger.warning("Using expired JWKS cache as fallback")
            return _jwks_cache
        raise


def validate_id_token(id_token: str, nonce: Optional[str] = None) -> Tuple[bool, Optional[dict]]:
    """
    Validate LinkedIn ID token signature and claims with immediate JWKS refresh on failure
    
    Args:
        id_token: The ID token to validate
        nonce: Optional nonce value to validate against token claims
    
    Returns:
        (is_valid, payload) tuple
    """
    try:
        # Decode without verification first to get header
        unverified = jwt.get_unverified_header(id_token)
        kid = unverified.get('kid')
        
        if not kid:
            logger.error("ID token missing 'kid' in header")
            return False, None
        
        # Get JWKS (may be cached)
        jwks = get_linkedin_jwks()
        
        # Find the key
        key = None
        for jwk_key in jwks.get('keys', []):
            if jwk_key.get('kid') == kid:
                key = jwk_key
                break
        
        if not key:
            logger.error(f"Key with kid '{kid}' not found in JWKS - forcing refresh")
            # Force refresh JWKS and try again
            jwks = get_linkedin_jwks(force_refresh=True)
            for jwk_key in jwks.get('keys', []):
                if jwk_key.get('kid') == kid:
                    key = jwk_key
                    break
            if not key:
                logger.error(f"Key with kid '{kid}' still not found after JWKS refresh")
                return False, None
        
        # Convert JWK to RSA key
        rsa_key = jwk.construct(key)
        
        try:
            # Verify and decode token
            payload = jwt.decode(
                id_token,
                rsa_key,
                algorithms=['RS256'],
                audience=LINKEDIN_CLIENT_ID,
                issuer='https://www.linkedin.com'
            )
        except jwt.JWTError as jwt_error:
            # If signature validation fails, try refreshing JWKS once
            if 'signature' in str(jwt_error).lower() or 'Invalid' in str(jwt_error):
                logger.warning(f"ID token signature validation failed, refreshing JWKS: {jwt_error}")
                jwks = get_linkedin_jwks(force_refresh=True)
                # Try to find key again
                key = None
                for jwk_key in jwks.get('keys', []):
                    if jwk_key.get('kid') == kid:
                        key = jwk_key
                        break
                if key:
                    rsa_key = jwk.construct(key)
                    payload = jwt.decode(
                        id_token,
                        rsa_key,
                        algorithms=['RS256'],
                        audience=LINKEDIN_CLIENT_ID,
                        issuer='https://www.linkedin.com'
                    )
                else:
                    raise
            else:
                raise
        
        # Additional claim validations
        now = int(time.time())
        if payload.get('exp', 0) <= now:
            logger.error("ID token has expired")
            return False, None
        
        if payload.get('iat', 0) > now + 60:  # Allow 60s clock skew
            logger.error("ID token issued in the future")
            return False, None
        
        # Validate nonce if provided
        if nonce and payload.get('nonce') != nonce:
            logger.error(f"ID token nonce mismatch - expected: {nonce[:16]}..., got: {payload.get('nonce', 'none')[:16] if payload.get('nonce') else 'none'}...")
            return False, None
        
        # Log success without exposing token
        logger.info(f"ID token validated successfully for subject: {payload.get('sub', 'unknown')[:16]}...")
        return True, payload
        
    except jwt.ExpiredSignatureError:
        logger.error("ID token signature expired")
        return False, None
    except jwt.JWTClaimsError as e:
        logger.error(f"ID token claims validation failed: {e}")
        return False, None
    except Exception as e:
        logger.error(f"ID token validation error: {e}", exc_info=True)
        return False, None


def exchange_code_for_token(code: str, redirect_uri: str) -> Tuple[bool, Optional[dict], Optional[str]]:
    """
    Exchange authorization code for access token with retry/backoff
    
    Returns:
        (success, token_data, error_message) tuple
    """
    import time
    max_retries = 3
    base_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                LINKEDIN_OAUTH_TOKEN,
                data={
                    'grant_type': 'authorization_code',
                    'code': code,
                    'redirect_uri': redirect_uri,
                    'client_id': LINKEDIN_CLIENT_ID,
                    'client_secret': LINKEDIN_CLIENT_SECRET
                },
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                timeout=10
            )
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', base_delay * (2 ** attempt)))
                if attempt < max_retries - 1:
                    logger.warning(f"Rate limited, retrying after {retry_after}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_after)
                    continue
                else:
                    error_msg = "Rate limit exceeded, please try again later"
                    logger.error(f"Token exchange rate limited: {error_msg}")
                    return False, None, error_msg
            
            # Handle server errors with retry
            if response.status_code >= 500:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Server error {response.status_code}, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
                else:
                    error_data = response.json() if response.content else {}
                    error_msg = error_data.get('error_description', f"Server error: {response.status_code}")
                    logger.error(f"Token exchange failed after {max_retries} attempts: {response.status_code} - {error_msg}")
                    return False, None, error_msg
            
            if not response.ok:
                error_data = response.json() if response.content else {}
                error_msg = error_data.get('error_description', response.text)
                logger.error(f"Token exchange failed: {response.status_code} - {error_msg}")
                return False, None, error_msg
            
            token_data = response.json()
            
            # Validate required fields
            if 'access_token' not in token_data:
                logger.error("Token response missing access_token")
                return False, None, "No access token in response"
            
            # Verify scopes
            returned_scopes = token_data.get('scope', '').split()
            required_scopes = ['openid', 'r_profile_basicinfo', 'r_verify']
            missing_scopes = [s for s in required_scopes if s not in returned_scopes]
            
            if missing_scopes:
                logger.warning(f"Missing required scopes: {missing_scopes}. Returned scopes: {returned_scopes}")
                # Don't fail, but log warning - some scopes may be optional
            
            # Log success without exposing tokens
            logger.info(f"Token exchange successful - scopes: {returned_scopes}")
            return True, token_data, None
            
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Token exchange timeout, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue
            else:
                logger.error("Token exchange timeout after all retries")
                return False, None, "Request timeout"
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Token exchange request failed, retrying in {delay}s: {e}")
                time.sleep(delay)
                continue
            else:
                logger.error(f"Token exchange request failed after {max_retries} attempts: {e}")
                return False, None, str(e)
        except Exception as e:
            logger.error(f"Token exchange error: {e}", exc_info=True)
            return False, None, str(e)
    
    return False, None, "Token exchange failed after all retries"


def get_identity_me(access_token: str) -> Tuple[bool, Optional[dict], Optional[str]]:
    """
    Fetch LinkedIn identityMe endpoint with retry/backoff
    
    Returns:
        (success, data, error_message) tuple
    """
    import time
    max_retries = 3
    base_delay = 1
    
    for attempt in range(max_retries):
        try:
            response = requests.get(
                f"{LINKEDIN_API_BASE}/identityMe",
                headers={
                    'Authorization': f'Bearer {access_token}',
                    'LinkedIn-Version': LINKEDIN_API_VERSION
                },
                timeout=10
            )
            
            # Handle rate limiting
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', base_delay * (2 ** attempt)))
                if attempt < max_retries - 1:
                    logger.warning(f"Rate limited on identityMe, retrying after {retry_after}s")
                    time.sleep(retry_after)
                    continue
                else:
                    return False, None, "Rate limit exceeded"
            
            # Handle server errors with retry
            if response.status_code >= 500:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"Server error {response.status_code} on identityMe, retrying in {delay}s")
                    time.sleep(delay)
                    continue
                else:
                    error_data = response.json() if response.content else {}
                    error_msg = error_data.get('message', f"Server error: {response.status_code}")
                    logger.error(f"identityMe call failed after {max_retries} attempts: {error_msg}")
                    return False, None, error_msg
            
            if not response.ok:
                error_data = response.json() if response.content else {}
                error_msg = error_data.get('message', response.text)
                logger.error(f"identityMe call failed: {response.status_code} - {error_msg}")
                return False, None, error_msg
            
            data = response.json()
            # Log success without exposing sensitive data
            logger.info(f"identityMe fetched successfully for linkedin_id: {data.get('id', 'unknown')[:16] if data.get('id') else 'unknown'}...")
            return True, data, None
            
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"identityMe timeout, retrying in {delay}s")
                time.sleep(delay)
                continue
            else:
                return False, None, "Request timeout"
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"identityMe request failed, retrying in {delay}s: {e}")
                time.sleep(delay)
                continue
            else:
                logger.error(f"identityMe request failed after {max_retries} attempts: {e}")
                return False, None, str(e)
        except Exception as e:
            logger.error(f"identityMe error: {e}", exc_info=True)
            return False, None, str(e)
    
    return False, None, "identityMe failed after all retries"


def get_verification_report(access_token: str, criteria: list = None) -> Tuple[bool, Optional[dict], Optional[str]]:
    """
    Fetch LinkedIn verificationReport endpoint
    
    Args:
        access_token: LinkedIn access token
        criteria: List of verification criteria (default: ['IDENTITY', 'WORKPLACE'])
    
    Returns:
        (success, data, error_message) tuple
    """
    if criteria is None:
        criteria = ['IDENTITY', 'WORKPLACE']
    
    try:
        # Build query string
        params = '&'.join([f'verificationCriteria={c}' for c in criteria])
        url = f"{LINKEDIN_API_BASE}/verificationReport?{params}"
        
        response = requests.get(
            url,
            headers={
                'Authorization': f'Bearer {access_token}',
                'LinkedIn-Version': LINKEDIN_API_VERSION
            },
            timeout=10
        )
        
        if not response.ok:
            error_data = response.json() if response.content else {}
            error_msg = error_data.get('message', response.text)
            logger.error(f"verificationReport call failed: {response.status_code} - {error_msg}")
            return False, None, error_msg
        
        data = response.json()
        logger.info("verificationReport fetched successfully")
        return True, data, None
        
    except requests.exceptions.RequestException as e:
        logger.error(f"verificationReport request failed: {e}")
        return False, None, str(e)
    except Exception as e:
        logger.error(f"verificationReport error: {e}")
        return False, None, str(e)


def generate_nonce() -> str:
    """Generate cryptographically secure nonce for ID token validation"""
    return secrets.token_urlsafe(32)


def store_nonce(nonce: str, state: str) -> None:
    """Store nonce associated with state (for validation on callback)"""
    global _redis_client, _redis_available
    
    if _redis_client is None:
        _init_redis()
    
    if _redis_available and _redis_client:
        try:
            key = f"linkedin:nonce:{state}"
            _redis_client.setex(key, 300, nonce)  # 5 minute TTL
        except Exception as e:
            logger.warning(f"Failed to store nonce in Redis: {e}")
    # Nonce is also stored in state data for in-memory fallback


def get_nonce(state: str) -> Optional[str]:
    """Retrieve nonce for state"""
    global _redis_client, _redis_available
    
    if _redis_client is None:
        _init_redis()
    
    if _redis_available and _redis_client:
        try:
            key = f"linkedin:nonce:{state}"
            nonce = _redis_client.get(key)
            if nonce:
                _redis_client.delete(key)  # One-time use
                return nonce
        except Exception as e:
            logger.warning(f"Failed to get nonce from Redis: {e}")
    
    return None


def build_authorization_url(state: str, nonce: Optional[str] = None) -> str:
    """Build LinkedIn OAuth authorization URL with optional nonce"""
    if nonce is None:
        nonce = generate_nonce()
        store_nonce(nonce, state)
    
    params = {
        'response_type': 'code',
        'client_id': LINKEDIN_CLIENT_ID,
        'redirect_uri': LINKEDIN_REDIRECT_URI,
        'state': state,
        'scope': LINKEDIN_SCOPES,
        'nonce': nonce
    }
    
    query_string = '&'.join([f"{k}={requests.utils.quote(str(v))}" for k, v in params.items()])
    return f"{LINKEDIN_OAUTH_AUTHORIZE}?{query_string}"

