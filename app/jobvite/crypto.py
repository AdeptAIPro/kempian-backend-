"""
Encryption utilities for Jobvite integration.
Handles at-rest encryption for secrets and RSA/AES for Onboarding API.
"""

import os
import base64
import time
from typing import Dict
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding, hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding as asym_padding
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import hashlib
import hmac
import json

# At-rest encryption key (should come from KMS in production)
def get_encryption_key() -> bytes:
    """Get encryption key from environment or generate from JWT secret"""
    key = os.getenv('JOBVITE_ENCRYPTION_KEY')
    if not key:
        # Fallback: derive from JWT secret (not ideal, but works)
        jwt_secret = os.getenv('JWT_SECRET_KEY', 'default-secret')
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'jobvite_salt',
            iterations=100000,
            backend=default_backend()
        )
        return kdf.derive(jwt_secret.encode())
    # If key is base64 encoded, decode it; otherwise use directly
    try:
        return base64.b64decode(key)
    except:
        # If not base64, pad or truncate to 32 bytes
        key_bytes = key.encode() if isinstance(key, str) else key
        return key_bytes.ljust(32, b'0')[:32]

def encrypt_at_rest(plaintext: str) -> str:
    """Encrypt plaintext for storage (AES-256-CBC)"""
    key = get_encryption_key()
    iv = os.urandom(16)
    
    cipher = Cipher(
        algorithms.AES(key),
        modes.CBC(iv),
        backend=default_backend()
    )
    encryptor = cipher.encryptor()
    
    # Pad plaintext
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(plaintext.encode()) + padder.finalize()
    
    # Encrypt
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()
    
    # Return base64(iv + ciphertext)
    return base64.b64encode(iv + ciphertext).decode()

def decrypt_at_rest(ciphertext: str) -> str:
    """Decrypt stored ciphertext"""
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
    
    # Unpad
    unpadder = padding.PKCS7(128).unpadder()
    plaintext = unpadder.update(padded_plaintext) + unpadder.finalize()
    
    return plaintext.decode()

# RSA Key Loading
def load_rsa_public_key(pem_string: str):
    """Load RSA public key from PEM string"""
    return serialization.load_pem_public_key(pem_string.encode())

def load_rsa_private_key(pem_string: str):
    """Load RSA private key from PEM string"""
    return serialization.load_pem_private_key(pem_string.encode(), password=None)

def generate_rsa_key_pair():
    """Generate RSA 2048 key pair"""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )
    public_key = private_key.public_key()
    
    # Serialize to PEM
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    
    return private_pem.decode(), public_pem.decode()

# Onboarding API Encryption (RSA + AES)
def encrypt_onboarding_payload(filter_json: dict, jobvite_public_key_pem: str) -> dict:
    """
    Encrypt payload for Jobvite Onboarding API.
    
    Steps:
    1. Serialize filter_json to JSON UTF-8 string
    2. Generate random 256-bit AES key
    3. Encrypt JSON with AES (ECB mode, PKCS5 padding)
    4. Encrypt AES key with Jobvite's RSA public key (PKCS1 padding)
    5. Base64 encode both
    """
    # Step 1: Serialize to JSON
    json_string = json.dumps(filter_json, separators=(',', ':'))
    
    # Step 2: Generate AES key (256-bit = 32 bytes)
    aes_key = os.urandom(32)
    
    # Step 3: Encrypt JSON with AES-256-ECB (Note: ECB is insecure but required by Jobvite spec)
    cipher = Cipher(
        algorithms.AES(aes_key),
        modes.ECB(),  # ECB mode as per Jobvite spec
        backend=default_backend()
    )
    encryptor = cipher.encryptor()
    
    # Pad to block size (16 bytes for AES)
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(json_string.encode('utf-8')) + padder.finalize()
    
    encrypted_payload = encryptor.update(padded_data) + encryptor.finalize()
    payload_base64 = base64.b64encode(encrypted_payload).decode()
    
    # Step 4: Encrypt AES key with RSA
    public_key = load_rsa_public_key(jobvite_public_key_pem)
    encrypted_aes_key = public_key.encrypt(
        aes_key,
        asym_padding.PKCS1v15()  # PKCS1 padding as per Jobvite spec
    )
    key_base64 = base64.b64encode(encrypted_aes_key).decode()
    
    # Step 5: Return structure
    return {
        "key": key_base64,
        "payload": payload_base64
    }

def decrypt_onboarding_response(response_json: dict, our_private_key_pem: str) -> dict:
    """
    Decrypt response from Jobvite Onboarding API.
    
    Steps:
    1. Base64 decode 'key', decrypt with our RSA private key → AES key
    2. Base64 decode 'payload', decrypt with AES key → JSON string
    3. Parse JSON → dict
    """
    # Step 1: Decrypt AES key
    encrypted_key_b64 = response_json.get("key")
    encrypted_payload_b64 = response_json.get("payload")
    
    if not encrypted_key_b64 or not encrypted_payload_b64:
        raise ValueError("Missing 'key' or 'payload' in response")
    
    encrypted_key = base64.b64decode(encrypted_key_b64)
    encrypted_payload = base64.b64decode(encrypted_payload_b64)
    
    private_key = load_rsa_private_key(our_private_key_pem)
    aes_key = private_key.decrypt(
        encrypted_key,
        asym_padding.PKCS1v15()
    )
    
    # Step 2: Decrypt payload
    cipher = Cipher(
        algorithms.AES(aes_key),
        modes.ECB(),
        backend=default_backend()
    )
    decryptor = cipher.decryptor()
    
    padded_plaintext = decryptor.update(encrypted_payload) + decryptor.finalize()
    
    # Unpad
    unpadder = padding.PKCS7(128).unpadder()
    json_string = unpadder.update(padded_plaintext) + unpadder.finalize()
    
    # Step 3: Parse JSON
    return json.loads(json_string.decode('utf-8'))

# HMAC Authentication Headers for Jobvite v2 API (per official PDF spec)
def build_jobvite_v2_hmac_headers(api_key: str, api_secret: str, epoch: int = None) -> Dict[str, str]:
    """
    Build HMAC authentication headers for Jobvite v2 API per official PDF specification.
    
    Returns headers:
    - X-JVI-API: API key
    - X-JVI-SIGN: Base64(HMAC_SHA256(apiSecret, apiKey + "|" + epoch))
    - X-JVI-EPOCH: Unix timestamp in seconds
    
    Formula: signature = Base64(HMAC_SHA256(apiSecret, apiKey + "|" + epoch))
    """
    if epoch is None:
        epoch = int(time.time())
    
    # Build signature: HMAC_SHA256(apiSecret, apiKey + "|" + epoch)
    to_hash = f"{api_key}|{epoch}"
    sig = hmac.new(
        api_secret.encode("utf-8"),
        to_hash.encode("utf-8"),
        hashlib.sha256
    ).digest()
    sig_b64 = base64.b64encode(sig).decode("utf-8")
    
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-JVI-API": api_key,
        "X-JVI-SIGN": sig_b64,
        "X-JVI-EPOCH": str(epoch),
        "User-Agent": "Kempian/3.0"
    }

# HMAC Signature for Webhooks
def verify_webhook_signature(raw_body: bytes, signature_header: str, signing_key: str) -> bool:
    """
    Verify Jobvite webhook signature.
    
    Signature format: Base64(HMAC-SHA256(raw_body, signing_key))
    """
    expected_signature = base64.b64encode(
        hmac.new(
            signing_key.encode(),
            raw_body,
            hashlib.sha256
        ).digest()
    ).decode()
    
    return hmac.compare_digest(expected_signature, signature_header)

