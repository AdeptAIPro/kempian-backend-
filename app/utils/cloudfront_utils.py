import os
import datetime
import base64
from urllib.parse import quote
from typing import Optional

import boto3
from botocore.exceptions import ClientError
from botocore.signers import CloudFrontSigner
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding

import logging

logger = logging.getLogger('cloudfront_utils')

def normalize_resume_key(key: str) -> str:
    """Ensure key starts with RESUME_PREFIX and remove any 'resumes/' legacy segment."""
    try:
        prefix = os.getenv('RESUME_PREFIX', 'career_resume/')
        # Remove duplicate leading slashes
        k = key.lstrip('/')
        # Strip legacy 'resumes/' if present after prefix or at root
        if k.startswith('resumes/'):
            k = k[len('resumes/') : ]
        if k.startswith(prefix + 'resumes/'):
            k = prefix + k[len(prefix + 'resumes/') : ]
        # Ensure single prefix
        if not k.startswith(prefix):
            k = prefix + k
        return k
    except Exception:
        return key

class CloudFrontURLSigner:
    """CloudFront URL signer for secure file downloads"""
    
    def __init__(self):
        self.aws_region = os.getenv('AWS_REGION', 'ap-south-1')
        self.cf_private_key_secret = os.getenv('CF_PRIVATE_KEY_SECRET', 'prod/cf_signing_private_key')
        self.cloudfront_domain = os.getenv('CLOUDFRONT_DOMAIN', '').rstrip('/')
        self.key_pair_id = os.getenv('KEY_PAIR_ID')
        self.cf_private_key_pem = os.getenv('CF_PRIVATE_KEY_PEM')
        self.url_ttl_minutes = int(os.getenv('URL_TTL_MINUTES', '60'))
        
        # Validate required configuration
        if not self.cloudfront_domain:
            raise ValueError("CLOUDFRONT_DOMAIN environment variable is required")
        if not self.key_pair_id:
            raise ValueError("KEY_PAIR_ID environment variable is required")
    
    def _get_private_key_pem_bytes(self) -> bytes:
        """Get private key PEM bytes from environment or Secrets Manager"""
        # If PEM provided directly via env var, use it
        if self.cf_private_key_pem:
            return self.cf_private_key_pem.encode("utf-8")
        
        # Otherwise fetch from Secrets Manager
        try:
            session = boto3.Session(region_name=self.aws_region)
            sm = session.client("secretsmanager")
            resp = sm.get_secret_value(SecretId=self.cf_private_key_secret)
            
            if "SecretString" in resp and resp["SecretString"]:
                return resp["SecretString"].encode("utf-8")
            if "SecretBinary" in resp and resp["SecretBinary"]:
                # SecretBinary is base64-encoded by the API
                return base64.b64decode(resp["SecretBinary"])
            
            raise ValueError("Secret has no SecretString or SecretBinary content")
            
        except ClientError as e:
            logger.error(f"Failed to read secret '{self.cf_private_key_secret}': {e}")
            raise ValueError(f"Failed to read secret '{self.cf_private_key_secret}': {e}")
    
    def _build_signer(self) -> CloudFrontSigner:
        """Build CloudFront signer with private key"""
        try:
            pem_bytes = self._get_private_key_pem_bytes()
            private_key = serialization.load_pem_private_key(pem_bytes, password=None)
            
            def _rsa_signer(message: bytes) -> bytes:
                # CloudFront requires RSA-SHA1 (PKCS#1 v1.5) for URL signing
                return private_key.sign(message, padding.PKCS1v15(), hashes.SHA1())
            
            return CloudFrontSigner(self.key_pair_id, _rsa_signer)
            
        except Exception as e:
            logger.error(f"Failed to build CloudFront signer: {e}")
            raise ValueError(f"Failed to build CloudFront signer: {e}")
    
    def generate_signed_url(self, object_key: str, ttl_minutes: Optional[int] = None) -> str:
        """
        Generate a signed CloudFront URL for the given object key
        
        Args:
            object_key: S3 object key (e.g., 'resumes/user123_job456_resume.pdf')
            ttl_minutes: URL expiration time in minutes (defaults to configured value)
        
        Returns:
            Signed CloudFront URL
        """
        try:
            ttl = ttl_minutes or self.url_ttl_minutes
            # Normalize key with configured prefix and strip legacy segments
            object_key = normalize_resume_key(object_key)
            
            # Keep slashes; encode only unsafe chars
            encoded_key = quote(object_key, safe="/")
            url = f"{self.cloudfront_domain}/{encoded_key}"
            
            # Calculate expiration time
            expires = datetime.datetime.utcnow() + datetime.timedelta(minutes=ttl)
            
            # Generate signed URL
            signer = self._build_signer()
            signed_url = signer.generate_presigned_url(url, date_less_than=expires)
            
            logger.info(f"Generated CloudFront signed URL for {object_key}, expires in {ttl} minutes")
            return signed_url
            
        except Exception as e:
            logger.error(f"Failed to generate CloudFront signed URL for {object_key}: {e}")
            raise ValueError(f"Failed to generate CloudFront signed URL: {e}")

# Global instance
_cloudfront_signer = None

def get_cloudfront_signer() -> CloudFrontURLSigner:
    """Get or create CloudFront signer instance"""
    global _cloudfront_signer
    if _cloudfront_signer is None:
        _cloudfront_signer = CloudFrontURLSigner()
    return _cloudfront_signer

def generate_resume_download_url(resume_s3_key: str, ttl_minutes: int = 60) -> str:
    """
    Generate a signed CloudFront URL for resume download
    
    Args:
        resume_s3_key: S3 key for the resume file
        ttl_minutes: URL expiration time in minutes
    
    Returns:
        Signed CloudFront URL for resume download
    """
    try:
        signer = get_cloudfront_signer()
        return signer.generate_signed_url(resume_s3_key, ttl_minutes)
    except Exception as e:
        logger.error(f"Failed to generate resume download URL for {resume_s3_key}: {e}")
        raise
