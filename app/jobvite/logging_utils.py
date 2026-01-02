"""
Structured logging utilities for Jobvite integration.
Includes redaction of sensitive data and consistent log format.
"""

import re
from typing import Dict, Any, Optional
from app.simple_logger import get_logger

logger = get_logger("jobvite_logging")

# Patterns for sensitive data redaction
SENSITIVE_PATTERNS = [
    (r'api[_-]?key["\']?\s*[:=]\s*["\']?([^"\'\s]+)', 'api_key'),
    (r'api[_-]?secret["\']?\s*[:=]\s*["\']?([^"\'\s]+)', 'api_secret'),
    (r'password["\']?\s*[:=]\s*["\']?([^"\'\s]+)', 'password'),
    (r'secret["\']?\s*[:=]\s*["\']?([^"\'\s]+)', 'secret'),
    (r'-----BEGIN.*?KEY-----.*?-----END.*?KEY-----', 'rsa_key'),
    (r'webhook[_-]?signing[_-]?key["\']?\s*[:=]\s*["\']?([^"\'\s]+)', 'webhook_key'),
]

def redact_sensitive_data(message: str) -> str:
    """
    Redact sensitive data from log messages.
    
    Args:
        message: Log message that may contain sensitive data
    
    Returns:
        Message with sensitive data redacted
    """
    redacted = message
    
    for pattern, data_type in SENSITIVE_PATTERNS:
        if data_type in ['api_key', 'api_secret', 'password', 'secret', 'webhook_key']:
            # Replace with [REDACTED:type]
            redacted = re.sub(pattern, f'[REDACTED:{data_type}]', redacted, flags=re.IGNORECASE | re.DOTALL)
        elif data_type == 'rsa_key':
            # Replace entire key block
            redacted = re.sub(pattern, '[REDACTED:rsa_key]', redacted, flags=re.IGNORECASE | re.DOTALL)
    
    return redacted

def log_jobvite_request(
    tenant_id: int,
    company_id: str,
    endpoint: str,
    method: str = 'GET',
    status_code: Optional[int] = None,
    duration_ms: Optional[float] = None,
    error: Optional[str] = None,
    items_count: Optional[int] = None,
    rate_limit_retries: int = 0
):
    """
    Log a Jobvite API request with structured fields.
    
    Args:
        tenant_id: Tenant ID
        company_id: Jobvite company ID
        endpoint: API endpoint
        method: HTTP method
        status_code: HTTP status code
        duration_ms: Request duration in milliseconds
        error: Error message (if any)
        items_count: Number of items returned
        rate_limit_retries: Number of rate limit retries
    """
    log_data = {
        'tenant_id': tenant_id,
        'company_id': company_id,
        'jobvite_endpoint': endpoint,
        'method': method,
        'status_code': status_code,
        'duration_ms': duration_ms,
        'items_count': items_count,
        'rate_limit_retries': rate_limit_retries
    }
    
    if error:
        log_data['error'] = redact_sensitive_data(error)
        logger.error(f"Jobvite API request failed: {log_data}")
    else:
        logger.info(f"Jobvite API request: {log_data}")

def log_sync_operation(
    tenant_id: int,
    sync_type: str,
    status: str,
    synced_count: int = 0,
    error_count: int = 0,
    duration_seconds: Optional[float] = None,
    error: Optional[str] = None
):
    """
    Log a sync operation with structured fields.
    
    Args:
        tenant_id: Tenant ID
        sync_type: Type of sync ('jobs', 'candidates', 'onboarding')
        status: Sync status ('success', 'partial', 'failed')
        synced_count: Number of items synced
        error_count: Number of errors
        duration_seconds: Sync duration in seconds
        error: Error message (if any)
    """
    log_data = {
        'tenant_id': tenant_id,
        'sync_type': sync_type,
        'status': status,
        'synced_count': synced_count,
        'error_count': error_count,
        'duration_seconds': duration_seconds
    }
    
    if error:
        log_data['error'] = redact_sensitive_data(error)
        logger.error(f"Sync operation failed: {log_data}")
    else:
        logger.info(f"Sync operation completed: {log_data}")

def log_webhook_event(
    tenant_id: int,
    company_id: str,
    event_type: str,
    source: str,
    signature_valid: bool,
    processed: bool,
    error: Optional[str] = None
):
    """
    Log a webhook event with structured fields.
    
    Args:
        tenant_id: Tenant ID
        company_id: Jobvite company ID
        event_type: Webhook event type
        source: Webhook source ('candidate', 'job', 'onboarding')
        signature_valid: Whether signature was valid
        processed: Whether webhook was processed
        error: Error message (if any)
    """
    log_data = {
        'tenant_id': tenant_id,
        'company_id': company_id,
        'event_type': event_type,
        'source': source,
        'signature_valid': signature_valid,
        'processed': processed
    }
    
    if error:
        log_data['error'] = redact_sensitive_data(error)
        logger.error(f"Webhook event failed: {log_data}")
    else:
        logger.info(f"Webhook event received: {log_data}")

# Security policy: Never log sensitive data
def safe_log(message: str, level: str = 'info', **kwargs):
    """
    Safe logging that automatically redacts sensitive data.
    
    Args:
        message: Log message
        level: Log level ('info', 'warning', 'error', 'debug')
        **kwargs: Additional structured fields
    """
    redacted_message = redact_sensitive_data(message)
    
    # Redact any sensitive values in kwargs
    safe_kwargs = {}
    for key, value in kwargs.items():
        if any(sensitive in key.lower() for sensitive in ['key', 'secret', 'password', 'token']):
            safe_kwargs[key] = '[REDACTED]'
        elif isinstance(value, str) and ('-----BEGIN' in value or len(value) > 50):
            # Likely a key or long secret
            safe_kwargs[key] = '[REDACTED]'
        else:
            safe_kwargs[key] = value
    
    if level == 'error':
        logger.error(f"{redacted_message} | {safe_kwargs}")
    elif level == 'warning':
        logger.warning(f"{redacted_message} | {safe_kwargs}")
    elif level == 'debug':
        logger.debug(f"{redacted_message} | {safe_kwargs}")
    else:
        logger.info(f"{redacted_message} | {safe_kwargs}")

