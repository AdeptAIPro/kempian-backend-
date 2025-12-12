"""
Unit tests for Jobvite v2 HMAC authentication headers.
"""

import pytest
import base64
import hmac
from hashlib import sha256
from app.jobvite.crypto import build_jobvite_v2_hmac_headers


def test_hmac_header_matches_expected():
    """Test that HMAC headers match the expected format per Jobvite PDF spec."""
    api_key = "testkey"
    api_secret = "testsecret"
    epoch = 1609459200  # Fixed epoch for deterministic testing (2021-01-01 00:00:00 UTC)
    
    headers = build_jobvite_v2_hmac_headers(api_key, api_secret, epoch=epoch)
    
    # Verify all required headers are present
    assert "X-JVI-API" in headers
    assert "X-JVI-SIGN" in headers
    assert "X-JVI-EPOCH" in headers
    assert "Content-Type" in headers
    assert "Accept" in headers
    
    # Verify header values
    assert headers["X-JVI-API"] == api_key
    assert headers["X-JVI-EPOCH"] == str(epoch)
    
    # Verify signature calculation matches expected formula
    # Formula: Base64(HMAC_SHA256(apiSecret, apiKey + "|" + epoch))
    expected_to_hash = f"{api_key}|{epoch}"
    expected_sig = base64.b64encode(
        hmac.new(
            api_secret.encode("utf-8"),
            expected_to_hash.encode("utf-8"),
            sha256
        ).digest()
    ).decode("utf-8")
    
    assert headers["X-JVI-SIGN"] == expected_sig


def test_hmac_header_uses_current_time_when_epoch_not_provided():
    """Test that HMAC headers use current time when epoch is not provided."""
    import time
    
    api_key = "testkey"
    api_secret = "testsecret"
    
    before = int(time.time())
    headers = build_jobvite_v2_hmac_headers(api_key, api_secret)
    after = int(time.time())
    
    epoch = int(headers["X-JVI-EPOCH"])
    assert before <= epoch <= after
    
    # Verify signature is valid for this epoch
    expected_to_hash = f"{api_key}|{epoch}"
    expected_sig = base64.b64encode(
        hmac.new(
            api_secret.encode("utf-8"),
            expected_to_hash.encode("utf-8"),
            sha256
        ).digest()
    ).decode("utf-8")
    
    assert headers["X-JVI-SIGN"] == expected_sig


def test_hmac_header_different_epochs_produce_different_signatures():
    """Test that different epochs produce different signatures."""
    api_key = "testkey"
    api_secret = "testsecret"
    
    headers1 = build_jobvite_v2_hmac_headers(api_key, api_secret, epoch=1000)
    headers2 = build_jobvite_v2_hmac_headers(api_key, api_secret, epoch=2000)
    
    # Signatures should be different
    assert headers1["X-JVI-SIGN"] != headers2["X-JVI-SIGN"]
    
    # But API key should be the same
    assert headers1["X-JVI-API"] == headers2["X-JVI-API"] == api_key


def test_hmac_header_different_secrets_produce_different_signatures():
    """Test that different secrets produce different signatures."""
    api_key = "testkey"
    secret1 = "secret1"
    secret2 = "secret2"
    epoch = 1609459200
    
    headers1 = build_jobvite_v2_hmac_headers(api_key, secret1, epoch=epoch)
    headers2 = build_jobvite_v2_hmac_headers(api_key, secret2, epoch=epoch)
    
    # Signatures should be different
    assert headers1["X-JVI-SIGN"] != headers2["X-JVI-SIGN"]
    
    # But API key and epoch should be the same
    assert headers1["X-JVI-API"] == headers2["X-JVI-API"] == api_key
    assert headers1["X-JVI-EPOCH"] == headers2["X-JVI-EPOCH"] == str(epoch)

