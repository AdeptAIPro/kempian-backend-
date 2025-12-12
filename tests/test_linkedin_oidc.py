"""
Unit tests for LinkedIn OIDC authentication
Tests token exchange, ID token validation, JWKS handling, and scope verification
"""

import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock
from jose import jwt
from app.auth.linkedin_oidc import (
    validate_id_token,
    exchange_code_for_token,
    get_linkedin_jwks,
    validate_state,
    store_state,
    encrypt_token,
    decrypt_token
)


class TestTokenExchange:
    """Tests for token exchange functionality"""
    
    @patch('app.auth.linkedin_oidc.requests.post')
    def test_exchange_token_success(self, mock_post):
        """Test successful token exchange"""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            'access_token': 'test_access_token',
            'expires_in': 5184000,
            'id_token': 'test_id_token',
            'scope': 'openid profile email r_profile_basicinfo r_verify'
        }
        mock_post.return_value = mock_response
        
        success, token_data, error = exchange_code_for_token('test_code', 'http://test.com/callback')
        
        assert success is True
        assert token_data is not None
        assert token_data['access_token'] == 'test_access_token'
        assert error is None
    
    @patch('app.auth.linkedin_oidc.requests.post')
    def test_exchange_token_failure(self, mock_post):
        """Test token exchange failure"""
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 400
        mock_response.json.return_value = {'error_description': 'Invalid code'}
        mock_response.text = 'Invalid code'
        mock_post.return_value = mock_response
        
        success, token_data, error = exchange_code_for_token('invalid_code', 'http://test.com/callback')
        
        assert success is False
        assert token_data is None
        assert error is not None
    
    @patch('app.auth.linkedin_oidc.requests.post')
    def test_exchange_token_rate_limit(self, mock_post):
        """Test token exchange with rate limiting and retry"""
        # First call: rate limited
        mock_response_429 = Mock()
        mock_response_429.status_code = 429
        mock_response_429.headers = {'Retry-After': '2'}
        
        # Second call: success
        mock_response_success = Mock()
        mock_response_success.ok = True
        mock_response_success.json.return_value = {
            'access_token': 'test_token',
            'scope': 'openid r_profile_basicinfo r_verify'
        }
        
        mock_post.side_effect = [mock_response_429, mock_response_success]
        
        with patch('app.auth.linkedin_oidc.time.sleep'):  # Mock sleep to speed up test
            success, token_data, error = exchange_code_for_token('test_code', 'http://test.com/callback')
        
        assert success is True
        assert mock_post.call_count == 2
    
    @patch('app.auth.linkedin_oidc.requests.post')
    def test_exchange_token_missing_scopes(self, mock_post):
        """Test token exchange with missing required scopes"""
        mock_response = Mock()
        mock_response.ok = True
        mock_response.json.return_value = {
            'access_token': 'test_token',
            'scope': 'openid profile'  # Missing r_verify
        }
        mock_post.return_value = mock_response
        
        success, token_data, error = exchange_code_for_token('test_code', 'http://test.com/callback')
        
        # Should still succeed but log warning about missing scopes
        assert success is True
        assert token_data is not None


class TestIDTokenValidation:
    """Tests for ID token validation"""
    
    def test_validate_id_token_expired(self):
        """Test validation of expired ID token"""
        # Create an expired token
        expired_payload = {
            'sub': 'test_user',
            'exp': int(time.time()) - 3600,  # Expired 1 hour ago
            'iat': int(time.time()) - 7200,
            'aud': '228358385',
            'iss': 'https://www.linkedin.com'
        }
        
        # This test would need a real signed token, so we'll mock the validation
        with patch('app.auth.linkedin_oidc.jwt.decode') as mock_decode:
            mock_decode.side_effect = jwt.ExpiredSignatureError("Token expired")
            
            is_valid, payload = validate_id_token('expired_token')
            
            assert is_valid is False
            assert payload is None
    
    def test_validate_id_token_wrong_audience(self):
        """Test validation with wrong audience"""
        with patch('app.auth.linkedin_oidc.jwt.decode') as mock_decode:
            mock_decode.side_effect = jwt.JWTClaimsError("Invalid audience")
            
            is_valid, payload = validate_id_token('invalid_token')
            
            assert is_valid is False
            assert payload is None
    
    def test_validate_id_token_with_nonce(self):
        """Test ID token validation with nonce"""
        test_nonce = 'test_nonce_value'
        
        with patch('app.auth.linkedin_oidc.jwt.decode') as mock_decode:
            # Token with matching nonce
            mock_decode.return_value = {
                'sub': 'test_user',
                'nonce': test_nonce,
                'exp': int(time.time()) + 3600,
                'iat': int(time.time()),
                'aud': '228358385',
                'iss': 'https://www.linkedin.com'
            }
            
            with patch('app.auth.linkedin_oidc.get_linkedin_jwks') as mock_jwks:
                mock_jwks.return_value = {'keys': []}
                with patch('app.auth.linkedin_oidc.jwt.get_unverified_header') as mock_header:
                    mock_header.return_value = {'kid': 'test_kid'}
                    with patch('app.auth.linkedin_oidc.jwk.construct') as mock_construct:
                        mock_construct.return_value = Mock()
                        
                        is_valid, payload = validate_id_token('test_token', nonce=test_nonce)
                        
                        # Should validate successfully with matching nonce
                        assert is_valid is True
                        assert payload is not None
                        assert payload['nonce'] == test_nonce


class TestJWKSHandling:
    """Tests for JWKS fetching and caching"""
    
    @patch('app.auth.linkedin_oidc.requests.get')
    def test_jwks_fetch_success(self, mock_get):
        """Test successful JWKS fetch"""
        mock_response = Mock()
        mock_response.json.return_value = {'keys': [{'kid': 'test_kid'}]}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        jwks = get_linkedin_jwks()
        
        assert jwks is not None
        assert 'keys' in jwks
        mock_get.assert_called_once()
    
    @patch('app.auth.linkedin_oidc.requests.get')
    def test_jwks_fetch_failure_with_cache(self, mock_get):
        """Test JWKS fetch failure with cached fallback"""
        # Set up cached JWKS
        from app.auth.linkedin_oidc import _jwks_cache, _jwks_cache_time
        _jwks_cache = {'keys': [{'kid': 'cached_kid'}]}
        _jwks_cache_time = time.time() - 100  # Old cache
        
        mock_get.side_effect = Exception("Network error")
        
        jwks = get_linkedin_jwks()
        
        # Should return cached JWKS
        assert jwks is not None
        assert jwks == _jwks_cache
    
    @patch('app.auth.linkedin_oidc.requests.get')
    def test_jwks_force_refresh(self, mock_get):
        """Test forced JWKS refresh"""
        mock_response = Mock()
        mock_response.json.return_value = {'keys': [{'kid': 'new_kid'}]}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        jwks = get_linkedin_jwks(force_refresh=True)
        
        assert jwks is not None
        mock_get.assert_called_once()


class TestStateManagement:
    """Tests for OAuth state management"""
    
    def test_store_and_validate_state(self):
        """Test state storage and validation"""
        test_state = 'test_state_123'
        
        # Store state
        store_state(test_state, 'test_session')
        
        # Validate state
        is_valid = validate_state(test_state)
        
        assert is_valid is True
        
        # State should be one-time use
        is_valid_again = validate_state(test_state)
        assert is_valid_again is False
    
    def test_validate_expired_state(self):
        """Test validation of expired state"""
        test_state = 'expired_state'
        
        # Store state with old timestamp
        from app.auth.linkedin_oidc import _state_store
        _state_store[test_state] = {
            'created_at': time.time() - 400,  # 400 seconds ago (expired)
            'session_id': 'test'
        }
        
        is_valid = validate_state(test_state)
        
        assert is_valid is False


class TestTokenEncryption:
    """Tests for token encryption/decryption"""
    
    def test_encrypt_decrypt_token(self):
        """Test token encryption and decryption"""
        original_token = 'test_access_token_12345'
        
        encrypted = encrypt_token(original_token)
        decrypted = decrypt_token(encrypted)
        
        assert encrypted != original_token
        assert decrypted == original_token
    
    def test_encrypt_empty_token(self):
        """Test encryption of empty token"""
        encrypted = encrypt_token('')
        assert encrypted == ''
        
        decrypted = decrypt_token('')
        assert decrypted == ''
    
    def test_decrypt_with_wrong_key(self):
        """Test decryption with wrong key fails"""
        original_token = 'test_token'
        encrypted = encrypt_token(original_token)
        
        # Change encryption key
        import os
        original_key = os.getenv('LINKEDIN_ENCRYPTION_KEY')
        os.environ['LINKEDIN_ENCRYPTION_KEY'] = 'wrong_key_base64_encoded_32_bytes_long'
        
        try:
            # Re-import to get new key
            from importlib import reload
            import app.auth.linkedin_oidc
            reload(app.auth.linkedin_oidc)
            
            # Decryption should fail or produce garbage
            decrypted = app.auth.linkedin_oidc.decrypt_token(encrypted)
            assert decrypted != original_token
        finally:
            # Restore original key
            if original_key:
                os.environ['LINKEDIN_ENCRYPTION_KEY'] = original_key
            else:
                os.environ.pop('LINKEDIN_ENCRYPTION_KEY', None)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

