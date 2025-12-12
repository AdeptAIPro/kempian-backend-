"""
Unit tests for Jobvite crypto functions.
Tests encryption/decryption, HMAC, and RSA key operations.
"""

import unittest
import base64
from app.jobvite.crypto import (
    encrypt_at_rest,
    decrypt_at_rest,
    generate_rsa_key_pair,
    encrypt_onboarding_payload,
    decrypt_onboarding_response,
    verify_webhook_signature
)

class TestJobviteCrypto(unittest.TestCase):
    """Test crypto functions"""
    
    def test_encrypt_decrypt_at_rest(self):
        """Test AES-256-CBC encryption/decryption"""
        plaintext = "test_secret_key_12345"
        encrypted = encrypt_at_rest(plaintext)
        decrypted = decrypt_at_rest(encrypted)
        self.assertEqual(plaintext, decrypted)
    
    def test_encrypt_decrypt_different_values(self):
        """Test that different inputs produce different encrypted outputs"""
        plaintext1 = "secret1"
        plaintext2 = "secret2"
        encrypted1 = encrypt_at_rest(plaintext1)
        encrypted2 = encrypt_at_rest(plaintext2)
        self.assertNotEqual(encrypted1, encrypted2)
    
    def test_rsa_key_generation(self):
        """Test RSA key pair generation"""
        private_key, public_key = generate_rsa_key_pair()
        self.assertIn("BEGIN PRIVATE KEY", private_key)
        self.assertIn("BEGIN PUBLIC KEY", public_key)
        self.assertIn("END PRIVATE KEY", private_key)
        self.assertIn("END PUBLIC KEY", public_key)
    
    def test_onboarding_encryption_decryption(self):
        """Test RSA+AES encryption/decryption for Onboarding API"""
        # Generate key pair
        our_private_key, our_public_key = generate_rsa_key_pair()
        jobvite_public_key = our_public_key  # For testing, use same key
        
        # Test payload
        filter_json = {"candidateId": "12345", "processId": "67890"}
        
        # Encrypt
        encrypted = encrypt_onboarding_payload(filter_json, jobvite_public_key)
        self.assertIn("key", encrypted)
        self.assertIn("payload", encrypted)
        
        # Decrypt
        decrypted = decrypt_onboarding_response(
            encrypted["key"],
            encrypted["payload"],
            our_private_key
        )
        self.assertEqual(decrypted, filter_json)
    
    def test_webhook_signature_verification(self):
        """Test webhook signature verification"""
        signing_key = "test_signing_key_12345"
        raw_body = '{"eventType":"candidate.updated","id":"12345"}'
        
        # Generate signature
        import hmac
        import hashlib
        signature = base64.b64encode(
            hmac.new(
                signing_key.encode(),
                raw_body.encode(),
                hashlib.sha256
            ).digest()
        ).decode()
        
        # Verify
        is_valid = verify_webhook_signature(raw_body, signature, signing_key)
        self.assertTrue(is_valid)
        
        # Test invalid signature
        is_valid_fake = verify_webhook_signature(raw_body, "fake_signature", signing_key)
        self.assertFalse(is_valid_fake)

if __name__ == '__main__':
    unittest.main()

