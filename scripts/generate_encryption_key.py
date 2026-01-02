"""
Generate a secure encryption key for Jobvite integration.

This script generates a 32-byte (256-bit) encryption key and base64 encodes it
for use in the ENCRYPTION_KEY environment variable.

Usage:
    python backend/scripts/generate_encryption_key.py
"""

import secrets
import base64

def generate_encryption_key():
    """Generate a secure 32-byte encryption key"""
    # Generate 32 random bytes
    key_bytes = secrets.token_bytes(32)
    
    # Base64 encode for storage in environment variable
    key_base64 = base64.b64encode(key_bytes).decode('utf-8')
    
    return key_base64

if __name__ == '__main__':
    print("=" * 60)
    print("Jobvite Encryption Key Generator")
    print("=" * 60)
    print()
    
    key = generate_encryption_key()
    
    print("Generated encryption key (32 bytes, base64 encoded):")
    print()
    print(key)
    print()
    print("=" * 60)
    print("Add this to your .env file:")
    print(f"ENCRYPTION_KEY={key}")
    print("=" * 60)
    print()
    print("IMPORTANT:")
    print("1. Keep this key secure and secret")
    print("2. Do not commit it to version control")
    print("3. Store it securely (e.g., AWS Secrets Manager)")
    print("4. If you lose this key, you cannot decrypt existing data")
    print()

