#!/usr/bin/env python3
"""
Test JWT token debugging
"""

import sys
import os
sys.path.append('backend')

from backend.app.auth.jwt_utils import verify_cognito_jwt, decode_jwt_unsafe
from backend.app.simple_logger import get_logger

logger = get_logger("jwt_debug")

def test_jwt_token(token):
    """Test JWT token processing"""
    print(f"Testing JWT token: {token[:50]}...")
    
    try:
        # First, try unsafe decoding to see what's in the token
        print("\n1. Unsafe JWT decoding:")
        unsafe_payload = decode_jwt_unsafe(token)
        if unsafe_payload:
            print(f"   Email: {unsafe_payload.get('email', 'N/A')}")
            print(f"   Tenant ID: {unsafe_payload.get('custom:tenant_id', 'N/A')}")
            print(f"   Role: {unsafe_payload.get('custom:role', 'N/A')}")
            print(f"   User Type: {unsafe_payload.get('custom:user_type', 'N/A')}")
            print(f"   Issuer: {unsafe_payload.get('iss', 'N/A')}")
            print(f"   Audience: {unsafe_payload.get('aud', 'N/A')}")
        else:
            print("   Failed to decode token")
        
        # Now try proper verification
        print("\n2. Proper JWT verification:")
        verified_payload = verify_cognito_jwt(token)
        if verified_payload:
            print(f"   Email: {verified_payload.get('email', 'N/A')}")
            print(f"   Tenant ID: {verified_payload.get('custom:tenant_id', 'N/A')}")
            print(f"   Role: {verified_payload.get('custom:role', 'N/A')}")
            print(f"   User Type: {verified_payload.get('custom:user_type', 'N/A')}")
        else:
            print("   JWT verification failed")
            
    except Exception as e:
        print(f"Error testing JWT: {e}")

def main():
    # Test with a sample token (this would normally be a real token from the frontend)
    sample_token = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiYWRtaW4iOnRydWV9.EkN-DOsnsuRjRO6BxXemmJDm3HbxZR0X3MQL6OVQEN4c"
    
    print("JWT Token Debug Test")
    print("=" * 50)
    
    test_jwt_token(sample_token)
    
    print("\n" + "=" * 50)
    print("Test completed")

if __name__ == "__main__":
    main()
