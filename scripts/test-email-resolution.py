#!/usr/bin/env python3
"""
Test script to verify email resolution in Cognito
"""
import sys
import os
sys.path.append('backend')

from backend.app.auth.cognito import resolve_email_to_username, get_user_by_email
from backend.app.simple_logger import get_logger

logger = get_logger("test")

def test_email_resolution():
    """Test email resolution for both emails"""
    
    # Test vapfull@gmail.com
    print("=" * 50)
    print("Testing vapfull@gmail.com")
    print("=" * 50)
    
    try:
        username1 = resolve_email_to_username("vapfull@gmail.com")
        print(f"✅ Username for vapfull@gmail.com: {username1}")
        
        user_info1 = get_user_by_email("vapfull@gmail.com")
        attrs1 = {attr['Name']: attr['Value'] for attr in user_info1['UserAttributes']}
        print(f"✅ User attributes: {attrs1}")
        
    except Exception as e:
        print(f"❌ Error with vapfull@gmail.com: {e}")
    
    print("\n" + "=" * 50)
    print("Testing sicokaf444@obirah.com")
    print("=" * 50)
    
    try:
        username2 = resolve_email_to_username("sicokaf444@obirah.com")
        print(f"✅ Username for sicokaf444@obirah.com: {username2}")
        
        user_info2 = get_user_by_email("sicokaf444@obirah.com")
        attrs2 = {attr['Name']: attr['Value'] for attr in user_info2['UserAttributes']}
        print(f"✅ User attributes: {attrs2}")
        
    except Exception as e:
        print(f"❌ Error with sicokaf444@obirah.com: {e}")

if __name__ == "__main__":
    test_email_resolution()
