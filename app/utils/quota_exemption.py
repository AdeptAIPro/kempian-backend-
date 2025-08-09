import os
import json

def is_quota_exempt(email):
    """
    Check if an email is exempt from quota restrictions.
    This function reads from the whitelist file created by the quota_whitelist.py script.
    """
    
    # Path to whitelist file
    whitelist_file = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'quota_whitelist.json')
    
    try:
        if os.path.exists(whitelist_file):
            with open(whitelist_file, 'r') as f:
                data = json.load(f)
                whitelist_emails = data.get('whitelist_emails', [])
                return email.lower() in [e.lower() for e in whitelist_emails]
    except Exception:
        pass
    
    return False
