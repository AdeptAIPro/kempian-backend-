import os
import json

def is_quota_exempt(email):
    """
    Check if an email is exempt from quota restrictions.
    This function now only allows users who are also in the unlimited quota system.
    """
    
    # First check if user is in the unlimited quota system (production)
    try:
        from .unlimited_quota_production import is_unlimited_quota_user
from app.simple_logger import get_logger
        if is_unlimited_quota_user(email):
            return True
    except ImportError:
        pass
    
    # Path to whitelist file (legacy - only for emergency access)
    whitelist_file = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'quota_whitelist.json')
    
    try:
        if os.path.exists(whitelist_file):
            with open(whitelist_file, 'r') as f:
                data = json.load(f)
                whitelist_emails = data.get('whitelist_emails', [])
                # Only allow whitelisted users if they are also in unlimited quota system
                if email.lower() in [e.lower() for e in whitelist_emails]:
                    try:
                        from .unlimited_quota_production import is_unlimited_quota_user
                        return is_unlimited_quota_user(email)
                    except ImportError:
                        return False
    except Exception:
        pass
    
    return False
