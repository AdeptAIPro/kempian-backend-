#!/usr/bin/env python3
"""
Quota Whitelist Management Script
This script manages a whitelist of email accounts that are exempt from quota restrictions.
"""

import os
import sys
import json
from datetime import datetime

# Add the backend directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app import create_app, db
from app.models import User, Tenant, Plan, JDSearchLog, TenantAlert

# Default whitelist emails (you can modify this list)
DEFAULT_WHITELIST_EMAILS = [
   
    "vapfull@gmail.com",
    "api@adeptaipro.com",
    # Add more emails as needed
]

WHITELIST_FILE = os.path.join(os.path.dirname(__file__), 'quota_whitelist.json')

def load_whitelist():
    """Load whitelist from JSON file"""
    try:
        if os.path.exists(WHITELIST_FILE):
            with open(WHITELIST_FILE, 'r') as f:
                data = json.load(f)
                return data.get('whitelist_emails', [])
        else:
            # Create default whitelist file
            save_whitelist(DEFAULT_WHITELIST_EMAILS)
            return DEFAULT_WHITELIST_EMAILS
    except Exception as e:
        print(f"Error loading whitelist: {e}")
        return []

def save_whitelist(emails):
    """Save whitelist to JSON file"""
    try:
        data = {
            'whitelist_emails': emails,
            'last_updated': datetime.utcnow().isoformat(),
            'description': 'Email accounts exempt from quota restrictions'
        }
        with open(WHITELIST_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✅ Whitelist saved to {WHITELIST_FILE}")
    except Exception as e:
        print(f"❌ Error saving whitelist: {e}")

def is_email_whitelisted(email):
    """Check if an email is in the whitelist"""
    whitelist = load_whitelist()
    return email.lower() in [e.lower() for e in whitelist]

def add_to_whitelist(email):
    """Add an email to the whitelist"""
    whitelist = load_whitelist()
    if email.lower() not in [e.lower() for e in whitelist]:
        whitelist.append(email)
        save_whitelist(whitelist)
        print(f"✅ Added {email} to whitelist")
    else:
        print(f"⚠️ {email} is already in whitelist")

def remove_from_whitelist(email):
    """Remove an email from the whitelist"""
    whitelist = load_whitelist()
    original_length = len(whitelist)
    whitelist = [e for e in whitelist if e.lower() != email.lower()]
    if len(whitelist) < original_length:
        save_whitelist(whitelist)
        print(f"✅ Removed {email} from whitelist")
    else:
        print(f"⚠️ {email} not found in whitelist")

def list_whitelist():
    """List all whitelisted emails"""
    whitelist = load_whitelist()
    if whitelist:
        print("📋 Whitelisted emails:")
        for i, email in enumerate(whitelist, 1):
            print(f"  {i}. {email}")
    else:
        print("📋 No emails in whitelist")

def check_user_quota_status():
    """Check quota status for all users and identify whitelisted ones"""
    app = create_app()
    with app.app_context():
        users = User.query.all()
        print("👥 User Quota Status:")
        print("-" * 60)
        
        for user in users:
            is_whitelisted = is_email_whitelisted(user.email)
            status = "🟢 WHITELISTED" if is_whitelisted else "🔴 QUOTA APPLIED"
            print(f"{user.email:<30} | {status}")
        
        print("-" * 60)

def create_quota_exemption_function():
    """Create a function that can be imported to check quota exemption"""
    function_code = '''
def is_quota_exempt(email):
    """
    Check if an email is exempt from quota restrictions.
    This function should be imported and used in the search routes.
    """
    import os
    import json
    
    # Path to whitelist file
    whitelist_file = os.path.join(os.path.dirname(__file__), 'scripts', 'quota_whitelist.json')
    
    try:
        if os.path.exists(whitelist_file):
            with open(whitelist_file, 'r') as f:
                data = json.load(f)
                whitelist_emails = data.get('whitelist_emails', [])
                return email.lower() in [e.lower() for e in whitelist_emails]
    except Exception:
        pass
    
    return False
'''
    
    # Save the function to a file that can be imported
    function_file = os.path.join(os.path.dirname(__file__), '..', 'app', 'utils', 'quota_exemption.py')
    os.makedirs(os.path.dirname(function_file), exist_ok=True)
    
    with open(function_file, 'w') as f:
        f.write(function_code)
    
    print(f"✅ Created quota exemption function at {function_file}")

def show_usage():
    """Show script usage"""
    print("""
🔧 Quota Whitelist Management Script

Usage:
  python quota_whitelist.py [command] [email]

Commands:
  list                    - Show all whitelisted emails
  add <email>            - Add email to whitelist
  remove <email>         - Remove email from whitelist
  check                  - Check quota status for all users
  create-function        - Create quota exemption function
  help                   - Show this help message

Examples:
  python quota_whitelist.py list
  python quota_whitelist.py add admin@company.com
  python quota_whitelist.py remove test@company.com
  python quota_whitelist.py check
  python quota_whitelist.py create-function
""")

def main():
    """Main function"""
    if len(sys.argv) < 2:
        show_usage()
        return
    
    command = sys.argv[1].lower()
    
    if command == 'list':
        list_whitelist()
    
    elif command == 'add':
        if len(sys.argv) < 3:
            print("❌ Please provide an email address")
            return
        email = sys.argv[2]
        add_to_whitelist(email)
    
    elif command == 'remove':
        if len(sys.argv) < 3:
            print("❌ Please provide an email address")
            return
        email = sys.argv[2]
        remove_from_whitelist(email)
    
    elif command == 'check':
        check_user_quota_status()
    
    elif command == 'create-function':
        create_quota_exemption_function()
    
    elif command == 'help':
        show_usage()
    
    else:
        print(f"❌ Unknown command: {command}")
        show_usage()

if __name__ == "__main__":
    main()
