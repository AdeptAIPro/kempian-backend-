"""
Migration script to create linkedin_recruiter_integrations table

Usage:
    Option 1: Run from backend directory
        python migrations/create_linkedin_recruiter_table.py
    
    Option 2: Run SQL script directly
        mysql -u your_user -p your_database < migrations/create_linkedin_recruiter_table.sql
    
    Option 3: Using Flask CLI (if using Flask-Migrate)
        flask db upgrade
"""

import sys
import os

# Add the backend directory to the path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

def create_table():
    """Create the linkedin_recruiter_integrations table"""
    try:
        from app import create_app, db
        from app.models import LinkedInRecruiterIntegration
        
        app = create_app()
        
        with app.app_context():
            # Create the table
            db.create_all()
            print("✅ Successfully created linkedin_recruiter_integrations table")
            print("\n✅ Table structure:")
            print("   - id (Primary Key, Auto Increment)")
            print("   - user_id (Foreign Key to users, Unique)")
            print("   - client_id, client_secret (VARCHAR 255)")
            print("   - company_id, contract_id (VARCHAR 255)")
            print("   - access_token, refresh_token (TEXT)")
            print("   - token_expires_at (DATETIME)")
            print("   - account_name, account_email (VARCHAR 255)")
            print("   - account_user_id, account_organization_id (VARCHAR 255)")
            print("   - created_at, updated_at (DATETIME)")
            print("\n✅ Indexes created:")
            print("   - idx_user_id")
            print("   - idx_company_id")
            print("   - idx_contract_id")
            return True
    except Exception as e:
        print(f"❌ Error creating table: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = create_table()
    sys.exit(0 if success else 1)

