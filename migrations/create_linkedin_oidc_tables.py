#!/usr/bin/env python3
"""
Database migration script to create LinkedIn OIDC authentication tables
Run this script to create the users_linkedin table for storing LinkedIn OAuth tokens

Usage:
    python backend/migrations/create_linkedin_oidc_tables.py
    OR
    cd backend && python migrations/create_linkedin_oidc_tables.py
"""

import sys
import os

# Add backend directory to path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app import create_app, db
from app.models import UserLinkedIn
from app.simple_logger import get_logger

logger = get_logger("migration")

def create_linkedin_oidc_tables():
    """Create LinkedIn OIDC authentication tables"""
    app = create_app()
    
    with app.app_context():
        try:
            logger.info("Creating LinkedIn OIDC tables...")
            
            # Create the table
            db.create_all()
            
            # Verify table was created
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            tables = inspector.get_table_names()
            
            if 'users_linkedin' in tables:
                logger.info("✅ users_linkedin table created successfully")
                
                # Check columns
                columns = [col['name'] for col in inspector.get_columns('users_linkedin')]
                logger.info(f"Table columns: {', '.join(columns)}")
                
                return True
            else:
                logger.error("❌ users_linkedin table not found")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error creating LinkedIn OIDC tables: {e}")
            import traceback
            traceback.print_exc()
            db.session.rollback()
            return False

if __name__ == '__main__':
    success = create_linkedin_oidc_tables()
    if success:
        print("\n✅ Migration completed successfully!")
        print("   - users_linkedin table created")
        print("\n[INFO] Table is ready for LinkedIn OIDC authentication.")
    else:
        print("\n❌ Migration failed. Check logs for details.")
        sys.exit(1)

