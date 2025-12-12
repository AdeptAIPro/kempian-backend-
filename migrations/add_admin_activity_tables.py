#!/usr/bin/env python3
"""
Database migration to add admin activity logging tables
Run this script to create the admin activity logging tables
"""

import sys
import os
sys.path.append('.')

from app import create_app, db
from app.models import AdminActivityLog, AdminSession
from app.simple_logger import get_logger

logger = get_logger("migration")

def create_admin_activity_tables():
    """Create admin activity logging tables"""
    try:
        app = create_app()
        with app.app_context():
            # Create the tables
            db.create_all()
            
            # Verify tables were created
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            tables = inspector.get_table_names()
            
            if 'admin_activity_logs' in tables and 'admin_sessions' in tables:
                logger.info("✅ Admin activity tables created successfully")
                logger.info(f"Created tables: {[t for t in tables if 'admin' in t]}")
                return True
            else:
                logger.error("❌ Failed to create admin activity tables")
                return False
                
    except Exception as e:
        logger.error(f"❌ Error creating admin activity tables: {e}")
        return False

def main():
    """Main migration function"""
    print("🔄 Creating admin activity logging tables...")
    
    success = create_admin_activity_tables()
    
    if success:
        print("✅ Migration completed successfully!")
        print("📊 Admin activity logging is now enabled")
        print("🔍 You can view activity logs at /admin/activity-logs")
    else:
        print("❌ Migration failed!")
        print("Please check the logs for more details")
        sys.exit(1)

if __name__ == "__main__":
    main()
