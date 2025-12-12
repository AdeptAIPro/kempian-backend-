"""
Migration script to create user_module_access table
Run this with: python backend/migrations/create_user_module_access_table.py
"""

import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app import create_app, db
from app.models import UserModuleAccess
from app.simple_logger import get_logger

logger = get_logger("migration")

def create_user_module_access_table():
    """Create the user_module_access table"""
    try:
        app = create_app()
        
        with app.app_context():
            # Check if table already exists
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            existing_tables = inspector.get_table_names()
            
            if 'user_module_access' in existing_tables:
                logger.info("Table 'user_module_access' already exists. Skipping creation.")
                return
            
            # Create the table
            logger.info("Creating user_module_access table...")
            db.create_all()
            logger.info("Successfully created user_module_access table")
            
    except Exception as e:
        logger.error(f"Error creating user_module_access table: {e}")
        raise

if __name__ == '__main__':
    create_user_module_access_table()
    print("Migration completed successfully!")

