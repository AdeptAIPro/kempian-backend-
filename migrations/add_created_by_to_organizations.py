"""
Migration Script: Add created_by column to organization_metadata table
This script adds the created_by field to track who created each organization.

Usage:
    python backend/migrations/add_created_by_to_organizations.py
    OR
    cd backend && python migrations/add_created_by_to_organizations.py
"""
import sys
import os

# Add backend directory to path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app import create_app, db
from sqlalchemy import text, inspect

def add_created_by_column():
    """Add created_by column to organization_metadata table if it doesn't exist"""
    app = create_app()
    
    with app.app_context():
        try:
            print("[INFO] Checking organization_metadata table...")
            
            # Check if column already exists
            inspector = inspect(db.engine)
            columns = [col['name'] for col in inspector.get_columns('organization_metadata')]
            
            if 'created_by' in columns:
                print("[INFO] Column 'created_by' already exists. Skipping migration.")
                return
            
            print("[INFO] Adding 'created_by' column to organization_metadata table...")
            
            # Add the column
            db.session.execute(text("""
                ALTER TABLE organization_metadata 
                ADD COLUMN created_by INTEGER NULL,
                ADD CONSTRAINT fk_organization_metadata_created_by 
                FOREIGN KEY (created_by) REFERENCES users(id)
            """))
            
            db.session.commit()
            
            print("\n[SUCCESS] Successfully added 'created_by' column!")
            print("[INFO] The column is now available for tracking organization creators.")
            
        except Exception as e:
            db.session.rollback()
            print(f"\n[ERROR] Error adding column: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

if __name__ == '__main__':
    add_created_by_column()

