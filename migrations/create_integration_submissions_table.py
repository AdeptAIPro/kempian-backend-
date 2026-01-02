"""
Migration script to create integration_submissions table
Run this script to create the table in your database
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.db import db
from app.models import IntegrationSubmission

def create_integration_submissions_table():
    """Create the integration_submissions table"""
    app = create_app()
    
    with app.app_context():
        try:
            # Create the table
            db.create_all()
            
            # Verify table creation
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            tables = inspector.get_table_names()
            
            if 'integration_submissions' in tables:
                print("✅ Successfully created integration_submissions table")
                
                # Show table structure
                columns = inspector.get_columns('integration_submissions')
                print("\nTable structure:")
                for column in columns:
                    print(f"  - {column['name']}: {column['type']}")
            else:
                print("❌ Table integration_submissions was not created")
                
        except Exception as e:
            print(f"❌ Error creating table: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    create_integration_submissions_table()

