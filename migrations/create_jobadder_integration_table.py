"""
Migration script to create jobadder_integrations table
Run this script to create the table in your database
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.db import db
from app.models import JobAdderIntegration

def create_jobadder_integration_table():
    """Create the jobadder_integrations table"""
    app = create_app()
    
    with app.app_context():
        try:
            print("Creating jobadder_integrations table...")
            
            # Create the table
            JobAdderIntegration.__table__.create(db.engine, checkfirst=True)
            
            # Verify table creation
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            tables = inspector.get_table_names()
            
            if 'jobadder_integrations' in tables:
                print("✅ Successfully created jobadder_integrations table")
                
                # Show table structure
                columns = inspector.get_columns('jobadder_integrations')
                print("\nTable structure:")
                for column in columns:
                    print(f"  - {column['name']}: {column['type']}")
            else:
                print("❌ Table jobadder_integrations was not created")
                print(f"Available tables: {tables}")
                
        except Exception as e:
            print(f"❌ Error creating table: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    create_jobadder_integration_table()

