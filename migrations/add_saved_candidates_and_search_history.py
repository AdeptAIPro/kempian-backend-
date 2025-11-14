#!/usr/bin/env python3
"""
Database migration script to add SavedCandidate and SearchHistory tables
Run this script to create the new tables for saved candidates and search history functionality
"""

import os
import sys
from datetime import datetime, timedelta

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.db import db
from app.models import SavedCandidate, SearchHistory

def create_tables():
    """Create the new tables"""
    app = create_app()
    
    with app.app_context():
        try:
            # Create the tables
            db.create_all()
            print("✅ Successfully created SavedCandidate and SearchHistory tables")
            
            # Verify tables were created
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            tables = inspector.get_table_names()
            
            if 'saved_candidates' in tables:
                print("✅ saved_candidates table created successfully")
            else:
                print("❌ saved_candidates table not found")
                
            if 'search_history' in tables:
                print("✅ search_history table created successfully")
            else:
                print("❌ search_history table not found")
                
        except Exception as e:
            print(f"❌ Error creating tables: {str(e)}")
            return False
            
    return True

def verify_tables():
    """Verify the table structure"""
    app = create_app()
    
    with app.app_context():
        try:
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            
            # Check saved_candidates table structure
            if 'saved_candidates' in inspector.get_table_names():
                columns = inspector.get_columns('saved_candidates')
                print("\n📋 saved_candidates table structure:")
                for col in columns:
                    print(f"  - {col['name']}: {col['type']}")
                    
            # Check search_history table structure
            if 'search_history' in inspector.get_table_names():
                columns = inspector.get_columns('search_history')
                print("\n📋 search_history table structure:")
                for col in columns:
                    print(f"  - {col['name']}: {col['type']}")
                    
        except Exception as e:
            print(f"❌ Error verifying tables: {str(e)}")
            return False
            
    return True

def main():
    """Main migration function"""
    print("🚀 Starting database migration for SavedCandidate and SearchHistory tables...")
    print("=" * 60)
    
    # Create tables
    if create_tables():
        print("\n🔍 Verifying table structure...")
        if verify_tables():
            print("\n✅ Migration completed successfully!")
            print("\n📝 Next steps:")
            print("1. Restart your backend server")
            print("2. Test the save candidate functionality")
            print("3. Test the search history functionality")
        else:
            print("\n❌ Migration completed but table verification failed")
    else:
        print("\n❌ Migration failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
