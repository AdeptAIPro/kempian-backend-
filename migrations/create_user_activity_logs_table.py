"""
Migration Script: Create User Activity Logs Table
Run this script to create the user_activity_logs table for tracking all user activities.

Usage:
    python backend/migrations/create_user_activity_logs_table.py
    OR
    cd backend && python migrations/create_user_activity_logs_table.py
"""
import sys
import os

# Add backend directory to path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app import create_app, db
from sqlalchemy import inspect, text

def create_user_activity_logs_table():
    """Create user_activity_logs table using SQLAlchemy models"""
    app = create_app()
    
    with app.app_context():
        try:
            print("=" * 80)
            print("USER ACTIVITY LOGS - DATABASE TABLE CREATION")
            print("=" * 80)
            print()
            
            # Test database connection
            print("[1/4] Testing database connection...")
            db.session.execute(text("SELECT 1"))
            print("   ✅ Database connection successful")
            print()
            
            # Check if table exists
            print("[2/4] Checking existing tables...")
            inspector = inspect(db.engine)
            existing_tables = inspector.get_table_names()
            
            if 'user_activity_logs' in existing_tables:
                print("   ⚠️  Table 'user_activity_logs' already exists")
                print()
                
                # Show current table structure
                print("[3/4] Current table structure:")
                columns = inspector.get_columns('user_activity_logs')
                for column in columns:
                    nullable = "NULL" if column['nullable'] else "NOT NULL"
                    print(f"   - {column['name']}: {column['type']} ({nullable})")
                print()
                
                # Check indexes
                indexes = inspector.get_indexes('user_activity_logs')
                if indexes:
                    print("   Indexes:")
                    for idx in indexes:
                        print(f"   - {idx['name']}: {', '.join(idx['column_names'])}")
                print()
                
                print("[4/4] Table already exists. No action needed.")
                print("   ✅ user_activity_logs table is ready for use.")
                return True
            else:
                print(f"   Found {len(existing_tables)} existing tables")
                print("   Table 'user_activity_logs' does not exist - will create it")
                print()
            
            # Import model to register it with SQLAlchemy
            print("[3/4] Importing UserActivityLog model...")
            from app.models import UserActivityLog
            print("   ✅ Model imported successfully")
            print()
            
            # Create the table
            print("[4/4] Creating user_activity_logs table...")
            UserActivityLog.__table__.create(db.engine, checkfirst=True)
            print("   ✅ Table created successfully")
            print()
            
            # Verify table creation
            print("[VERIFICATION] Verifying table structure...")
            inspector = inspect(db.engine)
            if 'user_activity_logs' in inspector.get_table_names():
                columns = inspector.get_columns('user_activity_logs')
                print(f"   ✅ Table 'user_activity_logs' created with {len(columns)} columns")
                print()
                print("   Table columns:")
                for column in columns:
                    nullable = "NULL" if column['nullable'] else "NOT NULL"
                    default = f" DEFAULT {column['default']}" if column.get('default') else ""
                    print(f"   - {column['name']}: {column['type']} ({nullable}){default}")
                print()
                
                # Show indexes
                indexes = inspector.get_indexes('user_activity_logs')
                if indexes:
                    print("   Indexes created:")
                    for idx in indexes:
                        unique = "UNIQUE" if idx['unique'] else ""
                        print(f"   - {idx['name']} ({unique}): {', '.join(idx['column_names'])}")
                print()
                
                print("=" * 80)
                print("✅ SUCCESS: user_activity_logs table created successfully!")
                print("=" * 80)
                print()
                print("The table is now ready to track user activities across the platform.")
                print("You can start using the @log_user_activity decorator in your routes.")
                return True
            else:
                print("   ❌ Table was not created")
                return False
                
        except Exception as e:
            db.session.rollback()
            print()
            print("=" * 80)
            print("❌ ERROR: Failed to create table")
            print("=" * 80)
            print(f"Error: {str(e)}")
            print()
            import traceback
            traceback.print_exc()
            return False

if __name__ == '__main__':
    success = create_user_activity_logs_table()
    sys.exit(0 if success else 1)

