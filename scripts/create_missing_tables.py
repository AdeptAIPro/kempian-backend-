#!/usr/bin/env python3
"""
Script to create missing tables identified by check_tables.py

This script will create all tables that are defined in models.py but don't exist in the database.

Usage:
    python create_missing_tables.py
"""

import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app import create_app, db
    # Import all models to ensure relationships are properly set up
    # This must be at module level, not inside a function
    from app.models import *  # noqa: F401, F403
except ImportError as e:
    print(f"Error importing app: {e}")
    print("Make sure you're running this script from the backend directory")
    sys.exit(1)


def create_missing_tables():
    """Create all missing tables in the database."""
    app = create_app()
    
    with app.app_context():
        try:
            print("=" * 80)
            print("CREATING MISSING TABLES")
            print("=" * 80)
            
            # List of missing tables (from check_tables.py output)
            missing_tables = [
                'deduction_types',
                'employee_deductions',
                'employee_tax_profiles',
                'fraud_alerts',
                'holiday_calendars',
                'leave_balances',
                'leave_requests',
                'leave_types',
                'pay_run_payslips',
                'pay_runs',
                'payment_transactions',
                'payroll_settings',
                'tax_configurations',
                'user_activity_logs'
            ]
            
            print(f"\nCreating {len(missing_tables)} missing table(s)...")
            print("-" * 80)
            
            # All models are already imported at module level
            # This ensures foreign key relationships are properly created
            # Create all tables (SQLAlchemy will only create those that don't exist)
            print("\nExecuting db.create_all()...")
            db.create_all()
            
            # Verify tables were created
            from sqlalchemy import inspect
            inspector = inspect(db.engine)
            existing_tables = set(inspector.get_table_names())
            
            created_count = 0
            still_missing = []
            
            for table_name in missing_tables:
                if table_name in existing_tables:
                    print(f"  ✓ Created: {table_name}")
                    created_count += 1
                else:
                    print(f"  ✗ Failed: {table_name}")
                    still_missing.append(table_name)
            
            print("-" * 80)
            print(f"\nSummary:")
            print(f"  Tables created: {created_count}/{len(missing_tables)}")
            
            if still_missing:
                print(f"  Still missing: {len(still_missing)}")
                print("\nTables that could not be created:")
                for table in still_missing:
                    print(f"  - {table}")
                return 1
            else:
                print("\n✅ SUCCESS: All missing tables have been created!")
                print("\nYou can now run 'python check_tables.py' again to verify.")
                return 0
                
        except Exception as e:
            print(f"❌ ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return 1


if __name__ == '__main__':
    print("\n⚠️  WARNING: This script will create missing tables in your database.")
    print("   Make sure you have a database backup before proceeding.\n")
    
    response = input("Do you want to continue? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        exit_code = create_missing_tables()
        sys.exit(exit_code)
    else:
        print("Operation cancelled.")
        sys.exit(0)

