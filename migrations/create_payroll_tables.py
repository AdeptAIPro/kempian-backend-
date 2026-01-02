"""
Migration Script: Create Payroll Tables
Run this script to create the necessary tables for payroll functionality:
- employee_profiles
- organization_metadata
- timesheets
- payslips

Usage:
    python backend/migrations/create_payroll_tables.py
    OR
    cd backend && python migrations/create_payroll_tables.py
"""
import sys
import os

# Add backend directory to path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app import create_app, db
from sqlalchemy import text

def create_payroll_tables():
    """Create payroll tables using SQLAlchemy models"""
    app = create_app()
    
    with app.app_context():
        try:
            print("[INFO] Creating payroll tables...")
            
            # Import models to register them with SQLAlchemy
            from app.models import EmployeeProfile, OrganizationMetadata, Timesheet, Payslip
            
            # Create all tables
            db.create_all()
            
            print("\n[SUCCESS] Successfully created payroll tables!")
            print("   - employee_profiles")
            print("   - organization_metadata")
            print("   - timesheets")
            print("   - payslips")
            print("\n[INFO] Tables are ready for use.")
            
        except Exception as e:
            db.session.rollback()
            print(f"\n[ERROR] Error creating tables: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

if __name__ == '__main__':
    create_payroll_tables()

