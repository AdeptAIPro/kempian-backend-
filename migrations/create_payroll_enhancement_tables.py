"""
Migration Script: Create Payroll Enhancement Tables
Run this script to create the necessary tables for enhanced payroll functionality:
- tax_configurations
- employee_tax_profiles
- deduction_types
- employee_deductions
- pay_runs
- pay_run_payslips
- payroll_settings
- holiday_calendars
- leave_types
- leave_balances
- leave_requests

Usage:
    python backend/migrations/create_payroll_enhancement_tables.py
    OR
    cd backend && python migrations/create_payroll_enhancement_tables.py
"""
import sys
import os

# Add backend directory to path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app import create_app, db
from sqlalchemy import text

def create_payroll_enhancement_tables():
    """Create payroll enhancement tables using SQLAlchemy models"""
    app = create_app()
    
    with app.app_context():
        try:
            print("[INFO] Creating payroll enhancement tables...")
            
            # Import models to register them with SQLAlchemy
            from app.models import (
                TaxConfiguration, EmployeeTaxProfile,
                DeductionType, EmployeeDeduction,
                PayRun, PayRunPayslip,
                PayrollSettings, HolidayCalendar,
                LeaveType, LeaveBalance, LeaveRequest
            )
            
            # Create all tables
            db.create_all()
            
            print("\n[SUCCESS] Successfully created payroll enhancement tables!")
            print("   - tax_configurations")
            print("   - employee_tax_profiles")
            print("   - deduction_types")
            print("   - employee_deductions")
            print("   - pay_runs")
            print("   - pay_run_payslips")
            print("   - payroll_settings")
            print("   - holiday_calendars")
            print("   - leave_types")
            print("   - leave_balances")
            print("   - leave_requests")
            print("\n[INFO] Tables are ready for use.")
            
        except Exception as e:
            db.session.rollback()
            print(f"\n[ERROR] Error creating tables: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

if __name__ == '__main__':
    create_payroll_enhancement_tables()

