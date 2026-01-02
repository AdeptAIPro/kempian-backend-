"""
Migration Script: Create ALL Payroll Tables
Run this script to create ALL tables for complete payroll functionality:
- employee_profiles
- organization_metadata
- timesheets
- payslips
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
- employer_wallet_balances
- fraud_alerts
- payment_transactions
- user_bank_accounts

Usage:
    python backend/migrations/create_all_payroll_tables.py
    OR
    cd backend && python migrations/create_all_payroll_tables.py
"""
import sys
import os

# Add backend directory to path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app import create_app, db
from sqlalchemy import text, inspect

def table_exists(engine, table_name):
    """Check if a table exists in the database"""
    inspector = inspect(engine)
    return table_name in inspector.get_table_names()

def create_all_payroll_tables():
    """Create all payroll tables using SQLAlchemy models"""
    app = create_app()
    
    with app.app_context():
        try:
            print("[INFO] Creating all payroll tables...")
            
            # Import all payroll-related models to register them with SQLAlchemy
            from app.models import (
                # Core payroll models
                EmployeeProfile, OrganizationMetadata, Timesheet, Payslip,
                # Tax & Deductions
                TaxConfiguration, EmployeeTaxProfile,
                DeductionType, EmployeeDeduction,
                # Pay Run Processing
                PayRun, PayRunPayslip,
                # Settings & Configuration
                PayrollSettings, HolidayCalendar,
                # Leave Management
                LeaveType, LeaveBalance, LeaveRequest,
                # Payment Processing
                EmployerWalletBalance, FraudAlert, PaymentTransaction,
                # Bank Accounts
                UserBankAccount
            )
            
            # Get list of tables before creation
            engine = db.engine
            existing_tables = set(inspect(engine).get_table_names())
            
            # Create all tables
            db.create_all()
            
            # Get list of tables after creation
            new_tables = set(inspect(engine).get_table_names())
            created_tables = new_tables - existing_tables
            
            print("\n[SUCCESS] Payroll tables setup complete!")
            
            if created_tables:
                print("\n[CREATED] New tables:")
                for table in sorted(created_tables):
                    print(f"   + {table}")
            else:
                print("\n[INFO] All tables already exist.")
            
            # List all payroll-related tables
            payroll_tables = [
                'employee_profiles',
                'organization_metadata', 
                'timesheets',
                'payslips',
                'tax_configurations',
                'employee_tax_profiles',
                'deduction_types',
                'employee_deductions',
                'pay_runs',
                'pay_run_payslips',
                'payroll_settings',
                'holiday_calendars',
                'leave_types',
                'leave_balances',
                'leave_requests',
                'employer_wallet_balances',
                'fraud_alerts',
                'payment_transactions',
                'user_bank_accounts'
            ]
            
            print("\n[STATUS] Payroll table status:")
            all_exist = True
            for table in payroll_tables:
                exists = table_exists(engine, table)
                status = "✓" if exists else "✗"
                print(f"   {status} {table}")
                if not exists:
                    all_exist = False
            
            if all_exist:
                print("\n[SUCCESS] All payroll tables are ready!")
            else:
                print("\n[WARNING] Some tables are missing. Check for model definition issues.")
                
        except Exception as e:
            db.session.rollback()
            print(f"\n[ERROR] Error creating tables: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

if __name__ == '__main__':
    create_all_payroll_tables()

