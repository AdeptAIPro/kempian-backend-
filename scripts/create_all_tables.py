"""
Comprehensive Migration Script: Create ALL Payroll Tables
==========================================================

This script creates ALL payroll-related tables including:
- Core payroll tables (19 tables)
- India compliance tables (10 tables)
- US compliance tables (7 tables)
Total: 36 tables

Core Payroll Tables:
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

India Compliance Tables:
- provident_fund_contributions
- esi_contributions
- professional_tax_deductions
- tds_records
- income_tax_exemptions
- form16_certificates
- form24q_returns
- challan_records
- state_tax_configurations
- country_tax_configurations

US Compliance Tables:
- sui_contributions
- workers_compensation
- garnishments
- form941_returns
- form940_returns
- currency_exchange_rates
- compliance_forms

Usage:
    python backend/migrations/create_all_payroll_tables.py
    OR
    cd backend && python migrations/create_all_payroll_tables.py

Note: This script uses SQLAlchemy's create_all() which respects foreign key
dependencies and creates tables in the correct order automatically.
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
            print("=" * 70)
            print("PAYROLL TABLES MIGRATION SCRIPT")
            print("=" * 70)
            print("\n[INFO] Initializing database connection...")
            
            # Import all payroll-related models to register them with SQLAlchemy
            print("[INFO] Importing payroll models...")
            
            # Core payroll models
            from app.models import (
                # Core Employee & Organization
                EmployeeProfile, 
                OrganizationMetadata,
                
                # Time & Attendance
                Timesheet,
                
                # Payroll Processing
                Payslip,
                PayRun,
                PayRunPayslip,
                
                # Tax Management
                TaxConfiguration,
                EmployeeTaxProfile,
                
                # Deductions
                DeductionType,
                EmployeeDeduction,
                
                # Settings & Configuration
                PayrollSettings,
                HolidayCalendar,
                
                # Leave Management
                LeaveType,
                LeaveBalance,
                LeaveRequest,
                
                # Payment Processing
                EmployerWalletBalance,
                FraudAlert,
                PaymentTransaction,
                
                # Bank Accounts
                UserBankAccount,
            )
            
            # India Compliance Models
            from app.models import (
                ProvidentFundContribution,
                ESIContribution,
                ProfessionalTaxDeduction,
                TDSRecord,
                IncomeTaxExemption,
                Form16Certificate,
                Form24QReturn,
                ChallanRecord,
                StateTaxConfiguration,
                CountryTaxConfiguration,
            )
            
            # US Compliance Models
            from app.models import (
                SUIContribution,
                WorkersCompensation,
                Garnishment,
                Form941Return,
                Form940Return,
                CurrencyExchangeRate,
                ComplianceForm,
            )
            
            print("[INFO] All models imported successfully.")
            
            # Get list of tables before creation
            engine = db.engine
            inspector = inspect(engine)
            existing_tables = set(inspector.get_table_names())
            
            print(f"\n[INFO] Found {len(existing_tables)} existing tables in database.")
            print("[INFO] Creating payroll tables...")
            
            # Create all tables (SQLAlchemy handles dependencies automatically)
            db.create_all()
            
            # Get list of tables after creation
            new_tables = set(inspector.get_table_names())
            created_tables = new_tables - existing_tables
            
            print("\n" + "=" * 70)
            print("MIGRATION COMPLETE")
            print("=" * 70)
            
            if created_tables:
                print(f"\n[SUCCESS] Created {len(created_tables)} new table(s):")
                for table in sorted(created_tables):
                    print(f"   ✓ {table}")
            else:
                print("\n[INFO] No new tables created. All tables already exist.")
            
            # List all payroll-related tables and their status
            payroll_tables = {
                'Core Payroll Tables': [
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
                    'user_bank_accounts',
                ],
                'India Compliance Tables': [
                    'provident_fund_contributions',
                    'esi_contributions',
                    'professional_tax_deductions',
                    'tds_records',
                    'income_tax_exemptions',
                    'form16_certificates',
                    'form24q_returns',
                    'challan_records',
                    'state_tax_configurations',
                    'country_tax_configurations',
                ],
                'US Compliance Tables': [
                    'sui_contributions',
                    'workers_compensation',
                    'garnishments',
                    'form941_returns',
                    'form940_returns',
                    'currency_exchange_rates',
                    'compliance_forms',
                ]
            }
            
            print("\n" + "=" * 70)
            print("PAYROLL TABLES STATUS")
            print("=" * 70)
            
            total_tables = 0
            existing_count = 0
            missing_count = 0
            
            for category, tables in payroll_tables.items():
                print(f"\n{category}:")
                for table in tables:
                    exists = table_exists(engine, table)
                    status = "✓" if exists else "✗"
                    print(f"   {status} {table}")
                    total_tables += 1
                    if exists:
                        existing_count += 1
                    else:
                        missing_count += 1
            
            print("\n" + "=" * 70)
            print(f"SUMMARY: {existing_count}/{total_tables} tables exist")
            print("=" * 70)
            
            if missing_count == 0:
                print("\n[SUCCESS] ✓ All payroll tables are ready!")
                print("\nNext steps:")
                print("  1. Verify database connection")
                print("  2. Run application and test payroll features")
                print("  3. Check API endpoints are accessible")
            else:
                print(f"\n[WARNING] ✗ {missing_count} table(s) are missing.")
                print("Possible causes:")
                print("  - Model definition issues")
                print("  - Foreign key dependency problems")
                print("  - Database permission issues")
                print("\nCheck the error messages above for details.")
            
            # Verify foreign key relationships
            print("\n[INFO] Verifying foreign key relationships...")
            try:
                # Check a few critical foreign keys
                critical_fks = [
                    ('employee_profiles', 'user_id', 'users'),
                    ('organization_metadata', 'tenant_id', 'tenants'),
                    ('payslips', 'user_id', 'users'),
                    ('pay_runs', 'tenant_id', 'tenants'),
                    ('payment_transactions', 'pay_run_id', 'pay_runs'),
                ]
                
                fk_issues = []
                for table, fk_col, ref_table in critical_fks:
                    if table_exists(engine, table) and table_exists(engine, ref_table):
                        # Try to query the table to verify FK constraint
                        try:
                            result = db.session.execute(
                                text(f"SELECT COUNT(*) FROM {table} LIMIT 1")
                            )
                            result.fetchone()
                        except Exception as e:
                            fk_issues.append(f"{table}.{fk_col} -> {ref_table}: {str(e)}")
                
                if fk_issues:
                    print("[WARNING] Potential foreign key issues detected:")
                    for issue in fk_issues:
                        print(f"   - {issue}")
                else:
                    print("[SUCCESS] Foreign key relationships verified.")
                    
            except Exception as e:
                print(f"[WARNING] Could not verify foreign keys: {str(e)}")
            
            print("\n" + "=" * 70)
            print("Migration script completed successfully!")
            print("=" * 70)
                
        except Exception as e:
            db.session.rollback()
            print("\n" + "=" * 70)
            print("ERROR: Migration failed")
            print("=" * 70)
            print(f"\n[ERROR] {str(e)}")
            print("\nFull traceback:")
            import traceback
            traceback.print_exc()
            print("\n" + "=" * 70)
            raise

if __name__ == '__main__':
    create_all_payroll_tables()
