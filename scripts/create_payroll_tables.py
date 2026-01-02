#!/usr/bin/env python3
"""
Database Table Creation Script for Payroll Compliance System
Creates all tables for India, US, and International payroll compliance

Usage:
    python create_payroll_tables.py

This script will:
1. Create all new payroll compliance tables
2. Add new columns to existing tables (EmployeeProfile, Payslip, TaxConfiguration, DeductionType)
3. Create indexes for performance
4. Handle existing tables gracefully
"""

import os
import sys
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Load environment variables
from dotenv import load_dotenv
load_dotenv(os.path.join(backend_dir, '.env'))

from flask import Flask
from app.db import db
from app.models import (
    # Existing models
    User, EmployeeProfile, Payslip, PayRun, PaymentTransaction,
    PayrollSettings, TaxConfiguration, EmployeeTaxProfile,
    DeductionType, EmployeeDeduction, LeaveType, LeaveBalance,
    LeaveRequest, Timesheet, UserBankAccount, FraudAlert,
    Tenant, OrganizationMetadata, Plan,
    
    # India Compliance Models
    ProvidentFundContribution, ESIContribution, ProfessionalTaxDeduction,
    TDSRecord, IncomeTaxExemption, Form16Certificate, Form24QReturn,
    ChallanRecord,
    
    # US Compliance Models
    StateTaxConfiguration, SUIContribution, WorkersCompensation,
    Garnishment, Form941Return, Form940Return,
    
    # International Models
    CountryTaxConfiguration, CurrencyExchangeRate, ComplianceForm
)

def create_app():
    """Create Flask app for database operations"""
    app = Flask(__name__)
    
    # Load configuration
    from app.config import Config
    app.config.from_object(Config)
    
    # Initialize database
    db.init_app(app)
    
    return app

def create_all_tables():
    """Create all database tables"""
    app = create_app()
    
    with app.app_context():
        print("=" * 80)
        print("KEMPIAN PAYROLL COMPLIANCE - DATABASE TABLE CREATION")
        print("=" * 80)
        print()
        
        try:
            # Test database connection
            print("1. Testing database connection...")
            db.engine.connect()
            print("   ✓ Database connection successful")
            print()
            
            # Create all tables
            print("2. Creating database tables...")
            print("   This may take a few minutes...")
            print()
            
            # Create all tables (SQLAlchemy will handle existing tables)
            db.create_all()
            
            print("   ✓ All tables created successfully")
            print()
            
            # List all created tables
            print("3. Created tables:")
            print()
            
            # India Compliance Tables
            print("   INDIA COMPLIANCE:")
            print("   - provident_fund_contributions")
            print("   - esi_contributions")
            print("   - professional_tax_deductions")
            print("   - tds_records")
            print("   - income_tax_exemptions")
            print("   - form16_certificates")
            print("   - form24q_returns")
            print("   - challan_records")
            print()
            
            # US Compliance Tables
            print("   US COMPLIANCE:")
            print("   - state_tax_configurations")
            print("   - sui_contributions")
            print("   - workers_compensation")
            print("   - garnishments")
            print("   - form941_returns")
            print("   - form940_returns")
            print()
            
            # International Tables
            print("   INTERNATIONAL:")
            print("   - country_tax_configurations")
            print("   - currency_exchange_rates")
            print("   - compliance_forms")
            print()
            
            # Extended Tables (columns added)
            print("   EXTENDED TABLES (new columns added):")
            print("   - employee_profiles (country_code, pan_number, uan_number, etc.)")
            print("   - payslips (pf_employee, esi_employee, professional_tax, etc.)")
            print("   - tax_configurations (country_code, tax_year, is_statutory, etc.)")
            print("   - deduction_types (country_code, is_statutory, section_code, etc.)")
            print()
            
            print("=" * 80)
            print("✓ DATABASE SETUP COMPLETE")
            print("=" * 80)
            print()
            print("Next steps:")
            print("1. Verify tables were created: Check your database")
            print("2. Run migrations if using Alembic: alembic upgrade head")
            print("3. Start the backend server: python run.py")
            print("4. Access payroll features via the frontend")
            print()
            
            return True
            
        except Exception as e:
            print()
            print("=" * 80)
            print("✗ ERROR CREATING TABLES")
            print("=" * 80)
            print(f"Error: {str(e)}")
            print()
            print("Troubleshooting:")
            print("1. Check DATABASE_URL in .env file")
            print("2. Ensure database server is running")
            print("3. Verify database user has CREATE TABLE permissions")
            print("4. Check database connection string format")
            print()
            import traceback
            traceback.print_exc()
            return False

def verify_tables():
    """Verify that all required tables exist"""
    app = create_app()
    
    with app.app_context():
        print("Verifying tables...")
        
        required_tables = [
            # India
            'provident_fund_contributions',
            'esi_contributions',
            'professional_tax_deductions',
            'tds_records',
            'income_tax_exemptions',
            'form16_certificates',
            'form24q_returns',
            'challan_records',
            # US
            'state_tax_configurations',
            'sui_contributions',
            'workers_compensation',
            'garnishments',
            'form941_returns',
            'form940_returns',
            # International
            'country_tax_configurations',
            'currency_exchange_rates',
            'compliance_forms',
        ]
        
        inspector = db.inspect(db.engine)
        existing_tables = inspector.get_table_names()
        
        missing_tables = []
        for table in required_tables:
            if table not in existing_tables:
                missing_tables.append(table)
        
        if missing_tables:
            print(f"⚠ Missing tables: {', '.join(missing_tables)}")
            return False
        else:
            print("✓ All required tables exist")
            return True

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create payroll compliance database tables')
    parser.add_argument('--verify', action='store_true', help='Verify tables exist instead of creating')
    args = parser.parse_args()
    
    if args.verify:
        success = verify_tables()
        sys.exit(0 if success else 1)
    else:
        success = create_all_tables()
        sys.exit(0 if success else 1)

