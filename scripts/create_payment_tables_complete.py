#!/usr/bin/env python3
"""
Complete Database Table Creation Script for Payroll Payment System

This script creates ALL necessary tables and columns for the payment system.
It handles both new table creation and adding columns to existing tables.

Usage:
    python scripts/create_payment_tables_complete.py

Requirements:
    - Database connection configured in .env
    - SQLAlchemy models imported
    - Database user has CREATE TABLE and ALTER TABLE permissions
"""

import os
import sys
from pathlib import Path
from decimal import Decimal

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from dotenv import load_dotenv
from app import create_app, db
from sqlalchemy import text, inspect
from sqlalchemy.exc import OperationalError, ProgrammingError

# Load environment variables
load_dotenv(os.path.join(backend_dir, '.env'))

def execute_sql(sql, description):
    """Execute SQL with error handling"""
    try:
        db.session.execute(text(sql))
        db.session.commit()
        print(f"   ✅ {description}")
        return True
    except ProgrammingError as e:
        error_msg = str(e).lower()
        if 'duplicate' in error_msg or 'already exists' in error_msg:
            print(f"   ⚠️  {description} (already exists)")
            return True
        else:
            print(f"   ❌ {description}: {e}")
            db.session.rollback()
            return False
    except Exception as e:
        print(f"   ❌ {description}: {e}")
        db.session.rollback()
        return False

def column_exists(table_name, column_name):
    """Check if column exists in table"""
    try:
        inspector = inspect(db.engine)
        columns = [col['name'] for col in inspector.get_columns(table_name)]
        return column_name in columns
    except:
        return False

def table_exists(table_name):
    """Check if table exists"""
    try:
        inspector = inspect(db.engine)
        return table_name in inspector.get_table_names()
    except:
        return False

def create_tables():
    """Create all payment-related tables and columns"""
    app = create_app()
    
    with app.app_context():
        print("=" * 80)
        print("PAYROLL PAYMENT SYSTEM - COMPLETE DATABASE SETUP")
        print("=" * 80)
        print()
        
        try:
            # Test database connection
            print("1. Testing database connection...")
            db.session.execute(text("SELECT 1"))
            print("   ✅ Database connection successful")
            print()
            
            # ========================================================================
            # CREATE NEW TABLES
            # ========================================================================
            print("2. Creating new tables...")
            
            # EmployerWalletBalance
            if not table_exists('employer_wallet_balances'):
                sql = """
                CREATE TABLE employer_wallet_balances (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    tenant_id INT NOT NULL UNIQUE,
                    available_balance DECIMAL(15, 2) NOT NULL DEFAULT 0,
                    locked_balance DECIMAL(15, 2) NOT NULL DEFAULT 0,
                    total_balance DECIMAL(15, 2) NOT NULL DEFAULT 0,
                    razorpay_account_status VARCHAR(50),
                    razorpay_account_id VARCHAR(255),
                    kyc_status VARCHAR(50),
                    last_synced_at DATETIME,
                    sync_error TEXT,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE,
                    INDEX idx_wallet_tenant (tenant_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
                execute_sql(sql, "Created table 'employer_wallet_balances'")
            else:
                print("   ⚠️  Table 'employer_wallet_balances' already exists")

            # PayRun (must be created before payment_transactions and fraud_alerts)
            if not table_exists('pay_runs'):
                sql = """
                CREATE TABLE pay_runs (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    tenant_id INT NOT NULL,
                    pay_period_start DATE NOT NULL,
                    pay_period_end DATE NOT NULL,
                    pay_date DATE NOT NULL,
                    status ENUM(
                        'draft',
                        'approval_pending',
                        'funds_validated',
                        'payout_initiated',
                        'partially_completed',
                        'completed',
                        'failed',
                        'reversed'
                    ) NOT NULL DEFAULT 'draft',
                    total_gross DECIMAL(12, 2) NOT NULL DEFAULT 0,
                    total_net DECIMAL(12, 2) NOT NULL DEFAULT 0,
                    total_tax DECIMAL(12, 2) NOT NULL DEFAULT 0,
                    total_deductions DECIMAL(12, 2) NOT NULL DEFAULT 0,
                    total_employees INT NOT NULL DEFAULT 0,
                    currency VARCHAR(10) NOT NULL DEFAULT 'USD',
                    notes TEXT,
                    funds_locked BOOLEAN NOT NULL DEFAULT FALSE,
                    funds_locked_at DATETIME,
                    funds_locked_amount DECIMAL(12, 2),
                    payments_initiated INT NOT NULL DEFAULT 0,
                    payments_successful INT NOT NULL DEFAULT 0,
                    payments_failed INT NOT NULL DEFAULT 0,
                    payments_pending INT NOT NULL DEFAULT 0,
                    created_by INT NOT NULL,
                    approved_by INT,
                    funds_validated_by INT,
                    funds_validated_at DATETIME,
                    processed_at DATETIME,
                    correlation_id VARCHAR(255),
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE,
                    FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY (approved_by) REFERENCES users(id) ON DELETE SET NULL,
                    FOREIGN KEY (funds_validated_by) REFERENCES users(id) ON DELETE SET NULL,
                    INDEX idx_payrun_tenant (tenant_id),
                    INDEX idx_payrun_status (status),
                    INDEX idx_payrun_dates (pay_period_start, pay_period_end),
                    INDEX idx_correlation_id (correlation_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
                execute_sql(sql, "Created table 'pay_runs'")
            else:
                print("   ⚠️  Table 'pay_runs' already exists")

            # Check for required dependencies
            if not table_exists('payslips'):
                print("   ⚠️  WARNING: Table 'payslips' does not exist. Payment transactions require it.")
                print("      Please run: python migrations/create_payroll_tables.py")
                print("      Skipping payment_transactions and fraud_alerts creation.")
            else:
                # PaymentTransaction (create after pay_runs and payslips since it references them)
                if not table_exists('payment_transactions'):
                    sql = """
                    CREATE TABLE payment_transactions (
                        id INT PRIMARY KEY AUTO_INCREMENT,
                        pay_run_id INT NOT NULL,
                        payslip_id INT NOT NULL,
                        employee_id INT NOT NULL,
                        tenant_id INT NOT NULL,
                        amount DECIMAL(12, 2) NOT NULL,
                        currency VARCHAR(10) NOT NULL DEFAULT 'INR',
                        payment_mode VARCHAR(20) NOT NULL,
                        beneficiary_name VARCHAR(255) NOT NULL,
                        account_number VARCHAR(50) NOT NULL,
                        ifsc_code VARCHAR(11),
                        bank_name VARCHAR(255),
                        gateway VARCHAR(50),
                        gateway_transaction_id VARCHAR(255),
                        gateway_payout_id VARCHAR(255),
                        gateway_response JSON,
                        idempotency_key VARCHAR(255) UNIQUE,
                        retry_count INT NOT NULL DEFAULT 0,
                        max_retries INT NOT NULL DEFAULT 3,
                        last_retry_at DATETIME,
                        purpose_code VARCHAR(20) NOT NULL DEFAULT 'SALARY',
                        payout_category VARCHAR(20) NOT NULL DEFAULT 'salary',
                        fraud_risk_score DECIMAL(5, 2),
                        fraud_flags JSON,
                        requires_manual_review BOOLEAN NOT NULL DEFAULT FALSE,
                        reviewed_by INT,
                        reviewed_at DATETIME,
                        review_notes TEXT,
                        status VARCHAR(50) NOT NULL DEFAULT 'pending',
                        failure_reason TEXT,
                        initiated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        processed_at DATETIME,
                        completed_at DATETIME,
                        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        FOREIGN KEY (pay_run_id) REFERENCES pay_runs(id) ON DELETE CASCADE,
                        FOREIGN KEY (payslip_id) REFERENCES payslips(id) ON DELETE CASCADE,
                        FOREIGN KEY (employee_id) REFERENCES users(id) ON DELETE CASCADE,
                        FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE,
                        FOREIGN KEY (reviewed_by) REFERENCES users(id) ON DELETE SET NULL,
                        INDEX idx_payment_payrun (pay_run_id),
                        INDEX idx_payment_employee (employee_id),
                        INDEX idx_payment_status (status),
                        INDEX idx_idempotency_key (idempotency_key),
                        INDEX idx_fraud_review (requires_manual_review)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                    execute_sql(sql, "Created table 'payment_transactions'")
                else:
                    print("   ⚠️  Table 'payment_transactions' already exists")

                # FraudAlert (create after payment_transactions since it references it)
                if not table_exists('fraud_alerts'):
                    sql = """
                    CREATE TABLE fraud_alerts (
                        id INT PRIMARY KEY AUTO_INCREMENT,
                        tenant_id INT NOT NULL,
                        pay_run_id INT,
                        payment_transaction_id INT,
                        employee_id INT NOT NULL,
                        alert_type VARCHAR(50) NOT NULL,
                        severity VARCHAR(20) NOT NULL,
                        risk_score DECIMAL(5, 2) NOT NULL,
                        flags JSON,
                        status VARCHAR(20) NOT NULL DEFAULT 'pending',
                        reviewed_by INT,
                        reviewed_at DATETIME,
                        review_notes TEXT,
                        created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                        FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE,
                        FOREIGN KEY (pay_run_id) REFERENCES pay_runs(id) ON DELETE SET NULL,
                        FOREIGN KEY (payment_transaction_id) REFERENCES payment_transactions(id) ON DELETE SET NULL,
                        FOREIGN KEY (employee_id) REFERENCES users(id) ON DELETE CASCADE,
                        FOREIGN KEY (reviewed_by) REFERENCES users(id) ON DELETE SET NULL,
                        INDEX idx_fraud_tenant (tenant_id),
                        INDEX idx_fraud_payrun (pay_run_id),
                        INDEX idx_fraud_status (status),
                        INDEX idx_fraud_employee (employee_id)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                    """
                    execute_sql(sql, "Created table 'fraud_alerts'")
                else:
                    print("   ⚠️  Table 'fraud_alerts' already exists")

            print()
            
            # ========================================================================
            # ADD COLUMNS TO EXISTING TABLES
            # ========================================================================
            print("3. Adding columns to existing tables...")
            
            # user_bank_accounts
            if table_exists('user_bank_accounts'):
                print("   Processing user_bank_accounts...")
                
                columns_to_add = {
                    'ifsc_code': "ALTER TABLE user_bank_accounts ADD COLUMN ifsc_code VARCHAR(11) NULL",
                    'account_type': "ALTER TABLE user_bank_accounts ADD COLUMN account_type VARCHAR(20) NULL",
                    'bank_branch': "ALTER TABLE user_bank_accounts ADD COLUMN bank_branch VARCHAR(255) NULL",
                    'bank_address': "ALTER TABLE user_bank_accounts ADD COLUMN bank_address TEXT NULL",
                    'consent_given_at': "ALTER TABLE user_bank_accounts ADD COLUMN consent_given_at DATETIME NULL",
                    'consent_ip': "ALTER TABLE user_bank_accounts ADD COLUMN consent_ip VARCHAR(45) NULL",
                    'verified_by_penny_drop': "ALTER TABLE user_bank_accounts ADD COLUMN verified_by_penny_drop BOOLEAN NOT NULL DEFAULT FALSE",
                    'verification_reference_id': "ALTER TABLE user_bank_accounts ADD COLUMN verification_reference_id VARCHAR(255) NULL",
                    'verification_date': "ALTER TABLE user_bank_accounts ADD COLUMN verification_date DATETIME NULL",
                    'last_updated_by': "ALTER TABLE user_bank_accounts ADD COLUMN last_updated_by INT NULL, ADD FOREIGN KEY (last_updated_by) REFERENCES users(id) ON DELETE SET NULL",
                    'bank_change_cooldown_until': "ALTER TABLE user_bank_accounts ADD COLUMN bank_change_cooldown_until DATETIME NULL",
                }
                
                for col_name, sql in columns_to_add.items():
                    if not column_exists('user_bank_accounts', col_name):
                        execute_sql(sql, f"Added column '{col_name}' to user_bank_accounts")
                    else:
                        print(f"      ⚠️  Column '{col_name}' already exists")
            else:
                print("   ⚠️  Table 'user_bank_accounts' does not exist (skipping)")
            
            # payroll_settings
            if table_exists('payroll_settings'):
                print("   Processing payroll_settings...")
                
                columns_to_add = {
                    'payment_gateway': "ALTER TABLE payroll_settings ADD COLUMN payment_gateway VARCHAR(50) NULL",
                    'razorpay_key_id': "ALTER TABLE payroll_settings ADD COLUMN razorpay_key_id VARCHAR(255) NULL",
                    'razorpay_key_secret': "ALTER TABLE payroll_settings ADD COLUMN razorpay_key_secret TEXT NULL",
                    'razorpay_webhook_secret': "ALTER TABLE payroll_settings ADD COLUMN razorpay_webhook_secret VARCHAR(255) NULL",
                    'razorpay_fund_account_id': "ALTER TABLE payroll_settings ADD COLUMN razorpay_fund_account_id VARCHAR(255) NULL",
                    'payment_mode': "ALTER TABLE payroll_settings ADD COLUMN payment_mode VARCHAR(20) NOT NULL DEFAULT 'NEFT'",
                }
                
                for col_name, sql in columns_to_add.items():
                    if not column_exists('payroll_settings', col_name):
                        execute_sql(sql, f"Added column '{col_name}' to payroll_settings")
                    else:
                        print(f"      ⚠️  Column '{col_name}' already exists")
            else:
                print("   ⚠️  Table 'payroll_settings' does not exist (skipping)")
            
            # pay_runs
            if table_exists('pay_runs'):
                print("   Processing pay_runs...")
                
                columns_to_add = {
                    'funds_locked': "ALTER TABLE pay_runs ADD COLUMN funds_locked BOOLEAN NOT NULL DEFAULT FALSE",
                    'funds_locked_at': "ALTER TABLE pay_runs ADD COLUMN funds_locked_at DATETIME NULL",
                    'funds_locked_amount': "ALTER TABLE pay_runs ADD COLUMN funds_locked_amount DECIMAL(12, 2) NULL",
                    'payments_initiated': "ALTER TABLE pay_runs ADD COLUMN payments_initiated INT NOT NULL DEFAULT 0",
                    'payments_successful': "ALTER TABLE pay_runs ADD COLUMN payments_successful INT NOT NULL DEFAULT 0",
                    'payments_failed': "ALTER TABLE pay_runs ADD COLUMN payments_failed INT NOT NULL DEFAULT 0",
                    'payments_pending': "ALTER TABLE pay_runs ADD COLUMN payments_pending INT NOT NULL DEFAULT 0",
                    'funds_validated_by': "ALTER TABLE pay_runs ADD COLUMN funds_validated_by INT NULL, ADD FOREIGN KEY (funds_validated_by) REFERENCES users(id) ON DELETE SET NULL",
                    'funds_validated_at': "ALTER TABLE pay_runs ADD COLUMN funds_validated_at DATETIME NULL",
                    'correlation_id': "ALTER TABLE pay_runs ADD COLUMN correlation_id VARCHAR(255) NULL",
                }
                
                for col_name, sql in columns_to_add.items():
                    if not column_exists('pay_runs', col_name):
                        execute_sql(sql, f"Added column '{col_name}' to pay_runs")
                    else:
                        print(f"      ⚠️  Column '{col_name}' already exists")
            else:
                print("   ⚠️  Table 'pay_runs' does not exist (skipping)")
            
            # payment_transactions (add missing columns if table exists)
            if table_exists('payment_transactions'):
                print("   Processing payment_transactions...")
                
                columns_to_add = {
                    'idempotency_key': "ALTER TABLE payment_transactions ADD COLUMN idempotency_key VARCHAR(255) NULL",
                    'retry_count': "ALTER TABLE payment_transactions ADD COLUMN retry_count INT NOT NULL DEFAULT 0",
                    'max_retries': "ALTER TABLE payment_transactions ADD COLUMN max_retries INT NOT NULL DEFAULT 3",
                    'last_retry_at': "ALTER TABLE payment_transactions ADD COLUMN last_retry_at DATETIME NULL",
                    'purpose_code': "ALTER TABLE payment_transactions ADD COLUMN purpose_code VARCHAR(20) NOT NULL DEFAULT 'SALARY'",
                    'payout_category': "ALTER TABLE payment_transactions ADD COLUMN payout_category VARCHAR(20) NOT NULL DEFAULT 'salary'",
                    'fraud_risk_score': "ALTER TABLE payment_transactions ADD COLUMN fraud_risk_score DECIMAL(5, 2) NULL",
                    'fraud_flags': "ALTER TABLE payment_transactions ADD COLUMN fraud_flags JSON NULL",
                    'requires_manual_review': "ALTER TABLE payment_transactions ADD COLUMN requires_manual_review BOOLEAN NOT NULL DEFAULT FALSE",
                    'reviewed_by': "ALTER TABLE payment_transactions ADD COLUMN reviewed_by INT NULL, ADD FOREIGN KEY (reviewed_by) REFERENCES users(id) ON DELETE SET NULL",
                    'reviewed_at': "ALTER TABLE payment_transactions ADD COLUMN reviewed_at DATETIME NULL",
                    'review_notes': "ALTER TABLE payment_transactions ADD COLUMN review_notes TEXT NULL",
                }
                
                for col_name, sql in columns_to_add.items():
                    if not column_exists('payment_transactions', col_name):
                        execute_sql(sql, f"Added column '{col_name}' to payment_transactions")
                    else:
                        print(f"      ⚠️  Column '{col_name}' already exists")
            else:
                print("   ⚠️  Table 'payment_transactions' does not exist (created above)")
            
            print()
            
            # ========================================================================
            # CREATE INDEXES
            # ========================================================================
            print("4. Creating indexes...")
            
            indexes = [
                ("CREATE INDEX idx_cooldown ON user_bank_accounts(bank_change_cooldown_until)", "Index on bank_change_cooldown_until"),
                ("CREATE INDEX idx_verification ON user_bank_accounts(verified_by_penny_drop)", "Index on verified_by_penny_drop"),
                ("CREATE INDEX idx_payment_tenant ON payment_transactions(tenant_id)", "Index on payment_transactions.tenant_id"),
                ("CREATE INDEX idx_payment_initiated ON payment_transactions(initiated_at)", "Index on payment_transactions.initiated_at"),
            ]
            
            for index_sql, description in indexes:
                try:
                    db.session.execute(text(index_sql))
                    db.session.commit()
                    print(f"   ✅ {description}")
                except Exception as e:
                    if 'duplicate' in str(e).lower() or 'already exists' in str(e).lower():
                        print(f"   ⚠️  {description} (already exists)")
                    else:
                        print(f"   ⚠️  {description}: {e}")
            
            print()
            
            # ========================================================================
            # UPDATE STATUS ENUM (if MySQL)
            # ========================================================================
            print("5. Updating status enums...")
            
            # Try to update pay_runs status enum (MySQL specific)
            try:
                sql = """
                ALTER TABLE pay_runs 
                MODIFY COLUMN status ENUM(
                    'draft',
                    'approval_pending',
                    'funds_validated',
                    'payout_initiated',
                    'partially_completed',
                    'completed',
                    'failed',
                    'reversed'
                ) NOT NULL DEFAULT 'draft';
                """
                execute_sql(sql, "Updated pay_runs.status enum")
            except Exception as e:
                print(f"   ⚠️  Could not update status enum (may not be MySQL or already updated): {e}")
            
            print()
            
            # ========================================================================
            # VERIFICATION
            # ========================================================================
            print("6. Verifying setup...")
            
            inspector = inspect(db.engine)
            tables = inspector.get_table_names()
            
            required_tables = ['employer_wallet_balances', 'fraud_alerts', 'payment_transactions']
            all_exist = all(table in tables for table in required_tables)
            
            if all_exist:
                print("   ✅ All required tables exist")
            else:
                missing = [t for t in required_tables if t not in tables]
                print(f"   ⚠️  Missing tables: {', '.join(missing)}")
            
            print()
            print("=" * 80)
            print("DATABASE SETUP COMPLETE")
            print("=" * 80)
            print()
            print("✅ Next Steps:")
            print("   1. Configure Razorpay keys in PayrollSettings")
            print("   2. Set up webhook: /api/hr/payments/webhooks/razorpay")
            print("   3. Configure cron jobs (reconciliation, balance sync)")
            print("   4. Test with ₹1 payment before production")
            print("   5. Review PAYROLL_INCIDENT_RUNBOOK.md")
            print("   6. Set up alerts and monitoring")
            print()
            
        except OperationalError as e:
            print(f"❌ Database connection failed: {e}")
            print()
            print("Please check:")
            print("1. Database server is running")
            print("2. DATABASE_URL in .env is correct")
            print("3. Database user has CREATE/ALTER TABLE permissions")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            db.session.rollback()
            sys.exit(1)

if __name__ == '__main__':
    create_tables()

